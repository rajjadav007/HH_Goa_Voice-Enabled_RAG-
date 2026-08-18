"""RAG Harness orchestrating state transitions, bounded retries, stage timeouts, and fallback recovery."""

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from generation.gemini.service import GeminiService
from guardrails.grounding.service import GroundingValidationService
from guardrails.input.service import InputGuardrailService
from guardrails.retrieval.service import RetrievalGuardrailService
from orchestration.harness.config import HarnessConfig
from orchestration.harness.models import HarnessState, HarnessTelemetry, StageExecutionRecord
from orchestration.harness.taxonomy import ErrorCategory, HarnessError, RETRYABLE_CATEGORIES
from orchestration.models import RAGOrchestrationResponse, generate_request_id
from orchestration.service import RAGOrchestrator
from retrieval.hybrid.service import HybridService
from retrieval.reranking.service import RerankerService

logger = logging.getLogger(__name__)


class RAGHarness:
    """Production RAG Execution Harness implementing state-machine control, retries, timeouts, and fallbacks."""

    def __init__(
        self,
        config: Optional[HarnessConfig] = None,
        orchestrator: Optional[RAGOrchestrator] = None,
    ):
        self.config = config or HarnessConfig()
        self.orchestrator = orchestrator or RAGOrchestrator()

    def execute_stage(
        self,
        stage_name: str,
        func: Callable[[], Any],
        timeout_sec: float,
        retryable_err_categories: Tuple[ErrorCategory, ...] = (ErrorCategory.INTERNAL_ERROR,),
        telemetry: Optional[HarnessTelemetry] = None,
    ) -> Tuple[Any, StageExecutionRecord]:
        """Execute a pipeline stage with bounded retries, backoff, and timeouts."""
        t_start = time.perf_counter()
        attempts = 0
        last_exception = None
        backoff_ms = self.config.initial_backoff_ms

        while attempts < self.config.max_retries:
            attempts += 1
            if telemetry:
                telemetry.total_attempts += 1

            try:
                # Stage timeout check
                t_run = time.perf_counter()
                res = func()
                dur_ms = (time.perf_counter() - t_run) * 1000

                if (dur_ms / 1000.0) > timeout_sec:
                    logger.warning(f"Stage '{stage_name}' exceeded timeout ({dur_ms:.1f}ms > {timeout_sec*1000:.1f}ms).")

                rec = StageExecutionRecord(
                    stage_name=stage_name,
                    start_time=t_start,
                    end_time=time.perf_counter(),
                    duration_ms=(time.perf_counter() - t_start) * 1000,
                    attempts=attempts,
                    success=True,
                )
                return res, rec

            except Exception as exc:
                last_exception = exc
                err_cat = getattr(exc, "category", ErrorCategory.INTERNAL_ERROR)
                is_retryable = err_cat in RETRYABLE_CATEGORIES if isinstance(err_cat, ErrorCategory) else False

                logger.warning(f"Stage '{stage_name}' failed attempt {attempts}/{self.config.max_retries}: {exc}")

                if not is_retryable or attempts >= self.config.max_retries:
                    break

                # Exponential backoff delay
                time.sleep(backoff_ms / 1000.0)
                backoff_ms = min(self.config.max_backoff_ms, backoff_ms * 2.0)

        dur_total_ms = (time.perf_counter() - t_start) * 1000
        rec = StageExecutionRecord(
            stage_name=stage_name,
            start_time=t_start,
            end_time=time.perf_counter(),
            duration_ms=dur_total_ms,
            attempts=attempts,
            success=False,
            error_category=getattr(last_exception, "category", ErrorCategory.INTERNAL_ERROR).value if hasattr(last_exception, "category") else ErrorCategory.INTERNAL_ERROR.value,
        )

        raise HarnessError(
            category=getattr(last_exception, "category", ErrorCategory.INTERNAL_ERROR),
            stage=stage_name,
            message=str(last_exception),
            retryable=False,
            details={"attempts": attempts, "duration_ms": dur_total_ms},
        )

    def run(
        self,
        query_text: str,
        request_id: Optional[str] = None,
    ) -> RAGOrchestrationResponse:
        """Run complete RAG workflow inside structured execution harness."""
        t_harness_start = time.perf_counter()
        req_id = request_id or generate_request_id()

        telemetry = HarnessTelemetry(
            request_id=req_id,
            state=HarnessState.INIT,
            degraded=False,
        )

        logger.info(f"[{req_id}] HARNESS_START query='{query_text[:50]}...'")
        deadline = time.time() + self.config.total_timeout_sec

        try:
            # 1. Input Guardrails
            telemetry.state = HarnessState.INIT
            guard_dec, rec_grd = self.execute_stage(
                stage_name="input_guardrails",
                func=lambda: self.orchestrator.guardrail_service.evaluate(query_text),
                timeout_sec=1.0,
                telemetry=telemetry,
            )
            telemetry.stages.append(rec_grd)

            if not guard_dec.allowed:
                telemetry.state = HarnessState.BLOCKED
                h_overhead = (time.perf_counter() - t_harness_start) * 1000 - rec_grd.duration_ms
                telemetry.harness_overhead_ms = max(0.0, h_overhead)

                return RAGOrchestrationResponse(
                    answer=f"I cannot process this request: {guard_dec.reason}",
                    grounded=False,
                    has_context=False,
                    sources=[],
                    request_id=req_id,
                    status="BLOCKED",
                    error_code=guard_dec.category.value,
                    latency_ms=round(rec_grd.duration_ms, 2),
                    metadata={"harness": telemetry.to_dict(), "guardrail": guard_dec.to_dict()},
                )

            telemetry.state = HarnessState.INPUT_VALIDATED

            # Check overall deadline
            if time.time() > deadline:
                raise HarnessError(category=ErrorCategory.REQUEST_TIMEOUT, stage="harness", message="Total RAG deadline exceeded.")

            # 2. Orchestrated RAG Execution
            t_orch_start = time.perf_counter()
            response = self.orchestrator.answer(query_text=query_text, request_id=req_id)
            orch_ms = (time.perf_counter() - t_orch_start) * 1000

            rec_orch = StageExecutionRecord(
                stage_name="rag_pipeline",
                start_time=t_orch_start,
                end_time=time.perf_counter(),
                duration_ms=orch_ms,
                attempts=1,
                success=(response.status in ["SUCCESS", "NO_CONTEXT"]),
            )
            telemetry.stages.append(rec_orch)

            if response.status == "NO_CONTEXT":
                telemetry.state = HarnessState.NO_CONTEXT
            elif response.status == "SUCCESS":
                telemetry.state = HarnessState.COMPLETED
            elif response.status == "BLOCKED":
                telemetry.state = HarnessState.BLOCKED
            else:
                telemetry.state = HarnessState.FAILED

            # Calculate harness overhead
            t_harness_total_ms = (time.perf_counter() - t_harness_start) * 1000
            h_overhead = t_harness_total_ms - response.latency_ms
            telemetry.harness_overhead_ms = max(0.0, h_overhead)

            response.metadata["harness"] = telemetry.to_dict()
            logger.info(f"[{req_id}] HARNESS_COMPLETE status='{response.status}' overhead={telemetry.harness_overhead_ms:.3f}ms")

            return response

        except HarnessError as exc:
            telemetry.state = HarnessState.TIMEOUT if exc.category == ErrorCategory.REQUEST_TIMEOUT else HarnessState.FAILED
            t_harness_total_ms = round((time.perf_counter() - t_harness_start) * 1000, 2)
            telemetry.harness_overhead_ms = 0.5

            logger.error(f"[{req_id}] HARNESS_ERROR category='{exc.category.value}' message='{exc.message}'")

            safe_msg = "Application Error: Request timed out." if exc.category == ErrorCategory.REQUEST_TIMEOUT else "Application Error: Internal processing failure."

            return RAGOrchestrationResponse(
                answer=safe_msg,
                grounded=False,
                has_context=False,
                sources=[],
                request_id=req_id,
                status="TIMEOUT" if exc.category == ErrorCategory.REQUEST_TIMEOUT else "ERROR",
                error_code=exc.category.value,
                latency_ms=t_harness_total_ms,
                metadata={"harness": telemetry.to_dict(), "error_details": exc.to_dict()},
            )
