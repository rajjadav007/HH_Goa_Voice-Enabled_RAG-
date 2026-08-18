"""RAG Orchestrator coordinating input guardrails, hybrid retrieval, reranking, retrieval guardrails, generation, and grounding validation."""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from generation.gemini.context_builder import ContextBuilder
from generation.gemini.models import RAGResponse
from generation.gemini.service import GeminiService
from guardrails.grounding.service import GroundingValidationService
from guardrails.input.service import InputGuardrailService
from guardrails.retrieval.service import RetrievalGuardrailService
from orchestration.models import (
    ErrorCode,
    RAGOrchestrationResponse,
    RAGOrchestratorConfig,
    generate_request_id,
)
from retrieval.hybrid.service import HybridService
from retrieval.reranking.models import RerankedResultPoint
from retrieval.reranking.service import RerankerService

logger = logging.getLogger(__name__)


class RAGOrchestrator:
    """Central production RAG orchestrator coordinating text RAG pipeline with input, retrieval, and grounding guardrails."""

    def __init__(
        self,
        config: Optional[RAGOrchestratorConfig] = None,
        guardrail_service: Optional[InputGuardrailService] = None,
        retrieval_guardrail_service: Optional[RetrievalGuardrailService] = None,
        grounding_validation_service: Optional[GroundingValidationService] = None,
        hybrid_service: Optional[HybridService] = None,
        reranker_service: Optional[RerankerService] = None,
        gemini_service: Optional[GeminiService] = None,
    ):
        self.config = config or RAGOrchestratorConfig()
        self.guardrail_service = guardrail_service or InputGuardrailService()
        self.retrieval_guardrail_service = retrieval_guardrail_service or RetrievalGuardrailService()
        self.grounding_validation_service = grounding_validation_service or GroundingValidationService()
        self.hybrid_service = hybrid_service or HybridService()
        self.reranker_service = reranker_service or RerankerService()
        self.gemini_service = gemini_service or GeminiService()

    def validate_source_integrity(
        self,
        raw_sources: List[Any],
        valid_chunks: List[RerankedResultPoint],
    ) -> List[Dict[str, Any]]:
        """Validate that all cited sources belong to actual retrieved context chunks."""
        valid_chunk_map = {c.chunk_id: c for c in valid_chunks}
        validated_sources: List[Dict[str, Any]] = []

        for s in raw_sources:
            cid = getattr(s, "chunk_id", None) or (s.get("chunk_id") if isinstance(s, dict) else None)
            if cid and cid in valid_chunk_map:
                chunk = valid_chunk_map[cid]
                validated_sources.append({
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "rank": chunk.final_rank,
                })
            else:
                logger.warning(f"Rejected model-fabricated or invalid source chunk_id: '{cid}'.")

        # If cited sources was empty or invalid, include top retrieved chunk attributions
        if not validated_sources and valid_chunks:
            for c in valid_chunks[:3]:
                validated_sources.append({
                    "chunk_id": c.chunk_id,
                    "document_id": c.document_id,
                    "rank": c.final_rank,
                })

        return validated_sources

    def answer(
        self,
        query_text: str,
        request_id: Optional[str] = None,
    ) -> RAGOrchestrationResponse:
        """Execute end-to-end orchestrated RAG workflow with input, retrieval, and grounding guardrails."""
        t_start = time.perf_counter()
        req_id = request_id or generate_request_id()
        timing: Dict[str, float] = {}

        # 1. Input Guardrails Evaluation
        t_grd = time.perf_counter()
        guard_decision = self.guardrail_service.evaluate(query_text)
        timing["input_guardrail_ms"] = round((time.perf_counter() - t_grd) * 1000, 2)

        if not guard_decision.allowed:
            total_ms = round((time.perf_counter() - t_start) * 1000, 2)
            timing["total_latency_ms"] = total_ms
            logger.warning(f"[{req_id}] GUARDRAIL_BLOCKED category='{guard_decision.category.value}' reason='{guard_decision.reason}'")

            return RAGOrchestrationResponse(
                answer=f"I cannot process this request: {guard_decision.reason}",
                grounded=False,
                has_context=False,
                sources=[],
                request_id=req_id,
                status="BLOCKED",
                error_code=guard_decision.category.value,
                latency_ms=total_ms,
                timing_breakdown=timing,
                metadata={"guardrail": guard_decision.to_dict()},
            )

        q_clean = query_text.strip()
        logger.info(f"[{req_id}] RAG_START for query: '{q_clean[:60]}...'")

        # Check overall timeout deadline
        deadline = time.time() + self.config.timeout_sec

        try:
            # 2. Hybrid Retrieval
            t_ret = time.perf_counter()
            if time.time() > deadline:
                raise TimeoutError("RAG request timed out prior to retrieval stage.")

            hybrid_candidates, h_metrics = self.hybrid_service.search(
                query_text=q_clean,
                vector_top_k=self.config.vector_top_k,
                bm25_top_k=self.config.bm25_top_k,
                final_top_k=self.config.candidate_k,
            )
            timing["retrieval_ms"] = round((time.perf_counter() - t_ret) * 1000, 2)
            logger.info(f"[{req_id}] RETRIEVAL_COMPLETE: {len(hybrid_candidates)} candidates.")

            # 3. No-Context Short Circuit (if candidates empty)
            if not hybrid_candidates:
                logger.info(f"[{req_id}] NO_CONTEXT path triggered.")
                return RAGOrchestrationResponse(
                    answer="Insufficient context available to answer the query.",
                    grounded=False,
                    has_context=False,
                    sources=[],
                    request_id=req_id,
                    status="NO_CONTEXT",
                    error_code=ErrorCode.NO_CONTEXT.value,
                    latency_ms=round((time.perf_counter() - t_start) * 1000, 2),
                    timing_breakdown=timing,
                )

            # 4. Reranking
            t_rer = time.perf_counter()
            if time.time() > deadline:
                raise TimeoutError("RAG request timed out prior to reranking stage.")

            reranked_chunks, r_metrics = self.reranker_service.rerank(
                query_text=q_clean,
                candidates=hybrid_candidates,
                candidate_k=self.config.candidate_k,
                final_k=self.config.final_k,
            )
            timing["rerank_ms"] = round((time.perf_counter() - t_rer) * 1000, 2)
            logger.info(f"[{req_id}] RERANK_COMPLETE: {len(reranked_chunks)} reranked chunks.")

            # 5. Retrieval Guardrails Evaluation
            t_rgrd = time.perf_counter()
            ret_guard_decision = self.retrieval_guardrail_service.evaluate(
                query_text=q_clean,
                reranked_chunks=reranked_chunks,
            )
            timing["retrieval_guardrail_ms"] = round((time.perf_counter() - t_rgrd) * 1000, 2)

            if not ret_guard_decision.allowed or not ret_guard_decision.valid_chunks:
                total_ms = round((time.perf_counter() - t_start) * 1000, 2)
                timing["total_latency_ms"] = total_ms
                logger.warning(f"[{req_id}] RETRIEVAL_GUARDRAIL_REJECTED status='{ret_guard_decision.status.value}' reason='{ret_guard_decision.reason}'")

                return RAGOrchestrationResponse(
                    answer="Insufficient context available to answer the query.",
                    grounded=False,
                    has_context=False,
                    sources=[],
                    request_id=req_id,
                    status="NO_CONTEXT",
                    error_code=ret_guard_decision.status.value,
                    latency_ms=total_ms,
                    timing_breakdown=timing,
                    metadata={"retrieval_guardrail": ret_guard_decision.to_dict()},
                )

            valid_chunks = ret_guard_decision.valid_chunks

            # 6. Gemini Generation
            t_gen = time.perf_counter()
            if time.time() > deadline:
                raise TimeoutError("RAG request timed out prior to generation stage.")

            rag_response = self.gemini_service.generate(
                query_text=q_clean,
                chunks=valid_chunks,
            )
            timing["generation_ms"] = round((time.perf_counter() - t_gen) * 1000, 2)

            # 7. Grounding Validation
            t_gnd = time.perf_counter()
            grounding_decision = self.grounding_validation_service.evaluate(
                query_text=q_clean,
                answer_text=rag_response.answer,
                is_grounded_flag=rag_response.grounded,
                context_chunks=valid_chunks,
            )
            timing["grounding_validation_ms"] = round((time.perf_counter() - t_gnd) * 1000, 2)
            logger.info(f"[{req_id}] GROUNDING_VALIDATION: status='{grounding_decision.status.value}' grounded={grounding_decision.grounded}.")

            # 8. Response & Source Validation
            t_val_resp = time.perf_counter()
            validated_sources = self.validate_source_integrity(
                raw_sources=rag_response.sources,
                valid_chunks=valid_chunks,
            )
            timing["response_validation_ms"] = round((time.perf_counter() - t_val_resp) * 1000, 2)

            total_ms = round((time.perf_counter() - t_start) * 1000, 2)
            timing["total_latency_ms"] = total_ms

            logger.info(f"[{req_id}] RAG_COMPLETE in {total_ms}ms.")

            return RAGOrchestrationResponse(
                answer=grounding_decision.validated_answer,
                grounded=grounding_decision.grounded,
                has_context=True,
                sources=validated_sources,
                request_id=req_id,
                status="SUCCESS",
                error_code=None,
                latency_ms=total_ms,
                token_usage=rag_response.token_usage,
                timing_breakdown=timing,
                metadata={
                    "model": rag_response.model,
                    "retrieved_candidates": len(hybrid_candidates),
                    "reranked_chunks": len(reranked_chunks),
                    "valid_chunks": len(valid_chunks),
                    "guardrail": guard_decision.to_dict(),
                    "retrieval_guardrail": ret_guard_decision.to_dict(),
                    "grounding_validation": grounding_decision.to_dict(),
                },
            )

        except TimeoutError as exc:
            total_ms = round((time.perf_counter() - t_start) * 1000, 2)
            logger.error(f"[{req_id}] RAG_TIMEOUT: {exc}")
            return RAGOrchestrationResponse(
                answer="Application Error: Request timed out.",
                grounded=False,
                has_context=False,
                sources=[],
                request_id=req_id,
                status="TIMEOUT",
                error_code=ErrorCode.TIMEOUT.value,
                latency_ms=total_ms,
                timing_breakdown=timing,
            )
        except Exception as exc:
            total_ms = round((time.perf_counter() - t_start) * 1000, 2)
            logger.error(f"[{req_id}] RAG_ERROR: {exc}", exc_info=True)
            return RAGOrchestrationResponse(
                answer="Application Error: Internal processing failure.",
                grounded=False,
                has_context=False,
                sources=[],
                request_id=req_id,
                status="ERROR",
                error_code=ErrorCode.GENERATION_ERROR.value,
                latency_ms=total_ms,
                timing_breakdown=timing,
                metadata={"error_detail": str(exc)},
            )
