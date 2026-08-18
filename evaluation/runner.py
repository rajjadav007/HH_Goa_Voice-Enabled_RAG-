"""Reproducible Evaluation Runner for HH Goa Voice-Enabled RAG System."""

import csv
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from evaluation.datasets.loader import EvalDatasetLoader
from evaluation.datasets.schema import (
    AggregateEvaluationReport,
    EvalCase,
    EvalResultItem,
    RunMetadata,
)
from evaluation.metrics import (
    aggregate_generation_metrics,
    aggregate_grounding_metrics,
    aggregate_retrieval_metrics,
    compute_latency_percentiles,
    compute_mrr,
    compute_recall_at_k,
)

logger = logging.getLogger(__name__)


class EvaluationRunner:
    """Core reproducible evaluation runner executing benchmark datasets against RAG pipeline."""

    def __init__(
        self,
        loader: Optional[EvalDatasetLoader] = None,
        offline_mock: bool = False,
        seed: int = 42,
    ):
        self.loader = loader or EvalDatasetLoader()
        self.offline_mock = offline_mock
        self.seed = seed

    def _run_mock_item(self, case: EvalCase, mode: str) -> EvalResultItem:
        """Execute deterministic mock item for offline testing."""
        t_start = time.perf_counter()
        is_voice = mode in ["voice_e2e", "full"]

        # Simulate context retrieval
        has_ctx = case.category not in ["unanswerable", "offtopic", "injection"]
        grounded = has_ctx
        status = "SUCCESS" if has_ctx else "NO_CONTEXT"
        err_cat = None if has_ctx else "INSUFFICIENT_CONTEXT"

        retrieved_docs = case.relevant_document_ids if has_ctx else []
        retrieved_chunks = case.relevant_chunk_ids if has_ctx else []

        ans = case.expected_answer or "No relevant information found in dataset."
        rec_1 = compute_recall_at_k(retrieved_docs, case.relevant_document_ids, 1)
        rec_3 = compute_recall_at_k(retrieved_docs, case.relevant_document_ids, 3)
        mrr = compute_mrr(retrieved_docs, case.relevant_document_ids)

        lat_ms = 45.0 if not is_voice else 110.0
        stt_ms = 35.0 if is_voice else None
        rag_ms = 75.0 if is_voice else 45.0

        return EvalResultItem(
            test_id=case.test_id,
            mode=mode,
            language=case.language,
            category=case.category,
            query=case.query,
            transcript=case.query if is_voice else None,
            answer=ans,
            grounded=grounded,
            grounding_status="GROUNDED" if grounded else "NO_CONTEXT_GROUNDED",
            has_context=has_ctx,
            retrieved_chunk_ids=retrieved_chunks,
            retrieved_doc_ids=retrieved_docs,
            recall_at_1=rec_1,
            recall_at_3=rec_3,
            mrr=mrr,
            latency_ms=lat_ms,
            stt_latency_ms=stt_ms,
            rag_latency_ms=rag_ms,
            status=status,
            error_category=err_cat,
            timing_breakdown={
                "input_guardrail_ms": 2.0,
                "retrieval_ms": 25.0,
                "rerank_ms": 15.0,
                "generation_ms": 45.0 if not is_voice else 80.0,
                "grounding_validation_ms": 8.0,
                "stt_ms": stt_ms or 0.0,
            },
        )

    def _run_real_item(self, case: EvalCase, mode: str, harness: Any, voice_orch: Any) -> EvalResultItem:
        """Execute real service call against production RAG harness or Voice Orchestrator."""
        t_start = time.perf_counter()
        is_voice = mode in ["voice_e2e", "full"]

        try:
            if is_voice and voice_orch:
                dummy_audio = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80\x3e\x00\x00\x00\x7d\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
                v_res = voice_orch.answer(
                    audio_data=dummy_audio,
                    filename=f"{case.test_id}.wav",
                    language_code=case.language,
                )
                transcript = v_res.get("transcript", case.query)
                answer = v_res.get("answer", "")
                grounded = v_res.get("grounded", False)
                grounding_status = v_res.get("grounding_status", "UNGROUNDED")
                has_ctx = v_res.get("has_context", False)
                sources = v_res.get("sources", [])
                retrieved_chunks = [s.get("chunk_id", "") for s in sources]
                retrieved_docs = [s.get("document_id", "") for s in sources]
                tot_ms = v_res.get("latency_ms", round((time.perf_counter() - t_start) * 1000, 2))
                stt_ms = v_res.get("timing_breakdown", {}).get("stt_ms", 0.0)
                rag_ms = v_res.get("timing_breakdown", {}).get("rag_ms", 0.0)
                status = v_res.get("status", "SUCCESS")
                err_code = v_res.get("error_code")
            else:
                rag_res = harness.run(query_text=case.query)
                transcript = None
                answer = rag_res.answer
                grounded = rag_res.grounded
                grounding_status = getattr(rag_res, "grounding_status", "GROUNDED" if grounded else "UNGROUNDED")
                has_ctx = rag_res.has_context
                sources = rag_res.sources
                retrieved_chunks = [s.get("chunk_id", "") for s in sources]
                retrieved_docs = [s.get("document_id", "") for s in sources]
                tot_ms = rag_res.latency_ms
                stt_ms = None
                rag_ms = rag_res.latency_ms
                status = rag_res.status
                err_code = rag_res.error_code

            # Categorize error if failure occurred
            err_cat = None
            if status not in ["COMPLETED", "SUCCESS", "NO_CONTEXT"]:
                err_cat = err_code or status
            elif not has_ctx:
                err_cat = "INSUFFICIENT_CONTEXT"

            rec_1 = compute_recall_at_k(retrieved_docs, case.relevant_document_ids, 1)
            rec_3 = compute_recall_at_k(retrieved_docs, case.relevant_document_ids, 3)
            mrr = compute_mrr(retrieved_docs, case.relevant_document_ids)

            return EvalResultItem(
                test_id=case.test_id,
                mode=mode,
                language=case.language,
                category=case.category,
                query=case.query,
                transcript=transcript,
                answer=answer,
                grounded=grounded,
                grounding_status=grounding_status,
                has_context=has_ctx,
                retrieved_chunk_ids=retrieved_chunks,
                retrieved_doc_ids=retrieved_docs,
                recall_at_1=rec_1,
                recall_at_3=rec_3,
                mrr=mrr,
                latency_ms=tot_ms,
                stt_latency_ms=stt_ms,
                rag_latency_ms=rag_ms,
                status=status,
                error_category=err_cat,
                timing_breakdown=v_res.get("timing_breakdown", {}) if is_voice else (getattr(rag_res, "timing_breakdown", {}) or {}),
            )
        except Exception as exc:
            tot_ms = round((time.perf_counter() - t_start) * 1000, 2)
            return EvalResultItem(
                test_id=case.test_id,
                mode=mode,
                language=case.language,
                category=case.category,
                query=case.query,
                answer=f"Evaluation execution failure: {exc}",
                grounded=False,
                grounding_status="UNGROUNDED",
                has_context=False,
                latency_ms=tot_ms,
                status="ERROR",
                error_category="INTERNAL_ERROR",
            )

    def run(
        self,
        mode: str = "full",
        language: Optional[str] = None,
        category: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> AggregateEvaluationReport:
        """Run evaluation benchmark suite across dataset cases."""
        run_id = f"eval_run_{uuid.uuid4().hex[:8]}"
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        cases = self.loader.filter(
            language=language,
            category=category,
            limit=limit,
            seed=self.seed,
        )

        harness = None
        voice_orch = None
        if not self.offline_mock:
            from orchestration.harness.service import RAGHarness
            from voice.orchestrator import VoiceRAGOrchestrator
            harness = RAGHarness()
            voice_orch = VoiceRAGOrchestrator(rag_harness=harness)

        raw_results: List[EvalResultItem] = []
        for case in cases:
            if self.offline_mock:
                res = self._run_mock_item(case, mode)
            else:
                res = self._run_real_item(case, mode, harness, voice_orch)
            raw_results.append(res)

        # Aggregate Metrics Calculation
        dict_results = [r.model_dump() for r in raw_results]

        # Attach ground truth references to dict results for metric aggregators
        case_map = {c.test_id: c for c in cases}
        for d in dict_results:
            c_obj = case_map.get(d["test_id"])
            if c_obj:
                d["relevant_doc_ids"] = c_obj.relevant_document_ids
                d["relevant_chunk_ids"] = c_obj.relevant_chunk_ids
                d["expected_answer"] = c_obj.expected_answer

        retrieval_metrics = aggregate_retrieval_metrics(dict_results)
        generation_metrics = aggregate_generation_metrics(dict_results)
        grounding_metrics = aggregate_grounding_metrics(dict_results)
        latencies = [r.latency_ms for r in raw_results]
        latency_metrics = compute_latency_percentiles(latencies)

        total_cnt = len(raw_results)
        succ_cnt = sum(1 for r in raw_results if r.status in ["SUCCESS", "COMPLETED", "NO_CONTEXT"])
        succ_rate = round(float(succ_cnt / total_cnt), 4) if total_cnt > 0 else 0.0

        # Language Breakdown
        lang_map: Dict[str, List[EvalResultItem]] = {}
        for r in raw_results:
            lang_map.setdefault(r.language, []).append(r)

        lang_breakdown = {}
        for lang_code, items in lang_map.items():
            l_lats = [i.latency_ms for i in items]
            l_grounded = sum(1 for i in items if i.grounded)
            lang_breakdown[lang_code] = {
                "count": len(items),
                "p50_latency_ms": round(float(np.percentile(l_lats, 50)), 2) if l_lats else 0.0,
                "grounded_rate": round(float(l_grounded / len(items)), 2) if items else 0.0,
            }

        # Category Breakdown
        cat_map: Dict[str, List[EvalResultItem]] = {}
        for r in raw_results:
            cat_map.setdefault(r.category, []).append(r)

        cat_breakdown = {}
        for cat_name, items in cat_map.items():
            c_grounded = sum(1 for i in items if i.grounded)
            cat_breakdown[cat_name] = {
                "count": len(items),
                "grounded_rate": round(float(c_grounded / len(items)), 2) if items else 0.0,
            }

        # Failure Taxonomy
        failure_counts: Dict[str, int] = {}
        for r in raw_results:
            if r.error_category:
                failure_counts[r.error_category] = failure_counts.get(r.error_category, 0) + 1

        meta = RunMetadata(
            run_id=run_id,
            timestamp=timestamp,
            seed=self.seed,
            mode=mode,
            offline_mock=self.offline_mock,
        )

        return AggregateEvaluationReport(
            metadata=meta,
            total_cases=total_cnt,
            successful_cases=succ_cnt,
            success_rate=succ_rate,
            retrieval=retrieval_metrics,
            generation=generation_metrics,
            grounding=grounding_metrics,
            latency=latency_metrics,
            language_breakdown=lang_breakdown,
            category_breakdown=cat_breakdown,
            failure_taxonomy=failure_counts,
        )

    def export_report(self, report: AggregateEvaluationReport, output_dir: Path):
        """Export machine-readable JSON and CSV evaluation reports."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        json_path = output_dir / f"{report.metadata.run_id}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(), f, indent=2, ensure_ascii=False)

        csv_path = output_dir / f"{report.metadata.run_id}_summary.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])
            writer.writerow(["Run ID", report.metadata.run_id])
            writer.writerow(["Total Cases", report.total_cases])
            writer.writerow(["Success Rate", f"{report.success_rate * 100:.1f}%"])
            writer.writerow(["Recall@1", report.retrieval.recall_at_1])
            writer.writerow(["Recall@3", report.retrieval.recall_at_3])
            writer.writerow(["MRR", report.retrieval.mrr])
            writer.writerow(["Grounded Rate", f"{report.grounding.grounded_rate * 100:.1f}%"])
            writer.writerow(["P50 Latency (ms)", report.latency.p50])
            writer.writerow(["P90 Latency (ms)", report.latency.p90])
            writer.writerow(["P95 Latency (ms)", report.latency.p95])
            writer.writerow(["<200ms Target", "PASSED" if report.latency.target_200ms_passed else "FAILED"])
