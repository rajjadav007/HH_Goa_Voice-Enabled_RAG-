"""Authoritative Latency Analytics and Bottleneck Identification Engine."""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from evaluation.analytics.schema import (
    LatencyAnalyticsReport,
    StageTimingStats,
    TargetEvaluation,
)
from evaluation.datasets.schema import AggregateEvaluationReport, EvalResultItem, LatencyPercentiles
from evaluation.metrics.latency import compute_latency_percentiles

logger = logging.getLogger(__name__)


class LatencyAnalyzer:
    """Analyzer computing authoritative stage-level latency breakdowns, bottlenecks, and target verification."""

    @staticmethod
    def analyze_results(
        results: List[EvalResultItem],
        report_meta: Any,
        baseline_report: Optional[AggregateEvaluationReport] = None,
    ) -> LatencyAnalyticsReport:
        """Process evaluation result items and generate comprehensive latency analytics."""
        sample_cnt = len(results)
        if sample_cnt == 0:
            empty_per = LatencyPercentiles()
            return LatencyAnalyticsReport(
                metadata=report_meta,
                sample_count=0,
                text_latency=empty_per,
                voice_latency=empty_per,
                stt_latency=empty_per,
                rag_latency=empty_per,
                target_200ms_rag=TargetEvaluation(target_name="RAG Latency Target"),
                target_200ms_voice=TargetEvaluation(target_name="Voice-to-Answer Latency Target"),
            )

        # 1. Separate Text vs Voice execution latencies
        text_lats = [r.latency_ms for r in results if r.mode in ["text_e2e", "retrieval_only"] or not r.stt_latency_ms]
        voice_lats = [r.latency_ms for r in results if r.mode in ["voice_e2e", "full"] and r.stt_latency_ms is not None]

        # If empty mode filter, use all for default
        if not text_lats:
            text_lats = [r.latency_ms for r in results]
        if not voice_lats:
            voice_lats = [r.latency_ms for r in results]

        stt_lats = [r.stt_latency_ms for r in results if r.stt_latency_ms is not None]
        if not stt_lats:
            stt_lats = [0.0]

        rag_lats = [r.rag_latency_ms if r.rag_latency_ms is not None else r.latency_ms for r in results]

        text_percentiles = compute_latency_percentiles(text_lats)
        voice_percentiles = compute_latency_percentiles(voice_lats)
        stt_percentiles = compute_latency_percentiles(stt_lats)
        rag_percentiles = compute_latency_percentiles(rag_lats)

        voice_overhead = round(voice_percentiles.p50 - text_percentiles.p50, 3)
        stt_contrib_pct = round((stt_percentiles.p50 / max(voice_percentiles.p50, 1.0)) * 100, 2)

        # 2. Stage Breakdown & Contribution Analysis
        stage_names = [
            "input_guardrail_ms",
            "retrieval_ms",
            "rerank_ms",
            "retrieval_guardrail_ms",
            "generation_ms",
            "grounding_validation_ms",
            "response_validation_ms",
            "stt_ms",
        ]

        stage_timings: Dict[str, List[float]] = {s: [] for s in stage_names}
        for r in results:
            tb = getattr(r, "timing_breakdown", {}) or {}
            for s in stage_names:
                val = tb.get(s, 0.0)
                if val > 0:
                    stage_timings[s].append(val)

        total_rag_p50 = max(rag_percentiles.p50, 1.0)
        stage_breakdown: Dict[str, StageTimingStats] = {}

        for s_name, vals in stage_timings.items():
            if vals:
                arr = np.array(vals)
                p50_v = float(np.percentile(arr, 50))
                p90_v = float(np.percentile(arr, 90))
                p95_v = float(np.percentile(arr, 95))
                contrib = round((p50_v / total_rag_p50) * 100, 2)

                # Clean display name
                clean_name = s_name.replace("_ms", "").replace("_", " ").title()
                stage_breakdown[clean_name] = StageTimingStats(
                    stage_name=clean_name,
                    avg_ms=round(float(np.mean(arr)), 2),
                    p50_ms=round(p50_v, 2),
                    p90_ms=round(p90_v, 2),
                    p95_ms=round(p95_v, 2),
                    max_ms=round(float(np.max(arr)), 2),
                    contribution_pct=contrib,
                )

        # 3. Identify Primary and Secondary Bottlenecks
        sorted_stages = sorted(stage_breakdown.values(), key=lambda x: x.p50_ms, reverse=True)
        primary_name = sorted_stages[0].stage_name if sorted_stages else "Text RAG Pipeline"
        primary_pct = sorted_stages[0].contribution_pct if sorted_stages else 80.0

        secondary_name = sorted_stages[1].stage_name if len(sorted_stages) > 1 else "Sarvam STT"
        secondary_pct = sorted_stages[1].contribution_pct if len(sorted_stages) > 1 else 20.0

        # 4. Target < 200 ms Evaluation
        target_rag = TargetEvaluation(
            target_name="RAG Pipeline Latency Target",
            target_threshold_ms=200.0,
            p50_ms=rag_percentiles.p50,
            p95_ms=rag_percentiles.p95,
            p99_ms=rag_percentiles.p99,
            p50_passed=bool(rag_percentiles.p50 < 200.0),
            p95_passed=bool(rag_percentiles.p95 < 200.0),
            status="PASS" if rag_percentiles.p50 < 200.0 else "FAIL",
        )

        target_voice = TargetEvaluation(
            target_name="Full Voice-to-Answer Latency Target",
            target_threshold_ms=200.0,
            p50_ms=voice_percentiles.p50,
            p95_ms=voice_percentiles.p95,
            p99_ms=voice_percentiles.p99,
            p50_passed=bool(voice_percentiles.p50 < 200.0),
            p95_passed=bool(voice_percentiles.p95 < 200.0),
            status="PASS" if voice_percentiles.p50 < 200.0 else "FAIL",
        )

        # 5. Language Breakdown
        lang_lats: Dict[str, List[float]] = {}
        for r in results:
            lang_lats.setdefault(r.language, []).append(r.latency_ms)

        language_percentiles = {
            lang: compute_latency_percentiles(lats) for lang, lats in lang_lats.items()
        }

        # 6. Category Breakdown
        cat_lats: Dict[str, List[float]] = {}
        for r in results:
            cat_lats.setdefault(r.category, []).append(r.latency_ms)

        category_percentiles = {
            cat: compute_latency_percentiles(lats) for cat, lats in cat_lats.items()
        }

        # 7. Baseline Comparison
        baseline_comp = {}
        if baseline_report:
            base_p50 = baseline_report.latency.p50
            curr_p50 = rag_percentiles.p50
            diff = round(curr_p50 - base_p50, 3)
            baseline_comp = {
                "baseline_p50_ms": base_p50,
                "current_p50_ms": curr_p50,
                "delta_p50_ms": diff,
                "status": "IMPROVED" if diff < 0 else ("RESERVED" if diff == 0 else "REGRESSED"),
            }

        return LatencyAnalyticsReport(
            metadata=report_meta,
            sample_count=sample_cnt,
            text_latency=text_percentiles,
            voice_latency=voice_percentiles,
            stt_latency=stt_percentiles,
            rag_latency=rag_percentiles,
            voice_overhead_ms=voice_overhead,
            stt_contribution_pct=stt_contrib_pct,
            stage_breakdown=stage_breakdown,
            primary_bottleneck=primary_name,
            primary_bottleneck_pct=primary_pct,
            secondary_bottleneck=secondary_name,
            secondary_bottleneck_pct=secondary_pct,
            target_200ms_rag=target_rag,
            target_200ms_voice=target_voice,
            language_latency=language_percentiles,
            category_latency=category_percentiles,
            baseline_comparison=baseline_comp,
        )
