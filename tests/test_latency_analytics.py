"""Unit test suite for Latency Analytics and Bottleneck Identification."""

import pytest
from evaluation.analytics.analyzer import LatencyAnalyzer
from evaluation.analytics.schema import LatencyAnalyticsReport
from evaluation.datasets.schema import EvalResultItem, RunMetadata


def test_latency_analyzer_percentiles_and_bottlenecks():
    # Build sample execution records
    sample_results = [
        EvalResultItem(
            test_id=f"t_{i}",
            mode="voice_e2e",
            language="en-IN",
            category="short",
            query="Sample query",
            answer="Sample answer",
            grounded=True,
            grounding_status="GROUNDED",
            has_context=True,
            latency_ms=120.0 + i,
            stt_latency_ms=30.0,
            rag_latency_ms=90.0 + i,
            timing_breakdown={
                "input_guardrail_ms": 2.0,
                "retrieval_ms": 25.0,
                "rerank_ms": 15.0,
                "generation_ms": 40.0 + i,
                "grounding_validation_ms": 8.0,
                "stt_ms": 30.0,
            },
        )
        for i in range(10)
    ]

    meta = RunMetadata(
        run_id="test_run_01",
        timestamp="2026-08-18T20:00:00Z",
        seed=42,
        mode="full",
        offline_mock=True,
    )

    report = LatencyAnalyzer.analyze_results(sample_results, meta)

    assert report.sample_count == 10
    assert report.text_latency.p50 > 0
    assert report.voice_latency.p50 > 0
    assert report.stt_latency.p50 == 30.0

    # Verify < 200 ms target
    assert report.target_200ms_rag.p50_passed is True
    assert report.target_200ms_rag.status == "PASS"

    # Verify primary bottleneck identification (Generation is top stage ~45ms)
    assert "Generation" in report.primary_bottleneck or "Retrieval" in report.primary_bottleneck
    assert report.primary_bottleneck_pct > 0.0
