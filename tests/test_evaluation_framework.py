"""Unit and integration test suite for the evaluation framework."""

import pytest
from pathlib import Path
from evaluation.datasets.loader import EvalDatasetLoader, build_default_eval_cases
from evaluation.datasets.schema import EvalCase
from evaluation.metrics.retrieval import (
    compute_recall_at_k,
    compute_mrr,
    compute_precision_at_k,
    compute_hit_rate,
    aggregate_retrieval_metrics,
)
from evaluation.metrics.generation import (
    compute_exact_match,
    compute_token_f1,
    aggregate_generation_metrics,
)
from evaluation.metrics.latency import compute_latency_percentiles
from evaluation.runner import EvaluationRunner
from evaluation.baseline import record_baseline, load_baseline, check_regression


def test_recall_at_k_calculation():
    ground_truth = ["doc1", "doc2"]
    
    # Hit at rank 1
    retrieved_1 = ["doc1", "doc3", "doc4"]
    assert compute_recall_at_k(retrieved_1, ground_truth, 1) == 0.5
    assert compute_recall_at_k(retrieved_1, ground_truth, 3) == 0.5

    # Hit both in top 3
    retrieved_2 = ["doc1", "doc2", "doc4"]
    assert compute_recall_at_k(retrieved_2, ground_truth, 3) == 1.0


def test_mrr_calculation():
    ground_truth = ["doc_target"]

    # Rank 1 hit -> MRR = 1.0
    assert compute_mrr(["doc_target", "doc_b"], ground_truth) == 1.0

    # Rank 2 hit -> MRR = 0.5
    assert compute_mrr(["doc_a", "doc_target"], ground_truth) == 0.5

    # Rank 5 hit -> MRR = 0.2
    assert compute_mrr(["a", "b", "c", "d", "doc_target"], ground_truth) == 0.2

    # Miss -> MRR = 0.0
    assert compute_mrr(["x", "y", "z"], ground_truth) == 0.0


def test_generation_metrics():
    pred = "A corporation is a legal entity."
    ref = "A corporation is a legal entity."
    assert compute_exact_match(pred, ref) == 1.0
    assert compute_token_f1(pred, ref) == 1.0

    partial_pred = "A corporation is an entity."
    assert compute_exact_match(partial_pred, ref) == 0.0
    assert compute_token_f1(partial_pred, ref) > 0.7


def test_latency_percentiles():
    latencies = [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0]
    percentiles = compute_latency_percentiles(latencies)

    assert percentiles.p50 == 145.0
    assert percentiles.target_200ms_passed is True
    assert percentiles.p100 == 190.0


def test_dataset_loader_filtering():
    loader = EvalDatasetLoader()
    cases = loader.cases
    assert len(cases) == 30

    hi_cases = loader.filter(language="hi-IN")
    assert len(hi_cases) > 0
    assert all(c.language == "hi-IN" for c in hi_cases)

    short_cases = loader.filter(category="short")
    assert len(short_cases) > 0
    assert all(c.category == "short" for c in short_cases)

    sampled = loader.filter(limit=5, seed=42)
    assert len(sampled) == 5


def test_evaluation_runner_mock_mode(tmp_path):
    runner = EvaluationRunner(offline_mock=True, seed=42)
    report = runner.run(mode="full", limit=10)

    assert report.total_cases == 10
    assert report.success_rate == 1.0
    assert report.retrieval.recall_at_1 >= 0.0
    assert report.latency.p50 < 200.0
    assert report.latency.target_200ms_passed is True

    # Test report export
    runner.export_report(report, tmp_path)
    assert (tmp_path / f"{report.metadata.run_id}.json").exists()
    assert (tmp_path / f"{report.metadata.run_id}_summary.csv").exists()


def test_baseline_regression_detection(tmp_path):
    runner = EvaluationRunner(offline_mock=True, seed=42)
    report = runner.run(mode="full", limit=5)

    base_path = tmp_path / "baseline.json"
    record_baseline(report, base_path)
    loaded = load_baseline(base_path)
    assert loaded is not None
    assert loaded.metadata.run_id == report.metadata.run_id

    # Test no regression
    has_reg, _ = check_regression(report, loaded)
    assert has_reg is False
