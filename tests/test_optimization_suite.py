"""Unit test suite for optimization experiment runner and LRU caching."""

import pytest
from evaluation.optimization.experiment import OptimizationSuite
from retrieval.embeddings.service import EmbeddingService


def test_embedding_service_lru_cache():
    svc = EmbeddingService()
    query = "What is a corporation?"

    # First call: computes embedding
    v1 = svc.embed_text(query, is_query=True)
    assert len(v1) == svc.dimension
    assert query in svc._query_cache

    # Second call: returns cached vector instantly
    v2 = svc.embed_text(query, is_query=True)
    assert v1 == v2


def test_optimization_suite_experiment_acceptance(tmp_path):
    suite = OptimizationSuite(output_dir=tmp_path)
    record = suite.run_experiment(
        experiment_id="exp_test_01",
        hypothesis="Test hypothesis for parallel query execution.",
        target_component="HybridService",
    )

    assert record.decision in ["ACCEPTED", "REJECTED"]
    assert (tmp_path / "exp_test_01.json").exists()
