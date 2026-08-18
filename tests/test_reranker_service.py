"""Unit tests for Phase 5.1 Reranker Service and fallback mechanisms."""

import pytest
from unittest.mock import MagicMock

from retrieval.hybrid.models import HybridResultPoint
from retrieval.reranking.models import RerankerConfig, RerankedResultPoint
from retrieval.reranking.service import RerankerService


def test_reranker_service_disabled_fallback():
    """Test RerankerService clean fallback when disabled in configuration."""
    config = RerankerConfig(enabled=False, candidate_k=5, final_k=3)
    service = RerankerService(config=config)

    c1 = HybridResultPoint(
        chunk_id="chk_1", document_id="doc_1", score=0.03, rank=1, text="Text 1", sources=["vector"]
    )
    c2 = HybridResultPoint(
        chunk_id="chk_2", document_id="doc_2", score=0.02, rank=2, text="Text 2", sources=["bm25"]
    )

    results, metrics = service.rerank("test query", candidates=[c1, c2])

    assert len(results) == 2
    assert metrics["fallback_mode"] is True
    assert results[0].chunk_id == "chk_1"
    assert results[0].final_rank == 1
    assert results[1].chunk_id == "chk_2"
    assert results[1].final_rank == 2


def test_reranker_service_mock_scoring():
    """Test RerankerService batch prediction, sorting, and final rank assignment."""
    config = RerankerConfig(enabled=True, candidate_k=3, final_k=2)
    service = RerankerService(config=config)

    # Mock CrossEncoder model
    mock_model = MagicMock()
    # Predict scores for 3 candidates: candidate 2 gets highest score (5.0), candidate 1 gets (1.0), candidate 3 gets (0.0)
    mock_model.predict.return_value = [1.0, 5.0, 0.0]
    service.model = mock_model
    service._is_loaded = True

    c1 = HybridResultPoint(chunk_id="chk_1", document_id="doc_1", score=0.03, rank=1, text="Text 1", sources=["vector"])
    c2 = HybridResultPoint(chunk_id="chk_2", document_id="doc_2", score=0.02, rank=2, text="Text 2", sources=["bm25"])
    c3 = HybridResultPoint(chunk_id="chk_3", document_id="doc_3", score=0.01, rank=3, text="Text 3", sources=["vector"])

    results, metrics = service.rerank("test query", candidates=[c1, c2, c3], candidate_k=3, final_k=2)

    assert len(results) == 2
    assert metrics["fallback_mode"] is False
    # Candidate 2 scored highest (5.0), so it becomes final_rank 1
    assert results[0].chunk_id == "chk_2"
    assert results[0].rerank_score == 5.0
    assert results[0].final_rank == 1
    assert results[0].rrf_score == 0.02

    # Candidate 1 scored second (1.0), so it becomes final_rank 2
    assert results[1].chunk_id == "chk_1"
    assert results[1].rerank_score == 1.0
    assert results[1].final_rank == 2


def test_reranker_service_model_exception_fallback():
    """Test RerankerService handles prediction exceptions by falling back to candidate order."""
    config = RerankerConfig(enabled=True)
    service = RerankerService(config=config)

    mock_model = MagicMock()
    mock_model.predict.side_effect = Exception("CUDA out of memory")
    service.model = mock_model
    service._is_loaded = True

    c1 = HybridResultPoint(chunk_id="chk_1", document_id="doc_1", score=0.05, rank=1, text="Text 1", sources=["vector"])

    results, metrics = service.rerank("test query", candidates=[c1])

    assert len(results) == 1
    assert metrics["fallback_mode"] is True
    assert "CUDA out of memory" in metrics["error"]
    assert results[0].chunk_id == "chk_1"
