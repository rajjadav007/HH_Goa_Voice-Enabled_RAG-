"""Unit tests for Phase 4.4 Hybrid Retrieval Service and RRF module."""

import pytest
from unittest.mock import MagicMock

from retrieval.bm25.models import BM25ResultPoint
from retrieval.hybrid.models import HybridConfig, HybridResultPoint
from retrieval.hybrid.rrf import compute_rrf_scores
from retrieval.hybrid.service import HybridService
from retrieval.vector_db.models import SearchResultPoint


def test_compute_rrf_scores_mathematical_correctness():
    """Test RRF fusion score calculation for vector, BM25, and overlapping chunks."""
    v1 = SearchResultPoint(
        chunk_id="chk_A", document_id="doc_1", score=0.9, text="Chunk A text", chunk_index=0, chunk_strategy="semantic"
    )
    v2 = SearchResultPoint(
        chunk_id="chk_B", document_id="doc_2", score=0.8, text="Chunk B text", chunk_index=0, chunk_strategy="semantic"
    )

    b1 = BM25ResultPoint(
        chunk_id="chk_C", document_id="doc_3", score=5.0, rank=1, text="Chunk C text"
    )
    b2 = BM25ResultPoint(
        chunk_id="chk_A", document_id="doc_1", score=4.0, rank=2, text="Chunk A text"
    )

    # RRF with k=60
    # chk_A: vec_rank=1, bm25_rank=2 -> 1/(60+1) + 1/(60+2) = 0.016393 + 0.016129 = 0.032522
    # chk_B: vec_rank=2, bm25_rank=None -> 1/(60+2) = 0.016129
    # chk_C: vec_rank=None, bm25_rank=1 -> 1/(60+1) = 0.016393

    fused = compute_rrf_scores(
        vector_results=[v1, v2],
        bm25_results=[b1, b2],
        rrf_k=60,
        final_top_k=3,
    )

    assert len(fused) == 3
    # chk_A should be rank 1 because it appeared in both lists
    assert fused[0].chunk_id == "chk_A"
    assert fused[0].rank == 1
    assert fused[0].sources == ["vector", "bm25"]
    assert fused[0].vector_rank == 1
    assert fused[0].bm25_rank == 2
    assert fused[0].score == round(1.0 / 61.0 + 1.0 / 62.0, 6)

    # chk_C should be rank 2 (1/61)
    assert fused[1].chunk_id == "chk_C"
    assert fused[1].sources == ["bm25"]

    # chk_B should be rank 3 (1/62)
    assert fused[2].chunk_id == "chk_B"
    assert fused[2].sources == ["vector"]


def test_hybrid_service_partial_failure_handling():
    """Test HybridService handles vector or BM25 failures gracefully."""
    mock_emb = MagicMock()
    mock_qdrant = MagicMock()
    mock_bm25 = MagicMock()

    mock_emb.embed_text.return_value = [0.1] * 384
    mock_qdrant.search.side_effect = Exception("Qdrant connection error")

    mock_bm25.is_loaded = True
    mock_bm25.search.return_value = [
        BM25ResultPoint(chunk_id="chk_bm25", document_id="doc_bm25", score=3.5, rank=1, text="BM25 text")
    ]

    service = HybridService(
        config=HybridConfig(vector_top_k=5, bm25_top_k=5, final_top_k=3),
        embedding_service=mock_emb,
        qdrant_service=mock_qdrant,
        bm25_service=mock_bm25,
    )

    # Search with Qdrant failing
    results, metrics = service.search("sample query", parallel=False)

    assert len(results) == 1
    assert results[0].chunk_id == "chk_bm25"
    assert results[0].sources == ["bm25"]
    assert metrics["vector_candidates"] == 0
    assert metrics["bm25_candidates"] == 1


def test_hybrid_service_both_empty():
    """Test HybridService handles both vector and BM25 returning 0 results."""
    mock_emb = MagicMock()
    mock_qdrant = MagicMock()
    mock_bm25 = MagicMock()

    mock_emb.embed_text.return_value = [0.1] * 384
    mock_qdrant.search.return_value = []
    mock_bm25.is_loaded = True
    mock_bm25.search.return_value = []

    service = HybridService(
        embedding_service=mock_emb,
        qdrant_service=mock_qdrant,
        bm25_service=mock_bm25,
    )

    results, metrics = service.search("nonexistent query")
    assert results == []
    assert metrics["has_relevant_context"] is False
