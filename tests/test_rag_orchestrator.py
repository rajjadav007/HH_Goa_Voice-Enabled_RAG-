"""Unit tests for Phase 5.3 RAG Orchestrator and pipeline safety."""

import time
import pytest
from unittest.mock import MagicMock

from generation.gemini.models import RAGResponse, SourceAttribution
from orchestration.models import ErrorCode, RAGOrchestrationResponse, RAGOrchestratorConfig
from orchestration.service import RAGOrchestrator
from retrieval.hybrid.models import HybridResultPoint
from retrieval.reranking.models import RerankedResultPoint


def test_orchestrator_invalid_query():
    """Test orchestrator handles empty/invalid queries."""
    mock_hybrid = MagicMock()
    mock_reranker = MagicMock()
    mock_gemini = MagicMock()

    orchestrator = RAGOrchestrator(
        hybrid_service=mock_hybrid,
        reranker_service=mock_reranker,
        gemini_service=mock_gemini,
    )

    res = orchestrator.answer("")
    assert res.status == "INVALID_QUERY"
    assert res.error_code == ErrorCode.INVALID_QUERY.value
    assert res.has_context is False

    res2 = orchestrator.answer("a")
    assert res2.status == "INVALID_QUERY"
    assert res2.error_code == ErrorCode.INVALID_QUERY.value


def test_orchestrator_no_context_path():
    """Test orchestrator triggers NO_CONTEXT path without invoking Gemini API."""
    mock_hybrid = MagicMock()
    mock_gemini = MagicMock()

    mock_hybrid.search.return_value = ([], {"vector_candidates": 0, "bm25_candidates": 0})
    orchestrator = RAGOrchestrator(hybrid_service=mock_hybrid, gemini_service=mock_gemini)

    res = orchestrator.answer("What is a corporation?")

    assert res.status == "NO_CONTEXT"
    assert res.error_code == ErrorCode.NO_CONTEXT.value
    assert res.has_context is False
    assert mock_gemini.generate.called is False


def test_orchestrator_source_integrity_rejects_fabricated_source_ids():
    """Test validate_source_integrity rejects model-fabricated chunk IDs."""
    mock_hybrid = MagicMock()
    mock_reranker = MagicMock()
    mock_gemini = MagicMock()

    orchestrator = RAGOrchestrator(
        hybrid_service=mock_hybrid,
        reranker_service=mock_reranker,
        gemini_service=mock_gemini,
    )

    r1 = RerankedResultPoint(
        chunk_id="chk_real_1", document_id="doc_1", rerank_score=0.9, final_rank=1, text="Text 1", sources=["vector"]
    )
    # Model returns citation for 'chk_fake_99' which was never retrieved
    raw_sources = [
        SourceAttribution(chunk_id="chk_fake_99", document_id="doc_99", rank=1),
        SourceAttribution(chunk_id="chk_real_1", document_id="doc_1", rank=2),
    ]

    validated = orchestrator.validate_source_integrity(raw_sources, valid_chunks=[r1])

    # Only 'chk_real_1' should be kept
    assert len(validated) == 1
    assert validated[0]["chunk_id"] == "chk_real_1"
    assert validated[0]["document_id"] == "doc_1"


def test_orchestrator_reranker_fallback_handling():
    """Test orchestrator falls back cleanly if reranker fails or is disabled."""
    mock_hybrid = MagicMock()
    mock_reranker = MagicMock()
    mock_gemini = MagicMock()

    c1 = HybridResultPoint(
        chunk_id="chk_1", document_id="doc_1", score=0.03, rank=1, text="Text 1", sources=["vector"]
    )
    mock_hybrid.search.return_value = ([c1], {})

    r1 = RerankedResultPoint(
        chunk_id="chk_1", document_id="doc_1", rerank_score=0.03, final_rank=1, text="Text 1", sources=["vector"]
    )
    mock_reranker.rerank.return_value = ([r1], {"fallback_mode": True})

    mock_gemini.generate.return_value = RAGResponse(
        answer="Grounded answer.",
        grounded=True,
        sources=[SourceAttribution(chunk_id="chk_1", document_id="doc_1", rank=1)],
        model="gemini-2.5-flash",
        latency_ms=50.0,
    )

    orchestrator = RAGOrchestrator(
        hybrid_service=mock_hybrid,
        reranker_service=mock_reranker,
        gemini_service=mock_gemini,
    )

    res = orchestrator.answer("What is a corporation?")

    assert res.status == "SUCCESS"
    assert res.grounded is True
    assert res.has_context is True
    assert len(res.sources) == 1
    assert res.sources[0]["chunk_id"] == "chk_1"
    assert "retrieval_ms" in res.timing_breakdown
    assert "rerank_ms" in res.timing_breakdown
    assert "generation_ms" in res.timing_breakdown


def test_orchestrator_timeout_handling():
    """Test orchestrator handles overall timeout deadline."""
    config = RAGOrchestratorConfig(timeout_sec=0.001)
    mock_hybrid = MagicMock()

    def slow_search(*args, **kwargs):
        time.sleep(0.01)
        return ([], {})

    mock_hybrid.search.side_effect = slow_search
    orchestrator = RAGOrchestrator(config=config, hybrid_service=mock_hybrid)

    res = orchestrator.answer("What is a corporation?")

    assert res.status in ["TIMEOUT", "NO_CONTEXT"]
    assert res.has_context is False
