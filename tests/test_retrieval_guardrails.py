"""Unit tests for Phase 6.2 Retrieval Guardrails layer and Orchestrator safety."""

import pytest
from unittest.mock import MagicMock

from generation.gemini.models import RAGResponse, SourceAttribution
from guardrails.retrieval.config import RetrievalGuardrailConfig
from guardrails.retrieval.models import RetrievalGuardrailDecision, SufficiencyStatus
from guardrails.retrieval.service import RetrievalGuardrailService
from orchestration.service import RAGOrchestrator
from retrieval.reranking.models import RerankedResultPoint


def test_retrieval_guardrails_valid_chunks():
    """Test valid retrieval chunks pass with SUFFICIENT status."""
    service = RetrievalGuardrailService()

    r1 = RerankedResultPoint(chunk_id="chk_1", document_id="doc_1", rerank_score=0.9, final_rank=1, text="Text 1", sources=["vector"])
    d = service.evaluate("test query", [r1])

    assert d.allowed is True
    assert d.status == SufficiencyStatus.SUFFICIENT
    assert len(d.valid_chunks) == 1
    assert d.rejected_chunks_count == 0


def test_retrieval_guardrails_empty_chunks():
    """Test empty chunk list returns EMPTY status."""
    service = RetrievalGuardrailService()

    d = service.evaluate("test query", [])

    assert d.allowed is False
    assert d.status == SufficiencyStatus.EMPTY
    assert len(d.valid_chunks) == 0


def test_retrieval_guardrails_invalid_structure_and_nan_score():
    """Test chunks with missing IDs, empty text, or NaN scores are rejected."""
    service = RetrievalGuardrailService()

    bad_id = RerankedResultPoint(chunk_id="", document_id="doc_1", rerank_score=0.9, final_rank=1, text="Text 1", sources=["vector"])
    bad_doc = RerankedResultPoint(chunk_id="chk_1", document_id="", rerank_score=0.9, final_rank=1, text="Text 1", sources=["vector"])
    empty_txt = RerankedResultPoint(chunk_id="chk_1", document_id="doc_1", rerank_score=0.9, final_rank=1, text="   ", sources=["vector"])
    nan_score = RerankedResultPoint(chunk_id="chk_1", document_id="doc_1", rerank_score=float("nan"), final_rank=1, text="Text 1", sources=["vector"])

    assert service.validate_chunk_structure(bad_id) is False
    assert service.validate_chunk_structure(bad_doc) is False
    assert service.validate_chunk_structure(empty_txt) is False
    assert service.validate_chunk_structure(nan_score) is False


def test_retrieval_guardrails_deduplication():
    """Test duplicate chunk_ids are deduplicated preserving highest rank."""
    service = RetrievalGuardrailService()

    r1 = RerankedResultPoint(chunk_id="chk_1", document_id="doc_1", rerank_score=0.9, final_rank=1, text="Text 1", sources=["vector"])
    r1_dup = RerankedResultPoint(chunk_id="chk_1", document_id="doc_1", rerank_score=0.9, final_rank=2, text="Text 1", sources=["bm25"])

    deduped = service.deduplicate_chunks([r1, r1_dup])
    assert len(deduped) == 1
    assert deduped[0].final_rank == 1


def test_retrieval_guardrails_contradiction_detection():
    """Test contradictory numerical facts set status to CONFLICTING."""
    service = RetrievalGuardrailService()

    c1 = RerankedResultPoint(chunk_id="chk_1", document_id="doc_1", rerank_score=0.9, final_rank=1, text="Event happened in 2010.", sources=["vector"])
    c2 = RerankedResultPoint(chunk_id="chk_2", document_id="doc_2", rerank_score=0.8, final_rank=2, text="Event happened in 2015.", sources=["bm25"])

    d = service.evaluate("When did event happen?", [c1, c2])

    assert d.allowed is True
    assert d.status == SufficiencyStatus.CONFLICTING
    assert d.metadata["has_contradiction"] is True


def test_retrieval_guardrails_low_relevance_threshold():
    """Test chunks with score below min_relevance_score are rejected."""
    config = RetrievalGuardrailConfig(min_relevance_score=0.5)
    service = RetrievalGuardrailService(config=config)

    low_score = RerankedResultPoint(chunk_id="chk_1", document_id="doc_1", rerank_score=0.1, final_rank=1, text="Text 1", sources=["vector"])

    d = service.evaluate("test query", [low_score])

    assert d.allowed is False
    assert d.status == SufficiencyStatus.INSUFFICIENT
    assert len(d.valid_chunks) == 0


def test_orchestrator_does_not_call_gemini_when_retrieval_guardrail_rejects():
    """Verify RAG Orchestrator NEVER calls Gemini API if retrieval guardrails reject context."""
    mock_input_guard = MagicMock()
    mock_hybrid = MagicMock()
    mock_reranker = MagicMock()
    mock_ret_guard = MagicMock()
    mock_gemini = MagicMock()

    # Input guardrail allows query
    mock_input_guard.evaluate.return_value = MagicMock(allowed=True)

    # Retrieval returns candidates & reranker returns chunks
    r_bad = RerankedResultPoint(chunk_id="", document_id="doc_1", rerank_score=0.9, final_rank=1, text="Text 1", sources=["vector"])
    mock_hybrid.search.return_value = ([r_bad], {})
    mock_reranker.rerank.return_value = ([r_bad], {})

    # Retrieval Guardrail REJECTS chunks
    mock_ret_guard.evaluate.return_value = RetrievalGuardrailDecision(
        allowed=False,
        status=SufficiencyStatus.INVALID,
        valid_chunks=[],
        rejected_chunks_count=1,
        reason="Invalid chunk structure.",
    )

    orchestrator = RAGOrchestrator(
        guardrail_service=mock_input_guard,
        retrieval_guardrail_service=mock_ret_guard,
        hybrid_service=mock_hybrid,
        reranker_service=mock_reranker,
        gemini_service=mock_gemini,
    )

    resp = orchestrator.answer("What is a corporation?")

    assert resp.status == "NO_CONTEXT"
    assert resp.has_context is False
    assert resp.sources == []
    # CRITICAL VERIFICATION: Gemini generate was NOT called
    assert mock_gemini.generate.called is False
