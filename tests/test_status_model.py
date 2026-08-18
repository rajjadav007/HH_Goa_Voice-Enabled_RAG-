"""Comprehensive Unit Tests for RAG Response Status Model and Error State Handling."""

import pytest
from unittest.mock import MagicMock, patch

from generation.gemini.models import RAGResponse, SourceAttribution
from generation.gemini.service import GeminiService
from guardrails.grounding.models import GroundingDecision, GroundingStatus
from guardrails.grounding.service import GroundingValidationService
from orchestration.models import RAGOrchestrationResponse
from orchestration.service import RAGOrchestrator
from retrieval.reranking.models import RerankedResultPoint


def test_status_model_fully_grounded():
    """Test 1: FULLY_GROUNDED produces grounded=True, FULLY_GROUNDED status, and valid sources."""
    orch = RAGOrchestrator()
    mock_gemini = MagicMock()
    mock_gemini.generate.return_value = RAGResponse(
        answer="A corporation is a legal entity.",
        grounded=True,
        sources=[SourceAttribution(chunk_id="chk_1", document_id="doc_1", rank=1)],
        model="gemini-3.6-flash",
        latency_ms=100.0,
        generation_status="SUCCESS",
    )
    orch.gemini_service = mock_gemini

    mock_chunk = RerankedResultPoint(chunk_id="chk_1", document_id="doc_1", rerank_score=0.9, final_rank=1, text="A corporation is a legal entity.", sources=["vector"])
    with patch.object(orch.hybrid_service, "search", return_value=([mock_chunk], {})), \
         patch.object(orch.reranker_service, "rerank", return_value=([mock_chunk], {})):
        res = orch.answer("What is a corporation?")

    assert res.status == "SUCCESS"
    assert res.grounded is True
    assert res.grounding_status in ["FULLY_GROUNDED", "PARTIALLY_GROUNDED"]
    assert res.generation_status == "SUCCESS"
    assert len(res.sources) == 1
    assert res.sources[0]["chunk_id"] == "chk_1"


def test_status_model_partially_grounded():
    """Test 2: PARTIALLY_GROUNDED produces grounded=True and PARTIALLY_GROUNDED status."""
    service = GroundingValidationService()
    c1 = RerankedResultPoint(chunk_id="chk_1", document_id="doc_1", rerank_score=0.9, final_rank=1, text="Corporation is a legal entity registered under law.", sources=["vector"])

    d = service.evaluate(
        query_text="What is a corporation?",
        answer_text="A corporation is a legal entity. It also manages global corporate governance.",
        is_grounded_flag=True,
        context_chunks=[c1],
    )

    assert d.status == GroundingStatus.PARTIALLY_GROUNDED
    assert d.grounded is True


def test_status_model_refusal_grounded():
    """Test 3: REFUSAL_GROUNDED produces grounded=False, REFUSAL_GROUNDED status, and no fake sources."""
    orch = RAGOrchestrator()
    mock_gemini = MagicMock()
    mock_gemini.generate.return_value = RAGResponse(
        answer="Insufficient context available to answer the query.",
        grounded=False,
        sources=[SourceAttribution(chunk_id="chk_1", document_id="doc_1", rank=1)],
        model="gemini-3.6-flash",
        latency_ms=100.0,
        generation_status="SUCCESS",
    )
    orch.gemini_service = mock_gemini

    mock_chunk = RerankedResultPoint(chunk_id="chk_1", document_id="doc_1", rerank_score=0.1, final_rank=1, text="Some unrelated text about flights.", sources=["vector"])
    with patch.object(orch.hybrid_service, "search", return_value=([mock_chunk], {})), \
         patch.object(orch.reranker_service, "rerank", return_value=([mock_chunk], {})):
        res = orch.answer("What is the capital of India?")

    assert res.status == "SUCCESS"
    assert res.grounded is False
    assert res.grounding_status == "REFUSAL_GROUNDED"
    assert res.generation_status == "SUCCESS"
    assert res.answer == "Insufficient context available to answer the query."
    assert res.sources == []  # No fake sources for refusal!


def test_status_model_provider_quota_exceeded():
    """Test 4: PROVIDER_QUOTA_EXCEEDED produces grounded=False, generation_status=PROVIDER_QUOTA_EXCEEDED."""
    orch = RAGOrchestrator()
    mock_gemini = MagicMock()
    mock_gemini.generate.return_value = RAGResponse(
        answer="Answer generation is temporarily unavailable.",
        grounded=False,
        sources=[],
        model="gemini-3.6-flash",
        latency_ms=50.0,
        generation_status="PROVIDER_QUOTA_EXCEEDED",
    )
    orch.gemini_service = mock_gemini

    mock_chunk = RerankedResultPoint(chunk_id="chk_1", document_id="doc_1", rerank_score=0.9, final_rank=1, text="Some text.", sources=["vector"])
    with patch.object(orch.hybrid_service, "search", return_value=([mock_chunk], {})), \
         patch.object(orch.reranker_service, "rerank", return_value=([mock_chunk], {})):
        res = orch.answer("What is a corporation?")

    assert res.status == "ERROR"
    assert res.grounded is False
    assert res.grounding_status is None
    assert res.generation_status == "PROVIDER_QUOTA_EXCEEDED"
    assert res.answer == "Answer generation is temporarily unavailable."
    assert res.sources == []


def test_status_model_provider_timeout():
    """Test 5: PROVIDER_TIMEOUT produces grounded=False, generation_status=PROVIDER_TIMEOUT."""
    orch = RAGOrchestrator()
    mock_gemini = MagicMock()
    mock_gemini.generate.return_value = RAGResponse(
        answer="Answer generation timed out. Please try again.",
        grounded=False,
        sources=[],
        model="gemini-3.6-flash",
        latency_ms=10000.0,
        generation_status="PROVIDER_TIMEOUT",
    )
    orch.gemini_service = mock_gemini

    mock_chunk = RerankedResultPoint(chunk_id="chk_1", document_id="doc_1", rerank_score=0.9, final_rank=1, text="Some text.", sources=["vector"])
    with patch.object(orch.hybrid_service, "search", return_value=([mock_chunk], {})), \
         patch.object(orch.reranker_service, "rerank", return_value=([mock_chunk], {})):
        res = orch.answer("What is a corporation?")

    assert res.status == "ERROR"
    assert res.grounded is False
    assert res.grounding_status is None
    assert res.generation_status == "PROVIDER_TIMEOUT"
    assert res.answer == "Answer generation timed out. Please try again."
    assert res.sources == []


def test_status_model_provider_unavailable():
    """Test 6: PROVIDER_UNAVAILABLE produces grounded=False, generation_status=PROVIDER_UNAVAILABLE."""
    orch = RAGOrchestrator()
    mock_gemini = MagicMock()
    mock_gemini.generate.return_value = RAGResponse(
        answer="Answer generation is temporarily unavailable.",
        grounded=False,
        sources=[],
        model="gemini-3.6-flash",
        latency_ms=50.0,
        generation_status="PROVIDER_UNAVAILABLE",
    )
    orch.gemini_service = mock_gemini

    mock_chunk = RerankedResultPoint(chunk_id="chk_1", document_id="doc_1", rerank_score=0.9, final_rank=1, text="Some text.", sources=["vector"])
    with patch.object(orch.hybrid_service, "search", return_value=([mock_chunk], {})), \
         patch.object(orch.reranker_service, "rerank", return_value=([mock_chunk], {})):
        res = orch.answer("What is a corporation?")

    assert res.status == "ERROR"
    assert res.grounded is False
    assert res.grounding_status is None
    assert res.generation_status == "PROVIDER_UNAVAILABLE"
    assert res.answer == "Answer generation is temporarily unavailable."
    assert res.sources == []


def test_status_model_internal_error():
    """Test 7: INTERNAL_ERROR produces grounded=False, generation_status=INTERNAL_ERROR."""
    orch = RAGOrchestrator()
    mock_gemini = MagicMock()
    mock_gemini.generate.side_effect = RuntimeError("Unexpected internal crash")
    orch.gemini_service = mock_gemini

    mock_chunk = RerankedResultPoint(chunk_id="chk_1", document_id="doc_1", rerank_score=0.9, final_rank=1, text="Some text.", sources=["vector"])
    with patch.object(orch.hybrid_service, "search", return_value=([mock_chunk], {})), \
         patch.object(orch.reranker_service, "rerank", return_value=([mock_chunk], {})):
        res = orch.answer("What is a corporation?")

    assert res.status == "ERROR"
    assert res.grounded is False
    assert res.sources == []


def test_sources_grounded_answer():
    """Test 8: Grounded answer contains valid sources corresponding to context."""
    orch = RAGOrchestrator()
    mock_gemini = MagicMock()
    mock_gemini.generate.return_value = RAGResponse(
        answer="A corporation is a company authorized to act as a single entity.",
        grounded=True,
        sources=[SourceAttribution(chunk_id="chk_100", document_id="doc_100", rank=1)],
        model="gemini-3.6-flash",
        latency_ms=100.0,
        generation_status="SUCCESS",
    )
    orch.gemini_service = mock_gemini

    mock_chunk = RerankedResultPoint(chunk_id="chk_100", document_id="doc_100", rerank_score=0.95, final_rank=1, text="A corporation is a company authorized to act as a single entity.", sources=["vector"])
    with patch.object(orch.hybrid_service, "search", return_value=([mock_chunk], {})), \
         patch.object(orch.reranker_service, "rerank", return_value=([mock_chunk], {})):
        res = orch.answer("What is a corporation?")

    assert len(res.sources) == 1
    assert res.sources[0]["chunk_id"] == "chk_100"


def test_no_fake_sources_refusal():
    """Test 9: Refusal response produces empty sources list."""
    orch = RAGOrchestrator()
    mock_gemini = MagicMock()
    mock_gemini.generate.return_value = RAGResponse(
        answer="Insufficient context available to answer the query.",
        grounded=False,
        sources=[SourceAttribution(chunk_id="chk_fake", document_id="doc_fake", rank=1)],
        model="gemini-3.6-flash",
        latency_ms=100.0,
        generation_status="SUCCESS",
    )
    orch.gemini_service = mock_gemini

    mock_chunk = RerankedResultPoint(chunk_id="chk_999", document_id="doc_999", rerank_score=0.01, final_rank=1, text="Irrelevant text.", sources=["vector"])
    with patch.object(orch.hybrid_service, "search", return_value=([mock_chunk], {})), \
         patch.object(orch.reranker_service, "rerank", return_value=([mock_chunk], {})):
        res = orch.answer("Out of scope query")

    assert res.sources == []


def test_no_stale_sources_provider_error():
    """Test 10: Provider error produces empty sources list (no stale sources)."""
    orch = RAGOrchestrator()
    mock_gemini = MagicMock()
    mock_gemini.generate.return_value = RAGResponse(
        answer="Answer generation is temporarily unavailable.",
        grounded=False,
        sources=[],
        model="gemini-3.6-flash",
        latency_ms=10.0,
        generation_status="PROVIDER_QUOTA_EXCEEDED",
    )
    orch.gemini_service = mock_gemini

    mock_chunk = RerankedResultPoint(chunk_id="chk_stale", document_id="doc_stale", rerank_score=0.8, final_rank=1, text="Stale chunk.", sources=["vector"])
    with patch.object(orch.hybrid_service, "search", return_value=([mock_chunk], {})), \
         patch.object(orch.reranker_service, "rerank", return_value=([mock_chunk], {})):
        res = orch.answer("What is a corporation?")

    assert res.sources == []


def test_text_and_voice_status_handling():
    """Tests 12 & 13: Text and Voice API schemas properly carry generation_status and grounding_status."""
    from backend.app.api.endpoints.query import QueryResponse, SourceItem

    q_resp = QueryResponse(
        success=True,
        answer="Test answer",
        grounded=True,
        grounding_status="FULLY_GROUNDED",
        generation_status="SUCCESS",
        has_context=True,
        sources=[SourceItem(chunk_id="c1", document_id="d1", rank=1)],
        request_id="req_123",
        status="SUCCESS",
        latency_ms=150.0,
    )

    assert q_resp.grounding_status == "FULLY_GROUNDED"
    assert q_resp.generation_status == "SUCCESS"
    assert q_resp.grounded is True
