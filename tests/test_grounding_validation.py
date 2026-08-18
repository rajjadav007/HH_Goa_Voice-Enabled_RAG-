"""Unit tests for Phase 6.3 Grounding Validation layer and Orchestrator integration."""

import pytest
from unittest.mock import MagicMock

from guardrails.grounding.config import GroundingValidationConfig
from guardrails.grounding.models import GroundingDecision, GroundingStatus
from guardrails.grounding.service import GroundingValidationService
from orchestration.service import RAGOrchestrator
from retrieval.reranking.models import RerankedResultPoint


def test_grounding_validation_fully_grounded():
    """Test fully supported answer returns FULLY_GROUNDED status."""
    service = GroundingValidationService()

    c1 = RerankedResultPoint(chunk_id="chk_1", document_id="doc_1", rerank_score=0.9, final_rank=1, text="The capital of France is Paris.", sources=["vector"])

    d = service.evaluate(
        query_text="What is the capital of France?",
        answer_text="The capital of France is Paris.",
        is_grounded_flag=True,
        context_chunks=[c1],
    )

    assert d.grounded is True
    assert d.status == GroundingStatus.FULLY_GROUNDED
    assert d.support_score > 0.7
    assert len(d.unsupported_claims) == 0


def test_grounding_validation_refusal_grounded():
    """Test controlled refusal response returns REFUSAL_GROUNDED status."""
    service = GroundingValidationService()

    d = service.evaluate(
        query_text="What is the capital of Atlantis?",
        answer_text="Insufficient context available to answer the query.",
        is_grounded_flag=False,
        context_chunks=[],
    )

    assert d.grounded is False
    assert d.status == GroundingStatus.REFUSAL_GROUNDED


def test_grounding_validation_ungrounded_hallucination():
    """Test unsupported hallucination returns UNGROUNDED status and fallback answer."""
    service = GroundingValidationService()

    c1 = RerankedResultPoint(chunk_id="chk_1", document_id="doc_1", rerank_score=0.9, final_rank=1, text="Photosynthesis is the process by which plants turn light into sugar.", sources=["vector"])

    d = service.evaluate(
        query_text="What is quantum mechanics?",
        answer_text="Quantum mechanics is governed by Schrödinger equation and wave function collapse.",
        is_grounded_flag=True,
        context_chunks=[c1],
    )

    assert d.grounded is False
    assert d.status == GroundingStatus.UNGROUNDED
    assert "Insufficient context available" in d.validated_answer
    assert len(d.unsupported_claims) > 0


def test_orchestrator_integrates_grounding_validation():
    """Verify RAG Orchestrator runs grounding validation and attaches decision metadata."""
    mock_input_guard = MagicMock()
    mock_ret_guard = MagicMock()
    mock_grounding = MagicMock()
    mock_hybrid = MagicMock()
    mock_reranker = MagicMock()
    mock_gemini = MagicMock()

    mock_input_guard.evaluate.return_value = MagicMock(allowed=True)

    r1 = RerankedResultPoint(chunk_id="chk_1", document_id="doc_1", rerank_score=0.9, final_rank=1, text="Text 1", sources=["vector"])
    mock_hybrid.search.return_value = ([r1], {})
    mock_reranker.rerank.return_value = ([r1], {})

    mock_ret_guard.evaluate.return_value = MagicMock(allowed=True, valid_chunks=[r1], to_dict=lambda: {})

    mock_gemini.generate.return_value = MagicMock(
        answer="Paris is the capital.",
        grounded=True,
        sources=[],
        model="gemini-2.5-flash",
        token_usage={},
    )

    mock_grounding.evaluate.return_value = GroundingDecision(
        grounded=True,
        status=GroundingStatus.FULLY_GROUNDED,
        support_score=0.95,
        validated_answer="Paris is the capital.",
    )

    orchestrator = RAGOrchestrator(
        guardrail_service=mock_input_guard,
        retrieval_guardrail_service=mock_ret_guard,
        grounding_validation_service=mock_grounding,
        hybrid_service=mock_hybrid,
        reranker_service=mock_reranker,
        gemini_service=mock_gemini,
    )

    resp = orchestrator.answer("What is the capital?")

    assert resp.status == "SUCCESS"
    assert resp.grounded is True
    assert resp.answer == "Paris is the capital."
    assert mock_grounding.evaluate.called is True
