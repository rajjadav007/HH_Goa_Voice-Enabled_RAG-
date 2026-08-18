"""Unit tests for Phase 7.2 Voice -> Text -> RAG Orchestration and Integration."""

import os
import sys
from unittest.mock import MagicMock, patch

# Ensure backend directory is in sys.path so 'app' imports resolve
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

import pytest
from fastapi.testclient import TestClient

from app.main import app
from orchestration.harness.service import RAGHarness
from voice.orchestrator import VoiceRAGOrchestrator
from voice.stt.models import STTResponse
from voice.stt.service import SarvamSTTService


def test_voice_rag_orchestrator_success_flow():
    """Test VoiceRAGOrchestrator passes transcript to harness and returns unified response."""
    mock_stt_resp = STTResponse(
        text="What is a corporation?",
        language="en-IN",
        confidence=0.98,
        provider="sarvam_mock",
        model="saarika:v2",
        latency_ms=15.0,
    )
    mock_stt_service = MagicMock(spec=SarvamSTTService)
    mock_stt_service.transcribe.return_value = mock_stt_resp

    mock_rag_harness_resp = MagicMock()
    mock_rag_harness_resp.answer = "A corporation is a legal entity."
    mock_rag_harness_resp.grounded = True
    mock_rag_harness_resp.grounding_status = "GROUNDED"
    mock_rag_harness_resp.has_context = True
    mock_rag_harness_resp.sources = [{"chunk_id": "chk_1", "document_id": "doc_1", "rank": 1}]
    mock_rag_harness_resp.request_id = "req_voice_1234"
    mock_rag_harness_resp.status = "COMPLETED"
    mock_rag_harness_resp.error_code = None
    mock_rag_harness_resp.latency_ms = 40.0
    mock_rag_harness_resp.token_usage = {"total_tokens": 120}
    mock_rag_harness_resp.metadata = {}

    mock_rag_harness = MagicMock(spec=RAGHarness)
    mock_rag_harness.run.return_value = mock_rag_harness_resp

    orchestrator = VoiceRAGOrchestrator(
        stt_service=mock_stt_service,
        rag_harness=mock_rag_harness,
    )

    result = orchestrator.answer(
        audio_data=b"RIFF\x24\x00\x00\x00WAVEfmt ",
        filename="test.wav",
        language_code="en-IN",
        request_id="req_voice_1234",
    )

    assert result["transcript"] == "What is a corporation?"
    assert result["answer"] == "A corporation is a legal entity."
    assert result["grounded"] is True
    assert result["request_id"] == "req_voice_1234"
    assert result["timing_breakdown"]["stt_ms"] == 15.0
    assert result["timing_breakdown"]["rag_ms"] == 40.0
    assert result["latency_ms"] >= 55.0

    mock_stt_service.transcribe.assert_called_once()
    mock_rag_harness.run.assert_called_once_with(
        query_text="What is a corporation?",
        request_id="req_voice_1234",
    )


def test_voice_rag_empty_transcript_handling():
    """Test VoiceRAGOrchestrator handles empty transcript gracefully without calling RAG harness."""
    mock_stt_resp = STTResponse(
        text="",
        language="hi-IN",
        confidence=0.0,
        provider="sarvam_mock",
        model="saarika:v2",
        latency_ms=10.0,
    )
    mock_stt_service = MagicMock(spec=SarvamSTTService)
    mock_stt_service.transcribe.return_value = mock_stt_resp

    mock_rag_harness = MagicMock(spec=RAGHarness)

    orchestrator = VoiceRAGOrchestrator(
        stt_service=mock_stt_service,
        rag_harness=mock_rag_harness,
    )

    result = orchestrator.answer(audio_data=b"RIFF\x24\x00\x00\x00WAVEfmt ")

    assert result["transcript"] == ""
    assert "Could not understand the audio" in result["answer"]
    assert result["grounded"] is False
    assert result["status"] == "NO_CONTEXT"
    assert result["error_code"] == "EMPTY_TRANSCRIPT"

    # Verify RAG harness was NOT called
    mock_rag_harness.run.assert_not_called()


def test_voice_query_api_endpoint_integration():
    """Test /api/voice-query endpoint with mocked voice orchestrator."""
    mock_orchestrator = MagicMock(spec=VoiceRAGOrchestrator)
    mock_orchestrator.answer.return_value = {
        "transcript": "What is a corporation?",
        "answer": "A corporation is a legal entity.",
        "grounded": True,
        "grounding_status": "GROUNDED",
        "has_context": True,
        "sources": [{"chunk_id": "chk_1", "document_id": "doc_1", "rank": 1}],
        "request_id": "req_voice_test_5678",
        "status": "COMPLETED",
        "error_code": None,
        "latency_ms": 50.0,
        "timing_breakdown": {"stt_ms": 10.0, "rag_ms": 40.0, "total_voice_latency_ms": 50.0},
        "stt": {"provider": "sarvam_mock", "model": "saarika:v2"},
    }

    with patch("app.api.endpoints.voice.get_voice_orchestrator", return_value=mock_orchestrator):
        test_client = TestClient(app)
        dummy_wav = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80\x3e\x00\x00\x00\x7d\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"

        response = test_client.post(
            "/api/voice-query",
            files={"file": ("test.wav", dummy_wav, "audio/wav")},
            data={"language_code": "en-IN"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["transcript"] == "What is a corporation?"
        assert data["answer"] == "A corporation is a legal entity."
        assert data["grounded"] is True
        assert data["request_id"] == "req_voice_test_5678"


def test_text_rag_regression():
    """Verify text RAG pipeline remains fully functional and unimpacted by voice integration."""
    mock_rag_harness = MagicMock(spec=RAGHarness)
    mock_harness_resp = MagicMock()
    mock_harness_resp.answer = "Corporation definition."
    mock_harness_resp.grounded = True
    mock_harness_resp.has_context = True
    mock_harness_resp.sources = []
    mock_harness_resp.request_id = "req_text_999"
    mock_harness_resp.status = "SUCCESS"
    mock_harness_resp.error_code = None
    mock_harness_resp.latency_ms = 30.0
    mock_harness_resp.timing_breakdown = {}
    mock_harness_resp.token_usage = {}
    mock_harness_resp.metadata = {}

    with patch("app.api.endpoints.query.orchestrator.answer", return_value=mock_harness_resp):
        test_client = TestClient(app)
        response = test_client.post("/api/query", json={"query": "What is a corporation?"})
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["answer"] == "Corporation definition."
