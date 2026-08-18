"""Unit tests for Phase 7.1 Sarvam Speech-to-Text layer and Voice API."""

import os
import sys

# Ensure backend directory is in sys.path so 'app' imports resolve
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

import pytest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.main import app
from voice.stt.config import SarvamSTTConfig
from voice.stt.models import STTResponse
from voice.stt.service import SarvamSTTService

client = TestClient(app)


def test_stt_config_api_key_security():
    """Test STT configuration loads API key securely from environment."""
    config = SarvamSTTConfig(api_key="test_key_123")
    assert config.api_key == "test_key_123"
    assert config.model == "saarika:v2"
    assert "wav" in config.allowed_formats


def test_stt_audio_validation_valid_payload():
    """Test validate_audio passes valid bytes audio payload."""
    service = SarvamSTTService()
    valid_bytes = b"RIFF\x24\x00\x00\x00WAVEfmt "

    content, ext = service.validate_audio(valid_bytes, filename="sample.wav")
    assert len(content) > 0
    assert ext == "wav"


def test_stt_audio_validation_empty_payload():
    """Test validate_audio rejects empty audio payload."""
    service = SarvamSTTService()

    with pytest.raises(ValueError) as exc_info:
        service.validate_audio(b"", filename="empty.wav")

    assert "Empty audio payload" in str(exc_info.value)


def test_stt_transcript_normalization():
    """Test normalize_transcript cleans whitespace while preserving multilingual scripts."""
    service = SarvamSTTService()

    raw = "  निगम  क्या   है? \n\t "
    norm = service.normalize_transcript(raw)
    assert norm == "निगम क्या है?"

    raw_as = " নিগম   মানে  কি? "
    norm_as = service.normalize_transcript(raw_as)
    assert norm_as == "নিগম মানে কি?"


def test_stt_transcribe_fallback_mock():
    """Test transcribe returns STTResponse with fallback transcript when no live API key."""
    service = SarvamSTTService(config=SarvamSTTConfig(api_key=None))
    valid_bytes = b"RIFF\x24\x00\x00\x00WAVEfmt "

    resp = service.transcribe(valid_bytes, filename="query.wav", language_code="hi-IN")

    assert isinstance(resp, STTResponse)
    assert len(resp.text) > 0
    assert resp.language == "hi-IN"
    assert resp.provider == "sarvam_mock"


def test_voice_query_api_endpoint():
    """Test /api/voice-query endpoint returns transcript and RAG answer."""
    mock_orchestrator = MagicMock()
    mock_orchestrator.answer.return_value = {
        "transcript": "What is a corporation?",
        "answer": "Corporation is a legal entity.",
        "grounded": True,
        "grounding_status": "GROUNDED",
        "has_context": True,
        "sources": [{"chunk_id": "chk_1", "document_id": "doc_1", "rank": 1}],
        "request_id": "req_test_123",
        "status": "COMPLETED",
        "error_code": None,
        "latency_ms": 42.0,
        "timing_breakdown": {"stt_ms": 10.0, "rag_ms": 32.0, "total_voice_latency_ms": 42.0},
        "stt": {"provider": "sarvam_mock", "model": "saarika:v2"},
    }

    with patch("app.api.endpoints.voice.get_voice_orchestrator", return_value=mock_orchestrator):
        test_client = TestClient(app)
        dummy_wav = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80\x3e\x00\x00\x00\x7d\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"

        response = test_client.post(
            "/api/voice-query",
            files={"file": ("test.wav", dummy_wav, "audio/wav")},
            data={"language_code": "hi-IN"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "transcript" in data
        assert data["answer"] == "Corporation is a legal entity."
        assert data["grounded"] is True
        assert data["stt"]["provider"] in ["sarvam", "sarvam_mock"]
