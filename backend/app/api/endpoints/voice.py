"""FastAPI Voice Query API endpoint integrating Sarvam STT with RAG Harness."""

import logging
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from orchestration.harness.service import RAGHarness
from voice.stt.service import SarvamSTTService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Voice Query"])

_stt_service: Optional[SarvamSTTService] = None
_harness_service: Optional[RAGHarness] = None


def get_stt_service() -> SarvamSTTService:
    global _stt_service
    if _stt_service is None:
        _stt_service = SarvamSTTService()
    return _stt_service


def get_harness_service() -> RAGHarness:
    global _harness_service
    if _harness_service is None:
        _harness_service = RAGHarness()
    return _harness_service


@router.post("/voice-query")
async def process_voice_query(
    file: UploadFile = File(...),
    language_code: Optional[str] = Form(None),
):
    """Process voice audio upload via Sarvam STT and execute end-to-end RAG harness pipeline."""
    t_start = time.perf_counter()

    try:
        content = await file.read()
        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty audio file uploaded.",
            )

        stt_srv = get_stt_service()
        harness_srv = get_harness_service()

        # 1. Sarvam STT Transcription
        stt_resp = stt_srv.transcribe(
            audio_data=content,
            filename=file.filename or "audio.wav",
            language_code=language_code,
        )

        transcript = stt_resp.text.strip()

        # Handle empty/unusable transcript
        if not transcript or len(transcript) < 2:
            total_ms = round((time.perf_counter() - t_start) * 1000, 2)
            return {
                "transcript": "",
                "answer": "Could not understand the audio. Please try again.",
                "grounded": False,
                "has_context": False,
                "sources": [],
                "request_id": "req_stt_empty",
                "status": "NO_CONTEXT",
                "error_code": "EMPTY_TRANSCRIPT",
                "latency_ms": total_ms,
                "stt": stt_resp.to_dict(),
            }

        # 2. Execute RAG Harness with transcribed query
        rag_resp = harness_srv.run(query_text=transcript)
        total_ms = round((time.perf_counter() - t_start) * 1000, 2)

        return {
            "transcript": transcript,
            "answer": rag_resp.answer,
            "grounded": rag_resp.grounded,
            "has_context": rag_resp.has_context,
            "sources": rag_resp.sources,
            "request_id": rag_resp.request_id,
            "status": rag_resp.status,
            "error_code": rag_resp.error_code,
            "latency_ms": total_ms,
            "token_usage": rag_resp.token_usage,
            "timing_breakdown": {
                "stt_ms": stt_resp.latency_ms,
                "rag_ms": rag_resp.latency_ms,
                "total_voice_latency_ms": total_ms,
            },
            "stt": stt_resp.to_dict(),
            "metadata": rag_resp.metadata,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(f"Voice query endpoint failure: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Voice processing failed. Internal server error.",
        )
