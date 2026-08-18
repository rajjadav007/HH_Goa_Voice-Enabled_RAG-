"""FastAPI Voice Query API endpoint integrating Sarvam STT with Voice RAG Orchestrator."""

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.api.endpoints.query import orchestrator as text_orchestrator
from orchestration.harness.service import RAGHarness
from voice.orchestrator import VoiceRAGOrchestrator

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Voice Query"])

_voice_orchestrator: Optional[VoiceRAGOrchestrator] = None


def get_voice_orchestrator() -> VoiceRAGOrchestrator:
    global _voice_orchestrator
    if _voice_orchestrator is None:
        harness = RAGHarness(orchestrator=text_orchestrator)
        _voice_orchestrator = VoiceRAGOrchestrator(rag_harness=harness)
    return _voice_orchestrator



@router.post("/voice-query")
async def process_voice_query(
    file: UploadFile = File(...),
    language_code: Optional[str] = Form(None),
):
    """Process voice audio upload via Sarvam STT and execute end-to-end RAG harness pipeline."""
    try:
        content = await file.read()
        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty audio file uploaded.",
            )

        orchestrator = get_voice_orchestrator()
        result = orchestrator.answer(
            audio_data=content,
            filename=file.filename or "audio.wav",
            language_code=language_code,
        )

        return result

    except HTTPException:
        raise
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
