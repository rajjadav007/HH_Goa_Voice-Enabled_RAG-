"""Voice RAG Orchestrator connecting Sarvam STT directly to existing text RAG harness."""

import logging
import time
import uuid
from typing import Any, Dict, Optional, Union

from orchestration.harness.service import RAGHarness
from voice.stt.service import SarvamSTTService

logger = logging.getLogger(__name__)


class VoiceRAGOrchestrator:
    """Unified application-level orchestrator for voice-enabled RAG queries."""

    def __init__(
        self,
        stt_service: Optional[SarvamSTTService] = None,
        rag_harness: Optional[RAGHarness] = None,
    ):
        self.stt_service = stt_service or SarvamSTTService()
        self.rag_harness = rag_harness or RAGHarness()

    def answer(
        self,
        audio_data: Union[str, bytes],
        filename: str = "query.wav",
        language_code: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute end-to-end Voice RAG flow: Audio -> STT -> Transcript -> Harness -> Answer."""
        t_start = time.perf_counter()
        req_id = request_id or f"req_voice_{uuid.uuid4().hex[:8]}"

        # 1. Audio Validation & Sarvam STT Transcription
        stt_resp = self.stt_service.transcribe(
            audio_data=audio_data,
            filename=filename,
            language_code=language_code,
        )

        transcript = stt_resp.text.strip()

        # 2. Transcript Validation (Empty / Unusable check)
        if not transcript or len(transcript) < 2:
            total_ms = round((time.perf_counter() - t_start) * 1000, 2)
            return {
                "transcript": "",
                "answer": "Could not understand the audio query. Please try speaking clearly.",
                "grounded": False,
                "grounding_status": "NO_CONTEXT_GROUNDED",
                "has_context": False,
                "sources": [],
                "request_id": req_id,
                "status": "NO_CONTEXT",
                "error_code": "EMPTY_TRANSCRIPT",
                "latency_ms": total_ms,
                "timing_breakdown": {
                    "stt_ms": stt_resp.latency_ms,
                    "rag_ms": 0.0,
                    "total_voice_latency_ms": total_ms,
                },
                "stt": stt_resp.to_dict(),
            }

        # 3. Execute Existing RAG Harness with Transcribed Query
        rag_resp = self.rag_harness.run(
            query_text=transcript,
            request_id=req_id,
        )
        t_elapsed = (time.perf_counter() - t_start) * 1000
        total_ms = round(max(t_elapsed, stt_resp.latency_ms + getattr(rag_resp, "latency_ms", 0.0)), 2)

        # 4. Construct Unified Response Schema
        return {
            "transcript": transcript,
            "answer": rag_resp.answer,
            "grounded": rag_resp.grounded,
            "grounding_status": getattr(rag_resp, "grounding_status", "GROUNDED" if rag_resp.grounded else "UNGROUNDED"),
            "has_context": rag_resp.has_context,
            "sources": rag_resp.sources,
            "request_id": req_id,
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
