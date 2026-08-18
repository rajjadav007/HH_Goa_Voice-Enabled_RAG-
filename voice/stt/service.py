"""Sarvam Speech-to-Text service for multilingual audio transcription."""

import io
import logging
import os
import re
import tempfile
import time
from typing import Any, Dict, Optional, Tuple, Union

import requests

from voice.stt.config import SarvamSTTConfig
from voice.stt.models import STTResponse

logger = logging.getLogger(__name__)


class SarvamSTTService:
    """Service interfacing with Sarvam AI Speech-to-Text API."""

    def __init__(self, config: Optional[SarvamSTTConfig] = None):
        self.config = config or SarvamSTTConfig()

    def validate_audio(
        self,
        audio_data: Union[str, bytes, io.BytesIO],
        filename: str = "audio.wav",
    ) -> Tuple[bytes, str]:
        """Validate audio data size, format, and readability."""
        if isinstance(audio_data, str):
            if not os.path.exists(audio_data):
                raise ValueError(f"Audio file not found at path: {audio_data}")
            file_size_mb = os.path.getsize(audio_data) / (1024 * 1024)
            if file_size_mb > self.config.max_audio_size_mb:
                raise ValueError(f"Audio file size ({file_size_mb:.2f} MB) exceeds maximum allowed limit ({self.config.max_audio_size_mb} MB).")
            with open(audio_data, "rb") as f:
                content = f.read()
            ext = os.path.splitext(audio_data)[1].lstrip(".").lower() or "wav"
        elif isinstance(audio_data, bytes):
            content = audio_data
            ext = os.path.splitext(filename)[1].lstrip(".").lower() or "wav"
        elif isinstance(audio_data, io.BytesIO):
            content = audio_data.getvalue()
            ext = os.path.splitext(filename)[1].lstrip(".").lower() or "wav"
        else:
            raise ValueError("Unsupported audio data format provided.")

        if not content or len(content) == 0:
            raise ValueError("Empty audio payload provided.")

        size_mb = len(content) / (1024 * 1024)
        if size_mb > self.config.max_audio_size_mb:
            raise ValueError(f"Audio payload ({size_mb:.2f} MB) exceeds maximum limit ({self.config.max_audio_size_mb} MB).")

        if ext not in self.config.allowed_formats:
            logger.warning(f"Extension '{ext}' not explicitly in allowed set {self.config.allowed_formats}. Proceeding with payload validation.")

        return content, ext

    def normalize_transcript(self, raw_text: str) -> str:
        """Clean and normalize transcribed text while preserving non-English scripts and meaning."""
        if not raw_text or not raw_text.strip():
            return ""
        # Remove repeated whitespace and control characters, preserve unicode scripts
        cleaned = re.sub(r"[\r\n\t]+", " ", raw_text.strip())
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        return cleaned

    def transcribe(
        self,
        audio_data: Union[str, bytes, io.BytesIO],
        filename: str = "query.wav",
        language_code: Optional[str] = None,
    ) -> STTResponse:
        """Transcribe audio payload using Sarvam AI STT API."""
        t0 = time.perf_counter()

        # Validate audio input
        content, ext = self.validate_audio(audio_data, filename=filename)
        lang = language_code or self.config.language_code

        api_key = self.config.api_key or os.getenv("SARVAM_API_KEY")

        # If API key is missing or mock mode active, use deterministic fallback
        if not api_key or api_key.startswith("MOCK_") or api_key == "YOUR_SARVAM_API_KEY":
            logger.warning("SARVAM_API_KEY missing or mock key detected. Utilizing STT mock fallback.")
            eval_ms = round((time.perf_counter() - t0) * 1000, 2)
            mock_text = "What is a corporation?"
            return STTResponse(
                text=mock_text,
                language=lang,
                confidence=0.98,
                provider="sarvam_mock",
                model=self.config.model,
                duration_sec=3.5,
                latency_ms=eval_ms,
                metadata={"mock_fallback": True},
            )

        # Real Sarvam API Call
        try:
            headers = {
                "api-subscription-key": api_key,
            }
            files = {
                "file": (f"audio.{ext}", content, f"audio/{ext}"),
            }
            data = {
                "model": self.config.model,
                "language_code": lang,
            }

            resp = requests.post(
                self.config.api_endpoint,
                headers=headers,
                files=files,
                data=data,
                timeout=self.config.timeout_sec,
            )

            eval_ms = round((time.perf_counter() - t0) * 1000, 2)

            if resp.status_code == 200:
                body = resp.json()
                raw_transcript = body.get("transcript") or body.get("text", "")
                norm_text = self.normalize_transcript(raw_transcript)
                detected_lang = body.get("language_code", lang)

                return STTResponse(
                    text=norm_text,
                    language=detected_lang,
                    confidence=body.get("confidence", 0.95),
                    provider="sarvam",
                    model=self.config.model,
                    duration_sec=body.get("duration", 0.0),
                    latency_ms=eval_ms,
                    metadata={"status_code": 200},
                )
            else:
                logger.error(f"Sarvam STT API returned status {resp.status_code}: {resp.text}")
                raise RuntimeError(f"Sarvam STT API error (status {resp.status_code}): {resp.text}")

        except Exception as exc:
            eval_ms = round((time.perf_counter() - t0) * 1000, 2)
            logger.error(f"Sarvam STT request failure: {exc}")
            raise RuntimeError(f"Sarvam STT transcription failed: {exc}")
