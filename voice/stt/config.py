"""Configuration settings for Sarvam Speech-to-Text provider."""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set


@dataclass
class SarvamSTTConfig:
    """Centralized configuration for Sarvam STT integration."""

    enabled: bool = True
    model: str = "saarika:v2.5"
    language_code: str = "hi-IN"

    api_endpoint: str = "https://api.sarvam.ai/speech-to-text"
    max_audio_size_mb: float = 10.0
    max_duration_sec: float = 60.0
    timeout_sec: float = 10.0
    allowed_formats: Set[str] = field(default_factory=lambda: {"wav", "mp3", "m4a", "webm", "flac", "ogg"})

    # API key loaded securely from environment
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("SARVAM_API_KEY"))
