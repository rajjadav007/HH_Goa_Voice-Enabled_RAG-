"""Data models and response types for Sarvam Speech-to-Text layer."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class STTResponse:
    """Normalized response returned by Sarvam STT service."""

    text: str
    language: Optional[str] = None
    confidence: Optional[float] = None
    provider: str = "sarvam"
    model: str = "saarika:v2.5"
    duration_sec: float = 0.0

    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "language": self.language,
            "confidence": self.confidence,
            "provider": self.provider,
            "model": self.model,
            "duration_sec": round(self.duration_sec, 2),
            "latency_ms": round(self.latency_ms, 2),
            "metadata": self.metadata,
        }
