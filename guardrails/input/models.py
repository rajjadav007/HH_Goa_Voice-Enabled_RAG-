"""Data models and decisions for Input Guardrails layer."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class GuardrailCategory(str, Enum):
    VALID = "VALID"
    EMPTY = "EMPTY"
    TOO_LONG = "TOO_LONG"
    OFF_TOPIC = "OFF_TOPIC"
    PROMPT_INJECTION = "PROMPT_INJECTION"
    UNSAFE = "UNSAFE"
    INVALID = "INVALID"


@dataclass
class GuardrailDecision:
    """Structured decision returned by Input Guardrail evaluation."""

    allowed: bool
    category: GuardrailCategory
    reason: Optional[str] = None
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "category": self.category.value,
            "reason": self.reason,
            "latency_ms": float(round(self.latency_ms, 2)),
            "metadata": self.metadata,
        }
