"""Data models and decision types for Grounding Validation layer."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class GroundingStatus(str, Enum):
    FULLY_GROUNDED = "FULLY_GROUNDED"
    PARTIALLY_GROUNDED = "PARTIALLY_GROUNDED"
    UNGROUNDED = "UNGROUNDED"
    REFUSAL_GROUNDED = "REFUSAL_GROUNDED"
    NO_CONTEXT_GROUNDED = "NO_CONTEXT_GROUNDED"


@dataclass
class GroundingDecision:
    """Structured decision returned by Grounding Validation evaluation."""

    grounded: bool
    status: GroundingStatus
    support_score: float
    unsupported_claims: List[str] = field(default_factory=list)
    validated_answer: str = ""
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "grounded": self.grounded,
            "status": self.status.value,
            "support_score": float(round(self.support_score, 4)),
            "unsupported_claims": self.unsupported_claims,
            "validated_answer": self.validated_answer,
            "latency_ms": float(round(self.latency_ms, 3)),
            "metadata": self.metadata,
        }
