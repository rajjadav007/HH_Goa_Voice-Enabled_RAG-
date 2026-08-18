"""Data models and decision types for Retrieval Guardrails layer."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from retrieval.reranking.models import RerankedResultPoint


class SufficiencyStatus(str, Enum):
    SUFFICIENT = "SUFFICIENT"
    PARTIAL = "PARTIAL"
    INSUFFICIENT = "INSUFFICIENT"
    INVALID = "INVALID"
    EMPTY = "EMPTY"
    CONFLICTING = "CONFLICTING"


@dataclass
class RetrievalGuardrailDecision:
    """Structured decision returned by Retrieval Guardrail evaluation."""

    allowed: bool
    status: SufficiencyStatus
    valid_chunks: List[RerankedResultPoint] = field(default_factory=list)
    rejected_chunks_count: int = 0
    reason: Optional[str] = None
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "status": self.status.value,
            "valid_chunks_count": len(self.valid_chunks),
            "rejected_chunks_count": self.rejected_chunks_count,
            "reason": self.reason,
            "latency_ms": float(round(self.latency_ms, 3)),
            "metadata": self.metadata,
        }
