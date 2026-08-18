"""Data models and configuration for RAG Orchestrator layer."""

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ErrorCode(str, Enum):
    INVALID_QUERY = "INVALID_QUERY"
    NO_CONTEXT = "NO_CONTEXT"
    RETRIEVAL_ERROR = "RETRIEVAL_ERROR"
    RERANKING_ERROR = "RERANKING_ERROR"
    GENERATION_ERROR = "GENERATION_ERROR"
    RESPONSE_VALIDATION_ERROR = "RESPONSE_VALIDATION_ERROR"
    TIMEOUT = "TIMEOUT"


@dataclass
class RAGOrchestratorConfig:
    """Centralized configuration for RAG Orchestrator."""

    timeout_sec: float = 15.0
    vector_top_k: int = 10
    bm25_top_k: int = 10
    candidate_k: int = 10
    final_k: int = 5
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGOrchestrationResponse:
    """Unified application-level response returned by RAG Orchestrator."""

    answer: str
    grounded: bool
    has_context: bool
    sources: List[Dict[str, Any]]
    request_id: str
    status: str
    grounding_status: Optional[str] = None
    generation_status: str = "SUCCESS"
    error_code: Optional[str] = None
    latency_ms: float = 0.0
    token_usage: Dict[str, int] = field(default_factory=dict)
    timing_breakdown: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "grounded": self.grounded,
            "has_context": self.has_context,
            "sources": self.sources,
            "request_id": self.request_id,
            "status": self.status,
            "grounding_status": self.grounding_status,
            "generation_status": self.generation_status,
            "error_code": self.error_code,
            "latency_ms": float(round(self.latency_ms, 2)),
            "token_usage": self.token_usage,
            "timing_breakdown": self.timing_breakdown,
            "metadata": self.metadata,
        }


def generate_request_id() -> str:
    """Generate a unique request tracking ID."""
    return f"req_{uuid.uuid4().hex[:12]}"
