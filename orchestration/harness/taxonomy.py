"""Centralized error taxonomy and structured exception handling for RAG Harness."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class ErrorCategory(str, Enum):
    INVALID_INPUT = "INVALID_INPUT"
    GUARDRAIL_BLOCKED = "GUARDRAIL_BLOCKED"
    RETRIEVAL_ERROR = "RETRIEVAL_ERROR"
    QDRANT_ERROR = "QDRANT_ERROR"
    BM25_ERROR = "BM25_ERROR"
    EMBEDDING_ERROR = "EMBEDDING_ERROR"
    RERANKER_ERROR = "RERANKER_ERROR"
    CONTEXT_ERROR = "CONTEXT_ERROR"
    GENERATION_ERROR = "GENERATION_ERROR"
    GEMINI_TIMEOUT = "GEMINI_TIMEOUT"
    GEMINI_RATE_LIMIT = "GEMINI_RATE_LIMIT"
    GENERATION_VALIDATION_ERROR = "GENERATION_VALIDATION_ERROR"
    GROUNDING_ERROR = "GROUNDING_ERROR"
    GROUNDING_FAILED = "GROUNDING_FAILED"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    REQUEST_TIMEOUT = "REQUEST_TIMEOUT"


RETRYABLE_CATEGORIES = {
    ErrorCategory.QDRANT_ERROR,
    ErrorCategory.BM25_ERROR,
    ErrorCategory.EMBEDDING_ERROR,
    ErrorCategory.RERANKER_ERROR,
    ErrorCategory.GEMINI_TIMEOUT,
    ErrorCategory.GEMINI_RATE_LIMIT,
    ErrorCategory.GENERATION_ERROR,
}


@dataclass
class HarnessError(Exception):
    """Structured internal exception representation."""

    category: ErrorCategory
    stage: str
    message: str
    retryable: bool = False
    request_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.category.value}] stage='{self.stage}' message='{self.message}' retryable={self.retryable}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "stage": self.stage,
            "message": self.message,
            "retryable": self.retryable,
            "request_id": self.request_id,
            "details": self.details,
        }
