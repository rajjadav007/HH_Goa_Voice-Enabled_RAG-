"""Data models and configuration for Gemini RAG Generation layer."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GeminiConfig:
    """Centralized configuration for Gemini RAG Generation."""

    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.1
    max_output_tokens: int = 512
    timeout_sec: float = 10.0
    max_retries: int = 3
    context_token_budget: int = 2000
    api_key_env_var: str = "GEMINI_API_KEY"
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SourceAttribution:
    """Source chunk attribution for grounded RAG answer."""

    chunk_id: str
    document_id: str
    rank: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "rank": self.rank,
        }


@dataclass
class RAGResponse:
    """Standardized response object returned by RAG generation pipeline."""

    answer: str
    grounded: bool
    sources: List[SourceAttribution]
    model: str
    latency_ms: float
    token_usage: Dict[str, int] = field(default_factory=dict)
    timing_breakdown: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "grounded": self.grounded,
            "sources": [s.to_dict() for s in self.sources],
            "model": self.model,
            "latency_ms": float(round(self.latency_ms, 2)),
            "token_usage": self.token_usage,
            "timing_breakdown": self.timing_breakdown,
            "metadata": self.metadata,
        }
