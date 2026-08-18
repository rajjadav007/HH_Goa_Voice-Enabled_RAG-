"""Configuration for RAG Harness, retries, timeouts, and fallback policy."""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class HarnessConfig:
    """Centralized reliability settings for orchestration harness."""

    enabled: bool = True
    max_retries: int = 3
    initial_backoff_ms: float = 50.0
    max_backoff_ms: float = 1000.0

    # Stage & overall timeouts
    total_timeout_sec: float = 10.0
    retrieval_timeout_sec: float = 3.0
    rerank_timeout_sec: float = 2.0
    generation_timeout_sec: float = 5.0
    grounding_timeout_sec: float = 2.0

    # Fallback policies
    enable_qdrant_to_bm25_fallback: bool = True
    enable_reranker_to_hybrid_fallback: bool = True

    extra_params: Dict[str, Any] = field(default_factory=dict)
