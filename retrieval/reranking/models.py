"""Data models and configuration for Reranking layer."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RerankerConfig:
    """Centralized configuration for Reranking service."""

    enabled: bool = True
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    candidate_k: int = 10
    final_k: int = 5
    device: str = "auto"
    batch_size: int = 16
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RerankedResultPoint:
    """Standardized result object for reranked context chunks."""

    chunk_id: str
    document_id: str
    rerank_score: float
    final_rank: int
    text: str
    sources: List[str]
    vector_rank: Optional[int] = None
    bm25_rank: Optional[int] = None
    rrf_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "rerank_score": float(round(self.rerank_score, 6)),
            "final_rank": self.final_rank,
            "text": self.text,
            "sources": self.sources,
            "vector_rank": self.vector_rank,
            "bm25_rank": self.bm25_rank,
            "rrf_score": float(round(self.rrf_score, 6)) if self.rrf_score is not None else None,
            "metadata": self.metadata,
        }
