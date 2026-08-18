"""Data models and configuration for Hybrid Retrieval layer."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class HybridConfig:
    """Centralized configuration for Hybrid Retrieval."""

    vector_top_k: int = 10
    bm25_top_k: int = 10
    rrf_k: int = 60
    final_top_k: int = 5
    enable_parallel: bool = True
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HybridResultPoint:
    """Standardized result object for fused hybrid retrieval."""

    chunk_id: str
    document_id: str
    score: float  # RRF score
    rank: int  # Final hybrid rank
    text: str
    sources: List[str]  # e.g. ["vector", "bm25"]
    vector_rank: Optional[int] = None
    bm25_rank: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "score": float(round(self.score, 6)),
            "rank": self.rank,
            "text": self.text,
            "sources": self.sources,
            "vector_rank": self.vector_rank,
            "bm25_rank": self.bm25_rank,
            "metadata": self.metadata,
        }
