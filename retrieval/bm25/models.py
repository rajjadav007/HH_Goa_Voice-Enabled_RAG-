"""Data models and configuration for BM25 lexical retrieval layer."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BM25Config:
    """Configuration structure for BM25 lexical retriever."""

    k1: float = 1.5
    b: float = 0.75
    top_k: int = 10
    index_dir: str = "data/bm25_index"
    index_file_name: str = "bm25.pkl"
    manifest_file_name: str = "manifest.json"
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BM25ResultPoint:
    """Standardized result object for BM25 lexical search."""

    chunk_id: str
    document_id: str
    score: float
    rank: int
    text: str
    method: str = "bm25"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "score": float(round(self.score, 4)),
            "rank": self.rank,
            "method": self.method,
            "text": self.text,
            "metadata": self.metadata,
        }
