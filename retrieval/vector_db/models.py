"""Data models and configuration for Qdrant vector database layer."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class QdrantConfig:
    """Centralized configuration for Qdrant vector database."""

    url: Optional[str] = None  # Remote URL e.g. "http://localhost:6333", or None for local/memory
    path: Optional[str] = "data/qdrant_db"  # Local storage path or ":memory:"
    collection_name: str = "hh_goa_chunks"
    vector_size: int = 384  # Matches Phase 4.1 embedding dimension
    distance: str = "Cosine"  # Cosine similarity metric
    batch_size: int = 64
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResultPoint:
    """Standardized search result point returned from Qdrant vector search."""

    chunk_id: str
    document_id: str
    score: float
    text: str
    chunk_index: int
    chunk_strategy: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "score": float(round(self.score, 4)),
            "text": self.text,
            "chunk_index": self.chunk_index,
            "chunk_strategy": self.chunk_strategy,
            "metadata": self.metadata,
        }
