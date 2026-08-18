"""Data models and configurations for embedding service."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EmbeddingConfig:
    """Configuration for production embedding service."""

    model_name: str = "intfloat/multilingual-e5-small"
    batch_size: int = 32
    normalize_embeddings: bool = True
    device: str = "auto"  # 'auto', 'cpu', 'cuda'
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorArtifact:
    """Traceable vector representation artifact connecting chunk to embedding vector."""

    chunk_id: str
    document_id: str
    vector: List[float]
    embedding_model: str
    dimension: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "vector": self.vector,
            "embedding_model": self.embedding_model,
            "dimension": self.dimension,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VectorArtifact":
        return cls(
            chunk_id=data["chunk_id"],
            document_id=data["document_id"],
            vector=data["vector"],
            embedding_model=data["embedding_model"],
            dimension=data["dimension"],
            metadata=data.get("metadata", {}),
        )
