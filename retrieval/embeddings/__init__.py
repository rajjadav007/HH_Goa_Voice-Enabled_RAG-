"""Embeddings package exports."""

from retrieval.embeddings.models import EmbeddingConfig, VectorArtifact
from retrieval.embeddings.service import EmbeddingService
from retrieval.embeddings.processor import BatchEmbeddingProcessor

__all__ = [
    "EmbeddingConfig",
    "VectorArtifact",
    "EmbeddingService",
    "BatchEmbeddingProcessor",
]
