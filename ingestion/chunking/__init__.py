"""Chunking framework package."""

from ingestion.chunking.models import (
    Chunk,
    ChunkingConfig,
    generate_stable_chunk_id,
)
from ingestion.chunking.base import (
    BaseChunker,
    InvalidChunkError,
    PassthroughChunker,
    validate_chunk,
)
from ingestion.chunking.registry import (
    ChunkerRegistry,
    StrategyNotFoundError,
)
from ingestion.chunking.utils import (
    count_characters,
    count_tokens,
    split_paragraphs,
    split_sentences,
)
from ingestion.chunking.processor import (
    BatchChunkProcessor,
)
import ingestion.chunking.strategies  # Auto-registers all strategies

__all__ = [
    "Chunk",
    "ChunkingConfig",
    "generate_stable_chunk_id",
    "BaseChunker",
    "PassthroughChunker",
    "InvalidChunkError",
    "validate_chunk",
    "ChunkerRegistry",
    "StrategyNotFoundError",
    "count_characters",
    "count_tokens",
    "split_paragraphs",
    "split_sentences",
    "BatchChunkProcessor",
]
