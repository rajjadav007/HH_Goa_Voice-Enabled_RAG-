"""Chunk data models and deterministic ID generator for chunking framework."""

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ChunkingConfig:
    """Centralized configuration structure for chunking strategies."""

    strategy: str = "passthrough"
    target_chunk_size: int = 256  # target tokens/words
    max_chunk_size: int = 512     # max tokens/words
    overlap: int = 32             # overlap tokens/words
    semantic_threshold: float = 0.75
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """Normalized Chunk representation for vector indexing & RAG retrieval."""

    chunk_id: str
    document_id: str
    text: str
    chunk_index: int
    chunk_strategy: str
    token_count: int
    character_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk object to dictionary representation."""
        return asdict(self)


def generate_stable_chunk_id(
    document_id: str, strategy: str, chunk_idx: int, text: str
) -> str:
    """Generate deterministic, stable chunk ID.

    Combines document_id, strategy name, chunk index, and content SHA-256 hash.
    """
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"chk_{document_id}_{strategy}_{chunk_idx}_{content_hash}"
