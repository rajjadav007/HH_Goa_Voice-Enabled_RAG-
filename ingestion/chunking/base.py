"""Base Abstract Chunker interface, chunk validation, and passthrough chunker."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ingestion.chunking.models import (
    Chunk,
    ChunkingConfig,
    generate_stable_chunk_id,
)
from ingestion.chunking.utils import count_characters, count_tokens
from ingestion.preprocessor import ProcessedDocument


class InvalidChunkError(Exception):
    """Raised when generated Chunk fails validation checks."""


def validate_chunk(chunk: Chunk) -> bool:
    """Validate Chunk object fields, boundaries, and metadata structure."""
    if not isinstance(chunk, Chunk):
        raise InvalidChunkError(f"Expected Chunk object, got {type(chunk)}")

    if not chunk.chunk_id or not isinstance(chunk.chunk_id, str):
        raise InvalidChunkError("Chunk missing valid chunk_id")

    if not chunk.document_id or not isinstance(chunk.document_id, str):
        raise InvalidChunkError(f"Chunk {chunk.chunk_id} missing valid document_id")

    if not chunk.text or not isinstance(chunk.text, str) or not chunk.text.strip():
        raise InvalidChunkError(f"Chunk {chunk.chunk_id} contains empty or whitespace-only text")

    if chunk.chunk_index < 0:
        raise InvalidChunkError(f"Chunk {chunk.chunk_id} has invalid negative chunk_index")

    if not chunk.chunk_strategy:
        raise InvalidChunkError(f"Chunk {chunk.chunk_id} missing chunk_strategy")

    if chunk.character_count <= 0 or chunk.token_count <= 0:
        raise InvalidChunkError(f"Chunk {chunk.chunk_id} has invalid length metrics (token={chunk.token_count}, char={chunk.character_count})")

    if not isinstance(chunk.metadata, dict):
        raise InvalidChunkError(f"Chunk {chunk.chunk_id} metadata must be a dictionary")

    return True


class BaseChunker(ABC):
    """Abstract Base Class for all chunking strategy implementations."""

    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()

    @property
    @abstractmethod
    def name(self) -> str:
        """Name identifier of the chunking strategy."""

    @abstractmethod
    def chunk_document(self, document: ProcessedDocument) -> List[Chunk]:
        """Chunk a single ProcessedDocument into a list of Chunk objects."""

    def process(self, document: ProcessedDocument) -> List[Chunk]:
        """Execute chunking and validate all resulting Chunk objects."""
        chunks = self.chunk_document(document)
        for c in chunks:
            validate_chunk(c)
        return chunks


class PassthroughChunker(BaseChunker):
    """Foundation chunker strategy: 1 ProcessedDocument -> 1 Chunk.

    Preserves entire document content as a single chunk. Used for framework
    validation, baseline comparison, and testing without implementing Phase 3.2 strategies.
    """

    @property
    def name(self) -> str:
        return "passthrough"

    def chunk_document(self, document: ProcessedDocument) -> List[Chunk]:
        if not document or not document.text or not document.text.strip():
            return []

        text = document.text.strip()
        chk_id = generate_stable_chunk_id(
            document_id=document.document_id,
            strategy=self.name,
            chunk_idx=0,
            text=text,
        )

        metadata = dict(document.metadata) if document.metadata else {}
        metadata.update({
            "source_query_id": document.source_query_id,
            "passage_index": document.passage_index,
            "is_selected": document.is_selected,
            "language": document.language,
            "english_text": document.english_text,
        })

        chunk = Chunk(
            chunk_id=chk_id,
            document_id=document.document_id,
            text=text,
            chunk_index=0,
            chunk_strategy=self.name,
            token_count=count_tokens(text),
            character_count=count_characters(text),
            metadata=metadata,
        )
        return [chunk]
