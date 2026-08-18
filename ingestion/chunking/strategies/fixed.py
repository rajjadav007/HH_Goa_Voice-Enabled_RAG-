"""Fixed-size token/word sliding window chunking strategy."""

from typing import List, Optional

from ingestion.chunking.base import BaseChunker
from ingestion.chunking.models import Chunk, ChunkingConfig, generate_stable_chunk_id
from ingestion.chunking.utils import count_characters, count_tokens
from ingestion.preprocessor import ProcessedDocument


class FixedSizeChunker(BaseChunker):
    """Fixed-size chunker using a sliding token window with configurable overlap."""

    @property
    def name(self) -> str:
        return "fixed"

    def chunk_document(self, document: ProcessedDocument) -> List[Chunk]:
        if not document or not document.text or not document.text.strip():
            return []

        text = document.text.strip()
        words = text.split()

        if not words:
            return []

        chunk_size = max(10, self.config.target_chunk_size)
        overlap = max(0, min(self.config.overlap, chunk_size - 1))
        step = chunk_size - overlap

        chunks: List[Chunk] = []
        chunk_idx = 0

        i = 0
        while i < len(words):
            chunk_words = words[i : i + chunk_size]
            chunk_text = " ".join(chunk_words).strip()

            if chunk_text:
                chk_id = generate_stable_chunk_id(
                    document_id=document.document_id,
                    strategy=self.name,
                    chunk_idx=chunk_idx,
                    text=chunk_text,
                )

                metadata = dict(document.metadata) if document.metadata else {}
                metadata.update({
                    "source_query_id": document.source_query_id,
                    "passage_index": document.passage_index,
                    "is_selected": document.is_selected,
                    "language": document.language,
                    "english_text": document.english_text,
                    "start_word_index": i,
                    "end_word_index": i + len(chunk_words),
                })

                c = Chunk(
                    chunk_id=chk_id,
                    document_id=document.document_id,
                    text=chunk_text,
                    chunk_index=chunk_idx,
                    chunk_strategy=self.name,
                    token_count=count_tokens(chunk_text),
                    character_count=count_characters(chunk_text),
                    metadata=metadata,
                )
                chunks.append(c)
                chunk_idx += 1

            i += step
            if i >= len(words) and i - step + len(words[i - step :]) == len(words):
                break

        return chunks
