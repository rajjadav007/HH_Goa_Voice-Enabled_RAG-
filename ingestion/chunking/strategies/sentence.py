"""Sentence-aware chunking strategy."""

from typing import List, Optional

from ingestion.chunking.base import BaseChunker
from ingestion.chunking.models import Chunk, ChunkingConfig, generate_stable_chunk_id
from ingestion.chunking.utils import count_characters, count_tokens, split_sentences
from ingestion.preprocessor import ProcessedDocument


class SentenceChunker(BaseChunker):
    """Chunker that respects sentence boundaries when grouping text."""

    @property
    def name(self) -> str:
        return "sentence"

    def chunk_document(self, document: ProcessedDocument) -> List[Chunk]:
        if not document or not document.text or not document.text.strip():
            return []

        sentences = split_sentences(document.text)

        if not sentences:
            return []

        target_size = max(10, self.config.target_chunk_size)
        max_size = max(target_size, self.config.max_chunk_size)

        chunks: List[Chunk] = []
        current_sentences: List[str] = []
        current_token_count = 0
        chunk_idx = 0

        for sentence in sentences:
            s_tokens = count_tokens(sentence)

            # If a single sentence is larger than max_size, flush current and add sentence alone
            if s_tokens > max_size:
                if current_sentences:
                    chunk_text = " ".join(current_sentences).strip()
                    chunks.append(self._create_chunk(document, chunk_text, chunk_idx))
                    chunk_idx += 1
                    current_sentences = []
                    current_token_count = 0

                chunks.append(self._create_chunk(document, sentence, chunk_idx))
                chunk_idx += 1
                continue

            if current_token_count + s_tokens > target_size and current_sentences:
                chunk_text = " ".join(current_sentences).strip()
                chunks.append(self._create_chunk(document, chunk_text, chunk_idx))
                chunk_idx += 1
                current_sentences = [sentence]
                current_token_count = s_tokens
            else:
                current_sentences.append(sentence)
                current_token_count += s_tokens

        if current_sentences:
            chunk_text = " ".join(current_sentences).strip()
            chunks.append(self._create_chunk(document, chunk_text, chunk_idx))

        return chunks

    def _create_chunk(
        self, document: ProcessedDocument, text: str, chunk_idx: int
    ) -> Chunk:
        chk_id = generate_stable_chunk_id(
            document_id=document.document_id,
            strategy=self.name,
            chunk_idx=chunk_idx,
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

        return Chunk(
            chunk_id=chk_id,
            document_id=document.document_id,
            text=text,
            chunk_index=chunk_idx,
            chunk_strategy=self.name,
            token_count=count_tokens(text),
            character_count=count_characters(text),
            metadata=metadata,
        )
