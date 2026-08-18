"""Structure-aware paragraph and section chunking strategy."""

from typing import List, Optional

from ingestion.chunking.base import BaseChunker
from ingestion.chunking.models import Chunk, ChunkingConfig, generate_stable_chunk_id
from ingestion.chunking.utils import (
    count_characters,
    count_tokens,
    split_paragraphs,
    split_sentences,
)
from ingestion.preprocessor import ProcessedDocument


class StructureAwareChunker(BaseChunker):
    """Structure-aware chunker prioritizing paragraph boundaries before size limits."""

    @property
    def name(self) -> str:
        return "structure"

    def chunk_document(self, document: ProcessedDocument) -> List[Chunk]:
        if not document or not document.text or not document.text.strip():
            return []

        paragraphs = split_paragraphs(document.text)

        if not paragraphs:
            return []

        target_size = max(10, self.config.target_chunk_size)
        max_size = max(target_size, self.config.max_chunk_size)

        chunks: List[Chunk] = []
        current_paras: List[str] = []
        current_token_count = 0
        chunk_idx = 0

        for para in paragraphs:
            p_tokens = count_tokens(para)

            # If single paragraph exceeds max_size, break down by sentences
            if p_tokens > max_size:
                if current_paras:
                    chunk_text = "\n\n".join(current_paras).strip()
                    chunks.append(self._create_chunk(document, chunk_text, chunk_idx))
                    chunk_idx += 1
                    current_paras = []
                    current_token_count = 0

                s_chunks = self._chunk_large_paragraph(document, para, chunk_idx, target_size)
                chunks.extend(s_chunks)
                chunk_idx += len(s_chunks)
                continue

            if current_token_count + p_tokens > target_size and current_paras:
                chunk_text = "\n\n".join(current_paras).strip()
                chunks.append(self._create_chunk(document, chunk_text, chunk_idx))
                chunk_idx += 1
                current_paras = [para]
                current_token_count = p_tokens
            else:
                current_paras.append(para)
                current_token_count += p_tokens

        if current_paras:
            chunk_text = "\n\n".join(current_paras).strip()
            chunks.append(self._create_chunk(document, chunk_text, chunk_idx))

        return chunks

    def _chunk_large_paragraph(
        self, document: ProcessedDocument, para: str, start_idx: int, target_size: int
    ) -> List[Chunk]:
        sentences = split_sentences(para)
        chunks: List[Chunk] = []
        current_s: List[str] = []
        current_t = 0
        c_idx = start_idx

        for s in sentences:
            st = count_tokens(s)
            if current_t + st > target_size and current_s:
                chunk_text = " ".join(current_s).strip()
                chunks.append(self._create_chunk(document, chunk_text, c_idx))
                c_idx += 1
                current_s = [s]
                current_t = st
            else:
                current_s.append(s)
                current_t += st

        if current_s:
            chunk_text = " ".join(current_s).strip()
            chunks.append(self._create_chunk(document, chunk_text, c_idx))

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
            "structure_unit": "paragraph",
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
