"""Hybrid chunking strategy combining structural, sentence, and size/overlap constraints."""

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


class HybridChunker(BaseChunker):
    """Hybrid chunker combining paragraph boundaries, sentence grouping, token limits, and controlled overlap."""

    @property
    def name(self) -> str:
        return "hybrid"

    def chunk_document(self, document: ProcessedDocument) -> List[Chunk]:
        if not document or not document.text or not document.text.strip():
            return []

        # 1. Structural unit splitting (paragraphs)
        paragraphs = split_paragraphs(document.text)
        if not paragraphs:
            return []

        # 2. Extract sentence units from paragraphs
        all_sentences: List[str] = []
        for p in paragraphs:
            s_list = split_sentences(p)
            if s_list:
                all_sentences.extend(s_list)
            else:
                all_sentences.append(p)

        target_size = max(10, self.config.target_chunk_size)
        overlap_tokens = max(0, min(self.config.overlap, target_size - 1))

        chunks: List[Chunk] = []
        current_sentences: List[str] = []
        current_tokens = 0
        chunk_idx = 0

        i = 0
        while i < len(all_sentences):
            sentence = all_sentences[i]
            st = count_tokens(sentence)

            if current_tokens + st > target_size and current_sentences:
                chunk_text = " ".join(current_sentences).strip()
                chunks.append(self._create_chunk(document, chunk_text, chunk_idx))
                chunk_idx += 1

                # Calculate overlap: backtrack sentences to keep overlap_tokens
                overlap_acc: List[str] = []
                overlap_t = 0
                for prev_s in reversed(current_sentences):
                    pt = count_tokens(prev_s)
                    if overlap_t + pt <= overlap_tokens:
                        overlap_acc.insert(0, prev_s)
                        overlap_t += pt
                    else:
                        break

                current_sentences = overlap_acc + [sentence]
                current_tokens = overlap_t + st
            else:
                current_sentences.append(sentence)
                current_tokens += st

            i += 1

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
            "hybrid_components": ["structure", "sentence", "token_boundary", "overlap"],
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
