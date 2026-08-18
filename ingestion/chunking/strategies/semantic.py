"""Semantic boundary chunking strategy using sentence similarity thresholds."""

import math
from collections import Counter
from typing import Dict, List, Optional

from ingestion.chunking.base import BaseChunker
from ingestion.chunking.models import Chunk, ChunkingConfig, generate_stable_chunk_id
from ingestion.chunking.utils import count_characters, count_tokens, split_sentences
from ingestion.preprocessor import ProcessedDocument


def _text_to_vector(text: str) -> Counter:
    """Convert text into word n-gram frequency vector."""
    words = [w.lower() for w in text.split() if w.strip()]
    # Character 3-grams + word tokens for robust multilingual similarity
    ngrams = words + [text[i : i + 3].lower() for i in range(len(text) - 2)]
    return Counter(ngrams)


def _cosine_similarity(vec1: Counter, vec2: Counter) -> float:
    """Calculate cosine similarity between two term frequency vectors."""
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
    sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    return float(numerator) / denominator


class SemanticChunker(BaseChunker):
    """Semantic chunker using sentence-level vector distance & threshold boundaries."""

    @property
    def name(self) -> str:
        return "semantic"

    def chunk_document(self, document: ProcessedDocument) -> List[Chunk]:
        if not document or not document.text or not document.text.strip():
            return []

        sentences = split_sentences(document.text)

        if not sentences:
            return []

        if len(sentences) == 1:
            return [self._create_chunk(document, sentences[0], 0)]

        threshold = self.config.semantic_threshold
        target_size = max(10, self.config.target_chunk_size)

        # Compute term vectors for sentences
        sentence_vecs = [_text_to_vector(s) for s in sentences]

        chunks: List[Chunk] = []
        current_group: List[str] = [sentences[0]]
        current_tokens = count_tokens(sentences[0])
        chunk_idx = 0

        for i in range(len(sentences) - 1):
            sim = _cosine_similarity(sentence_vecs[i], sentence_vecs[i + 1])
            next_tokens = count_tokens(sentences[i + 1])

            # Split boundary if similarity drops below threshold or tokens reach limit
            if (sim < threshold or current_tokens + next_tokens > target_size) and current_group:
                chunk_text = " ".join(current_group).strip()
                chunks.append(self._create_chunk(document, chunk_text, chunk_idx))
                chunk_idx += 1
                current_group = [sentences[i + 1]]
                current_tokens = next_tokens
            else:
                current_group.append(sentences[i + 1])
                current_tokens += next_tokens

        if current_group:
            chunk_text = " ".join(current_group).strip()
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
            "semantic_threshold": self.config.semantic_threshold,
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
