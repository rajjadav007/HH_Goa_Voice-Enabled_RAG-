"""BM25 lexical retrieval service supporting indexing, persistence, and search."""

import json
import logging
import os
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from ingestion.chunking.models import Chunk
from retrieval.bm25.models import BM25Config, BM25ResultPoint
from retrieval.bm25.tokenizer import MultilingualBM25Tokenizer

logger = logging.getLogger(__name__)


class BM25Service:
    """Production BM25 lexical retriever handling tokenization, indexing, and search."""

    def __init__(
        self,
        config: Optional[BM25Config] = None,
        tokenizer: Optional[MultilingualBM25Tokenizer] = None,
    ):
        self.config = config or BM25Config()
        self.tokenizer = tokenizer or MultilingualBM25Tokenizer()
        self.bm25_model: Optional[BM25Okapi] = None
        self.chunks_corpus: List[Chunk] = []
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded and self.bm25_model is not None

    def tokenize(self, text: str) -> List[str]:
        return self.tokenizer.tokenize(text)

    def build_index(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """Build BM25Okapi index over a list of Chunk objects."""
        if not chunks:
            raise ValueError("Cannot build BM25 index over empty chunk list.")

        logger.info(f"Tokenizing {len(chunks)} chunks for BM25 indexing...")
        t0 = time.time()
        tokenized_corpus = [self.tokenize(c.text) for c in chunks]
        tok_time = round(time.time() - t0, 4)

        logger.info(f"Building BM25Okapi index [k1={self.config.k1}, b={self.config.b}]...")
        t1 = time.time()
        self.bm25_model = BM25Okapi(
            tokenized_corpus, k1=self.config.k1, b=self.config.b
        )
        build_time = round(time.time() - t1, 4)

        self.chunks_corpus = list(chunks)
        self._is_loaded = True

        stats = {
            "indexed_chunks": len(chunks),
            "tokenization_time_sec": tok_time,
            "index_build_time_sec": build_time,
            "total_time_sec": round(tok_time + build_time, 4),
        }
        logger.info(f"BM25 index build complete in {stats['total_time_sec']} sec.")
        return stats

    def save_index(self, index_dir: Optional[str] = None) -> Dict[str, str]:
        """Persist BM25 index model and corpus chunks to disk."""
        if not self.is_loaded:
            raise ValueError("BM25 index is not built or loaded. Build index before saving.")

        target_dir = index_dir or os.path.abspath(self.config.index_dir)
        os.makedirs(target_dir, exist_ok=True)

        pkl_path = os.path.join(target_dir, self.config.index_file_name)
        manifest_path = os.path.join(target_dir, self.config.manifest_file_name)

        data_to_save = {
            "bm25_model": self.bm25_model,
            "chunks_corpus": [c.to_dict() for c in self.chunks_corpus],
            "config": self.config,
        }

        with open(pkl_path, "wb") as f:
            pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)

        manifest = {
            "k1": self.config.k1,
            "b": self.config.b,
            "top_k": self.config.top_k,
            "indexed_chunk_count": len(self.chunks_corpus),
            "index_file": pkl_path,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved BM25 index ({len(self.chunks_corpus)} chunks) to '{pkl_path}'.")
        return {"index_file": pkl_path, "manifest_file": manifest_path}

    def load_index(self, index_dir: Optional[str] = None) -> bool:
        """Load persisted BM25 index model and chunk corpus from disk."""
        target_dir = index_dir or os.path.abspath(self.config.index_dir)
        pkl_path = os.path.join(target_dir, self.config.index_file_name)

        if not os.path.exists(pkl_path):
            logger.warning(f"BM25 index pickle file not found at '{pkl_path}'.")
            self._is_loaded = False
            return False

        logger.info(f"Loading persisted BM25 index from '{pkl_path}'...")
        with open(pkl_path, "rb") as f:
            loaded_data = pickle.load(f)

        self.bm25_model = loaded_data["bm25_model"]
        raw_chunks = loaded_data["chunks_corpus"]
        self.chunks_corpus = [Chunk.from_dict(c) for c in raw_chunks]
        self._is_loaded = True

        logger.info(f"Loaded BM25 index with {len(self.chunks_corpus)} chunks.")
        return True

    def health_check(self) -> Dict[str, Any]:
        """Return health and status information for BM25 retriever."""
        return {
            "status": "healthy" if self.is_loaded else "not_loaded",
            "is_loaded": self.is_loaded,
            "indexed_chunks": len(self.chunks_corpus) if self.is_loaded else 0,
            "k1": self.config.k1,
            "b": self.config.b,
            "top_k": self.config.top_k,
        }

    def search(
        self, query_text: str, top_k: Optional[int] = None
    ) -> List[BM25ResultPoint]:
        """Execute BM25 lexical search over indexed chunk corpus."""
        if not self.is_loaded:
            raise ValueError("BM25 index is not loaded. Load or build index before searching.")

        if not query_text or not query_text.strip():
            return []

        limit = top_k or self.config.top_k
        q_tokens = self.tokenize(query_text)

        if not q_tokens:
            return []

        scores = self.bm25_model.get_scores(q_tokens)
        max_score = float(np.max(scores)) if len(scores) > 0 else 0.0

        top_indices = np.argsort(scores)[::-1][:limit]

        results: List[BM25ResultPoint] = []
        for rank, idx in enumerate(top_indices, start=1):
            score = float(scores[idx])

            # If max_score > 0, filter out non-positive scores.
            # If max_score == 0 (e.g. tiny N=2 test corpus where Robertson IDF is 0.0), check token overlap.
            if max_score > 0.0 and score <= 0.0:
                continue

            chunk = self.chunks_corpus[idx]
            if max_score == 0.0:
                doc_tokens = set(self.tokenize(chunk.text))
                if not any(qt in doc_tokens for qt in q_tokens):
                    continue

            res = BM25ResultPoint(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                score=score,
                rank=rank,
                text=chunk.text,
                method="bm25",
                metadata=dict(chunk.metadata),
            )
            results.append(res)

        return results
