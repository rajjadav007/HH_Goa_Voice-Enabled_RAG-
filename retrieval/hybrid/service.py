"""Hybrid Retrieval service orchestrating parallel Qdrant & BM25 search with RRF fusion."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from retrieval.bm25.models import BM25ResultPoint
from retrieval.bm25.service import BM25Service
from retrieval.embeddings.service import EmbeddingService
from retrieval.hybrid.models import HybridConfig, HybridResultPoint
from retrieval.hybrid.rrf import compute_rrf_scores
from retrieval.vector_db.models import SearchResultPoint
from retrieval.vector_db.service import QdrantService

logger = logging.getLogger(__name__)


class HybridService:
    """Production Hybrid Retriever unifying semantic and lexical retrieval."""

    def __init__(
        self,
        config: Optional[HybridConfig] = None,
        embedding_service: Optional[EmbeddingService] = None,
        qdrant_service: Optional[QdrantService] = None,
        bm25_service: Optional[BM25Service] = None,
    ):
        self.config = config or HybridConfig()
        self.embedding_service = embedding_service or EmbeddingService()
        self.qdrant_service = qdrant_service or QdrantService()
        self.bm25_service = bm25_service or BM25Service()

    def _execute_vector_search(
        self, query_text: str, top_k: int
    ) -> Tuple[List[SearchResultPoint], float]:
        """Generate query embedding and perform Qdrant vector search."""
        t0 = time.time()
        try:
            query_vector = self.embedding_service.embed_text(query_text, is_query=True)
            results = self.qdrant_service.search(query_vector=query_vector, top_k=top_k)
            elapsed_ms = round((time.time() - t0) * 1000, 2)
            return results, elapsed_ms
        except Exception as exc:
            logger.error(f"Vector search failed: {exc}")
            elapsed_ms = round((time.time() - t0) * 1000, 2)
            return [], elapsed_ms

    def _execute_bm25_search(
        self, query_text: str, top_k: int
    ) -> Tuple[List[BM25ResultPoint], float]:
        """Perform BM25 lexical search."""
        t0 = time.time()
        try:
            if not self.bm25_service.is_loaded:
                self.bm25_service.load_index()
            results = self.bm25_service.search(query_text=query_text, top_k=top_k)
            elapsed_ms = round((time.time() - t0) * 1000, 2)
            return results, elapsed_ms
        except Exception as exc:
            logger.error(f"BM25 search failed: {exc}")
            elapsed_ms = round((time.time() - t0) * 1000, 2)
            return [], elapsed_ms

    def search(
        self,
        query_text: str,
        vector_top_k: Optional[int] = None,
        bm25_top_k: Optional[int] = None,
        rrf_k: Optional[int] = None,
        final_top_k: Optional[int] = None,
        parallel: Optional[bool] = None,
    ) -> Tuple[List[HybridResultPoint], Dict[str, Any]]:
        """Execute hybrid search using concurrent or sequential Qdrant + BM25 retrieval."""
        t_start = time.time()

        if not query_text or not query_text.strip():
            return [], {"total_ms": 0.0, "status": "empty_query"}

        v_k = vector_top_k or self.config.vector_top_k
        b_k = bm25_top_k or self.config.bm25_top_k
        k_rrf = rrf_k or self.config.rrf_k
        f_k = final_top_k or self.config.final_top_k
        use_parallel = parallel if parallel is not None else self.config.enable_parallel

        vector_results: List[SearchResultPoint] = []
        bm25_results: List[BM25ResultPoint] = []
        vec_ms, bm25_ms = 0.0, 0.0

        if use_parallel:
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_vec = executor.submit(self._execute_vector_search, query_text, v_k)
                future_bm25 = executor.submit(self._execute_bm25_search, query_text, b_k)

                vector_results, vec_ms = future_vec.result()
                bm25_results, bm25_ms = future_bm25.result()
        else:
            vector_results, vec_ms = self._execute_vector_search(query_text, v_k)
            bm25_results, bm25_ms = self._execute_bm25_search(query_text, b_k)

        # Execute Reciprocal Rank Fusion
        t_rrf = time.time()
        fused_results = compute_rrf_scores(
            vector_results=vector_results,
            bm25_results=bm25_results,
            rrf_k=k_rrf,
            final_top_k=f_k,
        )
        rrf_ms = round((time.time() - t_rrf) * 1000, 2)
        total_ms = round((time.time() - t_start) * 1000, 2)

        metrics = {
            "query": query_text,
            "vector_candidates": len(vector_results),
            "bm25_candidates": len(bm25_results),
            "fused_results": len(fused_results),
            "has_relevant_context": len(fused_results) > 0,
            "parallel_execution": use_parallel,
            "timing_ms": {
                "vector_ms": vec_ms,
                "bm25_ms": bm25_ms,
                "rrf_ms": rrf_ms,
                "total_ms": total_ms,
            },
        }

        return fused_results, metrics
