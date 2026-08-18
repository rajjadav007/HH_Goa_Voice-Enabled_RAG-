"""End-to-End RAG Pipeline orchestrating Hybrid Retrieval, Reranking, and Gemini Generation."""

import logging
import time
from typing import Any, Dict, List, Optional

from generation.gemini.models import GeminiConfig, RAGResponse
from generation.gemini.service import GeminiService
from retrieval.hybrid.models import HybridConfig
from retrieval.hybrid.service import HybridService
from retrieval.reranking.models import RerankerConfig
from retrieval.reranking.service import RerankerService

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Production text RAG pipeline orchestrating retrieval, reranking, and generation."""

    def __init__(
        self,
        hybrid_service: Optional[HybridService] = None,
        reranker_service: Optional[RerankerService] = None,
        gemini_service: Optional[GeminiService] = None,
    ):
        self.hybrid_service = hybrid_service or HybridService()
        self.reranker_service = reranker_service or RerankerService()
        self.gemini_service = gemini_service or GeminiService()

    def query(
        self,
        query_text: str,
        vector_top_k: int = 10,
        bm25_top_k: int = 10,
        candidate_k: int = 10,
        final_k: int = 5,
    ) -> RAGResponse:
        """Execute end-to-end RAG query pipeline."""
        t_start = time.time()

        if not query_text or not query_text.strip():
            return RAGResponse(
                answer="No query provided.",
                grounded=False,
                sources=[],
                model=self.gemini_service.config.model_name,
                latency_ms=0.0,
            )

        # 1. Hybrid Retrieval (Qdrant + BM25 + RRF)
        t_ret = time.time()
        hybrid_candidates, h_metrics = self.hybrid_service.search(
            query_text=query_text,
            vector_top_k=vector_top_k,
            bm25_top_k=bm25_top_k,
            final_top_k=candidate_k,
        )
        retrieval_ms = round((time.time() - t_ret) * 1000, 2)

        # 2. Reranking (CrossEncoder)
        t_rer = time.time()
        reranked_chunks, r_metrics = self.reranker_service.rerank(
            query_text=query_text,
            candidates=hybrid_candidates,
            candidate_k=candidate_k,
            final_k=final_k,
        )
        rerank_ms = round((time.time() - t_rer) * 1000, 2)

        # 3. Context Construction & Gemini Generation
        t_gen = time.time()
        rag_response = self.gemini_service.generate(
            query_text=query_text,
            chunks=reranked_chunks,
        )
        generation_ms = round((time.time() - t_gen) * 1000, 2)
        total_pipeline_ms = round((time.time() - t_start) * 1000, 2)

        # Update timing breakdown & overall latency
        rag_response.latency_ms = total_pipeline_ms
        rag_response.timing_breakdown.update(
            {
                "retrieval_ms": retrieval_ms,
                "rerank_ms": rerank_ms,
                "generation_stage_ms": generation_ms,
                "total_pipeline_ms": total_pipeline_ms,
            }
        )
        rag_response.metadata.update(
            {
                "vector_candidates": h_metrics.get("vector_candidates", 0),
                "bm25_candidates": h_metrics.get("bm25_candidates", 0),
                "hybrid_candidates": len(hybrid_candidates),
                "reranked_chunks": len(reranked_chunks),
            }
        )

        return rag_response
