"""RAG Orchestrator coordinating hybrid retrieval, reranking, and generation."""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from generation.gemini.context_builder import ContextBuilder
from generation.gemini.models import RAGResponse
from generation.gemini.service import GeminiService
from orchestration.models import (
    ErrorCode,
    RAGOrchestrationResponse,
    RAGOrchestratorConfig,
    generate_request_id,
)
from retrieval.hybrid.service import HybridService
from retrieval.reranking.models import RerankedResultPoint
from retrieval.reranking.service import RerankerService

logger = logging.getLogger(__name__)


class RAGOrchestrator:
    """Central production RAG orchestrator coordinating text RAG pipeline."""

    def __init__(

        self,
        config: Optional[RAGOrchestratorConfig] = None,
        hybrid_service: Optional[HybridService] = None,
        reranker_service: Optional[RerankerService] = None,
        gemini_service: Optional[GeminiService] = None,
    ):
        self.config = config or RAGOrchestratorConfig()
        self.hybrid_service = hybrid_service or HybridService()
        self.reranker_service = reranker_service or RerankerService()
        self.gemini_service = gemini_service or GeminiService()

    def validate_query(self, query_text: Optional[str]) -> Tuple[bool, str]:
        """Validate query text eligibility."""
        if not query_text or not isinstance(query_text, str) or not query_text.strip():
            return False, "Query text must be a non-empty string."
        if len(query_text.strip()) < 2:
            return False, "Query text is too short."
        return True, ""

    def validate_source_integrity(
        self,
        raw_sources: List[Any],
        valid_chunks: List[RerankedResultPoint],
    ) -> List[Dict[str, Any]]:
        """Validate that all cited sources belong to actual retrieved context chunks."""
        valid_chunk_map = {c.chunk_id: c for c in valid_chunks}
        validated_sources: List[Dict[str, Any]] = []

        for s in raw_sources:
            cid = getattr(s, "chunk_id", None) or (s.get("chunk_id") if isinstance(s, dict) else None)
            if cid and cid in valid_chunk_map:
                chunk = valid_chunk_map[cid]
                validated_sources.append({
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "rank": chunk.final_rank,
                })
            else:
                logger.warning(f"Rejected model-fabricated or invalid source chunk_id: '{cid}'.")

        # If cited sources was empty or invalid, include top retrieved chunk attributions
        if not validated_sources and valid_chunks:
            for c in valid_chunks[:3]:
                validated_sources.append({
                    "chunk_id": c.chunk_id,
                    "document_id": c.document_id,
                    "rank": c.final_rank,
                })

        return validated_sources

    def answer(
        self,
        query_text: str,
        request_id: Optional[str] = None,
    ) -> RAGOrchestrationResponse:
        """Execute end-to-end orchestrated RAG workflow."""
        t_start = time.time()
        req_id = request_id or generate_request_id()
        timing: Dict[str, float] = {}

        # 1. Query Validation
        t_val = time.time()
        is_valid, val_err = self.validate_query(query_text)
        timing["query_validation_ms"] = round((time.time() - t_val) * 1000, 2)

        if not is_valid:
            return RAGOrchestrationResponse(
                answer=val_err,
                grounded=False,
                has_context=False,
                sources=[],
                request_id=req_id,
                status="INVALID_QUERY",
                error_code=ErrorCode.INVALID_QUERY.value,
                latency_ms=round((time.time() - t_start) * 1000, 2),
                timing_breakdown=timing,
            )

        q_clean = query_text.strip()
        logger.info(f"[{req_id}] RAG_START for query: '{q_clean[:60]}...'")

        # Check overall timeout deadline
        deadline = t_start + self.config.timeout_sec

        try:
            # 2. Hybrid Retrieval
            t_ret = time.time()
            if time.time() > deadline:
                raise TimeoutError("RAG request timed out prior to retrieval stage.")

            hybrid_candidates, h_metrics = self.hybrid_service.search(
                query_text=q_clean,
                vector_top_k=self.config.vector_top_k,
                bm25_top_k=self.config.bm25_top_k,
                final_top_k=self.config.candidate_k,
            )
            timing["retrieval_ms"] = round((time.time() - t_ret) * 1000, 2)
            logger.info(f"[{req_id}] RETRIEVAL_COMPLETE: {len(hybrid_candidates)} candidates.")

            # 3. No-Context Path Handling
            if not hybrid_candidates:
                logger.info(f"[{req_id}] NO_CONTEXT path triggered.")
                return RAGOrchestrationResponse(
                    answer="Insufficient context available to answer the query.",
                    grounded=False,
                    has_context=False,
                    sources=[],
                    request_id=req_id,
                    status="NO_CONTEXT",
                    error_code=ErrorCode.NO_CONTEXT.value,
                    latency_ms=round((time.time() - t_start) * 1000, 2),
                    timing_breakdown=timing,
                )

            # 4. Reranking
            t_rer = time.time()
            if time.time() > deadline:
                raise TimeoutError("RAG request timed out prior to reranking stage.")

            reranked_chunks, r_metrics = self.reranker_service.rerank(
                query_text=q_clean,
                candidates=hybrid_candidates,
                candidate_k=self.config.candidate_k,
                final_k=self.config.final_k,
            )
            timing["rerank_ms"] = round((time.time() - t_rer) * 1000, 2)
            logger.info(f"[{req_id}] RERANK_COMPLETE: {len(reranked_chunks)} reranked chunks.")

            # 5. Gemini Generation
            t_gen = time.time()
            if time.time() > deadline:
                raise TimeoutError("RAG request timed out prior to generation stage.")

            rag_response = self.gemini_service.generate(
                query_text=q_clean,
                chunks=reranked_chunks,
            )
            timing["generation_ms"] = round((time.time() - t_gen) * 1000, 2)
            logger.info(f"[{req_id}] GENERATION_COMPLETE: grounded={rag_response.grounded}.")

            # 6. Response & Source Validation
            t_val_resp = time.time()
            validated_sources = self.validate_source_integrity(
                raw_sources=rag_response.sources,
                valid_chunks=reranked_chunks,
            )
            timing["response_validation_ms"] = round((time.time() - t_val_resp) * 1000, 2)

            total_ms = round((time.time() - t_start) * 1000, 2)
            timing["total_latency_ms"] = total_ms

            logger.info(f"[{req_id}] RAG_COMPLETE in {total_ms}ms.")

            return RAGOrchestrationResponse(
                answer=rag_response.answer,
                grounded=rag_response.grounded,
                has_context=True,
                sources=validated_sources,
                request_id=req_id,
                status="SUCCESS",
                error_code=None,
                latency_ms=total_ms,
                token_usage=rag_response.token_usage,
                timing_breakdown=timing,
                metadata={
                    "model": rag_response.model,
                    "retrieved_candidates": len(hybrid_candidates),
                    "reranked_chunks": len(reranked_chunks),
                },
            )

        except TimeoutError as exc:
            total_ms = round((time.time() - t_start) * 1000, 2)
            logger.error(f"[{req_id}] RAG_TIMEOUT: {exc}")
            return RAGOrchestrationResponse(
                answer="Application Error: Request timed out.",
                grounded=False,
                has_context=False,
                sources=[],
                request_id=req_id,
                status="TIMEOUT",
                error_code=ErrorCode.TIMEOUT.value,
                latency_ms=total_ms,
                timing_breakdown=timing,
            )
        except Exception as exc:
            total_ms = round((time.time() - t_start) * 1000, 2)
            logger.error(f"[{req_id}] RAG_ERROR: {exc}", exc_info=True)
            return RAGOrchestrationResponse(
                answer="Application Error: Internal processing failure.",
                grounded=False,
                has_context=False,
                sources=[],
                request_id=req_id,
                status="ERROR",
                error_code=ErrorCode.GENERATION_ERROR.value,
                latency_ms=total_ms,
                timing_breakdown=timing,
                metadata={"error_detail": str(exc)},
            )
