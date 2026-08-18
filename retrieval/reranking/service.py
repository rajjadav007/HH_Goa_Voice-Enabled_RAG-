"""Reranking service using CrossEncoder for second-stage context reranking."""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from sentence_transformers import CrossEncoder

from retrieval.hybrid.models import HybridResultPoint
from retrieval.reranking.models import RerankerConfig, RerankedResultPoint

logger = logging.getLogger(__name__)


class RerankerService:
    """Production CrossEncoder Reranker with batch scoring and fallback handling."""

    _model_cache: Dict[str, Any] = {}

    def __init__(self, config: Optional[RerankerConfig] = None):
        self.config = config or RerankerConfig()
        self.model: Optional[CrossEncoder] = None
        self._is_loaded = False
        self.device = self._resolve_device(self.config.device)

    def _resolve_device(self, requested_device: str) -> str:
        if requested_device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return requested_device

    def load_model(self) -> bool:
        """Load CrossEncoder reranker model safely."""
        if self._is_loaded and self.model is not None:
            return True

        if not self.config.enabled:
            logger.info("Reranker is disabled in config. Skipping model load.")
            return False

        cache_key = f"{self.config.model_name}:{self.device}"
        if cache_key in RerankerService._model_cache:
            self.model = RerankerService._model_cache[cache_key]
            self._is_loaded = True
            return True

        try:
            logger.info(
                f"Loading CrossEncoder reranker model '{self.config.model_name}' on device '{self.device}'..."
            )
            t0 = time.time()
            self.model = CrossEncoder(
                self.config.model_name,
                device=self.device,
            )
            self._is_loaded = True
            RerankerService._model_cache[cache_key] = self.model
            logger.info(f"Reranker model loaded successfully in {round(time.time() - t0, 2)}s.")
            return True
        except Exception as exc:
            logger.error(f"Failed to load CrossEncoder reranker model: {exc}")
            self._is_loaded = False
            return False

    def health_check(self) -> Dict[str, Any]:
        return {
            "enabled": self.config.enabled,
            "is_loaded": self._is_loaded,
            "model_name": self.config.model_name,
            "device": self.device,
            "candidate_k": self.config.candidate_k,
            "final_k": self.config.final_k,
        }

    def rerank(
        self,
        query_text: str,
        candidates: List[HybridResultPoint],
        candidate_k: Optional[int] = None,
        final_k: Optional[int] = None,
    ) -> Tuple[List[RerankedResultPoint], Dict[str, Any]]:
        """Rerank hybrid candidate list using CrossEncoder pair relevance scoring."""
        t_start = time.time()

        if not candidates or not query_text or not query_text.strip():
            return [], {"total_ms": 0.0, "status": "empty_input"}

        c_k = candidate_k or self.config.candidate_k
        f_k = final_k or self.config.final_k
        pool = candidates[:c_k]

        # Fallback if reranker is disabled or fails to load
        if not self.config.enabled or not self.load_model() or self.model is None:
            logger.info("Reranker disabled/unavailable. Falling back to Hybrid RRF candidate order.")
            results: List[RerankedResultPoint] = []
            for rank, c in enumerate(pool[:f_k], start=1):
                results.append(
                    RerankedResultPoint(
                        chunk_id=c.chunk_id,
                        document_id=c.document_id,
                        rerank_score=c.score,
                        final_rank=rank,
                        text=c.text,
                        sources=c.sources,
                        vector_rank=c.vector_rank,
                        bm25_rank=c.bm25_rank,
                        rrf_score=c.score,
                        metadata=dict(c.metadata or {}),
                    )
                )
            total_ms = round((time.time() - t_start) * 1000, 2)
            return results, {
                "candidate_count": len(pool),
                "reranked_count": len(results),
                "fallback_mode": True,
                "total_ms": total_ms,
            }

        try:
            # Construct (Query, Candidate_Text) pairs for batch inference
            pairs = [[query_text, c.text] for c in pool]
            scores = self.model.predict(
                pairs, batch_size=self.config.batch_size, show_progress_bar=False
            )

            # Attach predicted scores to candidates
            scored_candidates: List[Tuple[HybridResultPoint, float]] = []
            for idx, c in enumerate(pool):
                score_val = float(scores[idx])
                scored_candidates.append((c, score_val))

            # Sort by CrossEncoder rerank score descending
            scored_candidates.sort(key=lambda item: item[1], reverse=True)

            reranked_results: List[RerankedResultPoint] = []
            for rank, (c, score_val) in enumerate(scored_candidates[:f_k], start=1):
                point = RerankedResultPoint(
                    chunk_id=c.chunk_id,
                    document_id=c.document_id,
                    rerank_score=score_val,
                    final_rank=rank,
                    text=c.text,
                    sources=c.sources,
                    vector_rank=c.vector_rank,
                    bm25_rank=c.bm25_rank,
                    rrf_score=c.score,
                    metadata=dict(c.metadata or {}),
                )
                reranked_results.append(point)

            total_ms = round((time.time() - t_start) * 1000, 2)
            metrics = {
                "candidate_count": len(pool),
                "reranked_count": len(reranked_results),
                "fallback_mode": False,
                "total_ms": total_ms,
            }
            return reranked_results, metrics

        except Exception as exc:
            logger.error(f"Error during CrossEncoder reranking: {exc}. Falling back to RRF order.")
            results = []
            for rank, c in enumerate(pool[:f_k], start=1):
                results.append(
                    RerankedResultPoint(
                        chunk_id=c.chunk_id,
                        document_id=c.document_id,
                        rerank_score=c.score,
                        final_rank=rank,
                        text=c.text,
                        sources=c.sources,
                        vector_rank=c.vector_rank,
                        bm25_rank=c.bm25_rank,
                        rrf_score=c.score,
                        metadata=dict(c.metadata or {}),
                    )
                )
            total_ms = round((time.time() - t_start) * 1000, 2)
            return results, {
                "candidate_count": len(pool),
                "reranked_count": len(results),
                "fallback_mode": True,
                "error": str(exc),
                "total_ms": total_ms,
            }
