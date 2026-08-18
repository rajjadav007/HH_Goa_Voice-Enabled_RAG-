"""Reciprocal Rank Fusion (RRF) module for combining rank lists."""

from typing import Any, Dict, List, Optional, Tuple

from retrieval.bm25.models import BM25ResultPoint
from retrieval.hybrid.models import HybridResultPoint
from retrieval.vector_db.models import SearchResultPoint


def compute_rrf_scores(
    vector_results: List[SearchResultPoint],
    bm25_results: List[BM25ResultPoint],
    rrf_k: int = 60,
    final_top_k: int = 5,
) -> List[HybridResultPoint]:
    """Compute Reciprocal Rank Fusion scores across vector and BM25 search results."""

    # Map chunk_id -> info accumulator
    # info structure: {"document_id": str, "text": str, "vector_rank": Optional[int], "bm25_rank": Optional[int], "rrf_score": float, "metadata": Dict}
    chunk_accumulator: Dict[str, Dict[str, Any]] = {}

    # Accumulate vector ranks
    for v_rank, v_point in enumerate(vector_results, start=1):
        cid = v_point.chunk_id
        if not cid:
            continue

        if cid not in chunk_accumulator:
            chunk_accumulator[cid] = {
                "chunk_id": cid,
                "document_id": v_point.document_id,
                "text": v_point.text,
                "vector_rank": v_rank,
                "bm25_rank": None,
                "rrf_score": 0.0,
                "metadata": dict(v_point.metadata or {}),
            }

        chunk_accumulator[cid]["vector_rank"] = v_rank
        chunk_accumulator[cid]["rrf_score"] += 1.0 / (rrf_k + v_rank)

    # Accumulate BM25 ranks
    for b_rank, b_point in enumerate(bm25_results, start=1):
        cid = b_point.chunk_id
        if not cid:
            continue

        if cid not in chunk_accumulator:
            chunk_accumulator[cid] = {
                "chunk_id": cid,
                "document_id": b_point.document_id,
                "text": b_point.text,
                "vector_rank": None,
                "bm25_rank": b_rank,
                "rrf_score": 0.0,
                "metadata": dict(b_point.metadata or {}),
            }
        else:
            # Update text if vector payload was empty
            if not chunk_accumulator[cid]["text"] and b_point.text:
                chunk_accumulator[cid]["text"] = b_point.text
            chunk_accumulator[cid]["metadata"].update(b_point.metadata or {})

        chunk_accumulator[cid]["bm25_rank"] = b_rank
        chunk_accumulator[cid]["rrf_score"] += 1.0 / (rrf_k + b_rank)

    # Convert accumulator dict to list and sort by RRF score descending
    candidates = list(chunk_accumulator.values())
    candidates.sort(key=lambda item: item["rrf_score"], reverse=True)

    # Build final HybridResultPoint objects
    fused_results: List[HybridResultPoint] = []
    for rank, item in enumerate(candidates[:final_top_k], start=1):
        sources = []
        if item["vector_rank"] is not None:
            sources.append("vector")
        if item["bm25_rank"] is not None:
            sources.append("bm25")

        point = HybridResultPoint(
            chunk_id=item["chunk_id"],
            document_id=item["document_id"],
            score=round(float(item["rrf_score"]), 6),
            rank=rank,
            text=item["text"],
            sources=sources,
            vector_rank=item["vector_rank"],
            bm25_rank=item["bm25_rank"],
            metadata=item["metadata"],
        )
        fused_results.append(point)

    return fused_results
