"""Verified mathematical formulas for context retrieval evaluation."""

from typing import List, Optional
from evaluation.datasets.schema import RetrievalMetrics


def compute_recall_at_k(retrieved_ids: List[str], ground_truth_ids: List[str], k: int) -> float:
    """Compute Recall@K: proportion of ground truth relevant items present in top-K retrieved items."""
    if not ground_truth_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for gt in ground_truth_ids if gt in top_k)
    return float(hits / len(ground_truth_ids))


def compute_mrr(retrieved_ids: List[str], ground_truth_ids: List[str]) -> float:
    """Compute Mean Reciprocal Rank (MRR): reciprocal rank of the first relevant item."""
    if not ground_truth_ids:
        return 0.0
    for idx, item in enumerate(retrieved_ids, start=1):
        if item in ground_truth_ids:
            return float(1.0 / idx)
    return 0.0


def compute_precision_at_k(retrieved_ids: List[str], ground_truth_ids: List[str], k: int) -> float:
    """Compute Precision@K: proportion of retrieved top-K items that are relevant."""
    if not retrieved_ids or k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for item in top_k if item in ground_truth_ids)
    return float(hits / len(top_k))


def compute_hit_rate(retrieved_ids: List[str], ground_truth_ids: List[str], k: int = 5) -> float:
    """Compute Hit Rate@K: 1.0 if at least one ground truth item is in top-K, 0.0 otherwise."""
    if not ground_truth_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    return 1.0 if any(gt in top_k for gt in ground_truth_ids) else 0.0


def aggregate_retrieval_metrics(results: List[dict]) -> RetrievalMetrics:
    """Aggregate retrieval metrics over all evaluation items with valid ground truth."""
    evaluable = [r for r in results if r.get("relevant_doc_ids") or r.get("relevant_chunk_ids")]
    if not evaluable:
        return RetrievalMetrics()

    r1_list, r3_list, r5_list, r10_list, mrr_list, p5_list, hr_list = [], [], [], [], [], [], []

    for item in evaluable:
        retrieved = item.get("retrieved_doc_ids", [])
        ground_truth = item.get("relevant_doc_ids", [])
        if not ground_truth:
            retrieved = item.get("retrieved_chunk_ids", [])
            ground_truth = item.get("relevant_chunk_ids", [])

        r1_list.append(compute_recall_at_k(retrieved, ground_truth, 1))
        r3_list.append(compute_recall_at_k(retrieved, ground_truth, 3))
        r5_list.append(compute_recall_at_k(retrieved, ground_truth, 5))
        r10_list.append(compute_recall_at_k(retrieved, ground_truth, 10))
        mrr_list.append(compute_mrr(retrieved, ground_truth))
        p5_list.append(compute_precision_at_k(retrieved, ground_truth, 5))
        hr_list.append(compute_hit_rate(retrieved, ground_truth, 5))

    return RetrievalMetrics(
        recall_at_1=round(float(sum(r1_list) / len(r1_list)), 4),
        recall_at_3=round(float(sum(r3_list) / len(r3_list)), 4),
        recall_at_5=round(float(sum(r5_list) / len(r5_list)), 4),
        recall_at_10=round(float(sum(r10_list) / len(r10_list)), 4),
        mrr=round(float(sum(mrr_list) / len(mrr_list)), 4),
        precision_at_5=round(float(sum(p5_list) / len(p5_list)), 4),
        hit_rate=round(float(sum(hr_list) / len(hr_list)), 4),
    )
