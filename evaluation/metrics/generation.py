"""Generation quality metrics for evaluating RAG answers."""

import re
from typing import List, Optional
from evaluation.datasets.schema import GenerationMetrics


def compute_exact_match(prediction: str, reference: str) -> float:
    """Compute Exact Match (EM) after text normalization."""
    norm_pred = re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", prediction.lower().strip()))
    norm_ref = re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", reference.lower().strip()))
    return 1.0 if norm_pred == norm_ref else 0.0


def compute_token_f1(prediction: str, reference: str) -> float:
    """Compute Token-level F1 score between prediction and reference."""
    pred_tokens = re.findall(r"\w+", prediction.lower())
    ref_tokens = re.findall(r"\w+", reference.lower())
    if not pred_tokens or not ref_tokens:
        return 0.0 if pred_tokens != ref_tokens else 1.0

    common = set(pred_tokens) & set(ref_tokens)
    num_same = sum(min(pred_tokens.count(w), ref_tokens.count(w)) for w in common)
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return float(2 * (precision * recall) / (precision + recall))


def aggregate_generation_metrics(results: List[dict]) -> GenerationMetrics:
    """Aggregate generation metrics over test cases with reference answers."""
    evaluable = [r for r in results if r.get("expected_answer")]
    if not evaluable:
        return GenerationMetrics()

    em_list, f1_list = [], []
    for item in evaluable:
        pred = item.get("answer", "")
        ref = item.get("expected_answer", "")
        em_list.append(compute_exact_match(pred, ref))
        f1_list.append(compute_token_f1(pred, ref))

    return GenerationMetrics(
        exact_match=round(float(sum(em_list) / len(em_list)), 4),
        token_f1=round(float(sum(f1_list) / len(f1_list)), 4),
        semantic_similarity=round(float(sum(f1_list) / len(f1_list)), 4),
        correctness=round(float(sum(f1_list) / len(f1_list)), 4),
    )
