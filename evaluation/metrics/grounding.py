"""Grounding and reliability metrics calculation."""

from typing import List
from evaluation.datasets.schema import GroundingMetrics


def aggregate_grounding_metrics(results: List[dict]) -> GroundingMetrics:
    """Aggregate grounding, hallucination, and unanswerable query abstention rates."""
    if not results:
        return GroundingMetrics()

    total = len(results)
    grounded_count = sum(1 for r in results if r.get("grounded", False))
    contradicted_count = sum(1 for r in results if r.get("grounding_status") == "CONTRADICTED")
    unsupported_count = sum(1 for r in results if not r.get("grounded", False) and r.get("has_context", False))

    unanswerable = [r for r in results if r.get("category") in ["unanswerable", "offtopic", "injection", "no_context"]]
    correct_refusals = sum(1 for r in unanswerable if not r.get("has_context", False) or not r.get("grounded", False))
    abstention_rate = float(correct_refusals / len(unanswerable)) if unanswerable else 1.0

    return GroundingMetrics(
        grounded_rate=round(float(grounded_count / total), 4),
        unsupported_claim_rate=round(float(unsupported_count / total), 4),
        contradiction_rate=round(float(contradicted_count / total), 4),
        correct_abstention_rate=round(abstention_rate, 4),
    )
