"""Baseline configuration recording and regression detection utilities."""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

from evaluation.datasets.schema import AggregateEvaluationReport


DEFAULT_BASELINE_PATH = Path("data/evaluation_baseline.json")


def record_baseline(report: AggregateEvaluationReport, filepath: Path = DEFAULT_BASELINE_PATH):
    """Save an aggregate evaluation report as the authoritative baseline."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, indent=2, ensure_ascii=False)


def load_baseline(filepath: Path = DEFAULT_BASELINE_PATH) -> Optional[AggregateEvaluationReport]:
    """Load the saved baseline aggregate evaluation report."""
    filepath = Path(filepath)
    if not filepath.exists():
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return AggregateEvaluationReport(**data)


def check_regression(
    current: AggregateEvaluationReport,
    baseline: AggregateEvaluationReport,
    max_latency_increase_ms: float = 50.0,
    min_recall_drop: float = 0.05,
    min_grounded_drop: float = 0.05,
) -> Tuple[bool, Dict[str, str]]:
    """Compare current evaluation report against baseline to detect regressions.
    
    Returns:
        (has_regression: bool, details: Dict[str, str])
    """
    regressions: Dict[str, str] = {}

    # 1. Latency Regression Check
    lat_diff = current.latency.p50 - baseline.latency.p50
    if lat_diff > max_latency_increase_ms:
        regressions["latency_p50"] = f"P50 latency increased by {lat_diff:.2f}ms (current: {current.latency.p50}ms, baseline: {baseline.latency.p50}ms)"

    # 2. Retrieval Recall@1 Regression Check
    recall_diff = baseline.retrieval.recall_at_1 - current.retrieval.recall_at_1
    if recall_diff > min_recall_drop:
        regressions["recall_at_1"] = f"Recall@1 dropped by {recall_diff:.4f} (current: {current.retrieval.recall_at_1}, baseline: {baseline.retrieval.recall_at_1})"

    # 3. Grounded Rate Regression Check
    grounded_diff = baseline.grounding.grounded_rate - current.grounding.grounded_rate
    if grounded_diff > min_grounded_drop:
        regressions["grounded_rate"] = f"Grounded rate dropped by {grounded_diff:.4f} (current: {current.grounding.grounded_rate}, baseline: {baseline.grounding.grounded_rate})"

    has_reg = len(regressions) > 0
    return has_reg, regressions
