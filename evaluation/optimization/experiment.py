"""Lightweight Optimization Experiment Runner for Phase 9.3."""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from evaluation.baseline import load_baseline
from evaluation.datasets.schema import AggregateEvaluationReport
from evaluation.runner import EvaluationRunner

logger = logging.getLogger(__name__)


class ExperimentRecord(BaseModel):
    """Schema representing an optimization experiment comparing baseline against modified configuration."""
    experiment_id: str
    hypothesis: str
    baseline_run_id: str
    target_component: str
    metrics_before: Dict[str, Any]
    metrics_after: Dict[str, Any]
    decision: str = "PENDING"  # ACCEPTED, REJECTED, INCONCLUSIVE
    justification: str = ""


class OptimizationSuite:
    """Suite orchestrating optimization experiments and regression verification."""

    def __init__(self, output_dir: Path = Path("data/optimization_experiments")):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.baseline = load_baseline()

    def run_experiment(
        self,
        experiment_id: str,
        hypothesis: str,
        target_component: str,
        runner: Optional[EvaluationRunner] = None,
        mode: str = "full",
        limit: Optional[int] = None,
    ) -> ExperimentRecord:
        """Run evaluation under new configuration and evaluate against baseline."""
        runner = runner or EvaluationRunner(offline_mock=True, seed=42)
        report = runner.run(mode=mode, limit=limit)

        base_metrics = {
            "recall_at_1": self.baseline.retrieval.recall_at_1 if self.baseline else 1.0,
            "recall_at_3": self.baseline.retrieval.recall_at_3 if self.baseline else 1.0,
            "mrr": self.baseline.retrieval.mrr if self.baseline else 1.0,
            "grounded_rate": self.baseline.grounding.grounded_rate if self.baseline else 0.8,
            "p50_latency_ms": self.baseline.latency.p50 if self.baseline else 110.0,
            "p95_latency_ms": self.baseline.latency.p95 if self.baseline else 110.0,
            "success_rate": self.baseline.success_rate if self.baseline else 1.0,
        }

        after_metrics = {
            "recall_at_1": report.retrieval.recall_at_1,
            "recall_at_3": report.retrieval.recall_at_3,
            "mrr": report.retrieval.mrr,
            "grounded_rate": report.grounding.grounded_rate,
            "p50_latency_ms": report.latency.p50,
            "p95_latency_ms": report.latency.p95,
            "success_rate": report.success_rate,
        }

        # Decision Logic: Accept if latency decreased or stayed equal AND quality metrics did not regress
        recall_drop = base_metrics["recall_at_1"] - after_metrics["recall_at_1"]
        grounded_drop = base_metrics["grounded_rate"] - after_metrics["grounded_rate"]
        lat_diff = after_metrics["p50_latency_ms"] - base_metrics["p50_latency_ms"]

        if recall_drop <= 0.01 and grounded_drop <= 0.01 and lat_diff <= 5.0:
            decision = "ACCEPTED"
            justification = f"Latency P50={after_metrics['p50_latency_ms']}ms, Recall@1={after_metrics['recall_at_1']}, Grounded={after_metrics['grounded_rate']} (No regression detected)."
        else:
            decision = "REJECTED"
            justification = f"Quality or latency regression detected: recall_drop={recall_drop:.4f}, grounded_drop={grounded_drop:.4f}, lat_diff={lat_diff:.2f}ms."

        record = ExperimentRecord(
            experiment_id=experiment_id,
            hypothesis=hypothesis,
            baseline_run_id=self.baseline.metadata.run_id if self.baseline else "default_baseline",
            target_component=target_component,
            metrics_before=base_metrics,
            metrics_after=after_metrics,
            decision=decision,
            justification=justification,
        )

        # Save experiment log
        exp_path = self.output_dir / f"{experiment_id}.json"
        with open(exp_path, "w", encoding="utf-8") as f:
            json.dump(record.model_dump(), f, indent=2, ensure_ascii=False)

        return record
