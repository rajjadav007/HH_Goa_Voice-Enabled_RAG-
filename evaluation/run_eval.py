"""CLI runner for evaluation framework.

Usage:
    python -m evaluation.run_eval --mode full --mock
    python -m evaluation.run_eval --mode text_e2e --limit 15
"""

import argparse
import sys
from pathlib import Path

from evaluation.baseline import check_regression, load_baseline, record_baseline
from evaluation.datasets.loader import EvalDatasetLoader
from evaluation.runner import EvaluationRunner


def main():
    parser = argparse.ArgumentParser(description="HH Goa 2026 Evaluation Framework")
    parser.add_argument("--mode", type=str, default="full", choices=["retrieval_only", "text_e2e", "voice_e2e", "guardrail", "full"], help="Evaluation run mode")
    parser.add_argument("--language", type=str, default=None, help="Filter by language code (e.g. en-IN, hi-IN)")
    parser.add_argument("--category", type=str, default=None, help="Filter by category")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test cases")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--mock", action="store_true", help="Run in offline mock mode")
    parser.add_argument("--output-dir", type=str, default="data/eval_results", help="Output directory for reports")
    parser.add_argument("--save-baseline", action="store_true", help="Save run report as baseline")

    args = parser.parse_args()

    print(f"\n============================================================")
    print(f"  HH Goa 2026 — Phase 9.1 Reproducible Evaluation Runner")
    print(f"============================================================")
    print(f"  Mode: {args.mode} | Mock: {args.mock} | Seed: {args.seed} | Limit: {args.limit}")

    runner = EvaluationRunner(offline_mock=args.mock, seed=args.seed)
    report = runner.run(
        mode=args.mode,
        language=args.language,
        category=args.category,
        limit=args.limit,
    )

    out_dir = Path(args.output_dir)
    runner.export_report(report, out_dir)

    print("\n------------------------------------------------------------")
    print("EVALUATION RESULTS SUMMARY")
    print("------------------------------------------------------------")
    print(f"Run ID                 : {report.metadata.run_id}")
    print(f"Total Test Cases       : {report.total_cases}")
    print(f"Success Rate           : {report.success_rate * 100:.1f}%")
    print(f"Retrieval Recall@1     : {report.retrieval.recall_at_1:.4f}")
    print(f"Retrieval Recall@3     : {report.retrieval.recall_at_3:.4f}")
    print(f"Retrieval MRR          : {report.retrieval.mrr:.4f}")
    print(f"Grounded Answer Rate   : {report.grounding.grounded_rate * 100:.1f}%")
    print(f"Unsupported Claim Rate : {report.grounding.unsupported_claim_rate * 100:.1f}%")
    print(f"P50 Latency (Median)   : {report.latency.p50} ms")
    print(f"P90 Latency            : {report.latency.p90} ms")
    print(f"P95 Latency            : {report.latency.p95} ms")
    print(f"P99 Latency            : {report.latency.p99} ms")
    print(f"<200ms Target Result   : {'PASSED' if report.latency.target_200ms_passed else 'FAILED'}\n")

    if args.save_baseline:
        record_baseline(report)
        print(f"Saved run {report.metadata.run_id} as baseline.")

    baseline = load_baseline()
    if baseline:
        has_reg, reg_details = check_regression(report, baseline)
        if has_reg:
            print("WARNING: Regression detected compared to baseline:")
            for k, v in reg_details.items():
                print(f"  - {k}: {v}")
        else:
            print("Baseline Check: No regression detected.")


if __name__ == "__main__":
    main()
