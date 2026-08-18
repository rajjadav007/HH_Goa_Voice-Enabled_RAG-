"""CLI Entrypoint for Phase 5.4 End-to-End Text RAG Benchmark.

Usage:
    python -m generation.gemini.run_e2e_benchmark
"""

import json
import logging
import sys

from generation.gemini.benchmark_e2e import E2EBenchmarkRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    try:
        runner = E2EBenchmarkRunner()

        print("\n============================================================")
        print("  HH Goa 2026 — Phase 5.4 End-to-End Text RAG Benchmark")
        print("============================================================\n")

        results = runner.run_benchmark()

        print("\n============================================================")
        print("BENCHMARK METRICS SUMMARY")
        print("============================================================")
        print(f"Total Evaluation Queries : {results['query_count']}")
        print(f"Recall@1                 : {results['recall_at_1']}")
        print(f"Recall@5                 : {results['recall_at_5']}")
        print(f"MRR                      : {results['mrr']}")
        print(f"Groundedness Rate        : {results['groundedness_rate'] * 100:.1f}%")
        print(f"Source Accuracy Rate     : {results['source_accuracy_rate'] * 100:.1f}%\n")

        print("============================================================")
        print("LATENCY DISTRIBUTION (ms)")
        print("============================================================")
        p = results["latency_percentiles_ms"]
        print(f"P50  (Median)            : {p['P50']} ms")
        print(f"P70                      : {p['P70']} ms")
        print(f"P90                      : {p['P90']} ms")
        print(f"P95                      : {p['P95']} ms")
        print(f"P99                      : {p['P99']} ms")
        print(f"P100 (Max)               : {p['P100']} ms\n")

        print("============================================================")
        print("STAGE LATENCY BREAKDOWN & BOTTLENECK")
        print("============================================================")
        s = results["average_stage_latencies_ms"]
        print(f"Average Retrieval Latency : {s['retrieval']} ms")
        print(f"Average Reranking Latency : {s['reranking']} ms")
        print(f"Average Generation Latency: {s['generation']} ms")
        print(f"Primary Latency Bottleneck: {results['main_latency_bottleneck']}\n")

        print("End-to-End Text RAG benchmark completed successfully!\n")

    except Exception as exc:
        logger.error(f"E2E Benchmark failed: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
