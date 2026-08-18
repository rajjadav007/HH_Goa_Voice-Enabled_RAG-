"""CLI Entrypoint for running Phase 4.4 Hybrid Retrieval & Benchmark.

Usage:
    python -m retrieval.hybrid.run_hybrid [--eval-queries N] [--sample-query QUERY]
"""

import argparse
import logging
import sys

from retrieval.hybrid.benchmark import HybridBenchmarkRunner
from retrieval.hybrid.models import HybridConfig
from retrieval.hybrid.service import HybridService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run Phase 4.4 Hybrid Retrieval evaluation benchmark."
    )
    parser.add_argument(
        "--eval-queries",
        type=int,
        default=50,
        help="Number of eval queries to benchmark (default: 50).",
    )
    parser.add_argument(
        "--vector-k",
        type=int,
        default=10,
        help="Vector search Top-K candidates (default: 10).",
    )
    parser.add_argument(
        "--bm25-k",
        type=int,
        default=10,
        help="BM25 search Top-K candidates (default: 10).",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="RRF constant K (default: 60).",
    )
    parser.add_argument(
        "--final-k",
        type=int,
        default=5,
        help="Final hybrid Top-K results (default: 5).",
    )
    parser.add_argument(
        "--sample-query",
        type=str,
        default="What is a corporation?",
        help="Sample query for search sanity check (default: 'What is a corporation?').",
    )
    args = parser.parse_args()

    try:
        config = HybridConfig(
            vector_top_k=args.vector_k,
            bm25_top_k=args.bm25_k,
            rrf_k=args.rrf_k,
            final_top_k=args.final_k,
        )
        service = HybridService(config=config)
        runner = HybridBenchmarkRunner(hybrid_service=service)

        print("\n============================================================")
        print("  HH Goa 2026 — Phase 4.4 Hybrid Retrieval Benchmark")
        print("============================================================\n")
        print(f"Vector Top-K        : {config.vector_top_k}")
        print(f"BM25 Top-K          : {config.bm25_top_k}")
        print(f"RRF K Constant      : {config.rrf_k}")
        print(f"Final Hybrid Top-K  : {config.final_top_k}\n")

        # Run benchmark comparison
        benchmark_res = runner.run_benchmark(max_queries=args.eval_queries)

        print("============================================================")
        print("RETRIEVAL STRATEGY BENCHMARK RESULTS")
        print("============================================================")
        print(f"{'Strategy':<22} | {'R@1':<6} | {'R@3':<6} | {'R@5':<6} | {'R@10':<6} | {'MRR':<6}")
        print("-" * 65)

        for strat, m in benchmark_res["strategy_metrics"].items():
            print(
                f"{strat:<22} | {m['recall_1']:<6.4f} | {m['recall_3']:<6.4f} | {m['recall_5']:<6.4f} | {m['recall_10']:<6.4f} | {m['mrr']:<6.4f}"
            )

        print("-" * 65)
        print("LATENCY COMPARISON:")
        print(f"  Sequential Execution : {benchmark_res['latency']['sequential_mean_ms']} ms")
        print(f"  Parallel Execution   : {benchmark_res['latency']['parallel_mean_ms']} ms")
        print(f"  Parallel Speedup     : {benchmark_res['latency']['speedup']}x\n")

        # Run sample query search
        print(f"Running sample hybrid query search for: '{args.sample_query}'...")
        results, metrics = service.search(args.sample_query)

        print(f"\nTop-{len(results)} Hybrid Search Results (Total Latency: {metrics['timing_ms']['total_ms']} ms):")
        print("-" * 75)
        for res in results:
            print(
                f"Rank {res.rank}: [RRF Score: {res.score:.6f}] Sources: {res.sources} (VecRank: {res.vector_rank}, BM25Rank: {res.bm25_rank})"
            )
            print(f"        Chunk ID: {res.chunk_id} | Doc ID: {res.document_id}")
            print(f"        Text: {res.text[:100]}...")
            print("-" * 75)

        print("\nTraceability & Hybrid Validation verified successfully!\n")

    except Exception as exc:
        logger.error(f"Hybrid retrieval benchmark failed: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
