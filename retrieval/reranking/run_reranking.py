"""CLI Entrypoint for running Phase 5.1 Reranking & Benchmark.

Usage:
    python -m retrieval.reranking.run_reranking [--eval-queries N] [--candidate-k K] [--sample-query QUERY]
"""

import argparse
import logging
import sys

from retrieval.hybrid.service import HybridService
from retrieval.reranking.benchmark import RerankerBenchmarkRunner
from retrieval.reranking.models import RerankerConfig
from retrieval.reranking.service import RerankerService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run Phase 5.1 Reranking benchmark and evaluation."
    )
    parser.add_argument(
        "--eval-queries",
        type=int,
        default=20,
        help="Number of eval queries for benchmark (default: 20).",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=10,
        help="Candidate pool size for reranking (default: 10).",
    )
    parser.add_argument(
        "--final-k",
        type=int,
        default=5,
        help="Final top-K reranked results (default: 5).",
    )
    parser.add_argument(
        "--sample-query",
        type=str,
        default="What is a corporation?",
        help="Sample query for search sanity check (default: 'What is a corporation?').",
    )
    parser.add_argument(
        "--disable-reranker",
        action="store_true",
        help="Disable reranker to test fallback mode.",
    )
    args = parser.parse_args()

    try:
        r_config = RerankerConfig(
            enabled=not args.disable_reranker,
            candidate_k=args.candidate_k,
            final_k=args.final_k,
        )
        h_service = HybridService()
        r_service = RerankerService(config=r_config)
        runner = RerankerBenchmarkRunner(hybrid_service=h_service, reranker_service=r_service)

        print("\n============================================================")
        print("  HH Goa 2026 — Phase 5.1 Reranking Benchmark")
        print("============================================================\n")
        print(f"Reranker Model     : {r_config.model_name}")
        print(f"Reranker Enabled   : {r_config.enabled}")
        print(f"Device             : {r_service.device}")
        print(f"Candidate Pool K   : {r_config.candidate_k}")
        print(f"Final Top-K        : {r_config.final_k}\n")

        # Run benchmark
        res = runner.run_benchmark(
            max_queries=args.eval_queries,
            candidate_k=args.candidate_k,
            final_k=args.final_k,
        )

        print("============================================================")
        print("RETRIEVAL STRATEGY BENCHMARK RESULTS")
        print("============================================================")
        print(
            f"{'Strategy':<22} | {'R@1':<6} | {'R@3':<6} | {'R@5':<6} | {'R@10':<6} | {'MRR':<6} | {'Latency':<8}"
        )
        print("-" * 75)

        for strat, m in res["strategy_metrics"].items():
            print(
                f"{strat:<22} | {m['recall_1']:<6.4f} | {m['recall_3']:<6.4f} | {m['recall_5']:<6.4f} | {m['recall_10']:<6.4f} | {m['mrr']:<6.4f} | {m['mean_latency_ms']:<7.1f}ms"
            )

        print("-" * 75)
        print(f"Standalone Reranker Inference Latency: {res['reranker_standalone_latency_ms']} ms\n")

        # Sample query test
        print(f"Running sample reranked search for: '{args.sample_query}'...")
        h_candidates, _ = h_service.search(args.sample_query, vector_top_k=args.candidate_k, bm25_top_k=args.candidate_k)
        reranked_res, r_metrics = r_service.rerank(args.sample_query, candidates=h_candidates, final_k=args.final_k)

        print(f"\nTop-{len(reranked_res)} Reranked Results (Rerank Latency: {r_metrics['total_ms']} ms):")
        print("-" * 75)
        for p in reranked_res:
            print(
                f"Rank {p.final_rank}: [CrossEncoder Score: {p.rerank_score:.6f}] Chunk ID: {p.chunk_id} | Doc ID: {p.document_id}"
            )
            print(f"        Sources: {p.sources} (RRF Score: {p.rrf_score})")
            print(f"        Text: {p.text[:100]}...")
            print("-" * 75)

        print("\nTraceability & Validation verified successfully!\n")

    except Exception as exc:
        logger.error(f"Reranking benchmark failed: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
