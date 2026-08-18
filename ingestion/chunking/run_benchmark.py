"""CLI Entrypoint for running Phase 3.3 Chunking Benchmark & Final Selection.

Usage:
    python -m ingestion.chunking.run_benchmark
"""

import argparse
import logging
import os
import sys

from ingestion.chunking.benchmark import ChunkingBenchmarkRunner
from ingestion.chunking.models import ChunkingConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run chunking strategy benchmark, evaluate retrieval quality, and select final strategy."
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=500,
        help="Maximum evaluation queries to benchmark (default: 500).",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=2000,
        help="Maximum evaluation documents to benchmark (default: 2000).",
    )
    args = parser.parse_args()

    try:
        runner = ChunkingBenchmarkRunner()
        summary = runner.run_benchmark_matrix(
            max_queries=args.max_queries, max_documents=args.max_documents
        )

        winning_strat = summary["winning_strategy"]
        winning_cfg_dict = summary["winning_config"]
        winning_cfg = ChunkingConfig(**winning_cfg_dict)

        print("\n============================================================")
        print("  HH Goa 2026 — Phase 3.3 Chunking Benchmark Results")
        print("============================================================\n")
        print(f"Evaluated Queries   : {summary['evaluated_queries']}")
        print(f"Evaluated Documents : {summary['evaluated_documents']}\n")

        print(f"{'Strategy':<12} | {'Target Size':<11} | {'Chunks':<8} | {'Recall@5':<9} | {'MRR':<7} | {'Avg Token Size':<14} | {'Score':<6}")
        print("-" * 80)

        for r in summary["results_matrix"]:
            strat = r["strategy"]
            tsize = r["config"]["target_chunk_size"]
            total_c = r["total_chunks"]
            rec5 = r["retrieval_metrics"]["recall_5"]
            mrr = r["retrieval_metrics"]["mrr"]
            avg_tok = r["token_stats"]["mean"]
            score = r["composite_score"]
            print(f"{strat:<12} | {tsize:<11} | {total_c:<8} | {rec5:<9.4f} | {mrr:<7.4f} | {avg_tok:<14.2f} | {score:<6.4f}")

        print("\n============================================================")
        print(f"WINNING STRATEGY     : {winning_strat}")
        print(f"WINNING CONFIG       : {winning_cfg_dict}")
        print(f"COMPOSITE SCORE      : {summary['winning_composite_score']}")
        print("============================================================\n")

        print("Generating final production chunk dataset using winning strategy...")
        final_manifest = runner.generate_final_production_chunks(
            winning_strategy=winning_strat, winning_config=winning_cfg
        )

        print("\nFinal Production Chunks Generated:")
        print(f"  - Output Chunks File : {final_manifest['output_chunks_file']}")
        print(f"  - Total Chunks       : {final_manifest['output_chunk_count']}")
        print(f"  - Avg Tokens/Chunk   : {final_manifest['avg_tokens_per_chunk']}")
        print(f"  - Manifest File      : {final_manifest['output_chunks_file'].replace('final_chunks.jsonl', 'final_manifest.json')}")
        print("\n============================================================\n")

    except Exception as exc:
        logger.error(f"Benchmark execution failed: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
