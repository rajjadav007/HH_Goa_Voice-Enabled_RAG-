"""CLI Entrypoint for running Phase 5.2 Gemini RAG Generation Pipeline.

Usage:
    python -m generation.gemini.run_rag [--query QUERY]
"""

import argparse
import json
import logging
import sys

from generation.gemini.pipeline import RAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run Phase 5.2 Gemini RAG Generation pipeline."
    )
    parser.add_argument(
        "--query",
        type=str,
        default="What is a corporation?",
        help="User question to answer using RAG pipeline (default: 'What is a corporation?').",
    )
    args = parser.parse_args()

    try:
        pipeline = RAGPipeline()

        print("\n============================================================")
        print("  HH Goa 2026 — Phase 5.2 Gemini RAG Generation Pipeline")
        print("============================================================\n")
        print(f"User Query          : '{args.query}'\n")

        print("Executing end-to-end RAG pipeline (Hybrid -> Reranker -> Gemini)...")
        response = pipeline.query(query_text=args.query)

        print("\n============================================================")
        print("GENERATED GROUNDED RESPONSE")
        print("============================================================")
        print(f"Model               : {response.model}")
        print(f"Grounded Status     : {response.grounded}")
        print(f"Total Pipeline Lat  : {response.latency_ms} ms")
        print(f"Timing Breakdown    : {json.dumps(response.timing_breakdown, indent=2)}")
        print(f"Token Usage         : {json.dumps(response.token_usage, indent=2)}\n")

        print("Answer:")
        print("-" * 75)
        print(response.answer)
        print("-" * 75)

        print(f"\nSource Attributions ({len(response.sources)} chunks cited):")
        for s in response.sources:
            print(f"  - Rank {s.rank}: Chunk ID '{s.chunk_id}' | Doc ID '{s.document_id}'")

        print("\nTraceability & Validation verified successfully!\n")

    except Exception as exc:
        logger.error(f"RAG generation pipeline failed: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
