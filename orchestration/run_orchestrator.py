"""CLI Entrypoint for running Phase 5.3 RAG Orchestrator.

Usage:
    python -m orchestration.run_orchestrator [--query QUERY]
"""

import argparse
import json
import logging
import sys

from orchestration.service import RAGOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run Phase 5.3 RAG Orchestrator application entry point."
    )
    parser.add_argument(
        "--query",
        type=str,
        default="What is a corporation?",
        help="User query for RAG Orchestrator (default: 'What is a corporation?').",
    )
    args = parser.parse_args()

    try:
        orchestrator = RAGOrchestrator()

        print("\n============================================================")
        print("  HH Goa 2026 — Phase 5.3 RAG Orchestrator Execution")
        print("============================================================\n")
        print(f"User Query          : '{args.query}'\n")

        print("Executing orchestrated text RAG pipeline...")
        response = orchestrator.answer(query_text=args.query)

        print("\n============================================================")
        print("ORCHESTRATED RAG RESPONSE")
        print("============================================================")
        print(f"Request ID          : {response.request_id}")
        print(f"Status              : {response.status}")
        print(f"Error Code          : {response.error_code}")
        print(f"Grounded Status     : {response.grounded}")
        print(f"Has Context         : {response.has_context}")
        print(f"Total Latency       : {response.latency_ms} ms")
        print(f"Timing Breakdown    : {json.dumps(response.timing_breakdown, indent=2)}")
        print(f"Token Usage         : {json.dumps(response.token_usage, indent=2)}\n")

        print("Answer:")
        print("-" * 75)
        print(response.answer)
        print("-" * 75)

        print(f"\nValidated Sources ({len(response.sources)} cited):")
        for s in response.sources:
            print(f"  - Rank {s['rank']}: Chunk ID '{s['chunk_id']}' | Doc ID '{s['document_id']}'")

        print("\nTraceability & Source Integrity verified successfully!\n")

    except Exception as exc:
        logger.error(f"RAG Orchestrator execution failed: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
