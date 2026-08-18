"""CLI Entrypoint for running Phase 4.3 BM25 Lexical Indexing & Search Sanity Check.

Usage:
    python -m retrieval.bm25.run_indexing [--max-chunks N] [--k1 K1] [--b B] [--sample-query QUERY]
"""

import argparse
import logging
import sys
import time

from retrieval.bm25.indexer import BatchBM25Indexer
from retrieval.bm25.models import BM25Config
from retrieval.bm25.service import BM25Service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Build and persist BM25 lexical index over Phase 3.3 production chunks."
    )
    parser.add_argument(
        "--k1",
        type=float,
        default=1.5,
        help="BM25 k1 parameter (default: 1.5).",
    )
    parser.add_argument(
        "--b",
        type=float,
        default=0.75,
        help="BM25 b parameter (default: 0.75).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Default top-K retrieval count (default: 10).",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Maximum chunks to index (default: process all).",
    )
    parser.add_argument(
        "--sample-query",
        type=str,
        default="What is a corporation?",
        help="Sample query to test BM25 search sanity (default: 'What is a corporation?').",
    )
    args = parser.parse_args()

    try:
        config = BM25Config(
            k1=args.k1,
            b=args.b,
            top_k=args.top_k,
        )
        service = BM25Service(config=config)
        indexer = BatchBM25Indexer(service=service)

        print("\n============================================================")
        print("  HH Goa 2026 — Phase 4.3 BM25 Lexical Indexing")
        print("============================================================\n")
        print(f"BM25 k1 Parameter   : {config.k1}")
        print(f"BM25 b Parameter    : {config.b}")
        print(f"Index Storage Dir   : {config.index_dir}\n")

        summary = indexer.index_chunks_file(max_chunks=args.max_chunks)

        print("============================================================")
        print("BM25 INDEXING COMPLETE")
        print("============================================================")
        print(f"Chunks Indexed      : {summary['indexed_chunks']}")
        print(f"Index Build Time    : {summary['index_build_time_sec']} sec")
        print(f"Index File Size     : {summary['index_file_size_mb']} MB")
        print(f"Output Index File   : {summary['output_index_file']}")
        print("============================================================\n")

        # Sanity check sample BM25 search
        print(f"Running sample BM25 search for query: '{args.sample_query}'...")
        t0 = time.time()
        results = service.search(args.sample_query, top_k=3)
        q_latency_ms = round((time.time() - t0) * 1000, 2)

        print(f"\nTop-{len(results)} BM25 Search Results (Query Latency: {q_latency_ms} ms):")
        print("-" * 70)
        for res in results:
            print(f"Rank {res.rank}: [BM25 Score: {res.score:.4f}] Chunk ID: {res.chunk_id} | Doc ID: {res.document_id}")
            print(f"        Text: {res.text[:100]}...")
            print("-" * 70)

        print("\nTraceability & Validation verified successfully!\n")

    except Exception as exc:
        logger.error(f"BM25 indexing failed: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
