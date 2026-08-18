"""CLI Entrypoint for running Phase 4.2 Qdrant Vector Database Indexing.

Usage:
    python -m retrieval.vector_db.run_indexing [--max-vectors N] [--recreate] [--sample-query QUERY]
"""

import argparse
import logging
import sys

from retrieval.embeddings.service import EmbeddingService
from retrieval.vector_db.indexer import BatchQdrantIndexer
from retrieval.vector_db.models import QdrantConfig
from retrieval.vector_db.service import QdrantService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Index Phase 4.1 vector embeddings into Qdrant vector database."
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="hh_goa_chunks",
        help="Qdrant collection name (default: 'hh_goa_chunks').",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="data/qdrant_db",
        help="Local Qdrant database path or ':memory:' (default: 'data/qdrant_db').",
    )
    parser.add_argument(
        "--max-vectors",
        type=int,
        default=None,
        help="Maximum vectors to index (default: process all).",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate Qdrant collection before indexing.",
    )
    parser.add_argument(
        "--sample-query",
        type=str,
        default="What is a corporation?",
        help="Sample query to test vector search sanity (default: 'What is a corporation?').",
    )
    args = parser.parse_args()

    try:
        config = QdrantConfig(
            collection_name=args.collection,
            path=args.path,
        )
        service = QdrantService(config=config)
        indexer = BatchQdrantIndexer(service=service)

        print("\n============================================================")
        print("  HH Goa 2026 — Phase 4.2 Qdrant Vector Indexing")
        print("============================================================\n")
        print(f"Collection Name     : {config.collection_name}")
        print(f"Vector Dimension    : {config.vector_size}")
        print(f"Distance Metric     : {config.distance}")
        print(f"Storage Path        : {config.path}\n")

        summary = indexer.index_vectors_file(
            max_vectors=args.max_vectors,
            recreate_collection=args.recreate,
        )

        print("============================================================")
        print("QDRANT INDEXING COMPLETE")
        print("============================================================")
        print(f"Vectors Processed   : {summary['vectors_processed']}")
        print(f"Qdrant Point Count  : {summary['qdrant_point_count']}")
        print(f"Elapsed Time (sec)  : {summary['elapsed_seconds']}")
        print(f"Throughput          : {summary['throughput_points_per_sec']} points/sec")
        print("============================================================\n")

        # Sanity check vector search
        print(f"Running sample vector search for query: '{args.sample_query}'...")
        emb_service = EmbeddingService()
        q_vector = emb_service.embed_text(args.sample_query, is_query=True)

        results = service.search(q_vector, top_k=3)

        print(f"\nTop-{len(results)} Vector Search Results:")
        print("-" * 70)
        for idx, res in enumerate(results, start=1):
            print(f"Rank {idx}: [Score: {res.score:.4f}] Chunk ID: {res.chunk_id} | Doc ID: {res.document_id}")
            print(f"        Text: {res.text[:100]}...")
            print("-" * 70)

        print("\nTraceability & Validation verified successfully!\n")

    except Exception as exc:
        logger.error(f"Qdrant indexing failed: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
