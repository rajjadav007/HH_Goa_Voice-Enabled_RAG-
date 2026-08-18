"""CLI Entrypoint for running Phase 4.1 Production Vector Embeddings Generation.

Usage:
    python -m retrieval.embeddings.run_embeddings [--max-chunks N] [--batch-size N] [--model MODEL_NAME]
"""

import argparse
import logging
import os
import sys

from retrieval.embeddings.models import EmbeddingConfig
from retrieval.embeddings.processor import BatchEmbeddingProcessor
from retrieval.embeddings.service import EmbeddingService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate production vector embeddings for Phase 3.3 final chunks."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="intfloat/multilingual-e5-small",
        help="Embedding model name (default: 'intfloat/multilingual-e5-small').",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size (default: 32).",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Maximum chunks to process (default: process all).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Execution device ('auto', 'cpu', 'cuda').",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Overwrite existing vector output instead of resuming.",
    )
    args = parser.parse_args()

    try:
        config = EmbeddingConfig(
            model_name=args.model,
            batch_size=args.batch_size,
            device=args.device,
        )
        service = EmbeddingService(config=config)
        processor = BatchEmbeddingProcessor(service=service)

        print("\n============================================================")
        print("  HH Goa 2026 — Phase 4.1 Production Embedding Generation")
        print("============================================================\n")
        print(f"Model Name          : {service.model_name}")
        print(f"Vector Dimension    : {service.dimension}")
        print(f"Similarity Metric   : cosine")
        print(f"Batch Size          : {args.batch_size}")
        print(f"Device              : {service._device}\n")

        manifest = processor.process_chunks_file(
            max_chunks=args.max_chunks,
            resume=not args.no_resume,
        )

        print("============================================================")
        print("EMBEDDING GENERATION COMPLETE")
        print("============================================================")
        print(f"Chunks Processed    : {manifest['total_chunks_processed']}")
        print(f"Total Vector Count  : {manifest['total_vector_artifacts']}")
        print(f"Elapsed Time (sec)  : {manifest['elapsed_seconds']}")
        print(f"Throughput          : {manifest['throughput_chunks_per_sec']} chunks/sec")
        print(f"Output Vectors File : {manifest['output_vectors_file']}")
        print("============================================================\n")

    except Exception as exc:
        logger.error(f"Embedding generation failed: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
