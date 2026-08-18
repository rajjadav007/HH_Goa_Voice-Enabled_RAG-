"""CLI Entrypoint for running offline chunking using registered strategies.

Usage:
    python -m ingestion.chunking.run_chunking --strategy passthrough
"""

import argparse
import logging
import os
import sys
from typing import Optional

from ingestion.chunking import (
    BatchChunkProcessor,
    ChunkerRegistry,
    ChunkingConfig,
)
from ingestion.preprocessor import DEFAULT_PROCESSED_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_CHUNKS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "chunks")
)


def run_chunking_pipeline(
    strategy: str = "passthrough",
    input_documents_jsonl: Optional[str] = None,
    output_chunks_jsonl: Optional[str] = None,
    output_manifest_json: Optional[str] = None,
    max_documents: Optional[int] = None,
    target_chunk_size: int = 256,
    max_chunk_size: int = 512,
    overlap: int = 32,
):
    """Run offline chunking pipeline using registered strategy."""
    in_docs = input_documents_jsonl or os.path.join(DEFAULT_PROCESSED_DIR, "documents.jsonl")
    out_chunks = output_chunks_jsonl or os.path.join(DEFAULT_CHUNKS_DIR, f"chunks_{strategy}.jsonl")
    out_manifest = output_manifest_json or os.path.join(DEFAULT_CHUNKS_DIR, f"manifest_{strategy}.json")

    config = ChunkingConfig(
        strategy=strategy,
        target_chunk_size=target_chunk_size,
        max_chunk_size=max_chunk_size,
        overlap=overlap,
    )

    chunker = ChunkerRegistry.get(strategy, config=config)
    processor = BatchChunkProcessor(chunker=chunker)

    manifest = processor.process_jsonl_file(
        input_documents_jsonl=in_docs,
        output_chunks_jsonl=out_chunks,
        output_manifest_json=out_manifest,
        max_documents=max_documents,
    )
    return manifest


def main():
    import os  # local import for CLI
    parser = argparse.ArgumentParser(
        description="Chunk processed dataset using registered chunking strategies."
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="passthrough",
        help="Registered chunking strategy name (default: passthrough).",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Input processed documents JSONL file (default: data/processed/documents.jsonl).",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output chunks JSONL file path.",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Maximum documents to chunk (default: all).",
    )
    args = parser.parse_args()

    try:
        manifest = run_chunking_pipeline(
            strategy=args.strategy,
            input_documents_jsonl=args.input_file,
            output_chunks_jsonl=args.output_file,
            max_documents=args.max_documents,
        )

        print("\n============================================================")
        print("  HH Goa 2026 — Chunking Pipeline Execution Complete")
        print("============================================================\n")
        print(f"Strategy Used       : {manifest['chunk_strategy']}")
        print(f"Input Documents     : {manifest['input_document_count']}")
        print(f"Output Chunks       : {manifest['output_chunk_count']}")
        print(f"Rejected Documents  : {manifest['rejected_document_count']}")
        print(f"Avg Tokens/Chunk    : {manifest['avg_tokens_per_chunk']}")
        print(f"Avg Chars/Chunk     : {manifest['avg_chars_per_chunk']}")
        print(f"Output Chunks File  : {manifest['output_chunks_file']}")

        print("\n============================================================\n")

    except Exception as exc:
        logger.error(f"Chunking pipeline failed: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    import os
    main()
