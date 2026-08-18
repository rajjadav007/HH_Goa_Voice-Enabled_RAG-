"""Complete End-to-End Ingestion, Chunking, and Dual-Indexing Pipeline for HH Goa Voice RAG.

Usage:
    python -m ingestion.build_complete_pipeline --max-rows 10000 --split validation
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Set

from ingestion.dataset_loader import MSMARCODatasetLoader
from ingestion.preprocessor import DEFAULT_PROCESSED_DIR, MSMARCOPreprocessor, PreprocessingConfig
from ingestion.chunking.run_chunking import DEFAULT_CHUNKS_DIR, run_chunking_pipeline
from retrieval.embeddings.models import EmbeddingConfig
from retrieval.embeddings.processor import BatchEmbeddingProcessor
from retrieval.embeddings.service import EmbeddingService
from retrieval.vector_db.indexer import BatchQdrantIndexer
from retrieval.vector_db.models import QdrantConfig
from retrieval.vector_db.service import QdrantService
from retrieval.bm25.indexer import BatchBM25Indexer
from retrieval.bm25.service import BM25Service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def execute_full_pipeline(
    max_rows: int = 10000,
    split: str = "validation",
    chunk_strategy: str = "passthrough",
    batch_size: int = 32,
    recreate_indexes: bool = True,
) -> Dict[str, Any]:
    """Execute complete dataset ingestion, chunking, embedding, vector DB, and BM25 indexing pipeline."""
    t_start = time.time()
    logger.info(f"Starting complete pipeline build [split={split}, max_rows={max_rows}, chunk_strategy={chunk_strategy}]...")

    max_records = None if max_rows <= 0 else max_rows

    # 1. Dataset Ingestion & Preprocessing
    logger.info("--- STAGE 1: DATASET INGESTION & PREPROCESSING ---")
    loader = MSMARCODatasetLoader()
    raw_stream = loader.load_dataset(split=split, streaming=True)
    prep_config = PreprocessingConfig(processed_dir=DEFAULT_PROCESSED_DIR)
    preprocessor = MSMARCOPreprocessor(config=prep_config)

    prep_manifest = preprocessor.process_dataset_stream(record_stream=raw_stream, max_records=max_records)
    logger.info(f"Preprocessing complete: {prep_manifest['processed_documents']} documents, {prep_manifest['processed_queries']} queries.")

    # 2. Document Chunking
    logger.info("--- STAGE 2: DOCUMENT CHUNKING ---")
    in_docs = os.path.join(DEFAULT_PROCESSED_DIR, "documents.jsonl")
    out_chunks = os.path.join(DEFAULT_CHUNKS_DIR, "final_chunks.jsonl")

    chunk_manifest = run_chunking_pipeline(
        strategy=chunk_strategy,
        input_documents_jsonl=in_docs,
        output_chunks_jsonl=out_chunks,
    )
    logger.info(f"Chunking complete: {chunk_manifest['output_chunk_count']} chunks produced.")

    # 3. Vector Embeddings Generation
    logger.info("--- STAGE 3: VECTOR EMBEDDINGS GENERATION ---")
    emb_config = EmbeddingConfig(batch_size=batch_size)
    emb_service = EmbeddingService(config=emb_config)
    emb_processor = BatchEmbeddingProcessor(service=emb_service)

    emb_manifest = emb_processor.process_chunks_file(
        chunks_jsonl=out_chunks,
        resume=False,
    )
    logger.info(f"Embeddings complete: {emb_manifest['total_vector_artifacts']} vectors created.")

    # 4. Qdrant Vector Indexing
    logger.info("--- STAGE 4: QDRANT VECTOR DB INDEXING ---")
    q_service = QdrantService()
    q_indexer = BatchQdrantIndexer(service=q_service)

    qdrant_summary = q_indexer.index_vectors_file(
        vectors_jsonl=emb_manifest["output_vectors_file"],
        chunks_jsonl=out_chunks,
        recreate_collection=recreate_indexes,
    )
    logger.info(f"Qdrant indexing complete: {qdrant_summary['qdrant_point_count']} points stored.")

    # 5. BM25 Lexical Indexing
    logger.info("--- STAGE 5: BM25 LEXICAL INDEXING ---")
    bm25_service = BM25Service()
    bm25_indexer = BatchBM25Indexer(service=bm25_service)

    bm25_summary = bm25_indexer.index_chunks_file(
        input_chunks_jsonl=out_chunks,
    )
    logger.info(f"BM25 indexing complete: {bm25_summary['indexed_chunks']} chunks indexed.")

    # 6. Strict Validation & ID Consistency Verification
    logger.info("--- STAGE 6: DUAL-INDEX ID CONSISTENCY VALIDATION ---")
    qdrant_point_count = qdrant_summary["qdrant_point_count"]
    bm25_count = bm25_summary["indexed_chunks"]
    processed_chunk_count = chunk_manifest["output_chunk_count"]

    # Collect chunk IDs from chunk JSONL file
    chunk_ids: Set[str] = set()
    with open(out_chunks, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                chunk_ids.add(item["chunk_id"])

    # Load BM25 index IDs
    bm25_service.load_index()
    bm25_ids = set(c.chunk_id for c in bm25_service.chunks_corpus)

    # Perform set equality check
    id_consistency_pass = (
        qdrant_point_count == bm25_count == processed_chunk_count == len(chunk_ids)
        and chunk_ids == bm25_ids
    )

    elapsed_total = round(time.time() - t_start, 2)

    report = {
        "pipeline_status": "SUCCESS" if id_consistency_pass else "VALIDATION_FAILURE",
        "dataset_name": loader.dataset_name,
        "split": split,
        "raw_rows_requested": max_rows if max_rows > 0 else "ALL",
        "raw_rows_processed": prep_manifest["input_records"],
        "processed_documents": prep_manifest["processed_documents"],
        "processed_queries": prep_manifest["processed_queries"],
        "chunk_strategy": chunk_strategy,
        "processed_chunks": processed_chunk_count,
        "qdrant_point_count": qdrant_point_count,
        "bm25_document_count": bm25_count,
        "id_consistency_check": "PASS" if id_consistency_pass else "FAIL",
        "total_time_seconds": elapsed_total,
        "qdrant_collection": qdrant_summary["collection_name"],
        "bm25_index_file": bm25_summary["output_index_file"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if not id_consistency_pass:
        logger.error(f"ID consistency validation failed! Qdrant: {qdrant_point_count}, BM25: {bm25_count}, Chunks: {processed_chunk_count}")
        raise ValueError("Index consistency check failed between Qdrant, BM25, and processed chunks.")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Build complete MSMARCO-XI dataset ingestion, chunking, Qdrant, and BM25 pipeline."
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=int(os.getenv("DATASET_MAX_ROWS", "10000")),
        help="Maximum raw MSMARCO-XI dataset rows to process (default: 10000). Set to 0 for unlimited.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split ('validation' or 'train'). Default: validation.",
    )
    parser.add_argument(
        "--chunk-strategy",
        type=str,
        default="passthrough",
        help="Chunking strategy ('passthrough', 'sentence', 'semantic'). Default: passthrough.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size. Default: 32.",
    )
    args = parser.parse_args()

    try:
        report = execute_full_pipeline(
            max_rows=args.max_rows,
            split=args.split,
            chunk_strategy=args.chunk_strategy,
            batch_size=args.batch_size,
        )

        print("\n============================================================")
        print("  HH GOA 2026 — COMPLETE INDEXING PIPELINE REPORT")
        print("============================================================\n")
        print(f"Pipeline Status     : {report['pipeline_status']}")
        print(f"Dataset Name        : {report['dataset_name']} [split={report['split']}]")
        print(f"Raw Rows Processed  : {report['raw_rows_processed']}")
        print(f"Processed Documents : {report['processed_documents']}")
        print(f"Processed Queries   : {report['processed_queries']}")
        print(f"Total Chunks        : {report['processed_chunks']}")
        print(f"Qdrant Points       : {report['qdrant_point_count']}")
        print(f"BM25 Documents      : {report['bm25_document_count']}")
        print(f"ID Consistency Check: {report['id_consistency_check']}")
        print(f"Total Time (sec)    : {report['total_time_seconds']} s")
        print("============================================================\n")

    except Exception as exc:
        logger.error(f"Pipeline execution failed: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
