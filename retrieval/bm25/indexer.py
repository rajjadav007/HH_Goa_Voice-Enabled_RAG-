"""Batch indexer for building and persisting production BM25 index."""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from ingestion.chunking.models import Chunk
from retrieval.bm25.models import BM25Config
from retrieval.bm25.service import BM25Service

logger = logging.getLogger(__name__)

DEFAULT_FINAL_CHUNKS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "chunks", "final_chunks.jsonl")
)
DEFAULT_BM25_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "bm25_index")
)


class BatchBM25Indexer:
    """Indexer building and persisting BM25 index from production chunk JSONL file."""

    def __init__(self, service: Optional[BM25Service] = None):
        self.service = service or BM25Service()

    def index_chunks_file(
        self,
        input_chunks_jsonl: Optional[str] = None,
        output_index_dir: Optional[str] = None,
        max_chunks: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Load production chunks, build BM25 index, and persist artifacts."""
        input_path = input_chunks_jsonl or DEFAULT_FINAL_CHUNKS_PATH
        out_dir = output_index_dir or DEFAULT_BM25_DIR

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Final chunks file not found at '{input_path}'. Run Phase 3.3 chunking first.")

        chunks: List[Chunk] = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    c = Chunk.from_dict(json.loads(line))
                    chunks.append(c)
                    if max_chunks and len(chunks) >= max_chunks:
                        break

        logger.info(f"Loaded {len(chunks)} chunks for BM25 indexing from '{input_path}'.")

        build_stats = self.service.build_index(chunks)
        saved_paths = self.service.save_index(index_dir=out_dir)

        index_size_bytes = os.path.getsize(saved_paths["index_file"]) if os.path.exists(saved_paths["index_file"]) else 0

        summary = {
            "indexed_chunks": len(chunks),
            "k1": self.service.config.k1,
            "b": self.service.config.b,
            "index_build_time_sec": build_stats["total_time_sec"],
            "index_file_size_bytes": index_size_bytes,
            "index_file_size_mb": round(index_size_bytes / (1024 * 1024), 2),
            "output_index_file": saved_paths["index_file"],
            "output_manifest_file": saved_paths["manifest_file"],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        logger.info(
            f"BM25 indexing complete. Chunks: {len(chunks)}, Size: {summary['index_file_size_mb']} MB, Time: {build_stats['total_time_sec']}s."
        )
        return summary
