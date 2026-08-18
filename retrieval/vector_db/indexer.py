"""Batch vector indexer for populating Qdrant database."""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from ingestion.chunking.models import Chunk
from retrieval.embeddings.models import VectorArtifact
from retrieval.vector_db.models import QdrantConfig
from retrieval.vector_db.service import QdrantService

logger = logging.getLogger(__name__)

DEFAULT_VECTORS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "embeddings", "vectors.jsonl")
)
DEFAULT_CHUNKS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "chunks", "final_chunks.jsonl")
)


class BatchQdrantIndexer:
    """Indexer streaming production vector artifacts into Qdrant in configurable batches."""

    def __init__(self, service: Optional[QdrantService] = None):
        self.service = service or QdrantService()

    def load_chunks_map(self, chunks_path: str, max_chunks: Optional[int] = None) -> Dict[str, Chunk]:
        """Load source chunk lookup dict by chunk_id for rich payload construction."""
        chunks_map: Dict[str, Chunk] = {}
        if not os.path.exists(chunks_path):
            logger.warning(f"Chunks file '{chunks_path}' not found. Payload text will be minimal.")
            return chunks_map

        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    c = Chunk.from_dict(json.loads(line))
                    chunks_map[c.chunk_id] = c
                    if max_chunks and len(chunks_map) >= max_chunks:
                        break

        logger.info(f"Loaded {len(chunks_map)} source chunks for payload mapping.")
        return chunks_map

    def index_vectors_file(
        self,
        vectors_jsonl: Optional[str] = None,
        chunks_jsonl: Optional[str] = None,
        max_vectors: Optional[int] = None,
        batch_size: Optional[int] = None,
        recreate_collection: bool = False,
    ) -> Dict[str, Any]:
        """Stream and index vector artifacts JSONL file into Qdrant."""
        v_path = vectors_jsonl or DEFAULT_VECTORS_PATH
        c_path = chunks_jsonl or DEFAULT_CHUNKS_PATH
        b_size = batch_size or self.service.config.batch_size

        if not os.path.exists(v_path):
            raise FileNotFoundError(f"Vector artifacts file not found at '{v_path}'. Run Phase 4.1 embeddings first.")

        # Load source chunk lookup
        chunks_map = self.load_chunks_map(c_path, max_chunks=max_vectors)

        # Create/verify collection
        self.service.create_collection(recreate=recreate_collection)

        # Stream vector artifacts
        start_time = time.time()
        processed_count = 0
        batch: List[VectorArtifact] = []

        with open(v_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    art = VectorArtifact.from_dict(json.loads(line))
                    batch.append(art)

                    if len(batch) >= b_size:
                        self.service.upsert_batch(batch, chunks_map=chunks_map)
                        processed_count += len(batch)
                        batch = []
                        if processed_count % (b_size * 5) == 0:
                            logger.info(f"Indexed {processed_count} vectors into Qdrant...")

                    if max_vectors and processed_count + len(batch) >= max_vectors:
                        break

        if batch:
            self.service.upsert_batch(batch, chunks_map=chunks_map)
            processed_count += len(batch)

        elapsed_sec = round(time.time() - start_time, 4)
        final_point_count = self.service.count()
        throughput = round(processed_count / max(0.001, elapsed_sec), 2)

        summary = {
            "collection_name": self.service.config.collection_name,
            "vector_dimension": self.service.config.vector_size,
            "distance_metric": self.service.config.distance,
            "vectors_processed": processed_count,
            "qdrant_point_count": final_point_count,
            "batch_size": b_size,
            "elapsed_seconds": elapsed_sec,
            "throughput_points_per_sec": throughput,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        logger.info(
            f"Qdrant indexing complete. Processed {processed_count} vectors. Point count: {final_point_count}. Throughput: {throughput} points/sec."
        )
        return summary
