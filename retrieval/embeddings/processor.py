"""Batch embedding processor for persistence and resumability."""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Set

from ingestion.chunking.models import Chunk
from retrieval.embeddings.models import EmbeddingConfig, VectorArtifact
from retrieval.embeddings.service import EmbeddingService

logger = logging.getLogger(__name__)

DEFAULT_FINAL_CHUNKS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "chunks", "final_chunks.jsonl")
)
DEFAULT_EMBEDDINGS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "embeddings")
)


class BatchEmbeddingProcessor:
    """Batch processor generating versioned, traceable vector artifacts from chunks."""

    def __init__(
        self,
        service: Optional[EmbeddingService] = None,
        embeddings_dir: Optional[str] = None,
    ):
        self.service = service or EmbeddingService()
        self.embeddings_dir = embeddings_dir or DEFAULT_EMBEDDINGS_DIR
        os.makedirs(self.embeddings_dir, exist_ok=True)

    def process_chunks_file(
        self,
        input_chunks_jsonl: Optional[str] = None,
        output_vectors_jsonl: Optional[str] = None,
        output_manifest_json: Optional[str] = None,
        max_chunks: Optional[int] = None,
        resume: bool = True,
    ) -> Dict[str, Any]:
        """Process production chunks JSONL file into vector artifacts JSONL file."""
        input_path = input_chunks_jsonl or DEFAULT_FINAL_CHUNKS_PATH
        output_path = output_vectors_jsonl or os.path.join(self.embeddings_dir, "vectors.jsonl")
        manifest_path = output_manifest_json or os.path.join(self.embeddings_dir, "manifest.json")

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Final chunks file not found at '{input_path}'. Run Phase 3.3 chunking first.")

        existing_chunk_ids: Set[str] = set()
        if resume and os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        d = json.loads(line)
                        existing_chunk_ids.add(d["chunk_id"])
            logger.info(f"Resumability active. Found {len(existing_chunk_ids)} existing vector artifacts.")

        chunks_to_process: List[Chunk] = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunk = Chunk.from_dict(json.loads(line))
                    if chunk.chunk_id not in existing_chunk_ids:
                        chunks_to_process.append(chunk)
                    if max_chunks and len(chunks_to_process) >= max_chunks:
                        break

        total_to_process = len(chunks_to_process)
        logger.info(f"Processing {total_to_process} chunks into vectors using '{self.service.model_name}'...")

        batch_size = self.service.config.batch_size
        mode = "a" if resume and os.path.exists(output_path) else "w"

        start_time = time.time()
        processed_count = 0
        written_count = len(existing_chunk_ids)

        with open(output_path, mode, encoding="utf-8") as out_f:
            for i in range(0, total_to_process, batch_size):
                batch_chunks = chunks_to_process[i : i + batch_size]
                texts = [c.text for c in batch_chunks]

                vectors = self.service.embed_batch(texts, is_query=False)

                for chunk, vec in zip(batch_chunks, vectors):
                    artifact = VectorArtifact(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        vector=vec,
                        embedding_model=self.service.model_name,
                        dimension=self.service.dimension,
                        metadata={
                            "source_query_id": chunk.metadata.get("source_query_id"),
                            "is_selected": chunk.metadata.get("is_selected"),
                            "language": chunk.metadata.get("language"),
                        },
                    )
                    out_f.write(json.dumps(artifact.to_dict(), ensure_ascii=False) + "\n")
                    written_count += 1

                processed_count += len(batch_chunks)
                if (i // batch_size) % 10 == 0 or i + batch_size >= total_to_process:
                    logger.info(f"Embedded {processed_count}/{total_to_process} chunks...")

        elapsed_sec = round(time.time() - start_time, 4)
        throughput = round(processed_count / max(0.001, elapsed_sec), 2)

        manifest = {
            "model_name": self.service.model_name,
            "dimension": self.service.dimension,
            "normalize_embeddings": self.service.config.normalize_embeddings,
            "similarity_metric": "cosine",
            "total_chunks_processed": processed_count,
            "total_vector_artifacts": written_count,
            "elapsed_seconds": elapsed_sec,
            "throughput_chunks_per_sec": throughput,
            "output_vectors_file": output_path,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(manifest_path, "w", encoding="utf-8") as mf:
            json.dump(manifest, mf, indent=2, ensure_ascii=False)

        logger.info(
            f"Embedding complete. Saved {written_count} vector artifacts to '{output_path}'. Throughput: {throughput} chunks/sec."
        )
        return manifest
