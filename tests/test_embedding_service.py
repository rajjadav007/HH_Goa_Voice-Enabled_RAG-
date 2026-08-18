"""Unit tests for Phase 4.1 Embedding Service and Processor."""

import json
import math
import os
import numpy as np
import pytest

from retrieval.embeddings import (
    BatchEmbeddingProcessor,
    EmbeddingConfig,
    EmbeddingService,
    VectorArtifact,
)
from ingestion.chunking import Chunk


def cosine_similarity(v1, v2):
    a = np.array(v1)
    b = np.array(v2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_embedding_service_dimension_and_sanity():
    """Test embedding service returns finite, normalized vectors with correct dimension."""
    config = EmbeddingConfig(model_name="intfloat/multilingual-e5-small", batch_size=4)
    service = EmbeddingService(config=config)

    assert service.dimension > 0
    vec = service.embed_text("What is a corporation?", is_query=True)

    assert len(vec) == service.dimension
    assert all(math.isfinite(x) for x in vec)

    # Verify L2 norm is ~1.0
    norm = np.linalg.norm(vec)
    assert abs(norm - 1.0) < 1e-3


test_embedding_service_dimension_and_sanity()


def test_query_document_similarity_sanity():
    """Test query and passage embeddings produce higher cosine similarity for relevant content."""
    config = EmbeddingConfig(model_name="intfloat/multilingual-e5-small")
    service = EmbeddingService(config=config)

    q_vec = service.embed_text("How does vitamin D help bone health?", is_query=True)
    doc_rel_vec = service.embed_text("Vitamin D enhances intestinal absorption of calcium and phosphate for strong bones.", is_query=False)
    doc_irrel_vec = service.embed_text("A corporation is a legal entity created under company laws.", is_query=False)

    sim_rel = cosine_similarity(q_vec, doc_rel_vec)
    sim_irrel = cosine_similarity(q_vec, doc_irrel_vec)

    assert sim_rel > sim_irrel


def test_batch_embedding_processor(tmp_path):
    """Test BatchEmbeddingProcessor creates traceable vector artifacts and manifest."""
    chunks_path = str(tmp_path / "chunks.jsonl")
    embeddings_dir = str(tmp_path / "embeddings")

    c1 = Chunk(
        chunk_id="chk_doc1_1",
        document_id="doc1",
        text="Vitamin D is essential for bones.",
        chunk_index=0,
        chunk_strategy="semantic",
        token_count=6,
        character_count=33,
        metadata={"source_query_id": 101, "is_selected": 1, "language": "eng_Latn"},
    )
    c2 = Chunk(
        chunk_id="chk_doc2_1",
        document_id="doc2",
        text="A corporation is a legal entity.",
        chunk_index=0,
        chunk_strategy="semantic",
        token_count=6,
        character_count=32,
        metadata={"source_query_id": 102, "is_selected": 0, "language": "eng_Latn"},
    )

    with open(chunks_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(c1.to_dict()) + "\n")
        f.write(json.dumps(c2.to_dict()) + "\n")

    config = EmbeddingConfig(model_name="intfloat/multilingual-e5-small", batch_size=2)
    service = EmbeddingService(config=config)
    processor = BatchEmbeddingProcessor(service=service, embeddings_dir=embeddings_dir)

    manifest = processor.process_chunks_file(
        input_chunks_jsonl=chunks_path,
        resume=False,
    )

    assert manifest["total_vector_artifacts"] == 2
    assert manifest["dimension"] == service.dimension
    assert os.path.exists(os.path.join(embeddings_dir, "vectors.jsonl"))
    assert os.path.exists(os.path.join(embeddings_dir, "manifest.json"))

    # Test resumability
    manifest_resume = processor.process_chunks_file(
        input_chunks_jsonl=chunks_path,
        resume=True,
    )
    assert manifest_resume["total_chunks_processed"] == 0
    assert manifest_resume["total_vector_artifacts"] == 2
