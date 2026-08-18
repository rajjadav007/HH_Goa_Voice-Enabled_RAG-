"""Unit tests for Phase 4.2 Qdrant Vector Database Service and Indexer."""

import json
import os
import pytest

from ingestion.chunking import Chunk
from retrieval.embeddings import EmbeddingConfig, EmbeddingService, VectorArtifact
from retrieval.vector_db import (
    BatchQdrantIndexer,
    QdrantConfig,
    QdrantService,
    generate_deterministic_point_id,
)


def test_deterministic_point_id():
    """Test point ID generation is deterministic and stable."""
    pid1 = generate_deterministic_point_id("chk_doc1_fixed_0_123456789abc")
    pid2 = generate_deterministic_point_id("chk_doc1_fixed_0_123456789abc")
    pid3 = generate_deterministic_point_id("chk_doc1_fixed_1_123456789abc")

    assert pid1 == pid2
    assert pid1 != pid3


def test_qdrant_service_collection_and_upsert():
    """Test QdrantService collection creation, idempotent upsert, count, and vector search."""
    config = QdrantConfig(
        path=":memory:",
        collection_name="test_chunks",
        vector_size=384,
        distance="Cosine",
    )
    service = QdrantService(config=config)

    assert service.create_collection(recreate=True)
    assert service.collection_exists()

    # Create dummy vector artifacts
    art1 = VectorArtifact(
        chunk_id="chk_101_0",
        document_id="doc_101",
        vector=[0.1] * 384,
        embedding_model="intfloat/multilingual-e5-small",
        dimension=384,
        metadata={"text": "Vitamin D supports calcium absorption and bone health."},
    )
    art2 = VectorArtifact(
        chunk_id="chk_102_0",
        document_id="doc_102",
        vector=[-0.1] * 384,
        embedding_model="intfloat/multilingual-e5-small",
        dimension=384,
        metadata={"text": "A corporation is a separate legal entity."},
    )

    # First upsert
    service.upsert_batch([art1, art2])
    assert service.count() == 2

    # Idempotent second upsert (same chunk_ids) should not increase point count
    service.upsert_batch([art1, art2])
    assert service.count() == 2

    # Test vector search
    q_vector = [0.1] * 384
    results = service.search(query_vector=q_vector, top_k=2)

    assert len(results) == 2
    assert results[0].chunk_id == "chk_101_0"
    assert results[0].document_id == "doc_101"
    assert results[0].score > results[1].score
    assert "Vitamin D" in results[0].text


def test_batch_qdrant_indexer(tmp_path):
    """Test BatchQdrantIndexer indexes vectors JSONL file and validates point count."""
    vectors_path = str(tmp_path / "vectors.jsonl")
    chunks_path = str(tmp_path / "chunks.jsonl")

    c1 = Chunk(
        chunk_id="chk_1",
        document_id="doc_1",
        text="Sample passage text 1",
        chunk_index=0,
        chunk_strategy="fixed",
        token_count=4,
        character_count=21,
    )
    v1 = VectorArtifact(
        chunk_id="chk_1",
        document_id="doc_1",
        vector=[0.05] * 384,
        embedding_model="intfloat/multilingual-e5-small",
        dimension=384,
    )

    with open(chunks_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(c1.to_dict()) + "\n")

    with open(vectors_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(v1.to_dict()) + "\n")

    q_config = QdrantConfig(path=":memory:", collection_name="test_index_collection", vector_size=384)
    service = QdrantService(config=q_config)
    indexer = BatchQdrantIndexer(service=service)

    summary = indexer.index_vectors_file(
        vectors_jsonl=vectors_path,
        chunks_jsonl=chunks_path,
        recreate_collection=True,
    )

    assert summary["vectors_processed"] == 1
    assert summary["qdrant_point_count"] == 1
