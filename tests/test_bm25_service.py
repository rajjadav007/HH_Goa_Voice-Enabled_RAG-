"""Unit tests for Phase 4.3 BM25 Lexical Retrieval Service and Indexer."""

import json
import os
import pytest

from ingestion.chunking.models import Chunk
from retrieval.bm25.indexer import BatchBM25Indexer
from retrieval.bm25.models import BM25Config, BM25ResultPoint
from retrieval.bm25.service import BM25Service
from retrieval.bm25.tokenizer import MultilingualBM25Tokenizer


def test_multilingual_bm25_tokenizer():
    """Test tokenizer handles English, numbers, and Indic scripts."""
    tokenizer = MultilingualBM25Tokenizer()

    tokens = tokenizer.tokenize("Corporation 2026: এটা কোম্পানী")
    assert "corporation" in tokens
    assert "2026" in tokens
    assert "এটা" in tokens
    assert "কোম্পানী" in tokens

    # Test empty string
    assert tokenizer.tokenize("") == []
    assert tokenizer.tokenize("   ") == []


def test_bm25_service_build_save_load_search(tmp_path):
    """Test BM25Service index creation, persistence, loading, and lexical search."""
    c1 = Chunk(
        chunk_id="chk_101",
        document_id="doc_1",
        text="Vitamin D supports calcium absorption and bone strength.",
        chunk_index=0,
        chunk_strategy="semantic",
        token_count=8,
        character_count=57,
    )
    c2 = Chunk(
        chunk_id="chk_102",
        document_id="doc_2",
        text="A corporation is an independent legal entity formed by shareholders.",
        chunk_index=0,
        chunk_strategy="semantic",
        token_count=9,
        character_count=68,
    )
    c3 = Chunk(
        chunk_id="chk_103",
        document_id="doc_3",
        text="Cardiac health is essential for human longevity.",
        chunk_index=0,
        chunk_strategy="semantic",
        token_count=7,
        character_count=48,
    )

    index_dir = str(tmp_path / "bm25_index")
    config = BM25Config(k1=1.5, b=0.75, top_k=5, index_dir=index_dir)
    service = BM25Service(config=config)

    # Build index
    stats = service.build_index([c1, c2, c3])
    assert stats["indexed_chunks"] == 3
    assert service.is_loaded

    # Save index
    saved = service.save_index(index_dir=index_dir)
    assert os.path.exists(saved["index_file"])
    assert os.path.exists(saved["manifest_file"])

    # Load into new service instance
    new_service = BM25Service(config=config)
    assert new_service.load_index(index_dir=index_dir)
    assert new_service.is_loaded
    assert len(new_service.chunks_corpus) == 3

    # Execute search
    results = new_service.search("corporation shareholders", top_k=2)
    assert len(results) == 1
    assert results[0].chunk_id == "chk_102"
    assert results[0].document_id == "doc_2"
    assert results[0].method == "bm25"
    assert results[0].rank == 1
    assert results[0].score > 0.0

    # Test empty/invalid query
    assert new_service.search("") == []
    assert new_service.search("   ") == []
    assert new_service.search("nonexistentwordxyz123") == []


def test_batch_bm25_indexer(tmp_path):
    """Test BatchBM25Indexer builds index from JSONL file and validates count."""
    chunks_path = str(tmp_path / "chunks.jsonl")
    out_dir = str(tmp_path / "bm25_output")

    c1 = Chunk(
        chunk_id="chk_201",
        document_id="doc_201",
        text="Retrospective study on cardiac diseases in 2026.",
        chunk_index=0,
        chunk_strategy="semantic",
        token_count=7,
        character_count=48,
    )

    with open(chunks_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(c1.to_dict()) + "\n")

    indexer = BatchBM25Indexer()
    summary = indexer.index_chunks_file(
        input_chunks_jsonl=chunks_path,
        output_index_dir=out_dir,
    )

    assert summary["indexed_chunks"] == 1
    assert summary["index_file_size_bytes"] > 0
    assert os.path.exists(summary["output_index_file"])
