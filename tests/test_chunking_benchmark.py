"""Unit tests for Phase 3.3 Chunking Benchmark and Selection."""

import json
import os
import pytest

from ingestion.chunking import (
    Chunk,
    ChunkingBenchmarkRunner,
    ChunkingConfig,
    LightweightEvalIndex,
)
from ingestion.preprocessor import ProcessedDocument, ProcessedQuery


def test_lightweight_eval_index():
    """Test LightweightEvalIndex building and keyword-weighted retrieval."""
    chunks = [
        Chunk(
            chunk_id="c1",
            document_id="doc1",
            text="Vitamin D is essential for bone health and calcium absorption.",
            chunk_index=0,
            chunk_strategy="fixed",
            token_count=10,
            character_count=60,
        ),
        Chunk(
            chunk_id="c2",
            document_id="doc2",
            text="A corporation is a legal entity created by a group of people.",
            chunk_index=0,
            chunk_strategy="fixed",
            token_count=11,
            character_count=61,
        ),
    ]

    index = LightweightEvalIndex()
    index.build(chunks)

    results = index.search("vitamin D bone health", top_k=2)
    assert len(results) >= 1
    assert results[0][0].document_id == "doc1"


def test_benchmark_runner_matrix(tmp_path):
    """Test ChunkingBenchmarkRunner evaluates matrix and outputs JSON artifacts."""
    processed_dir = str(tmp_path / "data" / "processed")
    eval_dir = str(tmp_path / "evaluation" / "chunking")
    final_dir = str(tmp_path / "data" / "chunks")
    os.makedirs(processed_dir, exist_ok=True)

    queries_path = os.path.join(processed_dir, "queries.jsonl")
    docs_path = os.path.join(processed_dir, "documents.jsonl")

    sample_query = {
        "query_id": 100,
        "query_text": "What is a corporation?",
        "eng_query_text": "What is a corporation?",
        "query_type": "DESCRIPTION",
        "source_lang": "eng_Latn",
        "target_lang": "eng_Latn",
        "relevant_document_ids": ["doc_100_0_hash"],
        "all_document_ids": ["doc_100_0_hash"],
    }
    sample_doc = {
        "document_id": "doc_100_0_hash",
        "text": "A corporation is a company authorized to act as a single entity.",
        "english_text": "A corporation is a company authorized to act as a single entity.",
        "source_query_id": 100,
        "passage_index": 0,
        "is_selected": 1,
        "language": "eng_Latn",
        "metadata": {},
    }

    with open(queries_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(sample_query) + "\n")

    with open(docs_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(sample_doc) + "\n")

    runner = ChunkingBenchmarkRunner(
        processed_dir=processed_dir, eval_dir=eval_dir, final_dir=final_dir
    )

    summary = runner.run_benchmark_matrix(max_queries=10, max_documents=10)

    assert "winning_strategy" in summary
    assert "results_matrix" in summary
    assert os.path.exists(os.path.join(eval_dir, "results.json"))
    assert os.path.exists(os.path.join(eval_dir, "summary.json"))
    assert os.path.exists(os.path.join(eval_dir, "error_analysis.json"))

    # Test final production chunk generation
    winning_strat = summary["winning_strategy"]
    winning_cfg = ChunkingConfig(**summary["winning_config"])
    manifest = runner.generate_final_production_chunks(winning_strat, winning_cfg)

    assert manifest["output_chunk_count"] > 0
    assert os.path.exists(os.path.join(final_dir, "final_chunks.jsonl"))
    assert os.path.exists(os.path.join(final_dir, "final_manifest.json"))
