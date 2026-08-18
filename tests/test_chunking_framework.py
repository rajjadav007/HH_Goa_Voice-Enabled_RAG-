"""Unit tests for Chunking Framework (Phase 3.1)."""

import json
import os
import pytest

from ingestion.chunking import (
    BaseChunker,
    BatchChunkProcessor,
    Chunk,
    ChunkerRegistry,
    ChunkingConfig,
    InvalidChunkError,
    PassthroughChunker,
    StrategyNotFoundError,
    count_tokens,
    generate_stable_chunk_id,
    split_paragraphs,
    split_sentences,
    validate_chunk,
)
from ingestion.preprocessor import ProcessedDocument


def test_chunk_model_and_stable_id():
    """Test Chunk model creation and deterministic chunk ID generation."""
    id1 = generate_stable_chunk_id("doc_100", "passthrough", 0, "Test text content")
    id2 = generate_stable_chunk_id("doc_100", "passthrough", 0, "Test text content")
    id3 = generate_stable_chunk_id("doc_100", "passthrough", 1, "Test text content")

    assert id1 == id2
    assert id1 != id3
    assert id1.startswith("chk_doc_100_passthrough_0_")


def test_validate_chunk_valid():
    """Test Chunk validation with a valid Chunk object."""
    chunk = Chunk(
        chunk_id="chk_1",
        document_id="doc_1",
        text="Sample chunk text",
        chunk_index=0,
        chunk_strategy="passthrough",
        token_count=3,
        character_count=17,
        metadata={"source_query_id": 123},
    )
    assert validate_chunk(chunk) is True


def test_validate_chunk_invalid():
    """Test Chunk validation raises error for invalid chunks."""
    invalid_chunk = Chunk(
        chunk_id="",
        document_id="doc_1",
        text="Text",
        chunk_index=0,
        chunk_strategy="test",
        token_count=1,
        character_count=4,
    )
    with pytest.raises(InvalidChunkError, match="missing valid chunk_id"):
        validate_chunk(invalid_chunk)

    empty_text_chunk = Chunk(
        chunk_id="chk_1",
        document_id="doc_1",
        text="  ",
        chunk_index=0,
        chunk_strategy="test",
        token_count=0,
        character_count=0,
    )
    with pytest.raises(InvalidChunkError, match="empty or whitespace-only text"):
        validate_chunk(empty_text_chunk)


def test_text_utilities():
    """Test token counting and Indic/English sentence splitting."""
    text = "Hello world! This is test 1. क्या हाल है? यह टेस्ट २ है।"
    assert count_tokens("Hello world") == 2

    sentences = split_sentences(text)
    assert len(sentences) == 4
    assert "Hello world!" in sentences
    assert "क्या हाल है?" in sentences

    paras = split_paragraphs("Para 1 content.\n\nPara 2 content.")
    assert len(paras) == 2
    assert paras[0] == "Para 1 content."


def test_registry_registration_and_get():
    """Test ChunkerRegistry registration and strategy resolution."""
    assert "passthrough" in ChunkerRegistry.list_strategies()

    chunker = ChunkerRegistry.get("passthrough")
    assert isinstance(chunker, PassthroughChunker)
    assert chunker.name == "passthrough"

    with pytest.raises(StrategyNotFoundError):
        ChunkerRegistry.get("non_existent_strategy")


def test_passthrough_chunker_parent_traceability():
    """Test PassthroughChunker preserves document metadata and parent traceability."""
    doc = ProcessedDocument(
        document_id="doc_812940_0_abc123",
        text="विटामिन डी शरीर के लिए आवश्यक है।",
        english_text="Vitamin D is essential for body.",
        source_query_id=812940,
        passage_index=0,
        is_selected=1,
        language="hin_Deva",
        metadata={"query_type": "DESCRIPTION"},
    )

    chunker = PassthroughChunker()
    chunks = chunker.process(doc)

    assert len(chunks) == 1
    c = chunks[0]
    assert c.document_id == doc.document_id
    assert c.text == doc.text
    assert c.metadata["source_query_id"] == doc.source_query_id
    assert c.metadata["is_selected"] == doc.is_selected
    assert c.metadata["language"] == doc.language


def test_batch_processor_jsonl(tmp_path):
    """Test BatchChunkProcessor processes processed document JSONL into chunk JSONL."""
    doc_file = str(tmp_path / "documents.jsonl")
    chunk_file = str(tmp_path / "chunks.jsonl")
    manifest_file = str(tmp_path / "manifest.json")

    sample_doc = {
        "document_id": "doc_100_0_hash",
        "text": "Sample passage text for batch testing.",
        "english_text": "Sample passage text for batch testing.",
        "source_query_id": 100,
        "passage_index": 0,
        "is_selected": 1,
        "language": "eng_Latn",
        "metadata": {"query_type": "DESCRIPTION"},
    }

    with open(doc_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(sample_doc) + "\n")

    processor = BatchChunkProcessor()
    manifest = processor.process_jsonl_file(
        input_documents_jsonl=doc_file,
        output_chunks_jsonl=chunk_file,
        output_manifest_json=manifest_file,
    )

    assert manifest["input_document_count"] == 1
    assert manifest["output_chunk_count"] == 1
    assert manifest["rejected_document_count"] == 0
    assert os.path.exists(chunk_file)
    assert os.path.exists(manifest_file)
