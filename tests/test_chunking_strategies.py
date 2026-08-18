"""Unit tests for multiple chunking strategies (Phase 3.2)."""

import pytest

from ingestion.chunking import ChunkerRegistry, ChunkingConfig
from ingestion.chunking.strategies import (
    FixedSizeChunker,
    HybridChunker,
    SemanticChunker,
    SentenceChunker,
    StructureAwareChunker,
)
from ingestion.preprocessor import ProcessedDocument


@pytest.fixture
def sample_document():
    return ProcessedDocument(
        document_id="doc_812940_0_test123",
        text=(
            "First sentence about vitamin D and health. Second sentence describes calcium absorption. "
            "Third sentence mentions sunlight exposure for synthesis.\n\n"
            "Fourth sentence starts a new paragraph about immunity. Fifth sentence details bone health."
        ),
        english_text="First sentence about vitamin D...",
        source_query_id=812940,
        passage_index=0,
        is_selected=1,
        language="eng_Latn",
        metadata={"query_type": "DESCRIPTION"},
    )


def test_registry_lists_all_strategies():
    """Verify registry contains all 6 implemented strategies."""
    strategies = ChunkerRegistry.list_strategies()
    assert "passthrough" in strategies
    assert "fixed" in strategies
    assert "sentence" in strategies
    assert "structure" in strategies
    assert "semantic" in strategies
    assert "hybrid" in strategies


def test_fixed_size_chunker(sample_document):
    """Test FixedSizeChunker sliding window and overlap."""
    cfg = ChunkingConfig(strategy="fixed", target_chunk_size=10, overlap=3)
    chunker = ChunkerRegistry.get("fixed", config=cfg)
    chunks = chunker.process(sample_document)

    assert len(chunks) > 1
    for c in chunks:
        assert c.chunk_strategy == "fixed"
        assert c.document_id == sample_document.document_id
        assert c.token_count <= 10 or len(c.text.split()) <= 10


def test_fixed_size_chunker_determinism(sample_document):
    """Test FixedSizeChunker output determinism."""
    cfg = ChunkingConfig(strategy="fixed", target_chunk_size=15, overlap=4)
    c1 = FixedSizeChunker(config=cfg).process(sample_document)
    c2 = FixedSizeChunker(config=cfg).process(sample_document)

    assert len(c1) == len(c2)
    for chk1, chk2 in zip(c1, c2):
        assert chk1.chunk_id == chk2.chunk_id
        assert chk1.text == chk2.text


def test_sentence_chunker(sample_document):
    """Test SentenceChunker respects sentence boundaries."""
    cfg = ChunkingConfig(strategy="sentence", target_chunk_size=20)
    chunker = SentenceChunker(config=cfg)
    chunks = chunker.process(sample_document)

    assert len(chunks) >= 1
    for c in chunks:
        assert c.chunk_strategy == "sentence"
        assert c.metadata["source_query_id"] == 812940


def test_structure_aware_chunker(sample_document):
    """Test StructureAwareChunker preserves paragraph boundaries."""
    cfg = ChunkingConfig(strategy="structure", target_chunk_size=25)
    chunker = StructureAwareChunker(config=cfg)
    chunks = chunker.process(sample_document)

    assert len(chunks) >= 2
    for c in chunks:
        assert c.chunk_strategy == "structure"
        assert c.metadata["structure_unit"] == "paragraph"


def test_semantic_chunker(sample_document):
    """Test SemanticChunker threshold boundary splitting."""
    cfg = ChunkingConfig(strategy="semantic", semantic_threshold=0.3, target_chunk_size=15)
    chunker = SemanticChunker(config=cfg)
    chunks = chunker.process(sample_document)

    assert len(chunks) >= 1
    for c in chunks:
        assert c.chunk_strategy == "semantic"
        assert "semantic_threshold" in c.metadata


def test_hybrid_chunker(sample_document):
    """Test HybridChunker combining paragraph, sentence, and size constraints."""
    cfg = ChunkingConfig(strategy="hybrid", target_chunk_size=15, overlap=3)
    chunker = HybridChunker(config=cfg)
    chunks = chunker.process(sample_document)

    assert len(chunks) >= 1
    for c in chunks:
        assert c.chunk_strategy == "hybrid"
        assert "hybrid_components" in c.metadata


def test_edge_case_short_document():
    """Test chunkers on short 1-word or 1-sentence document."""
    short_doc = ProcessedDocument(
        document_id="doc_short",
        text="SingleShortSentence.",
        english_text="SingleShortSentence.",
        source_query_id=1,
        passage_index=0,
        is_selected=0,
        language="eng_Latn",
    )

    for strat in ["passthrough", "fixed", "sentence", "structure", "semantic", "hybrid"]:
        chunker = ChunkerRegistry.get(strat)
        chunks = chunker.process(short_doc)
        assert len(chunks) == 1
        assert chunks[0].text == "SingleShortSentence."
        assert chunks[0].document_id == "doc_short"


def test_edge_case_empty_document():
    """Test chunkers on empty text document."""
    empty_doc = ProcessedDocument(
        document_id="doc_empty",
        text="",
        english_text="",
        source_query_id=1,
        passage_index=0,
        is_selected=0,
        language="eng_Latn",
    )

    for strat in ["passthrough", "fixed", "sentence", "structure", "semantic", "hybrid"]:
        chunker = ChunkerRegistry.get(strat)
        chunks = chunker.process(empty_doc)
        assert len(chunks) == 0
