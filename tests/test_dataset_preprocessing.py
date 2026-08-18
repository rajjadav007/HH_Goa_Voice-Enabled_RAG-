"""Unit tests for MSMARCO dataset preprocessor."""

import json
import os
import pytest

from ingestion.preprocessor import (
    MSMARCOPreprocessor,
    PreprocessingConfig,
    ProcessedDocument,
    ProcessedQuery,
    generate_stable_document_id,
    normalize_text,
)


def test_generate_stable_document_id():
    """Test stable document ID generation is deterministic."""
    id1 = generate_stable_document_id(100, 0, "Test passage text", "hin_Deva")
    id2 = generate_stable_document_id(100, 0, "Test passage text", "hin_Deva")
    id3 = generate_stable_document_id(100, 1, "Test passage text", "hin_Deva")

    assert id1 == id2
    assert id1 != id3
    assert id1.startswith("doc_100_0_")


def test_normalize_text():
    """Test text normalization rules."""
    raw = "  Hello \x00 world!\n\n  This is   NFC test.  "
    norm = normalize_text(raw)
    assert norm == "Hello world! This is NFC test."


def test_process_record_valid():
    """Test processing a valid raw MSMARCO-XI record."""
    raw_record = {
        "query_id": 812940,
        "query": "क्या विटामिन डी स्वास्थ्य के लिए आवश्यक है?",
        "Eng_Query": "Is vitamin D essential for health?",
        "Answer": "हाँ, विटामिन डी आवश्यक है।",
        "Eng_Answer": "Yes, vitamin D is essential.",
        "query_type": "DESCRIPTION",
        "source_lang": "eng_Latn",
        "target_lang": "hin_Deva",
        "meta": {"model_name": "gpt-4"},
        "passages": {
            "is_selected": [1, 0],
            "English_passages": [
                "Vitamin D regulates calcium absorption.",
                "Exposure to sunlight helps synthesis of vitamin D.",
            ],
            "Translated_passages": [
                "विटामिन डी शरीर में कैल्शियम अवशोषण को नियंत्रित करता है।",
                "धूप के संपर्क में आने से विटामिन डी का संश्लेषण होता है।",
            ],
        },
    }

    preprocessor = MSMARCOPreprocessor()
    q_obj, docs, rej = preprocessor.process_record(raw_record)

    assert rej is None
    assert q_obj is not None
    assert q_obj.query_id == 812940
    assert len(docs) == 2
    assert len(q_obj.relevant_document_ids) == 1
    assert q_obj.relevant_document_ids[0] == docs[0].document_id
    assert docs[0].is_selected == 1
    assert docs[1].is_selected == 0


def test_process_record_missing_query_id():
    """Test rejection when query_id is missing."""
    raw_record = {"query": "test"}
    preprocessor = MSMARCOPreprocessor()
    q_obj, docs, rej = preprocessor.process_record(raw_record)

    assert q_obj is None
    assert docs == []
    assert rej == "missing_query_id"


def test_process_record_empty_passages():
    """Test rejection when passages struct is empty or missing."""
    raw_record = {"query_id": 123, "query": "test", "passages": {}}
    preprocessor = MSMARCOPreprocessor()
    q_obj, docs, rej = preprocessor.process_record(raw_record)

    assert q_obj is None
    assert docs == []
    assert rej == "empty_passages_list"


def test_process_dataset_stream(tmp_path):
    """Test end-to-end processing stream generates JSONL files and manifest."""
    processed_dir = str(tmp_path / "data" / "processed")
    config = PreprocessingConfig(processed_dir=processed_dir)
    preprocessor = MSMARCOPreprocessor(config=config)

    records = [
        {
            "query_id": 1,
            "query": "q1",
            "Eng_Query": "q1",
            "source_lang": "eng_Latn",
            "target_lang": "asm_Beng",
            "passages": {
                "is_selected": [1],
                "English_passages": ["Passage content text for document one."],
                "Translated_passages": ["translated passage text."],
            },
        }
    ]

    manifest = preprocessor.process_dataset_stream(records)

    assert manifest["input_records"] == 1
    assert manifest["processed_queries"] == 1
    assert manifest["processed_documents"] == 1
    assert manifest["rejected_records"] == 0

    assert os.path.exists(os.path.join(processed_dir, "queries.jsonl"))
    assert os.path.exists(os.path.join(processed_dir, "documents.jsonl"))
    assert os.path.exists(os.path.join(processed_dir, "manifest.json"))
