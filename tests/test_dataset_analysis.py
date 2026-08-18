"""Unit tests for MSMARCO dataset analyzer."""

import json
import os
import pytest
from unittest.mock import MagicMock, patch

from ingestion.dataset_analyzer import (
    MSMARCODatasetAnalyzer,
    calc_percentiles,
)


def test_calc_percentiles():
    """Test percentile calculations for numeric list."""
    vals = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    res = calc_percentiles(vals)
    assert res["count"] == 10
    assert res["min"] == 10.0
    assert res["max"] == 100.0
    assert res["mean"] == 55.0
    assert res["median"] == 55.0
    assert res["p25"] == 32.5
    assert res["p75"] == 77.5


def test_calc_percentiles_empty():
    """Test percentile calculations for empty list."""
    res = calc_percentiles([])
    assert res["count"] == 0
    assert res["min"] == 0
    assert res["mean"] == 0.0


def test_analyzer_with_mock_data(tmp_path):
    """Test dataset analyzer outputs schema.json, statistics.json, samples.json."""
    output_dir = str(tmp_path / "docs" / "dataset")
    mock_loader = MagicMock()

    mock_inspection = MagicMock()
    mock_inspection.dataset_name = "ai4bharat/MSMARCO-XI"
    mock_inspection.configs_available = ["default"]
    mock_inspection.splits_discovered = {"train": 1000, "validation": 100}
    mock_inspection.schema = {"query_id": "int64", "query": "string"}
    mock_inspection.nested_schemas = {}
    mock_inspection.has_query_fields = True
    mock_inspection.has_passage_fields = True
    mock_inspection.has_ids = True
    mock_inspection.has_relevance_ground_truth = True
    mock_inspection.has_language_info = True
    mock_inspection.has_metadata = True

    mock_loader.inspect.return_value = mock_inspection

    mock_sample = {
        "query_id": 1,
        "Eng_Query": "What is AI?",
        "query": "AI क्या है?",
        "Eng_Answer": "AI is Artificial Intelligence.",
        "Answer": "AI का मतलब कृत्रिम बुद्धिमत्ता है।",
        "source_lang": "eng_Latn",
        "target_lang": "hin_Deva",
        "query_type": "DESCRIPTION",
        "passages": {
            "English_passages": ["Artificial intelligence (AI) is intelligence demonstrated by machines."],
            "Translated_passages": ["कृत्रिम बुद्धिमत्ता (एआई) मशीनों द्वारा प्रदर्शित बुद्धिमत्ता है।"],
            "is_selected": [1],
        },
    }
    mock_loader.load_dataset.return_value = [mock_sample]

    analyzer = MSMARCODatasetAnalyzer(loader=mock_loader, output_dir=output_dir)
    res = analyzer.analyze(sample_size=10, split="validation")

    assert os.path.exists(os.path.join(output_dir, "schema.json"))
    assert os.path.exists(os.path.join(output_dir, "statistics.json"))
    assert os.path.exists(os.path.join(output_dir, "samples.json"))

    assert res["statistics"]["sample_size"] == 1
    assert res["statistics"]["text_distributions"]["eng_query_word"]["mean"] == 3.0
    assert res["statistics"]["duplicates"]["unique_query_ids"] == 1
