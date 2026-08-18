"""Unit tests for MSMARCO dataset loader."""

import os
import pytest
from unittest.mock import MagicMock, patch

from ingestion.dataset_loader import (
    MSMARCODatasetLoader,
    DatasetDownloadError,
    DatasetLoadError,
    InvalidDatasetStructureError,
)


def test_loader_initialization(tmp_path):
    """Test MSMARCODatasetLoader initialization and directory creation."""
    raw_dir = tmp_path / "data" / "raw"
    loader = MSMARCODatasetLoader(raw_data_dir=str(raw_dir))
    assert os.path.exists(loader.raw_data_dir)
    assert os.path.exists(loader.cache_dir)


@patch("ingestion.dataset_loader._hf_load")
def test_load_dataset_error_handling(mock_load):
    """Test custom DatasetLoadError wrapper when HF load_dataset fails."""
    mock_load.side_effect = RuntimeError("HuggingFace network timeout")
    loader = MSMARCODatasetLoader()

    with pytest.raises(DatasetLoadError, match="Error loading"):
        loader.load_dataset(config="hi", split="validation", streaming=False)


def test_inspect_method(tmp_path):
    """Test dataset loader inspect method returns structured result."""
    loader = MSMARCODatasetLoader(raw_data_dir=str(tmp_path))
    with patch("datasets.get_dataset_config_names", return_value=["default"]), \
         patch("datasets.load_dataset_builder") as mock_builder, \
         patch("ingestion.dataset_loader._hf_load") as mock_hf_load:
        
        mock_info = MagicMock()
        mock_info.features = None
        mock_info.splits = None
        mock_builder.return_value.info = mock_info

        mock_sample = {
            "query_id": 100,
            "query": "क",
            "Eng_Query": "q",
            "passages": {"English_passages": ["p1"], "is_selected": [1]},
            "source_lang": "eng",
            "target_lang": "hin",
        }
        mock_hf_load.return_value = [mock_sample]

        res = loader.inspect(config="default", num_samples=1)
        assert res.dataset_name == loader.dataset_name
        assert res.has_query_fields is True
        assert res.has_passage_fields is True
