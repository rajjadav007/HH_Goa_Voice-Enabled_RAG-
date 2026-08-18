"""Dataset Loader Module for AI4Bharat MSMARCO-XI Dataset.

Responsible strictly for loading, validating, and inspecting the raw Hugging
Face MSMARCO-XI dataset in a reproducible, modular manner.  No schema
assumptions are hardcoded — every field name, type, and count is discovered at
runtime from the actual dataset.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — only the dataset identifier is fixed; nothing about its schema
# ---------------------------------------------------------------------------

DATASET_NAME = "ai4bharat/MSMARCO-XI"

DEFAULT_RAW_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class DatasetLoaderError(Exception):
    """Base exception for dataset loader errors."""


class DatasetDownloadError(DatasetLoaderError):
    """Raised when dataset download / network access fails."""


class DatasetLoadError(DatasetLoaderError):
    """Raised when dataset loading fails for non-network reasons."""


class InvalidDatasetStructureError(DatasetLoaderError):
    """Raised when loaded dataset has an unexpected or unusable structure."""


# ---------------------------------------------------------------------------
# Result dataclass — populated purely from live inspection
# ---------------------------------------------------------------------------


@dataclass
class DatasetInspectionResult:
    """Holds programmatic inspection results discovered at runtime."""

    dataset_name: str
    configs_available: List[str]
    splits_discovered: Dict[str, int]          # split_name -> record count
    schema: Dict[str, str]                     # field_name -> type description
    nested_schemas: Dict[str, Dict[str, str]]  # nested field breakdowns
    has_query_fields: bool
    has_passage_fields: bool
    has_ids: bool
    has_relevance_ground_truth: bool
    has_language_info: bool
    has_metadata: bool
    sample_records: List[Dict[str, Any]] = field(default_factory=list)
    raw_features_repr: str = ""
    inspection_warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class MSMARCODatasetLoader:
    """Modular dataset loader for AI4Bharat/MSMARCO-XI.

    Uses the Hugging Face Datasets library.  All schema information is
    discovered programmatically — nothing about field names or types is
    assumed ahead of time.
    """

    def __init__(
        self,
        dataset_name: str = DATASET_NAME,
        raw_data_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        try:
            import datasets  # noqa: F401  # verify library present
        except ImportError as exc:
            raise DatasetLoadError(
                "The 'datasets' library is required. "
                "Install it via: pip install datasets"
            ) from exc

        self.dataset_name = dataset_name
        self.raw_data_dir = raw_data_dir or DEFAULT_RAW_DIR
        self.cache_dir = cache_dir or os.path.join(self.raw_data_dir, "cache")
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.debug(
            "MSMARCODatasetLoader initialised — raw_data_dir=%s cache_dir=%s",
            self.raw_data_dir,
            self.cache_dir,
        )

    # ------------------------------------------------------------------
    # Public API: load
    # ------------------------------------------------------------------

    def load_dataset(
        self,
        config: Optional[str] = None,
        split: Optional[str] = None,
        streaming: bool = True,
        data_files: Optional[Union[str, Dict[str, str]]] = None,
    ):
        """Load MSMARCO-XI from Hugging Face Hub.

        Args:
            config:     Dataset configuration name (None = default).
            split:      Split name, e.g. 'train' or 'validation'.
                        None returns a DatasetDict with all splits.
            streaming:  When True, returns an IterableDataset (no full download).
            data_files: Optional override for which parquet files to load.

        Returns:
            Dataset | DatasetDict | IterableDataset
        """
        from datasets import load_dataset as hf_load_dataset

        logger.info(
            "Loading '%s' [config=%s split=%s streaming=%s]",
            self.dataset_name,
            config,
            split,
            streaming,
        )

        load_kwargs: Dict[str, Any] = {
            "cache_dir": self.cache_dir,
            "streaming": streaming,
        }
        if config is not None:
            load_kwargs["name"] = config
        if split is not None:
            load_kwargs["split"] = split
        if data_files is not None:
            load_kwargs["data_files"] = data_files

        try:
            ds = hf_load_dataset(self.dataset_name, **load_kwargs)
            logger.info("Dataset loaded successfully.")
            return ds
        except Exception as exc:
            err = str(exc).lower()
            logger.error("Failed to load dataset: %s", exc, exc_info=True)
            if any(k in err for k in ("download", "connection", "timeout", "network", "http")):
                raise DatasetDownloadError(
                    f"Download failed for '{self.dataset_name}': {exc}"
                ) from exc
            raise DatasetLoadError(
                f"Error loading '{self.dataset_name}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Public API: inspect (no assumptions — everything discovered at runtime)
    # ------------------------------------------------------------------

    def inspect(
        self,
        config: Optional[str] = None,
        num_samples: int = 3,
    ) -> DatasetInspectionResult:
        """Programmatically inspect the dataset.

        Uses load_dataset_builder (no full download) to retrieve schema/features,
        then loads a small streaming slice to get representative samples and
        to confirm actual field content at runtime.

        Args:
            config:      Dataset config name.  None = use dataset default.
            num_samples: Number of sample records to retrieve.

        Returns:
            DatasetInspectionResult populated entirely from live data.
        """
        from datasets import (
            get_dataset_config_names,
            get_dataset_split_names,
            load_dataset_builder,
        )

        warnings: List[str] = []

        # ---- 1. Discover available configs --------------------------------
        logger.info("Discovering configs for '%s'", self.dataset_name)
        try:
            configs_available = get_dataset_config_names(
                self.dataset_name, cache_dir=self.cache_dir
            )
            logger.info("Configs found: %s", configs_available)
        except Exception as exc:
            logger.warning("Could not list configs (%s); continuing.", exc)
            configs_available = [config] if config else []
            warnings.append(f"Config discovery failed: {exc}")

        # ---- 2. Decide which config to inspect ----------------------------
        inspect_config = config
        if inspect_config is None and configs_available:
            inspect_config = configs_available[0]
            logger.info("No config specified; using first available: '%s'", inspect_config)

        # ---- 3. Discover schema via dataset builder (no download) ---------
        logger.info(
            "Loading dataset builder for schema (config=%s)", inspect_config
        )
        raw_features_repr = ""
        schema: Dict[str, str] = {}
        nested_schemas: Dict[str, Dict[str, str]] = {}
        builder = None

        try:
            builder_kwargs: Dict[str, Any] = {"cache_dir": self.cache_dir}
            if inspect_config:
                builder_kwargs["name"] = inspect_config
            builder = load_dataset_builder(self.dataset_name, **builder_kwargs)
            features = builder.info.features

            if features is not None:
                raw_features_repr = repr(features)
                schema, nested_schemas = _flatten_features(features)
                logger.info("Schema discovered: %s", list(schema.keys()))
            else:
                warnings.append("Dataset builder returned no features.")
                logger.warning("Dataset builder returned no features.")
        except Exception as exc:
            logger.warning("Builder schema inspection failed (%s); will infer from samples.", exc)
            warnings.append(f"Builder schema inspection failed: {exc}")

        # ---- 4. Discover splits and record counts -------------------------
        # Use builder.info.splits first (no download required).  Fall back to
        # get_dataset_split_names if that is unavailable.
        splits_discovered: Dict[str, int] = {}
        split_names: List[str] = []

        # Fast path: counts from builder metadata
        try:
            if builder and builder.info.splits:
                for sname, sinfo in builder.info.splits.items():
                    splits_discovered[sname] = sinfo.num_examples
                    split_names.append(sname)
                logger.info(
                    "Split counts from builder metadata: %s",
                    splits_discovered,
                )
        except Exception as exc:
            logger.debug("builder.info.splits unavailable (%s).", exc)

        # Fall back to API call if builder didn't provide splits
        if not split_names:
            try:
                split_names = get_dataset_split_names(
                    self.dataset_name,
                    config_name=inspect_config,
                    cache_dir=self.cache_dir,
                )
                logger.info("Splits found via API: %s", split_names)
                for sname in split_names:
                    splits_discovered[sname] = -1  # count unknown without download
            except Exception as exc:
                logger.warning("Could not list splits (%s).", exc)
                split_names = ["train", "validation"]
                for sname in split_names:
                    splits_discovered[sname] = -1
                warnings.append(f"Split discovery failed: {exc}")

        # ---- 5. Fetch sample records from first available split -----------
        sample_records: List[Dict[str, Any]] = []
        # Prefer 'validation' (smaller parquets) over 'train' for fast sample fetch
        if "validation" in split_names:
            sample_split = "validation"
        elif split_names:
            sample_split = split_names[0]
        else:
            sample_split = "validation"

        try:
            stream_kwargs: Dict[str, Any] = {
                "split": sample_split,
                "streaming": True,
                "cache_dir": self.cache_dir,
            }
            if inspect_config:
                stream_kwargs["name"] = inspect_config
            ds_stream = _hf_load(self.dataset_name, **stream_kwargs)

            for i, record in enumerate(ds_stream):
                if i >= num_samples:
                    break
                sample_records.append(dict(record))

            # If schema was not discovered via builder, infer from first record
            if not schema and sample_records:
                schema, nested_schemas = _infer_schema_from_record(sample_records[0])
                logger.info("Schema inferred from sample record: %s", list(schema.keys()))

            logger.info("Fetched %d sample records from split '%s'.", len(sample_records), sample_split)
        except Exception as exc:
            logger.warning("Could not fetch sample records (%s).", exc)
            warnings.append(f"Sample fetch failed: {exc}")

        # ---- 6. Introspect field semantics (purely from discovered names) -
        all_fields = set(schema.keys())
        has_query = _any_field_contains(all_fields, {"query", "question", "q"})
        has_passages = _any_field_contains(
            all_fields, {"passage", "document", "context", "text", "answer"}
        )
        has_ids = _any_field_contains(all_fields, {"id"})
        has_relevance = _any_field_contains(
            all_fields, {"relevant", "selected", "label", "score", "relevance"}
        )
        has_lang = _any_field_contains(all_fields, {"lang", "language"})
        has_meta = _any_field_contains(all_fields, {"meta", "type", "category"})

        # Also check nested schemas for passage-level relevance
        for nested_field_schema in nested_schemas.values():
            nested_keys = set(nested_field_schema.keys())
            if _any_field_contains(nested_keys, {"passage", "document", "text"}):
                has_passages = True
            if _any_field_contains(nested_keys, {"selected", "relevant", "label", "score"}):
                has_relevance = True

        return DatasetInspectionResult(
            dataset_name=self.dataset_name,
            configs_available=configs_available,
            splits_discovered=splits_discovered,
            schema=schema,
            nested_schemas=nested_schemas,
            has_query_fields=has_query,
            has_passage_fields=has_passages,
            has_ids=has_ids,
            has_relevance_ground_truth=has_relevance,
            has_language_info=has_lang,
            has_metadata=has_meta,
            sample_records=sample_records,
            raw_features_repr=raw_features_repr,
            inspection_warnings=warnings,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _hf_load(dataset_name: str, **kwargs):
    from datasets import load_dataset as hf_load_dataset
    return hf_load_dataset(dataset_name, **kwargs)


def _flatten_features(
    features,
    prefix: str = "",
) -> tuple[Dict[str, str], Dict[str, Dict[str, str]]]:
    """Recursively flatten HF Features into {field_name: type_str} dicts.

    Returns:
        (flat_schema, nested_schemas)
        - flat_schema: top-level {field_name: type_description}
        - nested_schemas: {field_name: {sub_field_name: type_description}} for
          Sequence/Struct fields
    """
    flat: Dict[str, str] = {}
    nested: Dict[str, Dict[str, str]] = {}

    try:
        from datasets import Sequence
        from datasets.features.features import ClassLabel, Value
    except ImportError:
        pass

    for name, feat in features.items():
        full_name = f"{prefix}{name}" if prefix else name
        type_str = _feature_type_str(feat)
        flat[full_name] = type_str

        # Recurse into Sequence/Struct-like features
        try:
            if hasattr(feat, "feature") and hasattr(feat.feature, "items"):
                sub_flat, _ = _flatten_features(feat.feature, prefix="")
                nested[full_name] = sub_flat
            elif hasattr(feat, "items"):
                sub_flat, _ = _flatten_features(feat, prefix="")
                nested[full_name] = sub_flat
        except Exception:
            pass

    return flat, nested


def _feature_type_str(feat) -> str:
    """Return a human-readable type string for a HF feature."""
    try:
        from datasets import Sequence
        from datasets.features.features import ClassLabel, Value
        if isinstance(feat, Value):
            return feat.dtype
        if isinstance(feat, ClassLabel):
            return f"ClassLabel(names={feat.names})"
        if isinstance(feat, Sequence):
            inner = _feature_type_str(feat.feature)
            return f"Sequence[{inner}]"
    except ImportError:
        pass
    # Fallback
    t = type(feat).__name__
    try:
        if hasattr(feat, "dtype"):
            return f"{t}(dtype={feat.dtype})"
    except Exception:
        pass
    return t


def _infer_schema_from_record(record: Dict[str, Any]) -> tuple[Dict[str, str], Dict[str, Dict[str, str]]]:
    """Infer schema from a single record when builder features are unavailable."""
    flat: Dict[str, str] = {}
    nested: Dict[str, Dict[str, str]] = {}

    for key, value in record.items():
        if isinstance(value, dict):
            flat[key] = f"dict (keys: {list(value.keys())})"
            nested[key] = {k: type(v).__name__ for k, v in value.items()}
        elif isinstance(value, list):
            inner_type = type(value[0]).__name__ if value else "unknown"
            flat[key] = f"list[{inner_type}]"
        else:
            flat[key] = type(value).__name__

    return flat, nested


def _any_field_contains(fields: set, keywords: set) -> bool:
    """Return True if any field name contains one of the keywords (case-insensitive)."""
    fields_lower = {f.lower() for f in fields}
    return any(
        kw in f
        for f in fields_lower
        for kw in keywords
    )
