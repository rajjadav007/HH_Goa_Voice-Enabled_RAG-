"""CLI entry-point and convenience wrappers for loading AI4Bharat/MSMARCO-XI.

Usage (from project root):
    python -m ingestion.load_dataset
    python -m ingestion.load_dataset --config hi --samples 5
"""

import argparse
import json
import logging
import sys
from typing import Any, Dict, List, Optional, Union

from ingestion.dataset_loader import (
    DEFAULT_RAW_DIR,
    DatasetDownloadError,
    DatasetInspectionResult,
    DatasetLoadError,
    InvalidDatasetStructureError,
    MSMARCODatasetLoader,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public convenience functions (used by downstream phases)
# ---------------------------------------------------------------------------


def load_msmarco_dataset(
    config: Optional[str] = None,
    split: Optional[str] = "validation",
    streaming: bool = True,
    raw_data_dir: Optional[str] = None,
):
    """Load MSMARCO-XI and return Dataset / DatasetDict / IterableDataset.

    Args:
        config:       Dataset configuration name.  None = dataset default.
        split:        Split name ('train', 'validation').  None = all splits.
        streaming:    When True, avoids full download (default: True).
        raw_data_dir: Override for local cache directory.

    Returns:
        Hugging Face Dataset / DatasetDict / IterableDataset object.
    """
    loader = MSMARCODatasetLoader(raw_data_dir=raw_data_dir)
    return loader.load_dataset(config=config, split=split, streaming=streaming)


def inspect_msmarco_dataset(
    config: Optional[str] = None,
    raw_data_dir: Optional[str] = None,
    num_samples: int = 3,
) -> DatasetInspectionResult:
    """Inspect MSMARCO-XI schema, splits, counts, and sample records.

    All returned information is discovered at runtime — nothing is assumed.

    Args:
        config:       Dataset configuration name.  None = dataset default.
        raw_data_dir: Override for local cache directory.
        num_samples:  Number of sample records to retrieve.

    Returns:
        DatasetInspectionResult populated from live dataset inspection.
    """
    loader = MSMARCODatasetLoader(raw_data_dir=raw_data_dir)
    return loader.inspect(config=config, num_samples=num_samples)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_inspection(result: DatasetInspectionResult) -> None:
    """Pretty-print inspection result to stdout."""
    sep = "=" * 60

    print(f"\n{sep}")
    print("  AI4Bharat MSMARCO-XI — Dataset Inspection Report")
    print(f"{sep}\n")

    print(f"Dataset        : {result.dataset_name}")
    print(f"Configs found  : {result.configs_available}")

    print("\nSplits & Record Counts:")
    for split, count in result.splits_discovered.items():
        count_str = str(count) if count >= 0 else "unknown (streaming only)"
        print(f"  {split:<15}: {count_str}")

    print("\nTop-level Schema:")
    for field_name, type_str in result.schema.items():
        print(f"  {field_name:<30}: {type_str}")

    if result.nested_schemas:
        print("\nNested Schemas:")
        for parent, sub_schema in result.nested_schemas.items():
            print(f"  [{parent}]")
            for sub_field, sub_type in sub_schema.items():
                print(f"    {sub_field:<28}: {sub_type}")

    print("\nCapabilities:")
    print(f"  Query/question fields    : {result.has_query_fields}")
    print(f"  Passage/document fields  : {result.has_passage_fields}")
    print(f"  ID fields                : {result.has_ids}")
    print(f"  Relevance / ground truth : {result.has_relevance_ground_truth}")
    print(f"  Language info            : {result.has_language_info}")
    print(f"  Metadata fields          : {result.has_metadata}")

    if result.sample_records:
        print(f"\nSample Records (showing {len(result.sample_records)}):")
        for i, rec in enumerate(result.sample_records):
            print(f"\n  --- Record {i + 1} ---")
            for k, v in rec.items():
                v_repr = repr(v)
                if len(v_repr) > 120:
                    v_repr = v_repr[:117] + "..."
                print(f"    {k:<30}: {v_repr}")
    else:
        print("\nSample Records: none retrieved.")

    if result.inspection_warnings:
        print("\nWarnings:")
        for w in result.inspection_warnings:
            print(f"  [WARN] {w}")

    print(f"\n{sep}")
    print("  Status: SUCCESS — inspection complete.")
    print(f"{sep}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load and inspect the AI4Bharat/MSMARCO-XI dataset."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Dataset config name (e.g. 'hi', 'bn', 'default'). Default: dataset's own default.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of sample records to retrieve (default: 3).",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=None,
        help="Directory for raw dataset cache (default: data/raw).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output inspection result as JSON.",
    )
    args = parser.parse_args()

    loader = MSMARCODatasetLoader(raw_data_dir=args.raw_dir)

    try:
        result = loader.inspect(config=args.config, num_samples=args.samples)
    except (DatasetDownloadError, DatasetLoadError, InvalidDatasetStructureError) as exc:
        print(f"\nFAILED: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"\nUnexpected error: {exc}", file=sys.stderr)
        logger.exception("Unexpected error during inspection.")
        sys.exit(2)

    if args.json:
        import dataclasses
        print(json.dumps(dataclasses.asdict(result), indent=2, default=str))
    else:
        _print_inspection(result)


if __name__ == "__main__":
    main()
