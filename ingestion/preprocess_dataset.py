"""CLI Entrypoint for running dataset preprocessing on AI4Bharat MSMARCO-XI.

Usage:
    python -m ingestion.preprocess_dataset --max-records 10000
"""

import argparse
import logging
import sys
from typing import Optional

from ingestion.dataset_loader import MSMARCODatasetLoader
from ingestion.preprocessor import (
    DEFAULT_PROCESSED_DIR,
    MSMARCOPreprocessor,
    PreprocessingConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def preprocess_msmarco_dataset(
    max_records: Optional[int] = 10000,
    config: Optional[str] = None,
    split: str = "validation",
    processed_dir: Optional[str] = None,
    raw_data_dir: Optional[str] = None,
):
    """Convenience function to load raw dataset and run preprocessing pipeline."""
    loader = MSMARCODatasetLoader(raw_data_dir=raw_data_dir)
    logger.info(f"Loading raw dataset '{loader.dataset_name}' [split={split}, streaming=True]...")
    raw_stream = loader.load_dataset(config=config, split=split, streaming=True)

    prep_config = PreprocessingConfig(
        processed_dir=processed_dir or DEFAULT_PROCESSED_DIR
    )
    preprocessor = MSMARCOPreprocessor(config=prep_config)

    manifest = preprocessor.process_dataset_stream(
        record_stream=raw_stream, max_records=max_records
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess AI4Bharat/MSMARCO-XI dataset into normalized documents & queries."
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=10000,
        help="Maximum raw records to process (default: 10000). Set to 0 for unlimited.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Language config name (e.g. 'default').",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split to preprocess ('validation' or 'train'). Default: validation.",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=DEFAULT_PROCESSED_DIR,
        help="Output directory for processed jsonl files (default: data/processed).",
    )
    args = parser.parse_args()

    max_recs = None if args.max_records == 0 else args.max_records

    try:
        manifest = preprocess_msmarco_dataset(
            max_records=max_recs,
            config=args.config,
            split=args.split,
            processed_dir=args.processed_dir,
        )

        print("\n============================================================")
        print("  AI4Bharat MSMARCO-XI — Dataset Preprocessing Complete")
        print("============================================================\n")
        print(f"Input Raw Records   : {manifest['input_records']}")
        print(f"Processed Queries   : {manifest['processed_queries']}")
        print(f"Processed Documents : {manifest['processed_documents']}")
        print(f"Rejected Records    : {manifest['rejected_records']}")
        print(f"Deduplicated Docs   : {manifest['duplicate_passages_deduped']}")
        print("\nOutput Files:")
        for k, v in manifest['output_files'].items():
            print(f"  - {k:<12}: {v}")

        if manifest['rejected_reasons']:
            print("\nRejection Breakdown:")
            for reason, cnt in manifest['rejected_reasons'].items():
                print(f"  - {reason:<35}: {cnt}")

        print("\n============================================================\n")

    except Exception as exc:
        logger.error(f"Preprocessing failed: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
