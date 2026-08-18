"""CLI Entrypoint for running dataset analysis on AI4Bharat MSMARCO-XI.

Usage:
    python -m ingestion.analyze_dataset --sample-size 5000
"""

import argparse
import logging
import sys
from typing import Optional

from ingestion.dataset_analyzer import MSMARCODatasetAnalyzer, DEFAULT_OUTPUT_DIR
from ingestion.dataset_loader import MSMARCODatasetLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def analyze_msmarco_dataset(
    sample_size: int = 5000,
    config: Optional[str] = None,
    split: str = "validation",
    output_dir: Optional[str] = None,
    raw_data_dir: Optional[str] = None,
):
    """Convenience function to perform dataset analysis."""
    loader = MSMARCODatasetLoader(raw_data_dir=raw_data_dir)
    analyzer = MSMARCODatasetAnalyzer(loader=loader, output_dir=output_dir)
    return analyzer.analyze(sample_size=sample_size, config=config, split=split)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze structure and statistics of AI4Bharat/MSMARCO-XI dataset."
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5000,
        help="Number of records to sample for statistical analysis (default: 5000).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Dataset config name (e.g. 'default').",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split to sample from ('validation' or 'train'). Default: validation.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for json reports (default: docs/dataset).",
    )
    args = parser.parse_args()

    try:
        results = analyze_msmarco_dataset(
            sample_size=args.sample_size,
            config=args.config,
            split=args.split,
            output_dir=args.output_dir,
        )
        print("\n============================================================")
        print("  AI4Bharat MSMARCO-XI — Dataset Analysis Complete")
        print("============================================================\n")
        print(f"Sample Size Analyzed : {results['statistics']['sample_size']}")
        print(f"Split Analyzed       : {results['statistics']['analyzed_split']}")
        print(f"Output Directory     : {args.output_dir}")
        print("\nGenerated Reports:")
        for name, path in results["file_paths"].items():
            print(f"  - {name:<12}: {path}")

        print("\nKey Text Length Metrics (Eng_Query Word Count):")
        eq_w = results['statistics']['text_distributions']['eng_query_word']
        print(f"  Min={eq_w['min']}, Mean={eq_w['mean']}, Median={eq_w['median']}, P95={eq_w['p95']}, Max={eq_w['max']}")

        print("\nKey Text Length Metrics (Eng_Passage Word Count):")
        ep_w = results['statistics']['text_distributions']['eng_passage_word']
        print(f"  Min={ep_w['min']}, Mean={ep_w['mean']}, Median={ep_w['median']}, P95={ep_w['p95']}, Max={ep_w['max']}")

        print("\nPassages & Ground Truth:")
        pass_rec = results['statistics']['passages_per_record']
        sel_rec = results['statistics']['ground_truth_selected_passages']
        print(f"  Passages/record: Mean={pass_rec['mean']}, Min={pass_rec['min']}, Max={pass_rec['max']}")
        print(f"  Selected/record: Mean={sel_rec['mean']}, Zero-Selected Count={results['statistics']['zero_selected_passages_count']}")

        print("\n============================================================\n")

    except Exception as exc:
        logger.error(f"Analysis failed: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
