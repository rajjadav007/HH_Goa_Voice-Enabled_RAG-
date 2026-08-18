"""Dataset Analyzer Module for AI4Bharat MSMARCO-XI.

Performs reproducible statistical and structural analysis on the MSMARCO-XI
dataset loaded via MSMARCODatasetLoader.
"""

import json
import logging
import math
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from ingestion.dataset_loader import (
    DatasetInspectionResult,
    MSMARCODatasetLoader,
)

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "dataset")
)


def calc_percentiles(values: List[Union[int, float]]) -> Dict[str, float]:
    """Calculate min, max, mean, median, P25, P75, P90, P95, P99 for numeric values."""
    if not values:
        return {
            "count": 0,
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "median": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }
    arr = np.array(values, dtype=np.float64)
    return {
        "count": len(values),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(round(np.mean(arr), 2)),
        "median": float(round(np.median(arr), 2)),
        "p25": float(round(np.percentile(arr, 25), 2)),
        "p75": float(round(np.percentile(arr, 75), 2)),
        "p90": float(round(np.percentile(arr, 90), 2)),
        "p95": float(round(np.percentile(arr, 95), 2)),
        "p99": float(round(np.percentile(arr, 99), 2)),
    }


class MSMARCODatasetAnalyzer:
    """Analyzer for AI4Bharat MSMARCO-XI dataset."""

    def __init__(
        self,
        loader: Optional[MSMARCODatasetLoader] = None,
        output_dir: Optional[str] = None,
    ):
        self.loader = loader or MSMARCODatasetLoader()
        self.output_dir = output_dir or DEFAULT_OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)

    def analyze(
        self,
        sample_size: int = 5000,
        config: Optional[str] = None,
        split: str = "validation",
    ) -> Dict[str, Any]:
        """Perform analysis on dataset and output schema, statistics, and samples."""
        logger.info("Starting dataset inspection and schema analysis...")
        inspection = self.loader.inspect(config=config, num_samples=0)

        logger.info(f"Streaming {sample_size} records from split '{split}' for statistical analysis...")
        ds_stream = self.loader.load_dataset(config=config, split=split, streaming=True)

        records: List[Dict[str, Any]] = []
        for i, rec in enumerate(ds_stream):
            if i >= sample_size:
                break
            records.append(dict(rec))

        actual_sample_count = len(records)
        logger.info(f"Collected {actual_sample_count} records for analysis.")

        # --- 1. Schema ---
        schema_data = {
            "dataset_name": inspection.dataset_name,
            "configs_available": inspection.configs_available,
            "splits_discovered": inspection.splits_discovered,
            "top_level_schema": inspection.schema,
            "nested_schemas": inspection.nested_schemas,
            "capabilities": {
                "has_query_fields": inspection.has_query_fields,
                "has_passage_fields": inspection.has_passage_fields,
                "has_ids": inspection.has_ids,
                "has_relevance_ground_truth": inspection.has_relevance_ground_truth,
                "has_language_info": inspection.has_language_info,
                "has_metadata": inspection.has_metadata,
            },
        }

        # --- 2. Field Missing Values ---
        missing_counts: Dict[str, int] = {}
        missing_pct: Dict[str, float] = {}

        if actual_sample_count > 0:
            top_keys = set(records[0].keys())
            for key in top_keys:
                cnt = sum(
                    1 for r in records if r.get(key) is None or r.get(key) == "" or (isinstance(r.get(key), (list, dict)) and len(r.get(key)) == 0)
                )
                missing_counts[key] = cnt
                missing_pct[key] = round((cnt / actual_sample_count) * 100, 2)

            # Check passage sub-fields
            passage_subkeys = ["English_passages", "Translated_passages", "is_selected"]
            for subkey in passage_subkeys:
                full_key = f"passages.{subkey}"
                cnt = sum(
                    1
                    for r in records
                    if not isinstance(r.get("passages"), dict)
                    or r.get("passages", {}).get(subkey) is None
                    or len(r.get("passages", {}).get(subkey, [])) == 0
                )
                missing_counts[full_key] = cnt
                missing_pct[full_key] = round((cnt / actual_sample_count) * 100, 2)

        # --- 3. Text Length Statistics ---
        eng_query_char_len: List[int] = []
        eng_query_word_len: List[int] = []
        query_char_len: List[int] = []
        query_word_len: List[int] = []

        eng_ans_char_len: List[int] = []
        eng_ans_word_len: List[int] = []
        ans_char_len: List[int] = []
        ans_word_len: List[int] = []

        eng_passage_char_len: List[int] = []
        eng_passage_word_len: List[int] = []
        trans_passage_char_len: List[int] = []
        trans_passage_word_len: List[int] = []

        passages_per_record: List[int] = []
        selected_passages_per_record: List[int] = []

        # Duplicates & Categories
        query_ids: List[Any] = []
        eng_queries: List[str] = []
        source_langs: Dict[str, int] = {}
        target_langs: Dict[str, int] = {}
        query_types: Dict[str, int] = {}

        html_pattern = re.compile(r"<[^>]+>")
        url_pattern = re.compile(r"https?://\S+|www\.\S+")

        html_count = 0
        url_count = 0

        for r in records:
            # IDs
            qid = r.get("query_id")
            if qid is not None:
                query_ids.append(qid)

            # Lang & Meta
            sl = str(r.get("source_lang", "unknown"))
            tl = str(r.get("target_lang", "unknown"))
            qt = str(r.get("query_type", "unknown"))

            source_langs[sl] = source_langs.get(sl, 0) + 1
            target_langs[tl] = target_langs.get(tl, 0) + 1
            query_types[qt] = query_types.get(qt, 0) + 1

            # Eng Query
            eq = r.get("Eng_Query", "")
            if eq:
                eng_queries.append(eq)
                eng_query_char_len.append(len(eq))
                eng_query_word_len.append(len(eq.split()))

            # Query
            q = r.get("query", "")
            if q:
                query_char_len.append(len(q))
                query_word_len.append(len(q.split()))

            # Eng Answer
            ea = r.get("Eng_Answer", "")
            if ea:
                eng_ans_char_len.append(len(ea))
                eng_ans_word_len.append(len(ea.split()))

            # Answer
            a = r.get("Answer", "")
            if a:
                ans_char_len.append(len(a))
                ans_word_len.append(len(a.split()))

            # Passages
            passages = r.get("passages", {})
            if isinstance(passages, dict):
                eng_pass = passages.get("English_passages", [])
                trans_pass = passages.get("Translated_passages", [])
                is_sel = passages.get("is_selected", [])

                num_pass = len(eng_pass)
                passages_per_record.append(num_pass)

                num_sel = sum(1 for s in is_sel if s == 1)
                selected_passages_per_record.append(num_sel)

                for p in eng_pass:
                    if isinstance(p, str):
                        eng_passage_char_len.append(len(p))
                        eng_passage_word_len.append(len(p.split()))
                        if html_pattern.search(p):
                            html_count += 1
                        if url_pattern.search(p):
                            url_count += 1

                for p in trans_pass:
                    if isinstance(p, str):
                        trans_passage_char_len.append(len(p))
                        trans_passage_word_len.append(len(p.split()))

        # Duplicates
        unique_qids = len(set(query_ids))
        unique_eng_queries = len(set(eng_queries))

        duplicate_qid_count = len(query_ids) - unique_qids
        duplicate_query_count = len(eng_queries) - unique_eng_queries

        stats_data = {
            "sample_size": actual_sample_count,
            "analyzed_split": split,
            "missing_values": {
                "counts": missing_counts,
                "percentage": missing_pct,
            },
            "duplicates": {
                "total_queries_analyzed": len(eng_queries),
                "unique_query_ids": unique_qids,
                "duplicate_query_ids": duplicate_qid_count,
                "unique_eng_queries": unique_eng_queries,
                "duplicate_eng_queries": duplicate_query_count,
                "duplicate_query_ratio": round((duplicate_query_count / max(1, len(eng_queries))) * 100, 2),
            },
            "text_distributions": {
                "eng_query_char": calc_percentiles(eng_query_char_len),
                "eng_query_word": calc_percentiles(eng_query_word_len),
                "query_char": calc_percentiles(query_char_len),
                "query_word": calc_percentiles(query_word_len),
                "eng_answer_char": calc_percentiles(eng_ans_char_len),
                "eng_answer_word": calc_percentiles(eng_ans_word_len),
                "answer_char": calc_percentiles(ans_char_len),
                "answer_word": calc_percentiles(ans_word_len),
                "eng_passage_char": calc_percentiles(eng_passage_char_len),
                "eng_passage_word": calc_percentiles(eng_passage_word_len),
                "translated_passage_char": calc_percentiles(trans_passage_char_len),
                "translated_passage_word": calc_percentiles(trans_passage_word_len),
            },
            "passages_per_record": calc_percentiles(passages_per_record),
            "ground_truth_selected_passages": calc_percentiles(selected_passages_per_record),
            "zero_selected_passages_count": sum(1 for s in selected_passages_per_record if s == 0),
            "categorical_distributions": {
                "source_languages": source_langs,
                "target_languages": target_langs,
                "query_types": query_types,
            },
            "text_structure_flags": {
                "passages_with_html_count": html_count,
                "passages_with_url_count": url_count,
            },
        }

        # --- 4. Samples ---
        # Pick start, middle, end, random sample
        sample_indices = []
        if actual_sample_count > 0:
            sample_indices = [
                0,
                actual_sample_count // 4,
                actual_sample_count // 2,
                (3 * actual_sample_count) // 4,
                actual_sample_count - 1,
            ]
            sample_indices = sorted(list(set(i for i in sample_indices if 0 <= i < actual_sample_count)))

        samples_data = {
            "sample_count": len(sample_indices),
            "records": [records[idx] for idx in sample_indices],
        }

        # --- Save JSON outputs ---
        schema_path = os.path.join(self.output_dir, "schema.json")
        stats_path = os.path.join(self.output_dir, "statistics.json")
        samples_path = os.path.join(self.output_dir, "samples.json")

        with open(schema_path, "w", encoding="utf-8") as f:
            json.dump(schema_data, f, indent=2, ensure_ascii=False)

        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats_data, f, indent=2, ensure_ascii=False)

        with open(samples_path, "w", encoding="utf-8") as f:
            json.dump(samples_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Analysis saved to: {self.output_dir}")
        return {
            "schema": schema_data,
            "statistics": stats_data,
            "samples": samples_data,
            "file_paths": {
                "schema": schema_path,
                "statistics": stats_path,
                "samples": samples_path,
            },
        }
