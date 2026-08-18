"""Chunking benchmark and strategy selection module for Phase 3.3.

Benchmarks candidate chunking strategies on actual processed MSMARCO-XI data
using retrieval quality (Recall@K, MRR), size distributions, processing latency,
and storage cost to select the optimal production strategy.
"""

import json
import logging
import math
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from ingestion.chunking.base import BaseChunker, validate_chunk
from ingestion.chunking.models import Chunk, ChunkingConfig
from ingestion.chunking.processor import BatchChunkProcessor
from ingestion.chunking.registry import ChunkerRegistry
from ingestion.chunking.utils import count_characters, count_tokens
from ingestion.preprocessor import DEFAULT_PROCESSED_DIR, ProcessedDocument, ProcessedQuery

logger = logging.getLogger(__name__)

DEFAULT_EVAL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "evaluation", "chunking")
)
DEFAULT_FINAL_CHUNKS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "chunks")
)


def calc_metrics(values: List[Union[int, float]]) -> Dict[str, float]:
    """Calculate min, max, mean, median, P25, P75, P90, P95, P99 for list."""
    if not values:
        return {
            "count": 0, "min": 0, "max": 0, "mean": 0.0,
            "median": 0.0, "p25": 0.0, "p75": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0
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


class LightweightEvalIndex:
    """In-memory term index for evaluation retrieval without heavy DB dependencies."""

    def __init__(self):
        self.chunks: List[Chunk] = []
        self.index: Dict[str, List[int]] = defaultdict(list)
        self.idf: Dict[str, float] = {}

    def build(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.index.clear()
        self.idf.clear()
        doc_count = len(chunks)

        if doc_count == 0:
            return

        df: Dict[str, int] = defaultdict(int)

        for idx, chunk in enumerate(chunks):
            terms = set(self._tokenize(chunk.text))
            for term in terms:
                self.index[term].append(idx)
                df[term] += 1

        for term, count in df.items():
            self.idf[term] = math.log((doc_count + 1) / (count + 1)) + 1.0

    def _tokenize(self, text: str) -> List[str]:
        return [w.lower() for w in re_words(text)]

    def search(self, query_text: str, top_k: int = 10) -> List[Tuple[Chunk, float]]:
        q_terms = self._tokenize(query_text)
        if not q_terms or not self.chunks:
            return []

        scores: Dict[int, float] = defaultdict(float)

        for term in q_terms:
            if term in self.index:
                weight = self.idf.get(term, 1.0)
                for chunk_idx in self.index[term]:
                    scores[chunk_idx] += weight

        if not scores:
            return []

        sorted_idxs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self.chunks[idx], score) for idx, score in sorted_idxs]


def re_words(text: str) -> List[str]:
    import re
    return re.findall(r"\w+", text.lower(), re.UNICODE)


class ChunkingBenchmarkRunner:
    """Benchmark runner evaluating chunking strategies on retrieval & physical metrics."""

    def __init__(
        self,
        processed_dir: Optional[str] = None,
        eval_dir: Optional[str] = None,
        final_dir: Optional[str] = None,
    ):
        self.processed_dir = processed_dir or DEFAULT_PROCESSED_DIR
        self.eval_dir = eval_dir or DEFAULT_EVAL_DIR
        self.final_dir = final_dir or DEFAULT_FINAL_CHUNKS_DIR

        os.makedirs(self.eval_dir, exist_ok=True)
        os.makedirs(self.final_dir, exist_ok=True)

    def load_eval_data(
        self, max_queries: int = 500, max_documents: int = 2000
    ) -> Tuple[List[ProcessedQuery], List[ProcessedDocument]]:
        """Load processed queries and documents produced in Milestone 2."""
        queries_path = os.path.join(self.processed_dir, "queries.jsonl")
        docs_path = os.path.join(self.processed_dir, "documents.jsonl")

        if not os.path.exists(queries_path) or not os.path.exists(docs_path):
            raise FileNotFoundError(
                f"Processed dataset files not found at '{self.processed_dir}'. Run Milestone 2 preprocessing first."
            )

        queries: List[ProcessedQuery] = []
        with open(queries_path, "r", encoding="utf-8") as f:
            for line in f:
                if len(queries) >= max_queries:
                    break
                line = line.strip()
                if line:
                    d = json.loads(line)
                    q = ProcessedQuery(
                        query_id=d["query_id"],
                        query_text=d["query_text"],
                        eng_query_text=d["eng_query_text"],
                        answer_text=d.get("answer_text"),
                        eng_answer_text=d.get("eng_answer_text"),
                        query_type=d["query_type"],
                        source_lang=d["source_lang"],
                        target_lang=d["target_lang"],
                        relevant_document_ids=d.get("relevant_document_ids", []),
                        all_document_ids=d.get("all_document_ids", []),
                    )
                    queries.append(q)

        docs: List[ProcessedDocument] = []
        with open(docs_path, "r", encoding="utf-8") as f:
            for line in f:
                if len(docs) >= max_documents:
                    break
                line = line.strip()
                if line:
                    d = json.loads(line)
                    doc = ProcessedDocument(
                        document_id=d["document_id"],
                        text=d["text"],
                        english_text=d.get("english_text"),
                        source_query_id=d["source_query_id"],
                        passage_index=d["passage_index"],
                        is_selected=d["is_selected"],
                        language=d["language"],
                        metadata=d.get("metadata", {}),
                    )
                    docs.append(doc)

        logger.info(f"Loaded {len(queries)} eval queries and {len(docs)} eval documents.")
        return queries, docs

    def benchmark_strategy(
        self,
        strategy_name: str,
        config: ChunkingConfig,
        queries: List[ProcessedQuery],
        documents: List[ProcessedDocument],
    ) -> Dict[str, Any]:
        """Benchmark a single strategy and configuration combination."""
        chunker = ChunkerRegistry.get(strategy_name, config=config)
        processor = BatchChunkProcessor(chunker=chunker)

        t0 = time.time()
        chunks = processor.process_documents(documents)
        chunking_time_sec = round(time.time() - t0, 4)

        # 1. Chunk statistics
        token_lens = [c.token_count for c in chunks]
        char_lens = [c.character_count for c in chunks]
        unique_texts = set(c.text for c in chunks)
        duplicate_rate = round(1.0 - (len(unique_texts) / max(1, len(chunks))), 4)

        # 2. Build temporary evaluation index
        t1 = time.time()
        eval_index = LightweightEvalIndex()
        eval_index.build(chunks)
        index_build_time_sec = round(time.time() - t1, 4)

        # 3. Evaluate retrieval quality metrics
        rec1, rec3, rec5, rec10 = 0, 0, 0, 0
        mrr_sum = 0.0
        eval_query_count = 0
        failed_queries: List[Dict[str, Any]] = []

        for q in queries:
            rel_doc_ids = set(q.relevant_document_ids)
            if not rel_doc_ids:
                continue

            eval_query_count += 1
            search_results = eval_index.search(q.query_text, top_k=10)
            retrieved_chunk_doc_ids = [c[0].document_id for c in search_results]

            # Check recall hits
            hit1 = any(doc_id in rel_doc_ids for doc_id in retrieved_chunk_doc_ids[:1])
            hit3 = any(doc_id in rel_doc_ids for doc_id in retrieved_chunk_doc_ids[:3])
            hit5 = any(doc_id in rel_doc_ids for doc_id in retrieved_chunk_doc_ids[:5])
            hit10 = any(doc_id in rel_doc_ids for doc_id in retrieved_chunk_doc_ids[:10])

            if hit1:
                rec1 += 1
            if hit3:
                rec3 += 1
            if hit5:
                rec5 += 1
            if hit10:
                rec10 += 1

            # MRR
            rank = 0
            for idx, doc_id in enumerate(retrieved_chunk_doc_ids, start=1):
                if doc_id in rel_doc_ids:
                    rank = idx
                    break

            if rank > 0:
                mrr_sum += 1.0 / rank
            else:
                if len(failed_queries) < 10:
                    failed_queries.append({
                        "query_id": q.query_id,
                        "query_text": q.query_text,
                        "expected_doc_ids": list(rel_doc_ids),
                        "retrieved_doc_ids": retrieved_chunk_doc_ids[:5],
                    })

        recall_1 = round(rec1 / max(1, eval_query_count), 4)
        recall_3 = round(rec3 / max(1, eval_query_count), 4)
        recall_5 = round(rec5 / max(1, eval_query_count), 4)
        recall_10 = round(rec10 / max(1, eval_query_count), 4)
        mrr = round(mrr_sum / max(1, eval_query_count), 4)

        result = {
            "strategy": strategy_name,
            "config": asdict(config),
            "total_chunks": len(chunks),
            "chunking_time_sec": chunking_time_sec,
            "index_build_time_sec": index_build_time_sec,
            "duplicate_rate": duplicate_rate,
            "token_stats": calc_metrics(token_lens),
            "character_stats": calc_metrics(char_lens),
            "eval_query_count": eval_query_count,
            "retrieval_metrics": {
                "recall_1": recall_1,
                "recall_3": recall_3,
                "recall_5": recall_5,
                "recall_10": recall_10,
                "mrr": mrr,
            },
            "failed_query_samples": failed_queries,
        }
        return result

    def run_benchmark_matrix(
        self, max_queries: int = 500, max_documents: int = 2000
    ) -> Dict[str, Any]:
        """Run complete benchmark matrix across strategies and size configs."""
        queries, docs = self.load_eval_data(max_queries=max_queries, max_documents=max_documents)

        experiments = [
            ("passthrough", ChunkingConfig(strategy="passthrough")),
            ("fixed", ChunkingConfig(strategy="fixed", target_chunk_size=128, overlap=16)),
            ("fixed", ChunkingConfig(strategy="fixed", target_chunk_size=256, overlap=32)),
            ("sentence", ChunkingConfig(strategy="sentence", target_chunk_size=256)),
            ("structure", ChunkingConfig(strategy="structure", target_chunk_size=256)),
            ("semantic", ChunkingConfig(strategy="semantic", target_chunk_size=256, semantic_threshold=0.5)),
            ("semantic", ChunkingConfig(strategy="semantic", target_chunk_size=256, semantic_threshold=0.75)),
            ("hybrid", ChunkingConfig(strategy="hybrid", target_chunk_size=256, overlap=32)),
        ]

        results: List[Dict[str, Any]] = []
        for strat_name, cfg in experiments:
            logger.info(f"Benchmarking strategy '{strat_name}' [target_size={cfg.target_chunk_size}]...")
            res = self.benchmark_strategy(strat_name, cfg, queries, docs)
            results.append(res)

        # Select winner based on composite score: Recall@5 (50%), MRR (30%), Efficiency/P50 size (20%)
        best_exp = None
        best_score = -1.0

        for r in results:
            rec5 = r["retrieval_metrics"]["recall_5"]
            mrr = r["retrieval_metrics"]["mrr"]
            # Composite score favoring high Recall@5 & MRR with moderate chunk count
            score = (rec5 * 0.6) + (mrr * 0.4)
            r["composite_score"] = round(score, 4)
            if score > best_score:
                best_score = score
                best_exp = r

        summary = {
            "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "evaluated_queries": len(queries),
            "evaluated_documents": len(docs),
            "winning_strategy": best_exp["strategy"] if best_exp else "hybrid",
            "winning_config": best_exp["config"] if best_exp else {},
            "winning_composite_score": best_score,
            "results_matrix": results,
        }

        # Save artifacts to evaluation/chunking/
        results_file = os.path.join(self.eval_dir, "results.json")
        summary_file = os.path.join(self.eval_dir, "summary.json")
        error_file = os.path.join(self.eval_dir, "error_analysis.json")

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        error_samples = [
            {"strategy": r["strategy"], "failed_samples": r["failed_query_samples"]}
            for r in results
        ]
        with open(error_file, "w", encoding="utf-8") as f:
            json.dump(error_samples, f, indent=2, ensure_ascii=False)

        logger.info(f"Benchmark complete. Winner: '{summary['winning_strategy']}' (score: {best_score})")
        return summary

    def generate_final_production_chunks(
        self, winning_strategy: str, winning_config: ChunkingConfig
    ) -> Dict[str, Any]:
        """Generate final production chunk dataset using the winning strategy."""
        input_docs_jsonl = os.path.join(self.processed_dir, "documents.jsonl")
        output_chunks_jsonl = os.path.join(self.final_dir, "final_chunks.jsonl")
        output_manifest_json = os.path.join(self.final_dir, "final_manifest.json")

        chunker = ChunkerRegistry.get(winning_strategy, config=winning_config)
        processor = BatchChunkProcessor(chunker=chunker)

        manifest = processor.process_jsonl_file(
            input_documents_jsonl=input_docs_jsonl,
            output_chunks_jsonl=output_chunks_jsonl,
            output_manifest_json=output_manifest_json,
        )

        logger.info(
            f"Final production chunks generated using '{winning_strategy}' at '{output_chunks_jsonl}'"
        )
        return manifest
