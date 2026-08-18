"""Benchmark module comparing Vector-only, BM25-only, and Hybrid RRF retrieval performance."""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from retrieval.bm25.service import BM25Service
from retrieval.embeddings.service import EmbeddingService
from retrieval.hybrid.models import HybridConfig
from retrieval.hybrid.service import HybridService
from retrieval.vector_db.service import QdrantService

logger = logging.getLogger(__name__)

DEFAULT_EVAL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed", "queries.jsonl")
)


def calculate_metrics(retrieved_doc_ids: List[str], target_doc_ids: List[str]) -> Dict[str, float]:
    """Calculate Recall@K and Reciprocal Rank for a single query given target document IDs."""
    if not target_doc_ids:
        return {"recall_1": 0.0, "recall_3": 0.0, "recall_5": 0.0, "recall_10": 0.0, "mrr": 0.0}

    target_set = set(target_doc_ids)
    rr = 0.0
    recalls = {1: 0.0, 3: 0.0, 5: 0.0, 10: 0.0}

    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in target_set:
            if rr == 0.0:
                rr = 1.0 / rank
            for k in recalls:
                if rank <= k:
                    recalls[k] = 1.0

    return {
        "recall_1": recalls[1],
        "recall_3": recalls[3],
        "recall_5": recalls[5],
        "recall_10": recalls[10],
        "mrr": rr,
    }


class HybridBenchmarkRunner:
    """Benchmark runner for evaluating and comparing retrieval strategies."""

    def __init__(self, hybrid_service: Optional[HybridService] = None):
        self.hybrid_service = hybrid_service or HybridService()

    def load_eval_queries(self, eval_path: str, max_queries: int = 100) -> List[Dict[str, Any]]:
        """Load preprocessed evaluation queries and ground truth document IDs."""
        queries = []
        if not os.path.exists(eval_path):
            logger.warning(f"Eval path '{eval_path}' not found.")
            return queries

        with open(eval_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    # Require queries with relevant_document_ids for ground truth validation
                    rel_ids = item.get("relevant_document_ids", [])
                    if not rel_ids and "document_id" in item:
                        rel_ids = [item["document_id"]]
                    if rel_ids:
                        item["_target_doc_ids"] = rel_ids
                        queries.append(item)
                    if max_queries and len(queries) >= max_queries:
                        break

        logger.info(f"Loaded {len(queries)} evaluation queries with ground truth.")
        return queries

    def run_benchmark(
        self,
        eval_path: Optional[str] = None,
        max_queries: int = 50,
        rrf_k_values: List[int] = [20, 40, 60, 80],
    ) -> Dict[str, Any]:
        """Run comprehensive benchmark matrix across Vector-only, BM25-only, and Hybrid RRF."""
        e_path = eval_path or DEFAULT_EVAL_PATH
        queries = self.load_eval_queries(e_path, max_queries=max_queries)

        if not queries:
            raise ValueError(f"No evaluation queries found at '{e_path}'.")

        logger.info(f"Starting retrieval benchmark across {len(queries)} queries...")

        strategies: Dict[str, List[Dict[str, float]]] = {
            "Vector-Only": [],
            "BM25-Only": [],
        }
        for k in rrf_k_values:
            strategies[f"Hybrid-RRF (k={k})"] = []

        seq_latencies: List[float] = []
        par_latencies: List[float] = []

        for idx, item in enumerate(queries, start=1):
            q_text = item.get("query_text", "")
            target_doc_ids = item.get("_target_doc_ids", [])
            if not q_text or not target_doc_ids:
                continue

            # 1. Vector-only
            v_res, _ = self.hybrid_service._execute_vector_search(q_text, top_k=10)
            v_docs = [r.document_id for r in v_res]
            strategies["Vector-Only"].append(calculate_metrics(v_docs, target_doc_ids))

            # 2. BM25-only
            b_res, _ = self.hybrid_service._execute_bm25_search(q_text, top_k=10)
            b_docs = [r.document_id for r in b_res]
            strategies["BM25-Only"].append(calculate_metrics(b_docs, target_doc_ids))

            # 3. Hybrid RRF for various K values
            for k in rrf_k_values:
                h_res, _ = self.hybrid_service.search(
                    q_text, vector_top_k=10, bm25_top_k=10, rrf_k=k, final_top_k=10, parallel=True
                )
                h_docs = [r.document_id for r in h_res]
                strategies[f"Hybrid-RRF (k={k})"].append(calculate_metrics(h_docs, target_doc_ids))

            # Measure sequential vs parallel latency sample
            t0 = time.time()
            self.hybrid_service.search(q_text, parallel=False)
            seq_latencies.append((time.time() - t0) * 1000)

            t1 = time.time()
            self.hybrid_service.search(q_text, parallel=True)
            par_latencies.append((time.time() - t1) * 1000)

        # Aggregate metrics
        results_summary: Dict[str, Dict[str, float]] = {}
        for name, metric_list in strategies.items():
            if not metric_list:
                continue
            r1 = float(np.mean([m["recall_1"] for m in metric_list]))
            r3 = float(np.mean([m["recall_3"] for m in metric_list]))
            r5 = float(np.mean([m["recall_5"] for m in metric_list]))
            r10 = float(np.mean([m["recall_10"] for m in metric_list]))
            mrr = float(np.mean([m["mrr"] for m in metric_list]))

            results_summary[name] = {
                "recall_1": round(r1, 4),
                "recall_3": round(r3, 4),
                "recall_5": round(r5, 4),
                "recall_10": round(r10, 4),
                "mrr": round(mrr, 4),
            }

        avg_seq_ms = round(float(np.mean(seq_latencies)), 2)
        avg_par_ms = round(float(np.mean(par_latencies)), 2)

        return {
            "eval_query_count": len(queries),
            "strategy_metrics": results_summary,
            "latency": {
                "sequential_mean_ms": avg_seq_ms,
                "parallel_mean_ms": avg_par_ms,
                "speedup": round(avg_seq_ms / max(0.01, avg_par_ms), 2),
            },
        }
