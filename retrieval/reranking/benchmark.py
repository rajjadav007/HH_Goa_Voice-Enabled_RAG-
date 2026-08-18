"""Benchmark runner comparing Vector, BM25, Hybrid RRF, and Hybrid + Reranker."""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np

from retrieval.hybrid.benchmark import calculate_metrics
from retrieval.hybrid.service import HybridService
from retrieval.reranking.models import RerankerConfig
from retrieval.reranking.service import RerankerService

logger = logging.getLogger(__name__)

DEFAULT_EVAL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed", "queries.jsonl")
)


class RerankerBenchmarkRunner:
    """Benchmark runner evaluating Quality vs Latency across 4 retrieval configurations."""

    def __init__(
        self,
        hybrid_service: Optional[HybridService] = None,
        reranker_service: Optional[RerankerService] = None,
    ):
        self.hybrid_service = hybrid_service or HybridService()
        self.reranker_service = reranker_service or RerankerService()

    def load_eval_queries(self, eval_path: str, max_queries: int = 50) -> List[Dict[str, Any]]:
        """Load evaluation queries with ground truth document IDs."""
        queries = []
        if not os.path.exists(eval_path):
            logger.warning(f"Eval path '{eval_path}' not found.")
            return queries

        with open(eval_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    rel_ids = item.get("relevant_document_ids", [])
                    if not rel_ids and "document_id" in item:
                        rel_ids = [item["document_id"]]
                    if rel_ids:
                        item["_target_doc_ids"] = rel_ids
                        queries.append(item)
                    if max_queries and len(queries) >= max_queries:
                        break

        logger.info(f"Loaded {len(queries)} evaluation queries.")
        return queries

    def run_benchmark(
        self,
        eval_path: Optional[str] = None,
        max_queries: int = 30,
        candidate_k: int = 10,
        final_k: int = 5,
    ) -> Dict[str, Any]:
        """Run 4-way strategy comparison across Vector, BM25, Hybrid, and Hybrid + Reranker."""
        e_path = eval_path or DEFAULT_EVAL_PATH
        queries = self.load_eval_queries(e_path, max_queries=max_queries)

        if not queries:
            raise ValueError(f"No evaluation queries found at '{e_path}'.")

        logger.info(f"Starting 4-way retrieval benchmark across {len(queries)} queries...")

        # Ensure reranker model loaded for benchmark
        self.reranker_service.load_model()

        strategies: Dict[str, List[Dict[str, float]]] = {
            "Vector-Only": [],
            "BM25-Only": [],
            "Hybrid-RRF": [],
            "Hybrid + Reranker": [],
        }

        latencies: Dict[str, List[float]] = {
            "Vector-Only": [],
            "BM25-Only": [],
            "Hybrid-RRF": [],
            "Hybrid + Reranker": [],
            "Reranker-Only-Inference": [],
        }

        for idx, item in enumerate(queries, start=1):
            q_text = item.get("query_text", "")
            target_doc_ids = item.get("_target_doc_ids", [])
            if not q_text or not target_doc_ids:
                continue

            # 1. Vector-only
            t0 = time.time()
            v_res, _ = self.hybrid_service._execute_vector_search(q_text, top_k=10)
            latencies["Vector-Only"].append((time.time() - t0) * 1000)
            v_docs = [r.document_id for r in v_res]
            strategies["Vector-Only"].append(calculate_metrics(v_docs, target_doc_ids))

            # 2. BM25-only
            t1 = time.time()
            b_res, _ = self.hybrid_service._execute_bm25_search(q_text, top_k=10)
            latencies["BM25-Only"].append((time.time() - t1) * 1000)
            b_docs = [r.document_id for r in b_res]
            strategies["BM25-Only"].append(calculate_metrics(b_docs, target_doc_ids))

            # 3. Hybrid RRF
            t2 = time.time()
            h_res, _ = self.hybrid_service.search(
                q_text, vector_top_k=candidate_k, bm25_top_k=candidate_k, final_top_k=candidate_k, parallel=True
            )
            latencies["Hybrid-RRF"].append((time.time() - t2) * 1000)
            h_docs = [r.document_id for r in h_res[:final_k]]
            strategies["Hybrid-RRF"].append(calculate_metrics(h_docs, target_doc_ids))

            # 4. Hybrid + Reranker
            t3 = time.time()
            r_res, r_metrics = self.reranker_service.rerank(
                q_text, candidates=h_res, candidate_k=candidate_k, final_k=final_k
            )
            latencies["Hybrid + Reranker"].append((time.time() - t2) * 1000)
            latencies["Reranker-Only-Inference"].append(r_metrics.get("total_ms", 0.0))

            r_docs = [r.document_id for r in r_res]
            strategies["Hybrid + Reranker"].append(calculate_metrics(r_docs, target_doc_ids))

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
            mean_lat = float(np.mean(latencies[name]))

            results_summary[name] = {
                "recall_1": round(r1, 4),
                "recall_3": round(r3, 4),
                "recall_5": round(r5, 4),
                "recall_10": round(r10, 4),
                "mrr": round(mrr, 4),
                "mean_latency_ms": round(mean_lat, 2),
            }

        avg_rerank_ms = round(float(np.mean(latencies["Reranker-Only-Inference"])), 2)

        return {
            "eval_query_count": len(queries),
            "candidate_k": candidate_k,
            "final_k": final_k,
            "reranker_model": self.reranker_service.config.model_name,
            "device": self.reranker_service.device,
            "strategy_metrics": results_summary,
            "reranker_standalone_latency_ms": avg_rerank_ms,
        }
