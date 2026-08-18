"""CLI Benchmark Runner for Phase 6.3 Grounding Validation.

Usage:
    python -m guardrails.grounding.run_benchmark
"""

import logging
import time
from typing import Any, Dict, List

import numpy as np

from guardrails.grounding.service import GroundingValidationService
from retrieval.reranking.models import RerankedResultPoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def generate_benchmark_test_cases() -> List[Dict[str, Any]]:
    """Generate 30 test scenarios covering fully grounded, partially grounded, ungrounded hallucinations, and refusals."""
    c1 = RerankedResultPoint(chunk_id="chk_1", document_id="doc_1", rerank_score=0.9, final_rank=1, text="The capital of France is Paris. Paris has a population of over 2 million people.", sources=["vector"])
    c2 = RerankedResultPoint(chunk_id="chk_2", document_id="doc_2", rerank_score=0.8, final_rank=2, text="A corporation is a legal entity that is separate and distinct from its owners.", sources=["bm25"])

    cases = []
    # 15 Fully Grounded Cases
    for i in range(15):
        cases.append({
            "query": "What is the capital of France?",
            "answer": "The capital of France is Paris.",
            "grounded_flag": True,
            "chunks": [c1],
            "expected_status": "FULLY_GROUNDED",
        })

    # 5 Refusal Grounded Cases
    for i in range(5):
        cases.append({
            "query": "What is quantum gravity?",
            "answer": "Insufficient context available to answer the query.",
            "grounded_flag": False,
            "chunks": [],
            "expected_status": "REFUSAL_GROUNDED",
        })

    # 5 Ungrounded Hallucination Cases (Fake facts not in context)
    for i in range(5):
        cases.append({
            "query": "What is the capital of France?",
            "answer": "The capital of France is Berlin, located in South America.",
            "grounded_flag": True,
            "chunks": [c1],
            "expected_status": "UNGROUNDED",
        })

    # 5 Partially Grounded Cases
    for i in range(5):
        cases.append({
            "query": "What is a corporation?",
            "answer": "A corporation is a legal entity distinct from owners. It was invented in 1400 AD by sailors.",
            "grounded_flag": True,
            "chunks": [c2],
            "expected_status": "PARTIALLY_GROUNDED",
        })

    return cases


def run_benchmark():
    service = GroundingValidationService()
    test_cases = generate_benchmark_test_cases()

    latencies: List[float] = []
    status_counts: Dict[str, int] = {}
    hallucination_caught = 0
    total_hallucinations = 5

    print("\n============================================================")
    print("  HH Goa 2026 — Phase 6.3 Grounding Validation Benchmark")
    print("============================================================\n")

    for idx, item in enumerate(test_cases, 1):
        q = item["query"]
        ans = item["answer"]
        flag = item["grounded_flag"]
        chunks = item["chunks"]

        decision = service.evaluate(q, ans, flag, chunks)
        latencies.append(decision.latency_ms)

        st = decision.status.value
        status_counts[st] = status_counts.get(st, 0) + 1

        if item["expected_status"] == "UNGROUNDED" and decision.status.value == "UNGROUNDED":
            hallucination_caught += 1

    lat_arr = np.array(latencies)
    p50 = float(np.percentile(lat_arr, 50))
    p70 = float(np.percentile(lat_arr, 70))
    p90 = float(np.percentile(lat_arr, 90))
    p95 = float(np.percentile(lat_arr, 95))
    p99 = float(np.percentile(lat_arr, 99))
    p100 = float(np.max(lat_arr))

    print("============================================================")
    print("GROUNDING VALIDATION BENCHMARK SUMMARY")
    print("============================================================")
    print(f"Total Evaluation Scenarios: {len(test_cases)}")
    for st_name, count in status_counts.items():
        print(f"Status '{st_name}': {count} scenarios")
    print(f"Hallucination Catch Rate : {hallucination_caught}/{total_hallucinations} ({hallucination_caught/total_hallucinations*100:.1f}%)\n")

    print("============================================================")
    print("GROUNDING VALIDATION LATENCY DISTRIBUTION (ms)")
    print("============================================================")
    print(f"P50  (Median)            : {p50:.3f} ms")
    print(f"P70                      : {p70:.3f} ms")
    print(f"P90                      : {p90:.3f} ms")
    print(f"P95                      : {p95:.3f} ms")
    print(f"P99                      : {p99:.3f} ms")
    print(f"P100 (Max)               : {p100:.3f} ms\n")

    print("Grounding Validation benchmark completed successfully!\n")


if __name__ == "__main__":
    run_benchmark()
