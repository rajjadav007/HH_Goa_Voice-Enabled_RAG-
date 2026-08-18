"""Latency distribution percentile metrics and target verification."""

from typing import List
import numpy as np
from evaluation.datasets.schema import LatencyPercentiles


def compute_latency_percentiles(latencies_ms: List[float]) -> LatencyPercentiles:
    """Compute P50, P70, P90, P95, P99, P100 latency percentiles and verify < 200ms target."""
    if not latencies_ms:
        return LatencyPercentiles()

    arr = np.array(latencies_ms, dtype=float)
    p50 = float(np.percentile(arr, 50))
    p70 = float(np.percentile(arr, 70))
    p90 = float(np.percentile(arr, 90))
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))
    p100 = float(np.max(arr))

    # Target is passed if P50 (median) online response latency is under 200 ms
    target_passed = bool(p50 < 200.0)

    return LatencyPercentiles(
        p50=round(p50, 3),
        p70=round(p70, 3),
        p90=round(p90, 3),
        p95=round(p95, 3),
        p99=round(p99, 3),
        p100=round(p100, 3),
        target_200ms_passed=target_passed,
    )
