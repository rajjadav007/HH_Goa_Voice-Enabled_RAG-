"""Pydantic schemas for Latency Analytics and Bottleneck Identification."""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from evaluation.datasets.schema import LatencyPercentiles, RunMetadata


class StageTimingStats(BaseModel):
    """Timing statistics and contribution percentage for a single pipeline stage."""
    stage_name: str
    avg_ms: float = 0.0
    p50_ms: float = 0.0
    p90_ms: float = 0.0
    p95_ms: float = 0.0
    max_ms: float = 0.0
    contribution_pct: float = 0.0


class TargetEvaluation(BaseModel):
    """Evaluation result against the <200ms challenge latency target."""
    target_name: str = "< 200 ms Target"
    target_threshold_ms: float = 200.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    p50_passed: bool = False
    p95_passed: bool = False
    status: str = "FAIL"


class LatencyAnalyticsReport(BaseModel):
    """Complete Latency Analytics and Bottleneck Report."""
    metadata: RunMetadata
    sample_count: int
    text_latency: LatencyPercentiles
    voice_latency: LatencyPercentiles
    stt_latency: LatencyPercentiles
    rag_latency: LatencyPercentiles
    voice_overhead_ms: float = 0.0
    stt_contribution_pct: float = 0.0
    stage_breakdown: Dict[str, StageTimingStats] = Field(default_factory=dict)
    primary_bottleneck: str = "Unknown"
    primary_bottleneck_pct: float = 0.0
    secondary_bottleneck: str = "Unknown"
    secondary_bottleneck_pct: float = 0.0
    target_200ms_rag: TargetEvaluation
    target_200ms_voice: TargetEvaluation
    language_latency: Dict[str, LatencyPercentiles] = Field(default_factory=dict)
    category_latency: Dict[str, LatencyPercentiles] = Field(default_factory=dict)
    baseline_comparison: Dict[str, Any] = Field(default_factory=dict)
