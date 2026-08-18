"""Harness state machine models and execution telemetry tracking."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class HarnessState(str, Enum):
    INIT = "INIT"
    INPUT_VALIDATED = "INPUT_VALIDATED"
    RETRIEVED = "RETRIEVED"
    RERANKED = "RERANKED"
    RETRIEVAL_GUARDED = "RETRIEVAL_GUARDED"
    GENERATED = "GENERATED"
    GROUNDING_VALIDATED = "GROUNDING_VALIDATED"
    COMPLETED = "COMPLETED"
    BLOCKED = "BLOCKED"
    NO_CONTEXT = "NO_CONTEXT"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"


@dataclass
class StageExecutionRecord:
    """Record of an individual pipeline stage execution."""

    stage_name: str
    start_time: float
    end_time: float
    duration_ms: float
    attempts: int = 1
    success: bool = True
    error_category: Optional[str] = None
    fallback_used: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage_name,
            "duration_ms": round(self.duration_ms, 2),
            "attempts": self.attempts,
            "success": self.success,
            "error_category": self.error_category,
            "fallback_used": self.fallback_used,
        }


@dataclass
class HarnessTelemetry:
    """Telemetry and execution metrics gathered during harness execution."""

    request_id: str
    state: HarnessState
    degraded: bool = False
    total_attempts: int = 0
    stages: List[StageExecutionRecord] = field(default_factory=list)
    latency_breakdown: Dict[str, float] = field(default_factory=dict)
    harness_overhead_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "state": self.state.value,
            "degraded": self.degraded,
            "total_attempts": self.total_attempts,
            "harness_overhead_ms": round(self.harness_overhead_ms, 3),
            "latency_breakdown": self.latency_breakdown,
            "stages": [s.to_dict() for s in self.stages],
        }
