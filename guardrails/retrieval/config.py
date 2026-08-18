"""Configuration for Retrieval Guardrail service."""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class RetrievalGuardrailConfig:
    """Centralized settings for retrieval guardrails."""

    enabled: bool = True
    min_valid_results: int = 1
    min_relevance_score: float = -10.0
    conflict_detection: bool = True
    extra_params: Dict[str, Any] = field(default_factory=dict)
