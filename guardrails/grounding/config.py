"""Configuration for Grounding Validation service."""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class GroundingValidationConfig:
    """Centralized settings for grounding validation."""

    enabled: bool = True
    min_support_score: float = 0.4
    strict_mode: bool = False
    extra_params: Dict[str, Any] = field(default_factory=dict)
