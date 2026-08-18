"""Configuration for Input Guardrail service."""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class InputGuardrailConfig:
    """Centralized settings for input guardrails."""

    enabled: bool = True
    max_chars: int = 1000
    max_tokens: int = 250
    prompt_injection_enabled: bool = True
    safety_enabled: bool = True
    scope_enabled: bool = True
    extra_params: Dict[str, Any] = field(default_factory=dict)
