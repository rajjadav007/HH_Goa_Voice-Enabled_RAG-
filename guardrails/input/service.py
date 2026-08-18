"""Input Guardrail service evaluating user queries prior to RAG execution."""

import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from guardrails.input.config import InputGuardrailConfig
from guardrails.input.models import GuardrailCategory, GuardrailDecision

logger = logging.getLogger(__name__)

# Prompt injection patterns
PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(?:[a-z]+\s+)*(instructions|prompts|rules|guidelines)",
    r"reveal\s+(?:the\s+|hidden\s+)*(system\s+prompt|developer\s+instructions|hidden\s+prompt|api\s+key)",
    r"system\s+prompt\s+is",
    r"print\s+(api_key|secret|env\s+vars|system\s+instructions)",
    r"override\s+(system|developer)\s+(rules|mode)",
    r"you\s+are\s+now\s+in\s+dan\s+mode",
    r"jailbreak",
    r"do\s+anything\s+now",
    r"\bdan\b",
]

# Explicit non-QA off-topic intent patterns
OFF_TOPIC_PATTERNS = [
    r"^(write|build|create|generate)\s+(?:a|an)?\s*(python|java|javascript|cpp|c\#|html|css|game|app|script|code|poem|song|story)\b",
    r"^(write|generate)\s+code\s+for",
    r"^who\s+won\s+yesterday's\s+(match|game|football|cricket|superbowl)",
]

# Unsafe content patterns
UNSAFE_PATTERNS = [
    r"\b(how\s+to\s+make|build|synthesize)\s+(a\s+bomb|explosives|poison|biological\s+weapon)\b",
    r"\b(hack|crack)\s+(into\s+a\s+bank|email\s+account|wifi\s+network)\b",
]


class InputGuardrailService:
    """Service evaluating input safety, formatting, prompt injection, and scope."""

    def __init__(self, config: Optional[InputGuardrailConfig] = None):
        self.config = config or InputGuardrailConfig()
        self._compiled_injection = [re.compile(p, re.IGNORECASE) for p in PROMPT_INJECTION_PATTERNS]
        self._compiled_off_topic = [re.compile(p, re.IGNORECASE) for p in OFF_TOPIC_PATTERNS]
        self._compiled_unsafe = [re.compile(p, re.IGNORECASE) for p in UNSAFE_PATTERNS]

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough 4 chars per token)."""
        return max(1, len(text) // 4)

    def validate_format_and_length(self, query_text: Any) -> Optional[GuardrailDecision]:
        """Validate input type, non-emptiness, and character/token limits."""
        if query_text is None or not isinstance(query_text, str):
            return GuardrailDecision(
                allowed=False,
                category=GuardrailCategory.INVALID,
                reason="Query input must be a non-null string.",
            )

        clean_text = query_text.strip()
        if not clean_text:
            return GuardrailDecision(
                allowed=False,
                category=GuardrailCategory.EMPTY,
                reason="Query input is empty or whitespace only.",
            )

        if len(clean_text) > self.config.max_chars:
            return GuardrailDecision(
                allowed=False,
                category=GuardrailCategory.TOO_LONG,
                reason=f"Query character length ({len(clean_text)}) exceeds limit ({self.config.max_chars}).",
                metadata={"length": len(clean_text), "max_chars": self.config.max_chars},
            )

        est_tokens = self.estimate_tokens(clean_text)
        if est_tokens > self.config.max_tokens:
            return GuardrailDecision(
                allowed=False,
                category=GuardrailCategory.TOO_LONG,
                reason=f"Estimated query tokens ({est_tokens}) exceed limit ({self.config.max_tokens}).",
                metadata={"estimated_tokens": est_tokens, "max_tokens": self.config.max_tokens},
            )

        return None

    def detect_prompt_injection(self, query_text: str) -> Optional[GuardrailDecision]:
        """Detect prompt injection patterns."""
        if not self.config.prompt_injection_enabled:
            return None

        for pattern in self._compiled_injection:
            if pattern.search(query_text):
                return GuardrailDecision(
                    allowed=False,
                    category=GuardrailCategory.PROMPT_INJECTION,
                    reason="Query contains potential prompt injection attempt.",
                    metadata={"matched_pattern": pattern.pattern},
                )
        return None

    def detect_unsafe_input(self, query_text: str) -> Optional[GuardrailDecision]:
        """Detect unsafe or harmful content."""
        if not self.config.safety_enabled:
            return None

        for pattern in self._compiled_unsafe:
            if pattern.search(query_text):
                return GuardrailDecision(
                    allowed=False,
                    category=GuardrailCategory.UNSAFE,
                    reason="Query contains unsafe or restricted instructions.",
                    metadata={"matched_pattern": pattern.pattern},
                )
        return None

    def detect_off_topic(self, query_text: str) -> Optional[GuardrailDecision]:
        """Detect non-QA or out-of-scope intent."""
        if not self.config.scope_enabled:
            return None

        for pattern in self._compiled_off_topic:
            if pattern.search(query_text):
                return GuardrailDecision(
                    allowed=False,
                    category=GuardrailCategory.OFF_TOPIC,
                    reason="Query intent is outside supported dataset Q&A scope.",
                    metadata={"matched_pattern": pattern.pattern},
                )
        return None

    def evaluate(self, query_text: Any) -> GuardrailDecision:
        """Run complete input guardrail evaluation pipeline."""
        t0 = time.perf_counter()

        if not self.config.enabled:
            eval_ms = round((time.perf_counter() - t0) * 1000, 3)
            return GuardrailDecision(
                allowed=True,
                category=GuardrailCategory.VALID,
                reason="Guardrails disabled.",
                latency_ms=eval_ms,
            )

        # Stage 1: Format & Length Validation
        fmt_res = self.validate_format_and_length(query_text)
        if fmt_res:
            fmt_res.latency_ms = round((time.perf_counter() - t0) * 1000, 3)
            return fmt_res

        clean_text = query_text.strip()

        # Stage 2: Prompt Injection Detection
        inj_res = self.detect_prompt_injection(clean_text)
        if inj_res:
            inj_res.latency_ms = round((time.perf_counter() - t0) * 1000, 3)
            return inj_res

        # Stage 3: Unsafe Content Detection
        safe_res = self.detect_unsafe_input(clean_text)
        if safe_res:
            safe_res.latency_ms = round((time.perf_counter() - t0) * 1000, 3)
            return safe_res

        # Stage 4: Scope / Off-Topic Detection
        topic_res = self.detect_off_topic(clean_text)
        if topic_res:
            topic_res.latency_ms = round((time.perf_counter() - t0) * 1000, 3)
            return topic_res

        eval_ms = round((time.perf_counter() - t0) * 1000, 3)
        return GuardrailDecision(
            allowed=True,
            category=GuardrailCategory.VALID,
            reason="Query passed all input guardrail checks.",
            latency_ms=eval_ms,
        )
