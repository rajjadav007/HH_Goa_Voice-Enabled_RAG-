"""Text boundary and token counting utilities for chunking strategies."""

import re
from typing import List

# Sentence splitting pattern supporting English (.!?), Indic purna viram (।), and double newlines
SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?।])\s+|\n\n+")
PARAGRAPH_SPLIT_REGEX = re.compile(r"\n\s*\n")


def count_tokens(text: str) -> int:
    """Count tokens in text.

    Uses word-level count fallback. Fast, lightweight, zero heavy model downloads.
    """
    if not text:
        return 0
    # Split on whitespace & punctuation boundaries
    words = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    return len(words)


def count_characters(text: str) -> int:
    """Count characters in text."""
    return len(text) if text else 0


def split_sentences(text: str) -> List[str]:
    """Split text into sentence units respecting Indic (।) and English boundary punctuation."""
    if not text or not text.strip():
        return []
    parts = SENTENCE_SPLIT_REGEX.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def split_paragraphs(text: str) -> List[str]:
    """Split text into paragraph units based on double newline boundaries."""
    if not text or not text.strip():
        return []
    parts = PARAGRAPH_SPLIT_REGEX.split(text.strip())
    return [p.strip() for p in parts if p.strip()]
