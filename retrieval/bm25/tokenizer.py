"""Multilingual tokenizer for BM25 lexical indexing and query matching."""

import re
from typing import List


class MultilingualBM25Tokenizer:
    """Tokenizer preserving terms, numbers, Indic graphemes, and multilingual word boundaries."""

    def __init__(self, lower_case: bool = True):
        self.lower_case = lower_case
        # Regex matching non-punctuation, non-whitespace token blocks
        self.token_regex = re.compile(r"[^\s\:\,\.\!\?\;\(\)\[\]\{\}\"\']+", re.UNICODE)

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text string into normalized word tokens."""
        if not text or not text.strip():
            return []

        target_text = text.lower() if self.lower_case else text
        tokens = self.token_regex.findall(target_text)
        return [t.strip() for t in tokens if len(t.strip()) > 0]
