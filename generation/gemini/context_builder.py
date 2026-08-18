"""Context builder formatting reranked retrieval chunks for Gemini prompt injection defense."""

import logging
from typing import Any, Dict, List, Optional, Tuple

from retrieval.reranking.models import RerankedResultPoint

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Formats retrieved context chunks with token budgeting and prompt injection boundaries."""

    def __init__(self, token_budget: int = 2000):
        self.token_budget = token_budget

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (approx 4 chars per token)."""
        return max(1, len(text) // 4)

    def build_context(
        self,
        chunks: List[RerankedResultPoint],
        query_text: str,
        token_budget: Optional[int] = None,
    ) -> Tuple[str, List[RerankedResultPoint], Dict[str, Any]]:
        """Construct structured context string respecting context token budget and chunk boundaries."""
        budget = token_budget or self.token_budget

        if not chunks:
            return "", [], {"selected_chunks": 0, "estimated_tokens": 0, "budget_exceeded": False}

        selected_chunks: List[RerankedResultPoint] = []
        formatted_blocks: List[str] = []
        current_token_count = 0

        for chunk in chunks:
            if not chunk.text or not chunk.text.strip():
                continue

            # Format chunk as untrusted DATA container
            block = (
                f'<document_chunk id="{chunk.chunk_id}" doc_id="{chunk.document_id}" rank="{chunk.final_rank}">\n'
                f"{chunk.text.strip()}\n"
                f"</document_chunk>"
            )

            est_tokens = self.estimate_tokens(block)
            if current_token_count + est_tokens > budget and selected_chunks:
                logger.info(
                    f"Context token budget limit ({budget}) reached. Truncating context to top-{len(selected_chunks)} chunks."
                )
                break

            selected_chunks.append(chunk)
            formatted_blocks.append(block)
            current_token_count += est_tokens

        context_str = "\n\n".join(formatted_blocks)
        stats = {
            "selected_chunks": len(selected_chunks),
            "estimated_tokens": current_token_count,
            "budget_exceeded": len(selected_chunks) < len(chunks),
        }

        return context_str, selected_chunks, stats
