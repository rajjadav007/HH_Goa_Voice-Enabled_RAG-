"""Retrieval Guardrail service evaluating context validity, traceability, and sufficiency."""

import logging
import math
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from guardrails.retrieval.config import RetrievalGuardrailConfig
from guardrails.retrieval.models import RetrievalGuardrailDecision, SufficiencyStatus
from retrieval.reranking.models import RerankedResultPoint

logger = logging.getLogger(__name__)

NUMERIC_FACT_PATTERN = re.compile(r"\b(\d{4}|\d+(?:\.\d+)?\s*(?:mg|ml|kg|km|m|%|dollars|USD|years|days))\b", re.IGNORECASE)


class RetrievalGuardrailService:
    """Service validating retrieved context quality, source traceability, and sufficiency."""

    def __init__(self, config: Optional[RetrievalGuardrailConfig] = None):
        self.config = config or RetrievalGuardrailConfig()

    def validate_chunk_structure(self, chunk: Any) -> bool:
        """Validate structure of retrieved chunk."""
        if chunk is None or not isinstance(chunk, RerankedResultPoint):
            return False
        if not chunk.chunk_id or not isinstance(chunk.chunk_id, str) or not chunk.chunk_id.strip():
            return False
        if not chunk.document_id or not isinstance(chunk.document_id, str) or not chunk.document_id.strip():
            return False
        if not chunk.text or not isinstance(chunk.text, str) or not chunk.text.strip():
            return False
        if chunk.final_rank is None or not isinstance(chunk.final_rank, int) or chunk.final_rank < 1:
            return False

        # Validate score is finite and numeric
        score = chunk.rerank_score
        if score is None or not isinstance(score, (int, float)) or math.isnan(score) or math.isinf(score):
            return False

        return True

    def validate_source_traceability(self, chunk: RerankedResultPoint) -> bool:
        """Verify chunk chunk_id and document_id conform to dataset traceability schema."""
        if not chunk.chunk_id.startswith("chk_") and "chunk" not in chunk.chunk_id:
            return False
        if not chunk.document_id.startswith("doc_") and "doc" not in chunk.document_id:
            return False
        return True

    def deduplicate_chunks(self, chunks: List[RerankedResultPoint]) -> List[RerankedResultPoint]:
        """Filter out duplicate chunk_ids preserving highest rank."""
        seen = set()
        deduped = []
        for c in chunks:
            if c.chunk_id not in seen:
                seen.add(c.chunk_id)
                deduped.append(c)
        return deduped

    def detect_contradictions(self, chunks: List[RerankedResultPoint]) -> bool:
        """Detect obvious numerical/factual contradictions across top retrieved chunks."""
        if not self.config.conflict_detection or len(chunks) < 2:
            return False

        facts_found = []
        for c in chunks[:3]:
            matches = NUMERIC_FACT_PATTERN.findall(c.text)
            if matches:
                facts_found.append(set(matches))

        if len(facts_found) >= 2:
            # Check if disjoint numeric sets exist in top 2 chunks on same topic
            s1, s2 = facts_found[0], facts_found[1]
            if s1 and s2 and not (s1 & s2) and len(s1) == 1 and len(s2) == 1:
                logger.info(f"Conflict detected between top chunks: '{s1}' vs '{s2}'.")
                return True

        return False

    def evaluate(
        self,
        query_text: str,
        reranked_chunks: List[RerankedResultPoint],
    ) -> RetrievalGuardrailDecision:
        """Evaluate retrieved context validity, score, rank, and sufficiency."""
        t0 = time.perf_counter()

        if not self.config.enabled:
            eval_ms = round((time.perf_counter() - t0) * 1000, 3)
            return RetrievalGuardrailDecision(
                allowed=True,
                status=SufficiencyStatus.SUFFICIENT,
                valid_chunks=reranked_chunks,
                rejected_chunks_count=0,
                reason="Retrieval guardrails disabled.",
                latency_ms=eval_ms,
            )

        if not reranked_chunks:
            eval_ms = round((time.perf_counter() - t0) * 1000, 3)
            return RetrievalGuardrailDecision(
                allowed=False,
                status=SufficiencyStatus.EMPTY,
                valid_chunks=[],
                rejected_chunks_count=0,
                reason="No retrieval chunks provided.",
                latency_ms=eval_ms,
            )

        valid_chunks: List[RerankedResultPoint] = []
        rejected_count = 0

        # Validate structure and traceability
        for chunk in reranked_chunks:
            if self.validate_chunk_structure(chunk) and self.validate_source_traceability(chunk):
                # Score threshold check
                if chunk.rerank_score >= self.config.min_relevance_score:
                    valid_chunks.append(chunk)
                else:
                    rejected_count += 1
                    logger.warning(f"Rejected chunk '{chunk.chunk_id}' due to low relevance score ({chunk.rerank_score}).")
            else:
                rejected_count += 1
                logger.warning(f"Rejected malformed or non-traceable chunk: '{getattr(chunk, 'chunk_id', 'UNKNOWN')}'.")

        # Deduplicate valid chunks
        valid_chunks = self.deduplicate_chunks(valid_chunks)

        # Re-sort valid chunks by final_rank / rerank_score to ensure correct rank ordering
        valid_chunks.sort(key=lambda c: (c.final_rank, -c.rerank_score))

        # Re-index final_rank strictly from 1..N
        for idx, c in enumerate(valid_chunks, 1):
            c.final_rank = idx

        if not valid_chunks or len(valid_chunks) < self.config.min_valid_results:
            eval_ms = round((time.perf_counter() - t0) * 1000, 3)
            return RetrievalGuardrailDecision(
                allowed=False,
                status=SufficiencyStatus.INSUFFICIENT,
                valid_chunks=[],
                rejected_chunks_count=rejected_count,
                reason=f"Insufficient valid retrieval context ({len(valid_chunks)} valid chunks < min {self.config.min_valid_results}).",
                latency_ms=eval_ms,
            )

        # Detect contradiction
        has_conflict = self.detect_contradictions(valid_chunks)
        status = SufficiencyStatus.CONFLICTING if has_conflict else SufficiencyStatus.SUFFICIENT

        eval_ms = round((time.perf_counter() - t0) * 1000, 3)
        return RetrievalGuardrailDecision(
            allowed=True,
            status=status,
            valid_chunks=valid_chunks,
            rejected_chunks_count=rejected_count,
            reason="Retrieved context meets quality and sufficiency criteria." if not has_conflict else "Conflicting factual claims detected in context.",
            latency_ms=eval_ms,
            metadata={"has_contradiction": has_conflict},
        )
