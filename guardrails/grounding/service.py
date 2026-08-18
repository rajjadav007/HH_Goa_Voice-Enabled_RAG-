"""Grounding Validation service verifying generated answer support against retrieved context."""

import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from guardrails.grounding.config import GroundingValidationConfig
from guardrails.grounding.models import GroundingDecision, GroundingStatus
from retrieval.reranking.models import RerankedResultPoint

logger = logging.getLogger(__name__)

STOP_WORDS = {"the", "a", "an", "is", "are", "was", "were", "of", "in", "to", "and", "or", "it", "by", "for", "with", "on", "at", "from"}


class GroundingValidationService:
    """Service evaluating factual support of model-generated answers against retrieved context chunks."""

    def __init__(self, config: Optional[GroundingValidationConfig] = None):
        self.config = config or GroundingValidationConfig()

    def extract_claims(self, answer_text: str) -> List[str]:
        """Extract individual sentence claims from answer text."""
        if not answer_text or not answer_text.strip():
            return []
        raw_sentences = re.split(r"(?<=[.!?])\s+", answer_text.strip())
        claims = [s.strip() for s in raw_sentences if len(s.strip()) > 5]
        return claims if claims else [answer_text.strip()]

    def verify_claim_support(
        self,
        claim: str,
        context_chunks: List[RerankedResultPoint],
    ) -> Tuple[float, str]:
        """Calculate claim support score against context chunks using content-word & n-gram overlap."""
        if not context_chunks:
            return 0.0, ""

        claim_clean = re.sub(r"[^\w\s]", "", claim.lower()).strip()
        all_claim_tokens = claim_clean.split()
        claim_words = set(w for w in all_claim_tokens if w not in STOP_WORDS)
        if not claim_words:
            return 1.0, ""

        best_score = 0.0
        best_chunk_id = ""

        combined_context = " ".join([c.text.lower() for c in context_chunks])
        combined_clean = re.sub(r"[^\w\s]", "", combined_context)
        context_words = set(w for w in combined_clean.split() if w not in STOP_WORDS)

        # 1. Content word overlap ratio
        overlap_count = len(claim_words & context_words)
        word_overlap_ratio = overlap_count / len(claim_words)

        # 2. Bigram / Substring overlap from all claim tokens
        bigram_matches = 0
        if len(all_claim_tokens) >= 2:
            bigrams = [" ".join(all_claim_tokens[i:i+2]) for i in range(len(all_claim_tokens)-1)]
            for bg in bigrams:
                if bg in combined_clean:
                    bigram_matches += 1
            bigram_ratio = bigram_matches / len(bigrams)
        else:
            bigram_ratio = word_overlap_ratio

        # Composite support score
        support_score = (0.5 * word_overlap_ratio) + (0.5 * bigram_ratio)

        for c in context_chunks:
            if support_score > best_score:
                best_score = support_score
                best_chunk_id = c.chunk_id

        return min(1.0, support_score), best_chunk_id

    def evaluate(
        self,
        query_text: str,
        answer_text: str,
        is_grounded_flag: bool,
        context_chunks: List[RerankedResultPoint],
    ) -> GroundingDecision:
        """Evaluate factual grounding of answer against context chunks."""
        t0 = time.perf_counter()

        if not self.config.enabled:
            eval_ms = round((time.perf_counter() - t0) * 1000, 3)
            return GroundingDecision(
                grounded=is_grounded_flag,
                status=GroundingStatus.FULLY_GROUNDED if is_grounded_flag else GroundingStatus.UNGROUNDED,
                support_score=1.0 if is_grounded_flag else 0.0,
                validated_answer=answer_text,
                latency_ms=eval_ms,
            )

        # Refusal check
        if "insufficient context" in answer_text.lower() or "cannot process this request" in answer_text.lower():
            eval_ms = round((time.perf_counter() - t0) * 1000, 3)
            return GroundingDecision(
                grounded=False,
                status=GroundingStatus.REFUSAL_GROUNDED,
                support_score=1.0,
                unsupported_claims=[],
                validated_answer=answer_text,
                latency_ms=eval_ms,
                metadata={"refusal": True},
            )

        if not context_chunks:
            eval_ms = round((time.perf_counter() - t0) * 1000, 3)
            return GroundingDecision(
                grounded=False,
                status=GroundingStatus.NO_CONTEXT_GROUNDED,
                support_score=0.0,
                unsupported_claims=[answer_text],
                validated_answer="Insufficient context available to answer the query.",
                latency_ms=eval_ms,
            )

        claims = self.extract_claims(answer_text)
        claim_scores = []
        unsupported = []

        for claim in claims:
            score, cid = self.verify_claim_support(claim, context_chunks)
            claim_scores.append(score)
            if score < self.config.min_support_score:
                unsupported.append(claim)

        avg_support = sum(claim_scores) / len(claim_scores) if claim_scores else 0.0

        if avg_support >= 0.65 and not unsupported:
            status = GroundingStatus.FULLY_GROUNDED
            is_grounded = True
            val_ans = answer_text
        elif avg_support >= self.config.min_support_score and len(unsupported) <= len(claims) // 2:
            status = GroundingStatus.PARTIALLY_GROUNDED
            is_grounded = not self.config.strict_mode
            val_ans = answer_text
        else:
            status = GroundingStatus.UNGROUNDED
            is_grounded = False
            val_ans = "Insufficient context available to verify the generated answer."
            logger.warning(f"Answer failed grounding validation (score={avg_support:.2f} < {self.config.min_support_score}).")

        eval_ms = round((time.perf_counter() - t0) * 1000, 3)
        return GroundingDecision(
            grounded=is_grounded,
            status=status,
            support_score=avg_support,
            unsupported_claims=unsupported,
            validated_answer=val_ans,
            latency_ms=eval_ms,
            metadata={"claims_count": len(claims), "unsupported_count": len(unsupported)},
        )
