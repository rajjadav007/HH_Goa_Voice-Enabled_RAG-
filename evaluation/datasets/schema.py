"""Pydantic schemas for the evaluation framework dataset cases, results, and metrics."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    """Schema representing a single evaluation test case."""
    test_id: str = Field(..., description="Unique evaluation case ID")
    query: str = Field(..., description="Input query text or reference question")
    language: str = Field("en-IN", description="Language code (e.g. en-IN, hi-IN, as-IN, ta-IN, bn-IN)")
    category: str = Field("factual", description="Query category: factual, multihop, short, long, paraphrased, entity, number, unanswerable, injection, offtopic")
    difficulty: str = Field("medium", description="Difficulty level: easy, medium, hard")
    expected_answer: Optional[str] = Field(None, description="Ground truth reference answer if available")
    relevant_document_ids: List[str] = Field(default_factory=list, description="Ground truth document IDs")
    relevant_chunk_ids: List[str] = Field(default_factory=list, description="Ground truth chunk IDs")
    voice_audio_path: Optional[str] = Field(None, description="Path to voice audio sample if testing voice STT")


class RetrievalMetrics(BaseModel):
    """Metrics for context retrieval evaluation."""
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0
    precision_at_5: float = 0.0
    hit_rate: float = 0.0


class GenerationMetrics(BaseModel):
    """Metrics for RAG answer generation evaluation."""
    exact_match: float = 0.0
    token_f1: float = 0.0
    semantic_similarity: float = 0.0
    correctness: float = 0.0


class GroundingMetrics(BaseModel):
    """Metrics for grounding validation and safety."""
    grounded_rate: float = 0.0
    unsupported_claim_rate: float = 0.0
    contradiction_rate: float = 0.0
    correct_abstention_rate: float = 0.0


class LatencyPercentiles(BaseModel):
    """Latency distribution percentiles in milliseconds."""
    p50: float = 0.0
    p70: float = 0.0
    p90: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    p100: float = 0.0
    target_200ms_passed: bool = False


class EvalResultItem(BaseModel):
    """Evaluation output result for a single test case execution."""
    test_id: str
    mode: str
    language: str
    category: str
    query: str
    transcript: Optional[str] = None
    answer: str
    grounded: bool
    grounding_status: str
    has_context: bool
    retrieved_chunk_ids: List[str] = Field(default_factory=list)
    retrieved_doc_ids: List[str] = Field(default_factory=list)
    recall_at_1: Optional[float] = None
    recall_at_3: Optional[float] = None
    recall_at_5: Optional[float] = None
    mrr: Optional[float] = None
    latency_ms: float = 0.0
    stt_latency_ms: Optional[float] = None
    rag_latency_ms: Optional[float] = None
    status: str = "SUCCESS"
    error_category: Optional[str] = None
    timing_breakdown: Dict[str, float] = Field(default_factory=dict)


class RunMetadata(BaseModel):
    """Metadata tracking reproducibility settings and environment parameters for an evaluation run."""
    run_id: str
    timestamp: str
    seed: int
    mode: str
    offline_mock: bool
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_model: str = "gemini-3.6-flash"
    stt_provider: str = "sarvam-saaras:v2"
    dataset_version: str = "v1.0"


class AggregateEvaluationReport(BaseModel):
    """Complete aggregated evaluation report."""
    metadata: RunMetadata
    total_cases: int
    successful_cases: int
    success_rate: float
    retrieval: RetrievalMetrics
    generation: GenerationMetrics
    grounding: GroundingMetrics
    latency: LatencyPercentiles
    language_breakdown: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    category_breakdown: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    failure_taxonomy: Dict[str, int] = Field(default_factory=dict)
