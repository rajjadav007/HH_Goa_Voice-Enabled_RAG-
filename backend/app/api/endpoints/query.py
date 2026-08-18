"""FastAPI Endpoint for Text RAG Query Execution."""

import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from orchestration.service import RAGOrchestrator

logger = logging.getLogger(__name__)

router = APIRouter()
orchestrator = RAGOrchestrator()


class QueryRequest(BaseModel):
    query: str = Field(description="User text query for RAG question answering.")


class SourceItem(BaseModel):
    chunk_id: str
    document_id: str
    rank: int


class QueryResponse(BaseModel):
    success: bool = True
    answer: str
    grounded: bool
    has_context: bool
    sources: List[SourceItem]
    request_id: str
    status: str
    error_code: Optional[str] = None
    latency_ms: float
    timing_breakdown: Dict[str, float] = {}
    token_usage: Dict[str, int] = {}


@router.post("/query", response_model=QueryResponse, summary="Execute Text RAG Query")
async def execute_text_rag_query(request: QueryRequest) -> QueryResponse:
    """Execute end-to-end text RAG query using production RAG orchestrator."""
    if not request.query or not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query parameter must be a non-empty string.",
        )

    try:
        res = orchestrator.answer(query_text=request.query)
        sources_list = [
            SourceItem(
                chunk_id=s.get("chunk_id", ""),
                document_id=s.get("document_id", ""),
                rank=s.get("rank", 0),
            )
            for s in res.sources
        ]

        return QueryResponse(
            success=res.status == "SUCCESS" or res.status == "NO_CONTEXT",
            answer=res.answer,
            grounded=res.grounded,
            has_context=res.has_context,
            sources=sources_list,
            request_id=res.request_id,
            status=res.status,
            error_code=res.error_code,
            latency_ms=res.latency_ms,
            timing_breakdown=res.timing_breakdown,
            token_usage=res.token_usage,
        )
    except Exception as exc:
        logger.error(f"Error in text RAG endpoint: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal server error occurred while processing the RAG query.",
        )
