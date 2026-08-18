"""Standard response schemas and foundational request models."""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class QdrantStatus(BaseModel):
    """Pydantic schema for Qdrant vector database connectivity status."""
    connected: bool = Field(..., example=True)
    host: str = Field(..., example="localhost")
    port: int = Field(..., example=6333)
    status: Optional[str] = Field(None, example="healthy")
    error: Optional[str] = Field(None, example=None)


class HealthResponse(BaseModel):
    """Pydantic schema for backend health check endpoint response."""
    status: str = Field(..., example="ok")
    version: str = Field(..., example="1.0.0")
    service: str = Field(..., example="HH Goa Voice RAG Backend")
    timestamp: str = Field(..., description="ISO 8601 server timestamp")
    qdrant: Optional[QdrantStatus] = None


class ErrorDetail(BaseModel):
    """Structured error detail schema."""
    code: str = Field(..., example="INTERNAL_SERVER_ERROR")
    message: str = Field(..., example="An unexpected error occurred.")
    details: Optional[Dict[str, Any]] = None


class BaseAPIResponse(BaseModel):
    """Standardized API envelope model as specified in ARCHITECTURE.md."""
    request_id: str = Field(..., description="Unique request tracing ID")
    success: bool = Field(True, description="Success flag")
    error: Optional[ErrorDetail] = None
