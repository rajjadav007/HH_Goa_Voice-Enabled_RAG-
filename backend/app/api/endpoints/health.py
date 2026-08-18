"""Health check API endpoint implementation."""

from datetime import datetime, timezone
from fastapi import APIRouter
from app.core.config import settings
from app.models.schemas import HealthResponse, QdrantStatus
from app.retrieval.vector.client import check_qdrant_health

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Backend & Infrastructure Health Check",
    description="Returns backend server health status, timestamp, and Qdrant vector database connectivity.",
)
async def health_check() -> HealthResponse:
    """Returns status indicating backend operational state and Qdrant infrastructure connectivity."""
    qdrant_info = check_qdrant_health()
    return HealthResponse(
        status="ok",
        version=settings.VERSION,
        service=settings.PROJECT_NAME,
        timestamp=datetime.now(timezone.utc).isoformat(),
        qdrant=QdrantStatus(**qdrant_info),
    )
