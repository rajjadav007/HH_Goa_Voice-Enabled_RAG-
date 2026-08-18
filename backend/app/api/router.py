"""Central API Router aggregating endpoint modules."""

from fastapi import APIRouter
from app.api.endpoints import health, query, voice

api_router = APIRouter()

# Include health check, RAG query, and Voice query routers
api_router.include_router(health.router, tags=["Health"])
api_router.include_router(query.router, tags=["RAG Query"])
api_router.include_router(voice.router, tags=["Voice Query"])
