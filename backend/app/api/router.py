"""Central API Router aggregating endpoint modules."""

from fastapi import APIRouter
from app.api.endpoints import health, query

api_router = APIRouter()

# Include health check & RAG query routers
api_router.include_router(health.router, tags=["Health"])
api_router.include_router(query.router, tags=["RAG Query"])
