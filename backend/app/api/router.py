"""Central API Router aggregating endpoint modules."""

from fastapi import APIRouter
from app.api.endpoints import health

api_router = APIRouter()

# Include health check router
api_router.include_router(health.router, tags=["Health"])
