"""Qdrant Vector Database Client and Infrastructure Connectivity Manager."""

import httpx
from typing import Dict, Any
from app.core.config import settings
from app.core.logging import logger


def check_qdrant_health() -> Dict[str, Any]:
    """
    Verifies backend -> Qdrant connectivity over local storage or REST HTTP endpoint.
    Returns connectivity status dict safely without raising unhandled exceptions.
    """
    try:
        from retrieval.vector_db.service import QdrantService
        qs = QdrantService()
        res = qs.health_check()
        if res.get("status") == "healthy":
            return {
                "connected": True,
                "host": "local_storage",
                "port": settings.QDRANT_PORT,
                "status": "healthy"
            }
    except Exception:
        pass

    qdrant_url = f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}/healthz"
    try:
        with httpx.Client(timeout=1.0) as client:
            response = client.get(qdrant_url)
            if response.status_code == 200:
                return {
                    "connected": True,
                    "host": settings.QDRANT_HOST,
                    "port": settings.QDRANT_PORT,
                    "status": "healthy"
                }
            return {
                "connected": False,
                "host": settings.QDRANT_HOST,
                "port": settings.QDRANT_PORT,
                "error": f"Qdrant returned HTTP status {response.status_code}"
            }
    except Exception:
        return {
            "connected": False,
            "host": settings.QDRANT_HOST,
            "port": settings.QDRANT_PORT,
            "error": "Qdrant server unreachable"
        }

