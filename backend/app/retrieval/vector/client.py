"""Qdrant Vector Database Client and Infrastructure Connectivity Manager."""

import httpx
from typing import Dict, Any
from app.core.config import settings
from app.core.logging import logger


def check_qdrant_health() -> Dict[str, Any]:
    """
    Verifies backend -> Qdrant connectivity over REST HTTP endpoint (/healthz).
    Returns connectivity status dict safely without raising unhandled exceptions.
    """
    qdrant_url = f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}/healthz"
    try:
        with httpx.Client(timeout=3.0) as client:
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
    except Exception as exc:
        logger.warning(f"Qdrant connectivity check failed: {str(exc)}")
        return {
            "connected": False,
            "host": settings.QDRANT_HOST,
            "port": settings.QDRANT_PORT,
            "error": "Qdrant server unreachable (check Docker container status)"
        }
