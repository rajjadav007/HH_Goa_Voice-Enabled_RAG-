"""Unit tests for backend health check endpoint."""

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check_endpoint():
    """Tests GET /api/health returns 200 OK with expected JSON schema including Qdrant status."""
    response = client.get("/api/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "service" in data
    assert "timestamp" in data
    assert "qdrant" in data
    assert "connected" in data["qdrant"]
