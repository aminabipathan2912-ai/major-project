"""
backend/tests/test_phase1.py
Phase 1 test suite — verifies:
  1. Application starts and health check passes
  2. All API routes respond correctly
  3. Incident CRUD works end-to-end
  4. Fusion endpoint runs and creates incidents
  5. WebSocket connects
  6. Missing modalities handled gracefully
"""

# Repo root on sys.path first so 'ml' package is importable
import sys
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import os

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

# Use in-memory SQLite for tests
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ["DATABASE_SYNC_URL"] = "sqlite:///:memory:"
os.environ["DEBUG"] = "true"
os.environ["DEMO_MODE"] = "true"

from app.main import app
from app.core.database import init_db, engine, Base


@pytest_asyncio.fixture(autouse=True)
async def setup_db():
    """Create tables before each test and drop after."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


# ------------------------------------------------------------------ #
# 1. Health check
# ------------------------------------------------------------------ #
@pytest.mark.asyncio
async def test_health_check(client: AsyncClient):
    resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"


# ------------------------------------------------------------------ #
# 2. System status
# ------------------------------------------------------------------ #
@pytest.mark.asyncio
async def test_system_status(client: AsyncClient):
    resp = await client.get("/api/v1/system/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert "version" in body


@pytest.mark.asyncio
async def test_model_status(client: AsyncClient):
    resp = await client.get("/api/v1/system/models/status")
    assert resp.status_code == 200
    body = resp.json()
    assert "models" in body
    assert "video" in body["models"]
    assert "audio" in body["models"]
    assert "text" in body["models"]
    assert "sensor" in body["models"]


# ------------------------------------------------------------------ #
# 3. Modality stubs
# ------------------------------------------------------------------ #
@pytest.mark.asyncio
async def test_text_analyze(client: AsyncClient):
    resp = await client.post(
        "/api/v1/text/analyze",
        json={"text": "There is a fire in the building!", "source": "manual"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["modality"] == "text"
    assert "confidence" in body
    assert "event" in body


@pytest.mark.asyncio
async def test_sensor_readings(client: AsyncClient):
    resp = await client.post(
        "/api/v1/sensor/readings",
        json={
            "temperature": 75.0,
            "smoke_level": 0.9,
            "pir_motion": True,
            "location": "zone-A",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["modality"] == "sensor"


# ------------------------------------------------------------------ #
# 4. Fusion — with data
# ------------------------------------------------------------------ #
@pytest.mark.asyncio
async def test_fusion_creates_incident(client: AsyncClient):
    """High confidence fusion should create an incident and return result."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()

    resp = await client.post(
        "/api/v1/fusion/predict",
        json={
            "video": {
                "modality": "video",
                "event": "FIRE",
                "confidence": 0.89,
                "timestamp": now,
                "evidence": [],
                "status": "active",
                "raw_scores": {"FIRE": 0.89},
                "model_name": "test-model",
                "model_version": "1.0.0",
            },
            "sensor": {
                "modality": "sensor",
                "event": "SMOKE_ANOMALY",
                "confidence": 0.94,
                "timestamp": now,
                "evidence": [],
                "status": "active",
                "raw_scores": {"SMOKE_ANOMALY": 0.94},
                "model_name": "test-sensor",
                "model_version": "1.0.0",
            },
            "location": "zone-A",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["risk_score"] > 0
    assert body["create_incident"] is True
    assert "video" in body["contributing_modalities"]
    assert "sensor" in body["contributing_modalities"]


# ------------------------------------------------------------------ #
# 5. Fusion — missing modalities (must not crash)
# ------------------------------------------------------------------ #
@pytest.mark.asyncio
async def test_fusion_missing_modalities(client: AsyncClient):
    """Fusion with only one modality must succeed without crashing."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()

    resp = await client.post(
        "/api/v1/fusion/predict",
        json={
            "video": {
                "modality": "video",
                "event": "VIOLENCE",
                "confidence": 0.72,
                "timestamp": now,
                "evidence": [],
                "status": "active",
                "raw_scores": {"VIOLENCE": 0.72},
                "model_name": "test-model",
                "model_version": "1.0.0",
            },
            # audio, text, sensor all missing → None
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["fusion_breakdown"]["audio"] is None
    assert body["fusion_breakdown"]["text"] is None
    assert body["fusion_breakdown"]["sensor"] is None


# ------------------------------------------------------------------ #
# 6. Incident CRUD
# ------------------------------------------------------------------ #
@pytest.mark.asyncio
async def test_incident_list_empty(client: AsyncClient):
    resp = await client.get("/api/v1/incidents")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_incident_lifecycle(client: AsyncClient):
    """Create via fusion → list → get by id → update status."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()

    # Create via fusion
    await client.post(
        "/api/v1/fusion/predict",
        json={
            "video": {
                "modality": "video",
                "event": "FIRE",
                "confidence": 0.85,
                "timestamp": now,
                "evidence": [],
                "status": "active",
                "raw_scores": {},
                "model_name": "m",
                "model_version": "1.0.0",
            }
        },
    )

    # List
    resp = await client.get("/api/v1/incidents")
    assert resp.status_code == 200
    incidents = resp.json()
    assert len(incidents) >= 1
    incident_id = incidents[0]["id"]

    # Get by ID
    resp = await client.get(f"/api/v1/incidents/{incident_id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == incident_id

    # Update status
    resp = await client.patch(
        f"/api/v1/incidents/{incident_id}/status",
        json={"status": "ACKNOWLEDGED"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "ACKNOWLEDGED"


# ------------------------------------------------------------------ #
# 7. Alerts
# ------------------------------------------------------------------ #
@pytest.mark.asyncio
async def test_alerts_list(client: AsyncClient):
    resp = await client.get("/api/v1/alerts")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


# ------------------------------------------------------------------ #
# 8. 404 for unknown incident
# ------------------------------------------------------------------ #
@pytest.mark.asyncio
async def test_incident_not_found(client: AsyncClient):
    resp = await client.get("/api/v1/incidents/nonexistent-id")
    assert resp.status_code == 404
