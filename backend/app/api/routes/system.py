"""
backend/app/api/routes/system.py
System health, model status, and metrics endpoints.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import get_db
from app.schemas.common import HealthResponse
from app.services.incident_service import IncidentService

router = APIRouter(prefix="/system", tags=["system"])
settings = get_settings()

_startup_time = datetime.now(timezone.utc)


@router.get("/status", response_model=HealthResponse)
async def system_status(db: AsyncSession = Depends(get_db)):
    """Return system health and basic stats."""
    svc = IncidentService(db)
    stats = await svc.get_stats()
    uptime_sec = (datetime.now(timezone.utc) - _startup_time).total_seconds()
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        timestamp=datetime.now(timezone.utc),
        demo_mode=settings.DEMO_MODE,
        message=(
            f"System operational. Uptime: {uptime_sec:.0f}s. "
            f"Total incidents: {stats['total']}. Active: {stats['active']}."
        ),
    )


@router.get("/models/status")
async def model_status():
    """
    Return the status of all registered modality models.
    Phase 1: returns config-driven status; models load in Phase 2–5.
    """
    import yaml
    from pathlib import Path

    config_path = settings.configs_dir / "models.yaml"
    with open(config_path) as f:
        model_cfg = yaml.safe_load(f)

    models_info = {}
    for modality, info in model_cfg.get("models", {}).items():
        models_info[modality] = {
            "name": info.get("name"),
            "version": info.get("version"),
            "status": info.get("status", "PLANNED"),
            "supported_events": info.get("supported_events", []),
            "threshold": info.get("threshold"),
        }
    return {"models": models_info, "timestamp": datetime.now(timezone.utc)}


@router.get("/metrics")
async def system_metrics(db: AsyncSession = Depends(get_db)):
    """
    Return high-level detection metrics.
    Phase 1: placeholder — full metrics available in Phase 10.
    """
    svc = IncidentService(db)
    stats = await svc.get_stats()
    return {
        "note": "Full precision/recall/F1 metrics available in Phase 10 (Evaluation).",
        "incident_stats": stats,
        "timestamp": datetime.now(timezone.utc),
    }
