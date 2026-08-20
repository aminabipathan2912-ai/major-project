"""
backend/app/api/routes/sensor.py
Sensor data ingestion endpoint — Phase 1 stub.
Full sensor pipeline implemented in Phase 5.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.schemas.common import ModalityPrediction, Modality

router = APIRouter(prefix="/sensor", tags=["sensor"])


class SensorReading(BaseModel):
    """Sensor reading payload from IoT device or simulator."""
    temperature: Optional[float] = Field(None, description="Temperature in °C")
    smoke_level: Optional[float] = Field(None, ge=0.0, le=1.0, description="Smoke density 0–1")
    pir_motion: Optional[bool] = Field(None, description="PIR motion sensor triggered")
    humidity: Optional[float] = Field(None, ge=0.0, le=100.0, description="Relative humidity %")
    location: Optional[str] = Field(None, description="Sensor zone / camera zone label")
    device_id: Optional[str] = Field(None, description="IoT device identifier")


@router.post("/readings", response_model=ModalityPrediction)
async def ingest_sensor_readings(reading: SensorReading):
    """
    Accept sensor readings and return anomaly detection result.
    Phase 1: returns a mock response. Full sensor pipeline in Phase 5.
    """
    return ModalityPrediction(
        modality=Modality.SENSOR,
        event="NORMAL",
        confidence=1.0,
        timestamp=datetime.now(timezone.utc),
        evidence=[reading.model_dump(exclude_none=True)],
        status="active",
        raw_scores={"NORMAL": 1.0, "MOTION_ANOMALY": 0.0, "SMOKE_ANOMALY": 0.0, "TEMPERATURE_ANOMALY": 0.0},
        model_name="IsolationForest-anomaly-detector",
        model_version="1.0.0",
    )
