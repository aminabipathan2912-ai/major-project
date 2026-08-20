"""
backend/app/schemas/incident.py
Pydantic schemas for Incident read/write operations.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.common import Severity, IncidentStatus


class IncidentBase(BaseModel):
    event_type: str
    severity: Severity
    risk_score: float = Field(..., ge=0.0, le=1.0)
    location: Optional[str] = None
    contributing_modalities: Optional[List[str]] = None
    fusion_breakdown: Optional[Dict[str, Any]] = None
    description: Optional[str] = None


class IncidentCreate(IncidentBase):
    timestamp: Optional[datetime] = None


class IncidentUpdate(BaseModel):
    status: Optional[IncidentStatus] = None
    description: Optional[str] = None


class IncidentRead(IncidentBase):
    model_config = ConfigDict(from_attributes=True)

    id: str
    timestamp: datetime
    status: IncidentStatus
    created_at: datetime
    updated_at: datetime


# ------------------------------------------------------------------ #
# Nested summaries for IncidentDetail
# ------------------------------------------------------------------ #

class PredictionSummary(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        protected_namespaces=(),   # allow model_name field
    )

    id: str
    modality: str
    model_name: str
    event_label: str
    confidence: float
    timestamp: datetime


class EvidenceSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    modality: str
    file_path: Optional[str] = None
    text_content: Optional[str] = None
    sensor_readings: Optional[Dict[str, Any]] = None
    captured_at: datetime


class AlertSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    severity: str
    message: str
    status: str
    sent_at: datetime
    acknowledged_at: Optional[datetime] = None


class IncidentDetail(IncidentRead):
    """Full incident with nested evidence and alert summaries."""
    predictions: List[PredictionSummary] = []
    evidence_items: List[EvidenceSummary] = []
    alerts: List[AlertSummary] = []

