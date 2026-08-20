"""
backend/app/schemas/alert.py
Pydantic schemas for Alert operations.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict

from app.schemas.common import Severity, AlertStatus


class AlertRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    incident_id: str
    severity: Severity
    message: str
    status: AlertStatus
    sent_at: datetime
    acknowledged_at: Optional[datetime] = None


class AlertAcknowledge(BaseModel):
    status: AlertStatus = AlertStatus.ACKNOWLEDGED
