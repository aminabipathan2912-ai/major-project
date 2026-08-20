"""
backend/app/schemas/common.py
Shared Pydantic types and base classes used across all schemas.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ------------------------------------------------------------------ #
# Enumerations (mirrors DB enums but lives in schemas layer)
# ------------------------------------------------------------------ #

class Modality(str, Enum):
    VIDEO = "video"
    AUDIO = "audio"
    TEXT = "text"
    SENSOR = "sensor"


class Severity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class IncidentStatus(str, Enum):
    ACTIVE = "ACTIVE"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"
    FALSE_ALARM = "FALSE_ALARM"


class AlertStatus(str, Enum):
    SENT = "SENT"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"


# ------------------------------------------------------------------ #
# Standardised modality prediction output
# This is the data contract every modality model must produce.
# ------------------------------------------------------------------ #

class ModalityPrediction(BaseModel):
    """
    Standardised output from any modality model.
    This is the input to the multimodal fusion engine.
    """
    model_config = ConfigDict(
        from_attributes=True,
        protected_namespaces=(),   # allow model_name / model_version fields
    )

    modality: Modality
    event: str = Field(..., description="Detected event label, e.g. 'fire', 'gunshot'")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence 0–1")
    timestamp: datetime
    evidence: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Modality-specific evidence: bboxes, mel paths, etc.",
    )
    status: str = Field(
        default="active",
        description="'active' | 'no_event' | 'error'",
    )
    raw_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Full per-class probability dict from the model",
    )
    model_name: str
    model_version: str = "1.0.0"


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime
    demo_mode: bool
    message: str
