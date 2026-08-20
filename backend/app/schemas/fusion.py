"""
backend/app/schemas/fusion.py
Pydantic schemas for the multimodal fusion engine I/O.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.common import ModalityPrediction, Severity


class FusionRequest(BaseModel):
    """
    Input to /api/v1/fusion/predict.
    Each modality field is optional — the engine must handle missing modalities.
    """
    video: Optional[ModalityPrediction] = None
    audio: Optional[ModalityPrediction] = None
    text: Optional[ModalityPrediction] = None
    sensor: Optional[ModalityPrediction] = None
    location: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)


class FusionResult(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    event_type: str
    severity: Severity
    risk_score: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    contributing_modalities: List[str]
    fusion_breakdown: Dict[str, Optional[float]]
    explanation: str = Field(..., description="Human-readable rationale for this result")
    timestamp: datetime
    create_incident: bool = Field(
        ..., description="Whether the fusion engine recommends creating an incident"
    )
    fusion_level_used: str = Field(
        default="rule_based_aggregation",
        description="Which fusion level produced this result",
    )
