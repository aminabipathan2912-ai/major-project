"""
backend/app/api/routes/fusion.py
Multimodal fusion endpoint — Phase 1 stub with Level 1 weighted fusion.
Full fusion engine (Level 2 + Level 3) implemented in Phase 6.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.logging import get_logger
from app.schemas.common import Severity
from app.schemas.fusion import FusionRequest, FusionResult
from app.services.alert_service import AlertService
from app.services.incident_service import IncidentService

router = APIRouter(prefix="/fusion", tags=["fusion"])
logger = get_logger(__name__)

# Phase 1: simple weighted late fusion weights
_WEIGHTS = {"video": 0.35, "audio": 0.25, "text": 0.20, "sensor": 0.20}
_NO_EVENT_LABELS = {"no_event", "NORMAL", "NON_EMERGENCY", ""}


def _weighted_fusion(request: FusionRequest) -> FusionResult:
    """Level 1: weighted average late fusion."""
    modalities = {
        "video": request.video,
        "audio": request.audio,
        "text": request.text,
        "sensor": request.sensor,
    }

    weighted_sum = 0.0
    total_weight = 0.0
    contributing = []
    breakdown: dict = {}

    for name, pred in modalities.items():
        if pred is None:
            breakdown[name] = None
            continue
        if pred.event in _NO_EVENT_LABELS or pred.confidence < 0.1:
            breakdown[name] = 0.0
            continue
        w = _WEIGHTS[name]
        weighted_sum += pred.confidence * w
        total_weight += w
        contributing.append(name)
        breakdown[name] = round(pred.confidence, 4)

    risk_score = weighted_sum / total_weight if total_weight > 0 else 0.0

    # Derive event type from highest-confidence modality
    event_type = "NO_EVENT"
    best_conf = 0.0
    for name, pred in modalities.items():
        if pred and pred.confidence > best_conf and pred.event not in _NO_EVENT_LABELS:
            best_conf = pred.confidence
            event_type = pred.event.upper()

    # Severity mapping
    if risk_score >= 0.85:
        severity = Severity.CRITICAL
    elif risk_score >= 0.65:
        severity = Severity.HIGH
    elif risk_score >= 0.45:
        severity = Severity.MEDIUM
    else:
        severity = Severity.LOW

    create_incident = risk_score >= 0.45 and len(contributing) >= 1

    explanation_parts = []
    for name, pred in modalities.items():
        if breakdown.get(name) and breakdown[name] > 0:
            explanation_parts.append(
                f"{name.title()}: {pred.event} {breakdown[name]:.0%}"
            )
    explanation = (
        f"{severity.value} {event_type.replace('_', ' ')} | "
        + " | ".join(explanation_parts)
        + f" → Final risk: {risk_score:.0%}"
        + "\n⚠️  AI-generated detection. Human verification required."
    )

    return FusionResult(
        event_type=event_type,
        severity=severity,
        risk_score=round(risk_score, 4),
        confidence=round(risk_score, 4),
        contributing_modalities=contributing,
        fusion_breakdown=breakdown,
        explanation=explanation,
        timestamp=datetime.now(timezone.utc),
        create_incident=create_incident,
        fusion_level_used="weighted_late_fusion",
    )


@router.post("/predict", response_model=FusionResult)
async def fusion_predict(
    request: FusionRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Run multimodal fusion on predictions from one or more modalities.
    Creates an incident and alert if the risk score exceeds threshold.
    Phase 1: Level 1 weighted late fusion. Rule engine added in Phase 6.
    """
    result = _weighted_fusion(request)
    logger.info(
        "Fusion result: event=%s severity=%s risk=%.3f modalities=%s",
        result.event_type, result.severity, result.risk_score, result.contributing_modalities,
    )

    if result.create_incident:
        incident_svc = IncidentService(db)
        alert_svc = AlertService(db)
        incident = await incident_svc.create_from_fusion(result, location=request.location)
        await alert_svc.create_alert(incident)

    return result
