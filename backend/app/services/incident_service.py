"""
backend/app/services/incident_service.py
Business logic for incident creation, retrieval, and status management.
All DB calls are async.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.logging import get_logger
from app.models.incident import Incident
from app.models.prediction import ModelPrediction
from app.schemas.incident import IncidentCreate, IncidentUpdate
from app.schemas.fusion import FusionResult

logger = get_logger(__name__)


class IncidentService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    # ------------------------------------------------------------------ #
    # Create
    # ------------------------------------------------------------------ #
    async def create_from_fusion(self, result: FusionResult, location: Optional[str] = None) -> Incident:
        """Create and persist an Incident from a FusionResult."""
        incident = Incident(
            event_type=result.event_type,
            severity=result.severity.value,
            risk_score=result.risk_score,
            timestamp=result.timestamp,
            location=location,
            status="ACTIVE",
            contributing_modalities=result.contributing_modalities,
            fusion_breakdown=result.fusion_breakdown,
            description=result.explanation,
        )
        self.db.add(incident)
        await self.db.flush()   # get the generated ID without committing
        logger.info(
            "Incident created: id=%s event=%s severity=%s risk=%.3f",
            incident.id, incident.event_type, incident.severity, incident.risk_score,
        )
        return incident

    async def create(self, data: IncidentCreate) -> Incident:
        """Create an incident directly from schema (e.g. during testing)."""
        incident = Incident(**data.model_dump(exclude_none=True))
        self.db.add(incident)
        await self.db.flush()
        return incident

    # ------------------------------------------------------------------ #
    # Read
    # ------------------------------------------------------------------ #
    async def list_incidents(
        self,
        limit: int = 50,
        offset: int = 0,
        severity: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Incident]:
        stmt = (
            select(Incident)
            .order_by(desc(Incident.timestamp))
            .offset(offset)
            .limit(limit)
        )
        if severity:
            stmt = stmt.where(Incident.severity == severity)
        if status:
            stmt = stmt.where(Incident.status == status)
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def get_incident(self, incident_id: str) -> Optional[Incident]:
        stmt = (
            select(Incident)
            .where(Incident.id == incident_id)
            .options(
                selectinload(Incident.predictions),
                selectinload(Incident.evidence_items),
                selectinload(Incident.alerts),
            )
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    # ------------------------------------------------------------------ #
    # Update
    # ------------------------------------------------------------------ #
    async def update_status(self, incident_id: str, update: IncidentUpdate) -> Optional[Incident]:
        incident = await self.get_incident(incident_id)
        if not incident:
            return None
        if update.status is not None:
            incident.status = update.status.value
        if update.description is not None:
            incident.description = update.description
        incident.updated_at = datetime.now(timezone.utc)
        await self.db.flush()
        return incident

    # ------------------------------------------------------------------ #
    # Stats (for dashboard analytics)
    # ------------------------------------------------------------------ #
    async def get_stats(self) -> dict:
        from sqlalchemy import func
        total = await self.db.scalar(select(func.count()).select_from(Incident))
        active = await self.db.scalar(
            select(func.count()).select_from(Incident).where(Incident.status == "ACTIVE")
        )
        return {"total": total or 0, "active": active or 0}
