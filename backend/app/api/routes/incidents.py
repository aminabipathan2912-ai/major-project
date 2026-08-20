"""
backend/app/api/routes/incidents.py
Incident CRUD endpoints.
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.incident import IncidentRead, IncidentUpdate, IncidentDetail
from app.services.incident_service import IncidentService

router = APIRouter(prefix="/incidents", tags=["incidents"])


@router.get("", response_model=List[IncidentRead])
async def list_incidents(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    severity: Optional[str] = Query(None, description="Filter by severity: LOW|MEDIUM|HIGH|CRITICAL"),
    status: Optional[str] = Query(None, description="Filter by status: ACTIVE|ACKNOWLEDGED|RESOLVED|FALSE_ALARM"),
    db: AsyncSession = Depends(get_db),
):
    """Return a paginated list of incidents, newest first."""
    svc = IncidentService(db)
    return await svc.list_incidents(limit=limit, offset=offset, severity=severity, status=status)


@router.get("/{incident_id}", response_model=IncidentDetail)
async def get_incident(incident_id: str, db: AsyncSession = Depends(get_db)):
    """Return a single incident with all evidence and alerts."""
    svc = IncidentService(db)
    incident = await svc.get_incident(incident_id)
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    return incident


@router.patch("/{incident_id}/status", response_model=IncidentRead)
async def update_incident_status(
    incident_id: str,
    update: IncidentUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Acknowledge, resolve, or mark an incident as false alarm."""
    svc = IncidentService(db)
    incident = await svc.update_status(incident_id, update)
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    return incident
