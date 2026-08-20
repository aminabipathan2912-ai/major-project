"""
backend/app/api/routes/alerts.py
Alert read and acknowledgement endpoints.
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.alert import AlertRead, AlertAcknowledge
from app.services.alert_service import AlertService

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.get("", response_model=List[AlertRead])
async def list_alerts(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    status: Optional[str] = Query(None, description="Filter by status: SENT|ACKNOWLEDGED|RESOLVED"),
    db: AsyncSession = Depends(get_db),
):
    """Return recent alerts, newest first."""
    svc = AlertService(db)
    return await svc.list_alerts(limit=limit, offset=offset, status=status)


@router.patch("/{alert_id}/acknowledge", response_model=AlertRead)
async def acknowledge_alert(alert_id: str, db: AsyncSession = Depends(get_db)):
    """Mark an alert as acknowledged."""
    svc = AlertService(db)
    alert = await svc.acknowledge(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    return alert
