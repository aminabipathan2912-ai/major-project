"""
backend/app/services/alert_service.py
Business logic for alert creation and acknowledgement.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.alert import Alert
from app.models.incident import Incident
from app.schemas.common import Severity

logger = get_logger(__name__)


class AlertService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def create_alert(self, incident: Incident) -> Alert:
        """Generate an alert for a confirmed incident."""
        message = self._build_message(incident)
        alert = Alert(
            incident_id=incident.id,
            severity=incident.severity,
            message=message,
            status="SENT",
        )
        self.db.add(alert)
        await self.db.flush()
        logger.warning(
            "ALERT SENT | severity=%s | event=%s | risk=%.2f | incident=%s",
            alert.severity, incident.event_type, incident.risk_score, incident.id,
        )
        return alert

    def _build_message(self, incident: Incident) -> str:
        breakdown = incident.fusion_breakdown or {}
        lines = [
            f"🚨 {incident.severity} — {incident.event_type.upper().replace('_', ' ')}",
            f"Risk score: {incident.risk_score:.0%}",
            f"Location: {incident.location or 'Unknown'}",
            f"Time: {incident.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
            "Contributing evidence:",
        ]
        for modality, score in breakdown.items():
            if score is not None:
                lines.append(f"  • {modality.title()}: {score:.0%}")
        if incident.description:
            lines.append(f"\nRationale: {incident.description}")
        lines.append("\n⚠️  AI-generated detection. Human verification required.")
        return "\n".join(lines)

    async def list_alerts(
        self,
        limit: int = 50,
        offset: int = 0,
        status: Optional[str] = None,
    ) -> List[Alert]:
        stmt = select(Alert).order_by(desc(Alert.sent_at)).offset(offset).limit(limit)
        if status:
            stmt = stmt.where(Alert.status == status)
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def acknowledge(self, alert_id: str) -> Optional[Alert]:
        result = await self.db.execute(select(Alert).where(Alert.id == alert_id))
        alert = result.scalar_one_or_none()
        if not alert:
            return None
        alert.status = "ACKNOWLEDGED"
        alert.acknowledged_at = datetime.now(timezone.utc)
        await self.db.flush()
        return alert
