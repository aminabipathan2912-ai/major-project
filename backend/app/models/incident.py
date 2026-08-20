"""
backend/app/models/incident.py
SQLAlchemy ORM model for public-safety incidents.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Enum, Float, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Incident(Base):
    __tablename__ = "incidents"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    event_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    severity: Mapped[str] = mapped_column(
        Enum("LOW", "MEDIUM", "HIGH", "CRITICAL", name="severity_enum"),
        nullable=False,
        default="LOW",
    )
    risk_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, index=True
    )
    location: Mapped[str | None] = mapped_column(String(128), nullable=True)
    status: Mapped[str] = mapped_column(
        Enum("ACTIVE", "ACKNOWLEDGED", "RESOLVED", "FALSE_ALARM", name="incident_status_enum"),
        nullable=False,
        default="ACTIVE",
    )
    contributing_modalities: Mapped[list | None] = mapped_column(JSON, nullable=True)
    fusion_breakdown: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )

    # Relationships
    predictions: Mapped[list["ModelPrediction"]] = relationship(  # noqa: F821
        "ModelPrediction", back_populates="incident", cascade="all, delete-orphan"
    )
    evidence_items: Mapped[list["Evidence"]] = relationship(  # noqa: F821
        "Evidence", back_populates="incident", cascade="all, delete-orphan"
    )
    alerts: Mapped[list["Alert"]] = relationship(  # noqa: F821
        "Alert", back_populates="incident", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Incident id={self.id} event={self.event_type} severity={self.severity}>"
