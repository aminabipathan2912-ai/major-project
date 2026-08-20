"""
backend/app/models/evidence.py
SQLAlchemy ORM model for incident evidence (frames, audio clips, text, sensor readings).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Enum, ForeignKey, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Evidence(Base):
    __tablename__ = "evidence"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    incident_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("incidents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    modality: Mapped[str] = mapped_column(
        Enum("video", "audio", "text", "sensor", name="evidence_modality_enum"),
        nullable=False,
    )
    file_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    text_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    sensor_readings: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    captured_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    # Relationships
    incident: Mapped["Incident"] = relationship(  # noqa: F821
        "Incident", back_populates="evidence_items"
    )

    def __repr__(self) -> str:
        return f"<Evidence id={self.id} modality={self.modality}>"
