"""
backend/app/models/prediction.py
SQLAlchemy ORM model for individual modality model predictions.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Enum, Float, ForeignKey, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ModelPrediction(Base):
    __tablename__ = "model_predictions"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    incident_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("incidents.id", ondelete="SET NULL"), nullable=True, index=True
    )
    modality: Mapped[str] = mapped_column(
        Enum("video", "audio", "text", "sensor", name="modality_enum"),
        nullable=False,
        index=True,
    )
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    model_version: Mapped[str] = mapped_column(String(32), nullable=False, default="1.0.0")
    event_label: Mapped[str] = mapped_column(String(64), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    raw_output: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, index=True
    )
    source_ref: Mapped[str | None] = mapped_column(String(512), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")

    # Relationships
    incident: Mapped["Incident | None"] = relationship(  # noqa: F821
        "Incident", back_populates="predictions"
    )

    def __repr__(self) -> str:
        return (
            f"<ModelPrediction modality={self.modality} "
            f"event={self.event_label} conf={self.confidence:.3f}>"
        )
