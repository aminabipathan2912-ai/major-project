"""
backend/app/models/__init__.py
Import all ORM models so SQLAlchemy metadata is populated before create_all().
"""

from app.models.incident import Incident
from app.models.prediction import ModelPrediction
from app.models.evidence import Evidence
from app.models.alert import Alert

__all__ = ["Incident", "ModelPrediction", "Evidence", "Alert"]
