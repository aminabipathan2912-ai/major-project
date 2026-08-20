"""
ml/base.py
Abstract base class that every modality model must implement.
Defines the canonical interface between modality models and the fusion engine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Optional

from dataclasses import dataclass, field


@dataclass
class ModalityPredictionData:
    """
    Standardised prediction output from any modality model.
    This is the data contract that feeds into the multimodal fusion engine.
    """
    modality: str            # "video" | "audio" | "text" | "sensor"
    event: str               # detected event label, e.g. "FIRE", "GUNSHOT"
    confidence: float        # 0.0–1.0
    timestamp: datetime      # detection time (UTC)
    evidence: list           # modality-specific evidence
    status: str              # "active" | "no_event" | "error"
    raw_scores: dict         # full class probability dict
    model_name: str
    model_version: str = "1.0.0"


class BaseModalityModel(ABC):
    """
    Every modality model (video, audio, text, sensor) must subclass this.
    This enforces the interface contract required by the fusion engine.
    """

    modality: str                    # must be set by subclass
    model_name: str                  # human-readable model name
    model_version: str = "1.0.0"
    supported_events: list[str] = [] # events this model can detect

    @abstractmethod
    def load(self) -> None:
        """Load model weights and prepare for inference."""
        ...

    @abstractmethod
    def preprocess(self, raw_input: Any) -> Any:
        """Transform raw input into model-ready tensor/features."""
        ...

    @abstractmethod
    def predict(self, preprocessed: Any) -> ModalityPredictionData:
        """Run inference and return standardised prediction."""
        ...

    @property
    def is_ready(self) -> bool:
        """Return True if the model is loaded and ready for inference."""
        raise NotImplementedError

    def predict_raw(self, raw_input: Any) -> ModalityPredictionData:
        """
        Full pipeline: preprocess → predict.
        Subclasses should override predict() not this method.
        Returns an error prediction if inference fails.
        """
        try:
            preprocessed = self.preprocess(raw_input)
            return self.predict(preprocessed)
        except Exception as exc:
            return ModalityPredictionData(
                modality=self.modality,
                event="error",
                confidence=0.0,
                timestamp=datetime.now(timezone.utc),
                evidence=[],
                status="error",
                raw_scores={},
                model_name=self.model_name,
                model_version=self.model_version,
            )

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} "
            f"modality={self.modality} "
            f"model={self.model_name} "
            f"ready={self.is_ready}>"
        )
