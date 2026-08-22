from __future__ import annotations

import time

from ...core.models import ModelPrediction, ModelStatus
from ..base import ClipInput, InferenceModel


class AudioModelAdapter(InferenceModel):
    """
    Reserved for future audio-event detection layer.

    This stub ensures the pipeline has a clean integration point,
    without assuming any dataset/provider/model yet.
    """

    def __init__(self) -> None:
        self._updated_at = time.time()

    @property
    def model_name(self) -> str:
        return "audio"

    def status(self) -> ModelStatus:
        return ModelStatus(
            model_name=self.model_name,
            loaded=False,
            reason="Audio model reserved (not implemented yet).",
            last_error="",
            updated_at_epoch_s=self._updated_at,
        )

    def predict(self, clip: ClipInput) -> ModelPrediction | None:
        self._updated_at = time.time()
        return None

