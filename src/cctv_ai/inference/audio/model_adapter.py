from __future__ import annotations

import time

from ...core.models import ModelPrediction, ModelStatus
from ..base import ClipInput, InferenceModel
from .audio_buffer import AudioChunk


class AudioModelAdapter(InferenceModel):
    """
    Reserved for future audio-event detection layer.

    This stub ensures the pipeline has a clean integration point,
    without assuming any dataset/provider/model yet.

    `predict_audio` is the seam a real classifier drops into: the phone source
    already collects bounded audio chunks and `AUDIO_EVENT` already exists in
    `EventType`, so the escalation path is wired but inert. Nothing here
    fabricates a prediction — it returns `None` until a real model exists.
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

    def predict_audio(self, chunks: list[AudioChunk]) -> ModelPrediction | None:
        """
        Classify recent audio. Returns `None` until a real model is wired in.

        A future implementation decodes `chunk.data` (per `chunk.mime_type`),
        runs its classifier, and returns a `ModelPrediction` with
        `model_name="audio"`; the existing verifier + emergency path then handles
        it exactly like the video models.
        """
        self._updated_at = time.time()
        return None
