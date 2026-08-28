from __future__ import annotations

import os
import time

import numpy as np

from ...core.models import ModelPrediction, ModelStatus
from ..base import ClipInput, InferenceModel
from . import features as feat
from .audio_buffer import AudioChunk
from .audio_classifier import load_audio_checkpoint, predict_waveform


class AudioModelAdapter(InferenceModel):
    """
    Audio event detection over raw PCM captured by the phone browser.

    Mirrors the video adapters exactly: it loads an EfficientNet-B0 checkpoint,
    reports `loaded=False` with a reason when the weights are absent, and never
    invents a prediction. `predict_audio` is the entry point; `predict(clip)`
    exists only to satisfy the shared `InferenceModel` contract and always
    returns None, because a video clip carries no audio.
    """

    def __init__(self, *, weights_path: str = "") -> None:
        self._weights_path = (weights_path or "").strip()
        self._loaded_model = None
        self._last_error = ""
        self._updated_at = time.time()
        self._loaded = False
        self._reason = ""
        self._version = None
        self._try_load()

    def _try_load(self) -> None:
        if not self._weights_path:
            self._reason = (
                "Audio model not loaded (AUDIO_MODEL_WEIGHTS_PATH is empty). "
                "Train one with training/audio/train_audio.ipynb."
            )
            return
        if not os.path.exists(self._weights_path):
            self._reason = f"Audio model not loaded (file not found: {self._weights_path})."
            return
        try:
            self._loaded_model = load_audio_checkpoint(self._weights_path)
            self._loaded = True
            self._reason = ""
            self._version = self._loaded_model.arch
        except Exception as e:
            self._loaded = False
            self._loaded_model = None
            self._last_error = str(e)
            self._reason = f"Audio model failed to load: {e}"

    @property
    def model_name(self) -> str:
        return "audio"

    def status(self) -> ModelStatus:
        return ModelStatus(
            model_name=self.model_name,
            loaded=self._loaded,
            reason=self._reason,
            last_error=self._last_error,
            model_version=self._version,
            updated_at_epoch_s=self._updated_at,
        )

    def predict(self, clip: ClipInput) -> ModelPrediction | None:
        """Video clips carry no audio; the audio path is `predict_audio`."""
        return None

    def predict_audio(
        self, chunks: list[AudioChunk], *, camera_id: str = ""
    ) -> ModelPrediction | None:
        """
        Classify the most recent audio. Returns None when unloaded or silent.

        Chunks are concatenated so a scream spanning a chunk boundary is not
        split in half, then the trailing `clip_seconds` are classified.
        """
        if not self._loaded or self._loaded_model is None or not chunks:
            return None

        try:
            wave = self._concat(chunks)
            if wave.size == 0:
                return None
            label, confidence, metadata = predict_waveform(self._loaded_model, wave)
        except Exception as e:
            self._last_error = str(e)
            self._updated_at = time.time()
            return None

        self._updated_at = time.time()
        return ModelPrediction(
            model_name=self.model_name,
            predicted_label=label,
            confidence=confidence,
            timestamp_epoch_s=chunks[-1].received_epoch_s,
            camera_id=camera_id or "",
            metadata=metadata,
        )

    def _concat(self, chunks: list[AudioChunk]) -> np.ndarray:
        sample_rate = self._loaded_model.sample_rate if self._loaded_model else feat.SAMPLE_RATE
        seconds = self._loaded_model.clip_seconds if self._loaded_model else feat.CLIP_SECONDS
        waves = [feat.pcm_int16_to_float(c.data) for c in chunks if c.data]
        if not waves:
            return np.zeros(0, dtype=np.float32)
        wave = np.concatenate(waves)
        keep = int(round(seconds * sample_rate))
        return wave[-keep:] if wave.size > keep else wave
