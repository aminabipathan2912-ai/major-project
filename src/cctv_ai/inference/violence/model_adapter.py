from __future__ import annotations

import os
import time

from ...core.models import ModelPrediction, ModelStatus
from ..base import ClipInput, InferenceModel
from ..clip_classifier import (
    load_clip_checkpoint,
    predict_from_batch,
    sample_and_preprocess_clip,
)


class ViolenceModelAdapter(InferenceModel):
    def __init__(self, *, weights_path: str, preprocess_fast: bool = False) -> None:
        self._weights_path = (weights_path or "").strip()
        self._preprocess_fast = bool(preprocess_fast)
        self._loaded_model = None
        self._last_error = ""
        self._updated_at = time.time()
        self._loaded = False
        self._reason = ""
        self._version = None
        self._try_load()

    def _try_load(self) -> None:
        if not self._weights_path:
            self._reason = "Violence model not loaded (VIOLENCE_MODEL_WEIGHTS_PATH is empty)."
            return
        if not os.path.exists(self._weights_path):
            self._reason = f"Violence model not loaded (file not found: {self._weights_path})."
            return
        try:
            self._loaded_model = load_clip_checkpoint(self._weights_path, positive_label="VIOLENCE")
            self._loaded = True
            self._reason = ""
            self._version = self._loaded_model.arch
        except Exception as e:
            self._loaded = False
            self._loaded_model = None
            self._last_error = str(e)
            self._reason = f"Violence model failed to load: {e}"

    @property
    def model_name(self) -> str:
        return "violence"

    def status(self) -> ModelStatus:
        return ModelStatus(
            model_name=self.model_name,
            loaded=self._loaded,
            reason=self._reason,
            last_error=self._last_error,
            model_version=self._version,
            updated_at_epoch_s=self._updated_at,
        )

    @property
    def clip_num_frames(self) -> int | None:
        return self._loaded_model.num_frames if self._loaded_model is not None else None

    def preprocess_clip(self, clip: ClipInput):
        if not self._loaded or self._loaded_model is None:
            return None
        return sample_and_preprocess_clip(
            self._loaded_model, clip, fast=self._preprocess_fast
        )

    def predict(self, clip: ClipInput) -> ModelPrediction | None:
        if not self._loaded or self._loaded_model is None:
            return None
        try:
            batch = sample_and_preprocess_clip(
                self._loaded_model, clip, fast=self._preprocess_fast
            )
            label, confidence, metadata = predict_from_batch(self._loaded_model, batch)
        except Exception as e:
            self._last_error = str(e)
            self._updated_at = time.time()
            return None
        return self._to_prediction(label, confidence, metadata, clip)

    def predict_preprocessed(self, batch, clip: ClipInput) -> ModelPrediction | None:
        if not self._loaded or self._loaded_model is None:
            return None
        try:
            label, confidence, metadata = predict_from_batch(self._loaded_model, batch)
        except Exception as e:
            self._last_error = str(e)
            self._updated_at = time.time()
            return None
        return self._to_prediction(label, confidence, metadata, clip)

    def _to_prediction(self, label, confidence, metadata, clip: ClipInput) -> ModelPrediction:
        self._updated_at = time.time()
        return ModelPrediction(
            model_name=self.model_name,
            predicted_label=label,
            confidence=confidence,
            timestamp_epoch_s=clip.clip_start_time_epoch_s,
            camera_id=clip.camera_id,
            metadata=metadata,
        )
