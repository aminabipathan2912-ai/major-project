from __future__ import annotations

import os
import time

from ...core.models import ModelPrediction, ModelStatus
from ..base import ClipInput, InferenceModel
from ..clip_classifier import load_clip_checkpoint, predict_clip


class AccidentModelAdapter(InferenceModel):
    def __init__(self, *, weights_path: str) -> None:
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
            self._reason = "Accident model not loaded (ACCIDENT_MODEL_WEIGHTS_PATH is empty)."
            return
        if not os.path.exists(self._weights_path):
            self._reason = f"Accident model not loaded (file not found: {self._weights_path})."
            return
        try:
            self._loaded_model = load_clip_checkpoint(self._weights_path, positive_label="ACCIDENT")
            self._loaded = True
            self._reason = ""
            self._version = self._loaded_model.arch
        except Exception as e:
            self._loaded = False
            self._loaded_model = None
            self._last_error = str(e)
            self._reason = f"Accident model failed to load: {e}"

    @property
    def model_name(self) -> str:
        return "accident"

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
        if not self._loaded or self._loaded_model is None:
            return None
        try:
            label, confidence, metadata = predict_clip(self._loaded_model, clip)
        except Exception as e:
            self._last_error = str(e)
            self._updated_at = time.time()
            return None

        self._updated_at = time.time()
        return ModelPrediction(
            model_name=self.model_name,
            predicted_label=label,
            confidence=confidence,
            timestamp_epoch_s=clip.clip_start_time_epoch_s,
            camera_id=clip.camera_id,
            metadata=metadata,
        )
