from __future__ import annotations

import os
import time

from ...core.models import ModelPrediction, ModelStatus
from ..base import ClipInput, InferenceModel
from .text_classifier import load_text_checkpoint, predict_text


class TextModelAdapter(InferenceModel):
    """
    Text modality: classifies short incident reports (helpline text, social posts)
    into the same event vocabulary the video models use.

    Same contract as every other adapter: loads a checkpoint, reports
    `loaded=False` with a reason when absent, never fabricates a prediction.
    `predict_message` is the entry point; `predict(clip)` returns None because a
    video clip carries no text.
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
                "Text model not loaded (TEXT_MODEL_WEIGHTS_PATH is empty). "
                "Train one with training/text/train_text.ipynb."
            )
            return
        if not os.path.exists(self._weights_path):
            self._reason = f"Text model not loaded (file not found: {self._weights_path})."
            return
        try:
            self._loaded_model = load_text_checkpoint(self._weights_path)
            self._loaded = True
            self._reason = ""
            self._version = self._loaded_model.arch
        except Exception as e:
            self._loaded = False
            self._loaded_model = None
            self._last_error = str(e)
            self._reason = f"Text model failed to load: {e}"

    @property
    def model_name(self) -> str:
        return "text"

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
        """Video clips carry no text; the text path is `predict_message`."""
        return None

    def predict_message(
        self, text: str, *, camera_id: str = ""
    ) -> ModelPrediction | None:
        if not self._loaded or self._loaded_model is None:
            return None
        if not (text or "").strip():
            return None
        try:
            label, confidence, metadata = predict_text(self._loaded_model, text)
        except Exception as e:
            self._last_error = str(e)
            self._updated_at = time.time()
            return None

        self._updated_at = time.time()
        return ModelPrediction(
            model_name=self.model_name,
            predicted_label=label,
            confidence=confidence,
            timestamp_epoch_s=time.time(),
            camera_id=camera_id or "",
            metadata=metadata,
        )
