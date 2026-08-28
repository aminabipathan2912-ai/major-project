from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.models import ModelPrediction, ModelStatus


@dataclass(frozen=True)
class ClipInput:
    """
    Temporal input contract for all video models.
    """

    frames_bgr: list[np.ndarray]
    frame_timestamps_epoch_s: list[float]
    clip_start_time_epoch_s: float
    camera_id: str


class InferenceModel(abc.ABC):
    """
    Common model interface:
    input → prediction → confidence + event metadata
    """

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def status(self) -> ModelStatus:
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, clip: ClipInput) -> ModelPrediction | None:
        """
        Returns `None` when the model is unavailable or can not produce predictions.
        """

    # ------------------------------------------------------------------
    # Optional shared-tensor fast path
    #
    # When several models sample the same number of frames and use the same
    # preprocessing, the caller can sample + preprocess one clip and hand the
    # resulting batch tensor to each model, instead of every model repeating
    # that work. Models that do not support it inherit the defaults below and
    # the caller falls back to `predict(clip)`.
    # ------------------------------------------------------------------

    @property
    def clip_num_frames(self) -> int | None:
        """Frames this model samples per clip, or `None` if not loaded / N/A."""
        return None

    def preprocess_clip(self, clip: ClipInput):
        """Return a sampled + preprocessed batch tensor, or `None` if unsupported."""
        return None

    def predict_preprocessed(self, batch, clip: ClipInput) -> ModelPrediction | None:
        """
        Predict from a batch produced by `preprocess_clip` (possibly by a
        different model instance). Raises `NotImplementedError` when unsupported.
        """
        raise NotImplementedError

