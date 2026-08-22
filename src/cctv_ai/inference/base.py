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

