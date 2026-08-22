from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Camera source
    # Default to pre-recorded file for low-cost local development.
    CAMERA_SOURCE_TYPE: Literal["webcam", "rtsp", "file"] = "file"
    CAMERA_SOURCE: str = "tests/fixtures/sample.mp4"
    CAMERA_ID: str = "camera-1"

    # Frame buffering / inference cadence
    INFERENCE_INTERVAL_MS: int = 1000
    CLIP_FRAME_COUNT: int = 8
    FRAME_BUFFER_MAXLEN: int = 128

    # Accident verification
    ACCIDENT_CONFIDENCE_THRESHOLD: float = 0.6
    ACCIDENT_MIN_HITS: int = 3
    ACCIDENT_WINDOW_SEC: float = 3.0
    ACCIDENT_COOLDOWN_SEC: float = 30.0

    # Violence verification
    VIOLENCE_CONFIDENCE_THRESHOLD: float = 0.6
    VIOLENCE_MIN_HITS: int = 2
    VIOLENCE_WINDOW_SEC: float = 3.0
    VIOLENCE_COOLDOWN_SEC: float = 20.0

    # Model weights from Kaggle notebooks (copy .pt files into models/).
    ACCIDENT_MODEL_WEIGHTS_PATH: str = "models/accident_best.pt"
    VIOLENCE_MODEL_WEIGHTS_PATH: str = "models/violence_best.pt"

    # Emergency integration abstraction
    EMERGENCY_MODE: Literal["log-only"] = "log-only"


def get_settings() -> Settings:
    return Settings()

