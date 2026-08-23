from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    CAMERA_SOURCE_TYPE: Literal["webcam", "rtsp", "file"] = "file"
    CAMERA_SOURCE: str = "tests/fixtures/sample.mp4"
    CAMERA_ID: str = "camera-1"
    INCIDENT_LOCATION: str = "Main Road, test site"
    # Uploaded/test clips should behave like a camera feed: play once at the
    # source FPS instead of racing through and repeatedly retriggering alerts.
    FILE_LOOP: bool = False
    FILE_REALTIME: bool = True

    INFERENCE_INTERVAL_MS: int = 1000
    CLIP_FRAME_COUNT: int = 8
    # 32 comfortably retains the 8-frame inference clip without keeping a
    # large number of full-resolution OpenCV frames in RAM.
    FRAME_BUFFER_MAXLEN: int = 32

    ACCIDENT_CONFIDENCE_THRESHOLD: float = 0.6
    ACCIDENT_MIN_HITS: int = 3
    ACCIDENT_WINDOW_SEC: float = 3.0
    ACCIDENT_COOLDOWN_SEC: float = 30.0

    VIOLENCE_CONFIDENCE_THRESHOLD: float = 0.6
    VIOLENCE_MIN_HITS: int = 2
    VIOLENCE_WINDOW_SEC: float = 3.0
    VIOLENCE_COOLDOWN_SEC: float = 20.0

    ACCIDENT_MODEL_WEIGHTS_PATH: str = "models/accident_best.pt"
    VIOLENCE_MODEL_WEIGHTS_PATH: str = "models/violence_best.pt"

    # remote posts verified events to the lightweight hosted alert backend.
    EMERGENCY_MODE: Literal["log-only", "remote"] = "log-only"
    ALERT_BACKEND_URL: str = ""
    ALERT_BACKEND_TOKEN: str = ""

    VIDEO_UPLOAD_DIR: str = "data/uploads"


def get_settings() -> Settings:
    return Settings()
