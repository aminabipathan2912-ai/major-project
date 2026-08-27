from __future__ import annotations

from functools import lru_cache
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

    # Frames are downscaled once at ingest to the shorter side the training
    # transform resizes to anyway, so the ring buffer holds ~20x less at 1080p
    # without changing the tensor the models receive. See
    # scripts/check_preprocess_equivalence.py.
    INFERENCE_FRAME_SIZE: int = 256
    # True uses PIL BICUBIC, which is bit-identical to the training transform
    # (EfficientNet_B0_Weights.DEFAULT.transforms() resizes with BICUBIC).
    # False uses cv2 INTER_AREA: faster, but only enable it after the
    # equivalence script reports an acceptable difference on your own clips.
    INGEST_RESIZE_EXACT: bool = True
    # Frames per second pushed into the inference ring. 0 disables decimation
    # and keeps every decoded frame, which reproduces the pre-optimization clip
    # exactly: snapshot_last_n(8) then spans the same fraction of a second it
    # does today. A non-zero value widens that span (fewer near-duplicate
    # frames per clip) and is a Phase 3 / B7 decision, not a Phase 1 one.
    INGEST_SAMPLE_FPS: float = 0.0

    # Browser preview only. No model ever sees these pixels, so they are
    # encoded once per new frame and shared by every connected client.
    PREVIEW_FPS: float = 10.0
    PREVIEW_MAX_WIDTH: int = 640
    PREVIEW_JPEG_QUALITY: int = 80

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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached: constructing Settings re-reads and re-parses .env every time."""
    return Settings()
