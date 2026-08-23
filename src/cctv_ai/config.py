from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    CAMERA_SOURCE_TYPE: Literal["webcam", "rtsp", "file"] = "file"
    CAMERA_SOURCE: str = "tests/fixtures/sample.mp4"
    CAMERA_ID: str = "camera-1"
    INCIDENT_LOCATION: str = "Main Road, test site"

    INFERENCE_INTERVAL_MS: int = 1000
    CLIP_FRAME_COUNT: int = 8
    FRAME_BUFFER_MAXLEN: int = 128

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

    # log-only: print only. voice: Sarvam TTS + Twilio call (test numbers only).
    EMERGENCY_MODE: Literal["log-only", "voice"] = "log-only"

    # Public HTTPS origin after you host (no trailing slash). Required for Twilio Play/Gather.
    PUBLIC_BASE_URL: str = ""

    # Twilio
    TWILIO_ACCOUNT_SID: str = ""
    TWILIO_AUTH_TOKEN: str = ""
    TWILIO_FROM_NUMBER: str = ""
    # Single destination phone in E.164, e.g. +91XXXXXXXXXX. Not 100/108/112.
    TWILIO_TO_NUMBER: str = ""
    TWILIO_CALL_RETRY_MAX: int = 2

    # Sarvam TTS (Bulbul)
    SARVAM_API_KEY: str = ""
    SARVAM_TTS_URL: str = "https://api.sarvam.ai/text-to-speech"
    SARVAM_LANGUAGE_CODE: str = "en-IN"
    SARVAM_SPEAKER: str = "shubh"
    SARVAM_MODEL: str = "bulbul:v3"

    INCIDENT_DB_PATH: str = "data/incidents.db"
    INCIDENT_AUDIO_DIR: str = "data/audio"


def get_settings() -> Settings:
    return Settings()
