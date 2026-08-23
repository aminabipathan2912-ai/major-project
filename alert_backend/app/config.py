from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    DATABASE_URL: str
    ALERT_INGEST_TOKEN: str
    PUBLIC_BASE_URL: str

    TWILIO_ACCOUNT_SID: str = ""
    TWILIO_AUTH_TOKEN: str = ""
    TWILIO_FROM_NUMBER: str = ""
    TWILIO_TO_NUMBER: str = ""
    # `trial-template` uses Twilio's permitted trial speech-recognition URL.
    # `custom` is the full Sarvam audio + acknowledgement webhook flow.
    TWILIO_CALL_MODE: Literal["custom", "trial-template"] = "custom"
    TWILIO_TRIAL_TEMPLATE_URL: str = (
        "https://webhooks.twilio.com/v1/Voice/Template/voice_speech_recognition"
    )

    SARVAM_API_KEY: str = ""
    SARVAM_TTS_URL: str = "https://api.sarvam.ai/text-to-speech"
    SARVAM_LANGUAGE_CODE: str = "en-IN"
    SARVAM_SPEAKER: str = "shubh"
    SARVAM_MODEL: str = "bulbul:v3"
    TWILIO_VALIDATE_WEBHOOKS: bool = True


settings = Settings()
