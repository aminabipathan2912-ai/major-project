from typing import Any, Literal

from pydantic import BaseModel, Field


class VerifiedEventIn(BaseModel):
    event_key: str = Field(min_length=8, max_length=255)
    event_type: Literal["ACCIDENT", "VIOLENCE"]
    confidence: float = Field(ge=0, le=1)
    camera_id: str = Field(min_length=1, max_length=120)
    location: str = Field(min_length=1, max_length=500)
    timestamp_epoch_s: float
    details: dict[str, Any] = Field(default_factory=dict)
