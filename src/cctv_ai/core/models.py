from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal


EventType = Literal["ACCIDENT", "VIOLENCE", "AUDIO_EVENT"]


@dataclass(frozen=True)
class ModelPrediction:
    """
    Common model output contract (must be stable across model training iterations).
    """

    model_name: str  # e.g. "accident" or "violence"
    predicted_label: str  # e.g. "ACCIDENT" / "NORMAL" or "VIOLENCE" / "NORMAL"
    confidence: float  # 0..1
    timestamp_epoch_s: float  # time.time() in seconds for the *clip* start
    camera_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VerifiedEvent:
    event_type: EventType
    verified_label: str
    confidence: float
    timestamp_epoch_s: float
    camera_id: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelStatus:
    model_name: str
    loaded: bool
    reason: str = ""
    last_error: str = ""
    # Useful later when you add versioning for trained weights.
    model_version: str | None = None
    updated_at_epoch_s: float = field(default_factory=lambda: time.time())

