from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

from ..core.models import ModelPrediction, VerifiedEvent


@dataclass(frozen=True)
class TemporalVerificationConfig:
    confidence_threshold: float
    min_hits: int
    window_sec: float
    cooldown_sec: float
    positive_label: str  # e.g. "ACCIDENT" or "VIOLENCE"


class TemporalVerifier:
    """
    Temporal/event verification layer.

    This layer never triggers on a single uncertain frame:
    - Requires persistence across multiple frame/clip predictions (`min_hits`)
    - Uses a sliding time window (`window_sec`)
    - Applies cooldown (`cooldown_sec`) to avoid repeated emergency triggers
    """

    def __init__(self, *, config: TemporalVerificationConfig) -> None:
        self._config = config
        self._hits: deque[tuple[float, float]] = deque()  # (timestamp, confidence)
        self._last_verified_epoch_s: float | None = None

    @property
    def config(self) -> TemporalVerificationConfig:
        return self._config

    @property
    def last_verified_epoch_s(self) -> float | None:
        return self._last_verified_epoch_s

    def update(self, prediction: ModelPrediction) -> VerifiedEvent | None:
        """
        Returns a `VerifiedEvent` when the verification criteria are met.
        Otherwise returns None.
        """

        now = time.time()
        is_positive = prediction.predicted_label == self._config.positive_label
        is_confident = prediction.confidence >= self._config.confidence_threshold

        if is_positive and is_confident:
            # A file source can decode an entire short clip much faster than real
            # time. Its frame timestamps may therefore be old by the time a CPU
            # model finishes predicting. Verification is about persistence of
            # detections, so measure the hit window at prediction-arrival time.
            self._hits.append((now, float(prediction.confidence)))
        else:
            # Do not fully reset; keep temporal context for jitter.
            # (If you want full reset behavior, you can change this policy.)
            pass

        # Drop hits outside the window (relative to *now*).
        window_start = now - self._config.window_sec
        while self._hits and self._hits[0][0] < window_start:
            self._hits.popleft()

        if len(self._hits) < self._config.min_hits:
            return None

        if self._last_verified_epoch_s is not None:
            if (now - self._last_verified_epoch_s) < self._config.cooldown_sec:
                return None

        # Verified: choose latest hit confidence as the representative.
        _, latest_hit_conf = self._hits[-1]
        self._last_verified_epoch_s = now

        return VerifiedEvent(
            event_type="ACCIDENT" if self._config.positive_label == "ACCIDENT" else "VIOLENCE",
            verified_label=self._config.positive_label,
            confidence=latest_hit_conf,
            timestamp_epoch_s=prediction.timestamp_epoch_s,
            camera_id=prediction.camera_id,
            details={
                "hits_in_window": len(self._hits),
                "confidence_threshold": self._config.confidence_threshold,
                "window_sec": self._config.window_sec,
                "cooldown_sec": self._config.cooldown_sec,
            },
        )
