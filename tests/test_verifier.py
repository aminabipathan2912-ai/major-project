import time

import pytest

from cctv_ai.core.models import ModelPrediction
from cctv_ai.event_engine.verifier import (
    TemporalVerificationConfig,
    TemporalVerifier,
)


def _pred(*, label: str, conf: float, ts: float, camera_id: str = "c1", model_name: str = "accident"):
    return ModelPrediction(
        model_name=model_name,
        predicted_label=label,
        confidence=conf,
        timestamp_epoch_s=ts,
        camera_id=camera_id,
        metadata={},
    )


def test_verifier_requires_persistence(monkeypatch: pytest.MonkeyPatch):
    now = 1_000.0
    monkeypatch.setattr(time, "time", lambda: now)

    verifier = TemporalVerifier(
        config=TemporalVerificationConfig(
            confidence_threshold=0.6,
            min_hits=3,
            window_sec=2.0,
            cooldown_sec=10.0,
            positive_label="ACCIDENT",
        )
    )

    # Hit #1
    assert verifier.update(_pred(label="ACCIDENT", conf=0.9, ts=now)) is None
    # Hit #2
    now += 0.2
    assert verifier.update(_pred(label="ACCIDENT", conf=0.8, ts=now)) is None
    # Hit #3 (verifies)
    now += 0.2
    verified = verifier.update(_pred(label="ACCIDENT", conf=0.85, ts=now))
    assert verified is not None
    assert verified.event_type == "ACCIDENT"
    assert verified.verified_label == "ACCIDENT"


def test_verifier_cooldown_blocks_repeated_emergency(monkeypatch: pytest.MonkeyPatch):
    now = 2_000.0
    monkeypatch.setattr(time, "time", lambda: now)

    verifier = TemporalVerifier(
        config=TemporalVerificationConfig(
            confidence_threshold=0.6,
            min_hits=2,
            window_sec=2.0,
            cooldown_sec=10.0,
            positive_label="ACCIDENT",
        )
    )

    now += 0.1
    assert verifier.update(_pred(label="ACCIDENT", conf=0.9, ts=now)) is None
    now += 0.1
    verified1 = verifier.update(_pred(label="ACCIDENT", conf=0.9, ts=now))
    assert verified1 is not None

    # Attempt to verify again within cooldown window
    now += 1.0
    assert verifier.update(_pred(label="ACCIDENT", conf=0.95, ts=now)) is None
    now += 0.1
    assert verifier.update(_pred(label="ACCIDENT", conf=0.95, ts=now)) is None

