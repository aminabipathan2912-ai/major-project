"""
Escalation policy at the pipeline level.

The design point under test: fusion is *additive*. A modality with its own
verifier still escalates independently even when fusion is enabled, and one
cooldown per event type keeps the independent path and the fusion path from
both dialling for a single incident.
"""
import asyncio
import time

import pytest

from cctv_ai.config import Settings
from cctv_ai.core.models import VerifiedEvent
from cctv_ai.core.pipeline import PipelineService


def _settings(**overrides) -> Settings:
    base = dict(
        _env_file=None,
        ACCIDENT_MODEL_WEIGHTS_PATH="",
        VIOLENCE_MODEL_WEIGHTS_PATH="",
        AUDIO_MODEL_WEIGHTS_PATH="",
        TEXT_MODEL_WEIGHTS_PATH="",
        EMERGENCY_MODE="log-only",
        FUSION_ENABLED=True,
        ESCALATION_COOLDOWN_SEC=30.0,
    )
    base.update(overrides)
    return Settings(**base)


class _RecordingProvider:
    def __init__(self) -> None:
        self.sent: list[VerifiedEvent] = []

    async def on_verified_emergency(self, event: VerifiedEvent) -> None:
        self.sent.append(event)


def _pipeline(**overrides):
    p = PipelineService(_settings(**overrides))
    provider = _RecordingProvider()
    p._emergency_provider = provider
    # 'phone' avoids the recorded-file one-shot suppression, which is a
    # separate guard from the cross-path cooldown under test here.
    p._active_source_type = "phone"
    return p, provider


def _event(event_type: str = "VIOLENCE", conf: float = 0.8) -> VerifiedEvent:
    return VerifiedEvent(
        event_type=event_type,  # type: ignore[arg-type]
        verified_label=event_type,
        confidence=conf,
        timestamp_epoch_s=time.time(),
        camera_id="cam-1",
    )


def test_independent_escalation_fires_with_fusion_enabled():
    p, provider = _pipeline()
    asyncio.run(p._escalate(_event("VIOLENCE")))
    assert [e.event_type for e in provider.sent] == ["VIOLENCE"]


def test_second_call_for_same_event_type_is_suppressed():
    p, provider = _pipeline()

    async def run():
        await p._escalate(_event("VIOLENCE"))  # e.g. the video verifier
        await p._escalate(_event("VIOLENCE"))  # e.g. fusion, same incident
        await p._escalate(_event("ACCIDENT"))  # unrelated type still allowed

    asyncio.run(run())
    assert [e.event_type for e in provider.sent] == ["VIOLENCE", "ACCIDENT"]


def test_same_type_allowed_again_after_cooldown():
    p, provider = _pipeline(ESCALATION_COOLDOWN_SEC=0.0)

    async def run():
        await p._escalate(_event("VIOLENCE"))
        await p._escalate(_event("VIOLENCE"))

    asyncio.run(run())
    assert [e.event_type for e in provider.sent] == ["VIOLENCE", "VIOLENCE"]


def test_switching_source_clears_the_cooldown():
    p, provider = _pipeline()
    asyncio.run(p._escalate(_event("VIOLENCE")))
    p._last_escalation_epoch_s.clear()  # what switch_to_* does
    asyncio.run(p._escalate(_event("VIOLENCE")))
    assert len(provider.sent) == 2
