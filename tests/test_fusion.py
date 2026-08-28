import time

import pytest

from cctv_ai.event_engine.fusion import (
    FusionConfig,
    FusionEngine,
    ModalityEvidence,
)

ACCIDENT = frozenset({"ACCIDENT"})
BOTH = frozenset({"ACCIDENT", "VIOLENCE"})


def _engine(**overrides) -> FusionEngine:
    cfg = FusionConfig(
        enabled=True,
        window_sec=5.0,
        min_modalities=2,
        threshold=0.6,
        cooldown_sec=30.0,
        support_threshold=0.6,
        weights={"video_accident": 1.0, "audio": 1.0, "text": 0.5},
    )
    return FusionEngine(config=FusionConfig(**{**cfg.__dict__, **overrides}))


def _ev(modality, conf, supports=ACCIDENT, ts=None):
    return ModalityEvidence(
        modality=modality,
        confidence=conf,
        timestamp_epoch_s=time.time() if ts is None else ts,
        supports=supports,
    )


def test_single_confident_modality_does_not_escalate():
    """The exact false positive seen in the field: video sure, nothing else agrees."""
    engine = _engine()
    engine.observe(_ev("video_accident", 0.95))
    assert engine.evaluate("ACCIDENT", camera_id="c1") is None

    last = engine.status()["last"]["ACCIDENT"]
    assert last["fired"] is False
    assert "corroborate" in last["reason"]


def test_two_modalities_agreeing_escalate():
    engine = _engine()
    engine.observe(_ev("video_accident", 0.8))
    engine.observe(_ev("audio", 0.9, supports=BOTH))

    event = engine.evaluate("ACCIDENT", camera_id="c1")
    assert event is not None
    assert event.event_type == "ACCIDENT"
    assert sorted(event.details["supporting_modalities"]) == ["audio", "video_accident"]
    # equal weights -> plain mean
    assert event.confidence == pytest.approx(0.85, abs=1e-6)


def test_silent_modality_dilutes_the_score():
    """A modality that reports but sees nothing must drag the fused score down."""
    engine = _engine(min_modalities=1)
    engine.observe(_ev("video_accident", 0.9))
    engine.observe(_ev("audio", 0.0, supports=BOTH))

    # 1 supporting modality clears min_modalities=1, but the mean is 0.45 < 0.6
    assert engine.evaluate("ACCIDENT", camera_id="c1") is None
    assert engine.status()["last"]["ACCIDENT"]["fused_score"] == pytest.approx(0.45)


def test_min_modalities_one_reproduces_single_modality_behaviour():
    engine = _engine(min_modalities=1)
    engine.observe(_ev("video_accident", 0.95))
    assert engine.evaluate("ACCIDENT", camera_id="c1") is not None


def test_stale_evidence_is_ignored():
    engine = _engine(window_sec=2.0)
    old = time.time() - 10.0
    engine.observe(_ev("audio", 0.99, supports=BOTH, ts=old))
    engine.observe(_ev("video_accident", 0.95))

    # audio has aged out, so video stands alone
    assert engine.evaluate("ACCIDENT", camera_id="c1") is None
    assert engine.status()["last"]["ACCIDENT"]["contributions"] == {
        "video_accident": pytest.approx(0.95)
    }


def test_cooldown_blocks_immediate_refire():
    engine = _engine()
    engine.observe(_ev("video_accident", 0.9))
    engine.observe(_ev("audio", 0.9, supports=BOTH))
    assert engine.evaluate("ACCIDENT", camera_id="c1") is not None

    engine.observe(_ev("video_accident", 0.9))
    engine.observe(_ev("audio", 0.9, supports=BOTH))
    assert engine.evaluate("ACCIDENT", camera_id="c1") is None
    assert engine.status()["last"]["ACCIDENT"]["reason"] == "within cooldown"


def test_audio_corroborates_both_event_types():
    """A scream supports an accident and a fight; it must count for either."""
    engine = _engine()
    engine.observe(_ev("audio", 0.9, supports=BOTH))
    engine.observe(_ev("video_violence", 0.8, supports=frozenset({"VIOLENCE"})))

    assert engine.evaluate("VIOLENCE", camera_id="c1") is not None
    # ...but it must not manufacture an accident on its own
    assert engine.evaluate("ACCIDENT", camera_id="c1") is None


def test_weights_are_applied():
    engine = _engine(min_modalities=2, threshold=0.0)
    engine.observe(_ev("video_accident", 1.0))
    engine.observe(_ev("text", 0.0))
    event = engine.evaluate("ACCIDENT", camera_id="c1")
    # (1.0*1.0 + 0.5*0.0) / 1.5 = 0.667, not the unweighted 0.5
    assert engine.status()["last"]["ACCIDENT"]["fused_score"] == pytest.approx(0.6667, abs=1e-3)
    assert event is None  # text at 0.0 is not "supporting"


def test_reset_clears_evidence():
    engine = _engine()
    engine.observe(_ev("video_accident", 0.9))
    engine.observe(_ev("audio", 0.9, supports=BOTH))
    engine.reset()
    assert engine.evaluate("ACCIDENT", camera_id="c1") is None
    assert engine.status()["last"]["ACCIDENT"]["reason"] == "no evidence"


def test_no_evidence_is_safe():
    assert _engine().evaluate("ACCIDENT", camera_id="c1") is None
