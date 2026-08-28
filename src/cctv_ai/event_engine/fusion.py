from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque

from ..core.models import VerifiedEvent


@dataclass(frozen=True)
class ModalityEvidence:
    """
    One modality's opinion at one instant.

    `supports` is the set of event types this evidence can corroborate. A video
    accident prediction supports only ACCIDENT; a scream supports both ACCIDENT
    and VIOLENCE, because either could produce one. Declaring it per-evidence
    keeps the mapping explicit instead of hidden in the fusion rules.
    """

    modality: str
    confidence: float
    timestamp_epoch_s: float
    supports: frozenset[str]
    label: str = ""


@dataclass(frozen=True)
class FusionConfig:
    """
    Decision-level fusion parameters.

    `min_modalities` is the corroboration requirement and the heart of the
    method: with 1 it degrades to single-modality behaviour, with 2 no single
    modality can escalate on its own.
    """

    enabled: bool = False
    window_sec: float = 5.0
    min_modalities: int = 2
    threshold: float = 0.6
    cooldown_sec: float = 30.0
    # Per-modality confidence a reading must reach before it counts as support.
    support_threshold: float = 0.6
    weights: dict[str, float] = field(default_factory=dict)
    default_weight: float = 1.0

    def weight_for(self, modality: str) -> float:
        return float(self.weights.get(modality, self.default_weight))


@dataclass(frozen=True)
class FusionOutcome:
    """Why fusion did or did not fire — surfaced in `/api/status` and event details."""

    event_type: str
    fused_score: float
    supporting: tuple[str, ...]
    contributions: dict[str, float]
    fired: bool
    reason: str


class FusionEngine:
    """
    Decision-level multimodal fusion.

    Each modality reports a confidence for the event types it can speak to. The
    engine keeps a short rolling window of those readings and combines them into
    one score per event type:

        fused(E) = sum(weight_m * conf_m) / sum(weight_m)     over modalities
                                                              that reported on E

    An event escalates only when **both** hold:

      * `fused >= threshold`                 — the combined evidence is strong
      * `len(supporting) >= min_modalities`  — enough independent modalities agree

    The second condition is what a single-modality pipeline cannot express, and
    it is what suppresses a confident-but-wrong video reading that no other
    modality corroborates.

    Weighted-mean rather than noisy-OR is deliberate: OR rewards any single loud
    modality, which is the failure mode being designed against. The mean makes a
    silent modality actively dilute the score.
    """

    def __init__(self, *, config: FusionConfig) -> None:
        self._config = config
        self._evidence: dict[str, dict[str, Deque[ModalityEvidence]]] = defaultdict(
            lambda: defaultdict(deque)
        )
        self._last_fired: dict[str, float] = {}
        self._last_outcome: dict[str, FusionOutcome] = {}

    @property
    def config(self) -> FusionConfig:
        return self._config

    def observe(self, evidence: ModalityEvidence) -> None:
        """Record one modality reading against every event type it supports."""
        for event_type in evidence.supports:
            self._evidence[event_type][evidence.modality].append(evidence)
        self._prune()

    def _prune(self, now: float | None = None) -> None:
        cutoff = (time.time() if now is None else now) - self._config.window_sec
        for by_modality in self._evidence.values():
            for queue in by_modality.values():
                while queue and queue[0].timestamp_epoch_s < cutoff:
                    queue.popleft()

    def evaluate(self, event_type: str, *, camera_id: str) -> VerifiedEvent | None:
        """
        Fuse current evidence for `event_type`. Returns a `VerifiedEvent` only
        when the combined evidence clears both the score and corroboration bars.
        """
        now = time.time()
        self._prune(now)

        by_modality = self._evidence.get(event_type, {})
        contributions: dict[str, float] = {}
        for modality, queue in by_modality.items():
            if queue:
                # Strongest recent reading: an event is a peak, not an average.
                contributions[modality] = max(e.confidence for e in queue)

        if not contributions:
            # Absence of input is not a decision. Keep the last real outcome so
            # the operator can still see why the previous call was made, instead
            # of it being wiped once evidence ages out of the window.
            self._last_outcome.setdefault(
                event_type, FusionOutcome(event_type, 0.0, (), {}, False, "no evidence")
            )
            return None

        total_weight = sum(self._config.weight_for(m) for m in contributions)
        fused = (
            sum(self._config.weight_for(m) * c for m, c in contributions.items())
            / total_weight
            if total_weight
            else 0.0
        )
        supporting = tuple(
            sorted(
                m
                for m, c in contributions.items()
                if c >= self._config.support_threshold
            )
        )

        def record(fired: bool, reason: str) -> FusionOutcome:
            outcome = FusionOutcome(
                event_type=event_type,
                fused_score=round(fused, 4),
                supporting=supporting,
                contributions={m: round(c, 4) for m, c in contributions.items()},
                fired=fired,
                reason=reason,
            )
            self._last_outcome[event_type] = outcome
            return outcome

        if len(supporting) < self._config.min_modalities:
            record(
                False,
                f"only {len(supporting)} modality/modalities corroborate "
                f"(need {self._config.min_modalities})",
            )
            return None

        if fused < self._config.threshold:
            record(False, f"fused {fused:.3f} below threshold {self._config.threshold}")
            return None

        last = self._last_fired.get(event_type)
        if last is not None and (now - last) < self._config.cooldown_sec:
            record(False, "within cooldown")
            return None

        self._last_fired[event_type] = now
        outcome = record(True, "corroborated")

        latest_ts = max(
            e.timestamp_epoch_s
            for queue in by_modality.values()
            for e in queue
        )
        return VerifiedEvent(
            event_type=event_type,  # type: ignore[arg-type]
            verified_label=event_type,
            confidence=fused,
            timestamp_epoch_s=latest_ts,
            camera_id=camera_id,
            details={
                "fusion": True,
                "fused_score": outcome.fused_score,
                "supporting_modalities": list(outcome.supporting),
                "contributions": outcome.contributions,
                "min_modalities": self._config.min_modalities,
                "threshold": self._config.threshold,
                "window_sec": self._config.window_sec,
            },
        )

    def status(self) -> dict:
        """Explainability for `/api/status` — why the last decision went the way it did."""
        return {
            "enabled": self._config.enabled,
            "min_modalities": self._config.min_modalities,
            "threshold": self._config.threshold,
            "window_sec": self._config.window_sec,
            "last": {
                k: {
                    "fused_score": o.fused_score,
                    "supporting": list(o.supporting),
                    "contributions": o.contributions,
                    "fired": o.fired,
                    "reason": o.reason,
                }
                for k, o in self._last_outcome.items()
            },
        }

    def reset(self) -> None:
        """Drop retained evidence — used when the media source changes."""
        self._evidence.clear()
        self._last_outcome.clear()


def build_fusion_config(settings) -> FusionConfig:
    return FusionConfig(
        enabled=bool(getattr(settings, "FUSION_ENABLED", False)),
        window_sec=float(getattr(settings, "FUSION_WINDOW_SEC", 5.0)),
        min_modalities=int(getattr(settings, "FUSION_MIN_MODALITIES", 2)),
        threshold=float(getattr(settings, "FUSION_THRESHOLD", 0.6)),
        cooldown_sec=float(getattr(settings, "FUSION_COOLDOWN_SEC", 30.0)),
        support_threshold=float(getattr(settings, "FUSION_SUPPORT_THRESHOLD", 0.6)),
        weights={
            "video_accident": float(getattr(settings, "FUSION_WEIGHT_VIDEO", 1.0)),
            "video_violence": float(getattr(settings, "FUSION_WEIGHT_VIDEO", 1.0)),
            "audio": float(getattr(settings, "FUSION_WEIGHT_AUDIO", 1.0)),
            "text": float(getattr(settings, "FUSION_WEIGHT_TEXT", 0.5)),
        },
    )


__all__ = [
    "ModalityEvidence",
    "FusionConfig",
    "FusionOutcome",
    "FusionEngine",
    "build_fusion_config",
]
