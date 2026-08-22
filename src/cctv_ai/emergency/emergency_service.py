from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod

from ..core.models import VerifiedEvent
from ..config import Settings


class EmergencyProvider(ABC):
    """
    Provider-agnostic abstraction for emergency actions.

    Future workflow:
    - VerifiedEvent -> Voice/Notification Agent -> Emergency Service
    """

    @abstractmethod
    async def on_verified_emergency(self, event: VerifiedEvent) -> None:
        raise NotImplementedError


class LogOnlyEmergencyProvider(EmergencyProvider):
    async def on_verified_emergency(self, event: VerifiedEvent) -> None:
        # Important: do not fabricate real emergency events. In log-only mode,
        # we only record/print verified events.
        print(f"[EMERGENCY][{time.strftime('%Y-%m-%d %H:%M:%S')}] {event.event_type} "
              f"camera={event.camera_id} conf={event.confidence:.3f} ts={event.timestamp_epoch_s:.3f} "
              f"label={event.verified_label}")
        await asyncio.sleep(0)


def create_emergency_provider(settings: Settings) -> EmergencyProvider:
    mode = settings.EMERGENCY_MODE
    if mode == "log-only":
        return LogOnlyEmergencyProvider()
    raise ValueError(f"Unsupported EMERGENCY_MODE: {mode}")

