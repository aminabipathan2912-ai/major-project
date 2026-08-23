from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod

import httpx

from ..config import Settings
from ..core.models import VerifiedEvent


class EmergencyProvider(ABC):
    @abstractmethod
    async def on_verified_emergency(self, event: VerifiedEvent) -> None:
        raise NotImplementedError


class LogOnlyEmergencyProvider(EmergencyProvider):
    async def on_verified_emergency(self, event: VerifiedEvent) -> None:
        print(
            f"[EMERGENCY][{time.strftime('%Y-%m-%d %H:%M:%S')}] {event.event_type} "
            f"camera={event.camera_id} conf={event.confidence:.3f} ts={event.timestamp_epoch_s:.3f} "
            f"label={event.verified_label}"
        )
        await asyncio.sleep(0)


class RemoteEmergencyProvider(EmergencyProvider):
    """Send only verified-event metadata to the hosted alert backend."""

    def __init__(self, settings: Settings) -> None:
        self._url = settings.ALERT_BACKEND_URL.rstrip("/")
        self._token = settings.ALERT_BACKEND_TOKEN
        self._location = settings.INCIDENT_LOCATION

    async def on_verified_emergency(self, event: VerifiedEvent) -> None:
        if not self._url or not self._token:
            print("[ALERT_BACKEND] not configured; event was not sent.")
            return
        payload = {
            "event_key": f"{event.camera_id}:{event.event_type}:{event.timestamp_epoch_s:.3f}",
            "event_type": event.event_type,
            "confidence": event.confidence,
            "camera_id": event.camera_id,
            "location": self._location,
            "timestamp_epoch_s": event.timestamp_epoch_s,
            "details": event.details,
        }
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    f"{self._url}/api/incidents",
                    json=payload,
                    headers={"Authorization": f"Bearer {self._token}"},
                )
                response.raise_for_status()
            print(f"[ALERT_BACKEND] event sent: {event.event_type} {event.confidence:.3f}")
        except httpx.HTTPStatusError as exc:
            # The hosted backend returns safe operational details (such as a
            # Twilio error code) so the local operator can diagnose a failed
            # escalation without needing access to its Render logs.
            detail = exc.response.text.strip()
            print(
                f"[ALERT_BACKEND] delivery failed: HTTP {exc.response.status_code} "
                f"{detail[:800]}"
            )
        except httpx.HTTPError as exc:
            print(f"[ALERT_BACKEND] delivery failed: {exc}")


def create_emergency_provider(settings: Settings) -> EmergencyProvider:
    if settings.EMERGENCY_MODE == "log-only":
        return LogOnlyEmergencyProvider()
    if settings.EMERGENCY_MODE == "remote":
        return RemoteEmergencyProvider(settings)
    raise ValueError(f"Unsupported EMERGENCY_MODE: {settings.EMERGENCY_MODE}")
