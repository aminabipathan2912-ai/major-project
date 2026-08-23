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
        self._latest_incident_id: str | None = None

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
                body = response.json()
                self._latest_incident_id = body.get("incident_id")
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

    async def latest_incident(self) -> dict | None:
        """Fetch status updates made by Twilio callbacks without exposing the token to JS."""
        if not self._latest_incident_id or not self._url or not self._token:
            return None
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                response = await client.get(
                    f"{self._url}/api/incidents/{self._latest_incident_id}",
                    headers={"Authorization": f"Bearer {self._token}"},
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as exc:
            print(f"[ALERT_BACKEND] status lookup failed: {exc}")
            return None


def create_emergency_provider(settings: Settings) -> EmergencyProvider:
    if settings.EMERGENCY_MODE == "log-only":
        return LogOnlyEmergencyProvider()
    if settings.EMERGENCY_MODE == "remote":
        return RemoteEmergencyProvider(settings)
    raise ValueError(f"Unsupported EMERGENCY_MODE: {settings.EMERGENCY_MODE}")
