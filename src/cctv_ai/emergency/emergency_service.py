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

    async def start(self) -> None:
        """Optional: spin up background work (e.g. status polling)."""
        return None

    async def aclose(self) -> None:
        """Optional: release long-lived resources (HTTP clients, tasks)."""
        return None


class LogOnlyEmergencyProvider(EmergencyProvider):
    async def on_verified_emergency(self, event: VerifiedEvent) -> None:
        print(
            f"[EMERGENCY][{time.strftime('%Y-%m-%d %H:%M:%S')}] {event.event_type} "
            f"camera={event.camera_id} conf={event.confidence:.3f} ts={event.timestamp_epoch_s:.3f} "
            f"label={event.verified_label}"
        )
        await asyncio.sleep(0)


class RemoteEmergencyProvider(EmergencyProvider):
    """Send only verified-event metadata to the hosted alert backend.

    One `httpx.AsyncClient` is reused for every call (connection pooling, no
    repeat TLS handshake). Incident status is refreshed by a background task
    into `_latest_incident`; `latest_incident()` only reads that cache and never
    touches the network, so `/api/status` cannot hang on a slow backend.
    """

    _TERMINAL_STATUSES = {
        "REPORTED",
        "NO_ACKNOWLEDGEMENT",
        "CALL_FAILED",
        "CALL_NOT_CONFIGURED",
        "TTS_FAILED",
    }

    _IDLE_POLL_SEC = 30.0

    def __init__(self, settings: Settings) -> None:
        self._url = settings.ALERT_BACKEND_URL.rstrip("/")
        self._token = settings.ALERT_BACKEND_TOKEN
        self._location = settings.INCIDENT_LOCATION
        self._poll_interval = max(
            0.5, float(getattr(settings, "INCIDENT_POLL_INTERVAL_SEC", 3.0))
        )
        self._latest_incident_id: str | None = None
        self._latest_delivery_state: str | None = None
        self._latest_incident: dict | None = None

        self._client: httpx.AsyncClient | None = None
        self._poll_task: asyncio.Task | None = None
        self._poll_wake = asyncio.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(10.0))
        return self._client

    async def start(self) -> None:
        if self._url and self._token and self._poll_task is None:
            self._poll_task = asyncio.create_task(self._poll_loop())

    async def aclose(self) -> None:
        if self._poll_task is not None:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # Escalation
    # ------------------------------------------------------------------

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
            response = await self._get_client().post(
                f"{self._url}/api/incidents",
                json=payload,
                headers={"Authorization": f"Bearer {self._token}"},
                timeout=15.0,
            )
            response.raise_for_status()
            body = response.json()
            self._latest_incident_id = body.get("incident_id")
            self._latest_delivery_state = body.get("delivery_state")
            self._latest_incident = None
            self._poll_wake.set()
            print(f"[ALERT_BACKEND] event sent: {event.event_type} {event.confidence:.3f}")
        except httpx.HTTPStatusError as exc:
            # The hosted backend returns safe operational details (such as a
            # Twilio error code) so the local operator can diagnose a failed
            # escalation without needing access to its Render logs.
            detail = exc.response.text.strip()
            try:
                body = exc.response.json()
                failure = body.get("detail", {})
                if isinstance(failure, dict):
                    self._latest_incident_id = failure.get("incident_id")
                    self._latest_delivery_state = failure.get("delivery_state", "call_failed")
                    self._latest_incident = None
                    self._poll_wake.set()
            except ValueError:
                pass
            print(
                f"[ALERT_BACKEND] delivery failed: HTTP {exc.response.status_code} "
                f"{detail[:800]}"
            )
        except httpx.HTTPError as exc:
            print(f"[ALERT_BACKEND] delivery failed: {exc}")

    # ------------------------------------------------------------------
    # Cached status
    # ------------------------------------------------------------------

    async def latest_incident(self) -> dict | None:
        """Return the cached incident. Non-blocking — `_poll_loop` refreshes it."""
        if not self._latest_incident:
            return None
        return dict(self._latest_incident)

    async def _poll_loop(self) -> None:
        while True:
            try:
                await self._refresh_incident_once()
            except asyncio.CancelledError:
                raise
            except httpx.HTTPError as exc:
                print(f"[ALERT_BACKEND] status lookup failed: {exc}")
            except Exception as exc:  # never let the poller die
                print(f"[ALERT_BACKEND] status poll error: {exc}")

            status = (self._latest_incident or {}).get("status")
            terminal = status in self._TERMINAL_STATUSES
            idle = terminal or not self._latest_incident_id
            timeout = self._IDLE_POLL_SEC if idle else self._poll_interval
            try:
                await asyncio.wait_for(self._poll_wake.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                pass
            self._poll_wake.clear()

    async def _refresh_incident_once(self) -> None:
        if not self._latest_incident_id or not self._url or not self._token:
            return
        if (
            self._latest_incident
            and self._latest_incident.get("status") in self._TERMINAL_STATUSES
        ):
            return
        response = await self._get_client().get(
            f"{self._url}/api/incidents/{self._latest_incident_id}",
            headers={"Authorization": f"Bearer {self._token}"},
            timeout=8.0,
        )
        response.raise_for_status()
        incident = response.json()
        if self._latest_delivery_state:
            incident["delivery_state"] = self._latest_delivery_state
        self._latest_incident = incident


def create_emergency_provider(settings: Settings) -> EmergencyProvider:
    if settings.EMERGENCY_MODE == "log-only":
        return LogOnlyEmergencyProvider()
    if settings.EMERGENCY_MODE == "remote":
        return RemoteEmergencyProvider(settings)
    raise ValueError(f"Unsupported EMERGENCY_MODE: {settings.EMERGENCY_MODE}")
