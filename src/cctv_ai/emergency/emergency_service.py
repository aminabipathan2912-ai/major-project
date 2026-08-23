from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from pathlib import Path

from ..config import Settings
from ..core.models import VerifiedEvent
from .messaging import build_emergency_message
from .sarvam_tts import SarvamTTSClient, SarvamTTSError
from .store import IncidentStore
from .twilio_voice import TwilioVoiceClient


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


class EmergencyEscalationService(EmergencyProvider):
    """
    Verified event -> incident -> Sarvam TTS -> Twilio outbound call(s).

    Does not dial official emergency numbers. Uses configured test/operator numbers only.
    """

    def __init__(self, settings: Settings, store: IncidentStore) -> None:
        self._settings = settings
        self._store = store
        self._tts = SarvamTTSClient(
            api_key=settings.SARVAM_API_KEY,
            url=settings.SARVAM_TTS_URL,
            language_code=settings.SARVAM_LANGUAGE_CODE,
            speaker=settings.SARVAM_SPEAKER,
            model=settings.SARVAM_MODEL,
        )
        self._twilio = TwilioVoiceClient(
            account_sid=settings.TWILIO_ACCOUNT_SID,
            auth_token=settings.TWILIO_AUTH_TOKEN,
            from_number=settings.TWILIO_FROM_NUMBER,
            public_base_url=settings.PUBLIC_BASE_URL,
        )
        self._audio_dir = Path(settings.INCIDENT_AUDIO_DIR)
        self._audio_dir.mkdir(parents=True, exist_ok=True)

    def _destination(self) -> tuple[str, str] | None:
        number = (self._settings.TWILIO_TO_NUMBER or "").strip()
        if not number:
            return None
        return ("operator", number)

    async def on_verified_emergency(self, event: VerifiedEvent) -> None:
        message = build_emergency_message(event=event, location=self._settings.INCIDENT_LOCATION)
        incident = self._store.create_incident(
            event_type=event.event_type,
            camera_id=event.camera_id,
            location=self._settings.INCIDENT_LOCATION,
            confidence=event.confidence,
            timestamp_epoch_s=event.timestamp_epoch_s,
            message_text=message,
        )
        incident_id = incident["id"]
        print(f"[INCIDENT] created {incident_id} status=DETECTED")

        audio_name = f"{incident_id}.wav"
        audio_path = self._audio_dir / audio_name
        try:
            await asyncio.to_thread(self._tts.synthesize_to_file, message, audio_path)
            self._store.update_incident(incident_id, audio_filename=audio_name)
            print(f"[SARVAM] wrote {audio_path}")
        except SarvamTTSError as e:
            print(f"[SARVAM] TTS skipped/failed: {e}. Twilio will use spoken fallback text.")

        self._store.update_incident(incident_id, status="AWAITING_ACKNOWLEDGEMENT")

        dest = self._destination()
        if not dest:
            print("[TWILIO] TWILIO_TO_NUMBER is empty; incident stored only.")
            return

        if not self._twilio.configured:
            print("[TWILIO] not configured (need SID, token, from-number, PUBLIC_BASE_URL).")
            return

        role, number = dest
        await self._place_with_retry(incident_id=incident_id, role=role, to_number=number)

    async def retry_call(self, call_id: str) -> None:
        call = self._store.get_call(call_id)
        if not call:
            return
        if call["attempts"] >= self._settings.TWILIO_CALL_RETRY_MAX:
            print(f"[TWILIO] max retries reached for {call_id}")
            return
        await self._place_with_retry(
            incident_id=call["incident_id"],
            role=call["role"],
            to_number=call["to_number"],
            existing_call_id=call_id,
        )

    async def _place_with_retry(
        self,
        *,
        incident_id: str,
        role: str,
        to_number: str,
        existing_call_id: str | None = None,
    ) -> None:
        call = (
            self._store.get_call(existing_call_id)
            if existing_call_id
            else self._store.create_call(incident_id=incident_id, role=role, to_number=to_number)
        )
        if not call:
            return
        attempts = int(call["attempts"]) + 1
        try:
            sid = await asyncio.to_thread(
                self._twilio.place_call,
                to_number=to_number,
                incident_id=incident_id,
                call_id=call["id"],
            )
            self._store.update_call(call["id"], twilio_sid=sid, status="initiated", attempts=attempts, last_error="")
            print(f"[TWILIO] {role} call {call['id']} sid={sid}")
        except Exception as e:
            self._store.update_call(call["id"], status="failed", attempts=attempts, last_error=str(e))
            print(f"[TWILIO] {role} call failed: {e}")
            if attempts < self._settings.TWILIO_CALL_RETRY_MAX:
                await asyncio.sleep(2)
                await self._place_with_retry(
                    incident_id=incident_id,
                    role=role,
                    to_number=to_number,
                    existing_call_id=call["id"],
                )


def create_emergency_provider(settings: Settings, store: IncidentStore | None = None) -> EmergencyProvider:
    if settings.EMERGENCY_MODE == "log-only":
        return LogOnlyEmergencyProvider()
    if settings.EMERGENCY_MODE == "voice":
        if store is None:
            store = IncidentStore(settings.INCIDENT_DB_PATH)
        return EmergencyEscalationService(settings, store)
    raise ValueError(f"Unsupported EMERGENCY_MODE: {settings.EMERGENCY_MODE}")
