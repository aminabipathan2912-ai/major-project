from __future__ import annotations

import asyncio
import hmac
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import FileResponse
from twilio.request_validator import RequestValidator

from .config import settings
from .models import VerifiedEventIn
from .services import alert_twiml, emergency_message, is_acknowledgement, place_call, say_twiml, synthesize_to_file
from .store import IncidentStore

AUDIO_DIR = Path("data/audio")
logger = logging.getLogger("cctv_alert_backend")


def xml(content: str) -> Response:
    return Response(content=content, media_type="application/xml")


def assert_ingest_auth(authorization: str | None) -> None:
    expected = f"Bearer {settings.ALERT_INGEST_TOKEN}"
    if not authorization or not hmac.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="invalid alert-backend token")


async def assert_twilio_signature(request: Request, form: dict) -> None:
    if not settings.TWILIO_VALIDATE_WEBHOOKS:
        return
    signature = request.headers.get("X-Twilio-Signature", "")
    url = f"{settings.PUBLIC_BASE_URL.rstrip('/')}{request.url.path}"
    validator = RequestValidator(settings.TWILIO_AUTH_TOKEN)
    if not signature or not validator.validate(url, form, signature):
        raise HTTPException(status_code=403, detail="invalid Twilio signature")


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = IncidentStore(settings.DATABASE_URL)
    await asyncio.to_thread(store.initialize)
    app.state.store = store
    yield


app = FastAPI(title="CCTV Alert Backend", lifespan=lifespan)


@app.get("/")
async def index():
    return {
        "ok": True,
        "service": "CCTV alert backend",
        "health": "/health",
    }


@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/api/incidents")
async def create_incident(event: VerifiedEventIn, authorization: str | None = Header(default=None)):
    assert_ingest_auth(authorization)
    event_data = event.model_dump()
    message = emergency_message(event_data)
    store: IncidentStore = app.state.store
    active_incident = await asyncio.to_thread(
        store.get_active_incident,
        event.camera_id,
        settings.ACTIVE_INCIDENT_TIMEOUT_SEC,
    )
    if active_incident:
        # Never start overlapping calls for the same camera. The caller still
        # receives this incident id, letting the dashboard show the call that
        # is currently awaiting acknowledgement.
        return {
            "ok": True,
            "duplicate": True,
            "active": True,
            "delivery_state": "already_active",
            "incident_id": str(active_incident["id"]),
            "call_started": False,
        }
    incident, created = await asyncio.to_thread(store.create_incident, event_data, message)
    if not created:
        return {
            "ok": True,
            "duplicate": True,
            "delivery_state": "duplicate",
            "incident_id": str(incident["id"]),
        }

    audio_filename = None
    try:
        candidate = f"{incident['id']}.wav"
        if await asyncio.to_thread(synthesize_to_file, settings, message, AUDIO_DIR / candidate):
            audio_filename = candidate
            await asyncio.to_thread(store.update_incident, str(incident["id"]), audio_filename=candidate)
    except Exception as exc:
        await asyncio.to_thread(store.update_incident, str(incident["id"]), status="TTS_FAILED")
        logger.exception("Sarvam TTS failed for incident %s: %s", incident["id"], exc)
        raise HTTPException(status_code=502, detail=f"Sarvam TTS failed: {exc}") from exc

    if not (settings.TWILIO_ACCOUNT_SID and settings.TWILIO_AUTH_TOKEN and settings.TWILIO_FROM_NUMBER and settings.TWILIO_TO_NUMBER):
        await asyncio.to_thread(store.update_incident, str(incident["id"]), status="CALL_NOT_CONFIGURED")
        return {
            "ok": True,
            "incident_id": str(incident["id"]),
            "call_started": False,
            "delivery_state": "not_configured",
        }

    call = await asyncio.to_thread(store.create_call, str(incident["id"]), settings.TWILIO_TO_NUMBER)
    try:
        sid = await asyncio.to_thread(place_call, settings, str(incident["id"]), str(call["id"]))
        await asyncio.to_thread(store.update_call, str(call["id"]), twilio_sid=sid, status="initiated")
        initial_status = (
            "TRIAL_CALL_STARTED"
            if settings.TWILIO_CALL_MODE == "trial-template"
            else "AWAITING_ACKNOWLEDGEMENT"
        )
        await asyncio.to_thread(store.update_incident, str(incident["id"]), status=initial_status)
    except Exception as exc:
        await asyncio.to_thread(store.update_call, str(call["id"]), status="failed", last_error=str(exc))
        await asyncio.to_thread(store.update_incident, str(incident["id"]), status="CALL_FAILED")
        # Render logs get Twilio's precise code/message. We deliberately do not
        # log credentials or the full request payload.
        logger.exception(
            "Twilio call start failed for incident %s, call %s: %s",
            incident["id"],
            call["id"],
            exc,
        )
        raise HTTPException(
            status_code=502,
            detail={
                "message": f"Twilio call could not be started: {exc}",
                "incident_id": str(incident["id"]),
                "delivery_state": "call_failed",
            },
        ) from exc
    return {
        "ok": True,
        "incident_id": str(incident["id"]),
        "call_started": True,
        "delivery_state": "call_started",
        "audio": bool(audio_filename),
    }


@app.get("/api/incidents/{incident_id}")
async def incident_status(incident_id: str, authorization: str | None = Header(default=None)):
    """Authenticated status endpoint used by the local dashboard."""
    assert_ingest_auth(authorization)
    incident = await asyncio.to_thread(app.state.store.get_incident, incident_id)
    if not incident:
        raise HTTPException(status_code=404, detail="incident not found")
    return incident


@app.api_route("/twilio/voice/{incident_id}", methods=["GET", "POST"])
async def twilio_voice(incident_id: str, request: Request):
    form = dict(await request.form()) if request.method == "POST" else {}
    await assert_twilio_signature(request, form)
    incident = await asyncio.to_thread(app.state.store.get_incident, incident_id)
    if not incident:
        return xml(say_twiml("Incident not found."))
    return xml(alert_twiml(settings, incident_id, incident["message_text"], incident["audio_filename"]))


@app.post("/twilio/acknowledge/{incident_id}")
async def acknowledge(incident_id: str, request: Request):
    form = dict(await request.form())
    await assert_twilio_signature(request, form)
    speech = str(form.get("SpeechResult") or "")
    incident = await asyncio.to_thread(app.state.store.get_incident, incident_id)
    if not incident:
        return xml(say_twiml("Incident not found."))
    if is_acknowledgement(speech):
        await asyncio.to_thread(
            app.state.store.update_incident,
            incident_id,
            status="REPORTED",
            speech_result=speech,
            acknowledged_at=datetime.now(timezone.utc),
        )
        return xml(say_twiml("Acknowledgement received. The incident has been reported."))
    await asyncio.to_thread(app.state.store.update_incident, incident_id, status="NO_ACKNOWLEDGEMENT")
    return xml(say_twiml("Acknowledgement was not recognized. The call will now end."))


@app.post("/twilio/status/{call_id}")
async def twilio_status(call_id: str, request: Request):
    form = dict(await request.form())
    await assert_twilio_signature(request, form)
    call_status = str(form.get("CallStatus") or "unknown").lower()
    existing = await asyncio.to_thread(app.state.store.get_call, call_id)
    await asyncio.to_thread(
        app.state.store.update_call,
        call_id,
        status=call_status,
        twilio_sid=str(form.get("CallSid") or ""),
    )
    # Twilio calls an answered outbound call "in-progress". This means the
    # phone/voicemail answered, not that a human has acknowledged the incident.
    if existing and call_status in {"answered", "in-progress"}:
        await asyncio.to_thread(
            app.state.store.update_incident,
            str(existing["incident_id"]),
            status="CALL_ANSWERED",
        )
    return Response(status_code=204)


@app.get("/audio/{filename}")
async def audio(filename: str):
    path = (AUDIO_DIR / filename).resolve()
    if path.parent != AUDIO_DIR.resolve() or not path.is_file():
        raise HTTPException(status_code=404, detail="audio not found")
    return FileResponse(path, media_type="audio/wav")
