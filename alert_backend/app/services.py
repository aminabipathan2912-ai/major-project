from __future__ import annotations

import base64
from datetime import datetime
from pathlib import Path

import httpx
from twilio.rest import Client
from twilio.twiml.voice_response import Gather, VoiceResponse

from .config import Settings

ACKNOWLEDGEMENTS = ("done", "reported", "yes", "acknowledged", "confirmed")


def emergency_message(event: dict) -> str:
    kind = "road accident" if event["event_type"] == "ACCIDENT" else "violent incident"
    when = datetime.fromtimestamp(event["timestamp_epoch_s"]).astimezone().strftime("%I:%M %p")
    return (
        "This is an automated emergency alert. "
        f"A possible {kind} has been detected at {event['location']}. "
        f"Camera ID is {event['camera_id']}. The incident was detected at {when}. "
        "Please say done to confirm that the incident has been reported."
    )


def is_acknowledgement(speech: str) -> bool:
    value = (speech or "").lower().strip()
    return any(word in value for word in ACKNOWLEDGEMENTS)


def synthesize_to_file(settings: Settings, text: str, destination: Path) -> bool:
    if not settings.SARVAM_API_KEY:
        return False
    response = httpx.post(
        settings.SARVAM_TTS_URL,
        headers={"api-subscription-key": settings.SARVAM_API_KEY},
        json={
            "text": text,
            "language_code": settings.SARVAM_LANGUAGE_CODE,
            "speaker": settings.SARVAM_SPEAKER,
            "model": settings.SARVAM_MODEL,
        },
        timeout=60,
    )
    response.raise_for_status()
    body = response.json()
    audio = body.get("audios") or body.get("audio")
    if isinstance(audio, str):
        raw = base64.b64decode(audio)
    elif isinstance(audio, list) and audio:
        raw = b"".join(base64.b64decode(chunk) for chunk in audio)
    else:
        raise RuntimeError("Sarvam response did not include audio data")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(raw)
    return True


def place_call(settings: Settings, incident_id: str, call_id: str) -> str:
    client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
    base = settings.PUBLIC_BASE_URL.rstrip("/")
    request: dict = {
        "to": settings.TWILIO_TO_NUMBER,
        "from_": settings.TWILIO_FROM_NUMBER,
    }
    if settings.TWILIO_CALL_MODE == "trial-template":
        # Match Twilio's restricted-trial call example exactly: no dynamic URL,
        # method override, or custom status callback parameters.
        request["url"] = settings.TWILIO_TRIAL_TEMPLATE_URL
    elif settings.TWILIO_CALL_MODE == "trial-custom":
        # Twilio's current Voice trial supports custom TwiML, including the
        # <Say>, <Play>, and <Gather> verbs used by this endpoint. Keep the
        # initial call request minimal; Twilio uses POST by default.
        request["url"] = f"{base}/twilio/voice/{incident_id}"
    else:
        request["url"] = f"{base}/twilio/voice/{incident_id}"
        request["method"] = "POST"
        request["status_callback"] = f"{base}/twilio/status/{call_id}"
        request["status_callback_method"] = "POST"
        request["status_callback_event"] = ["initiated", "ringing", "answered", "completed"]
    call = client.calls.create(
        **request,
    )
    return str(call.sid)


def alert_twiml(settings: Settings, incident_id: str, message: str, audio_filename: str | None) -> str:
    response = VoiceResponse()
    base = settings.PUBLIC_BASE_URL.rstrip("/")
    gather = Gather(
        input="speech",
        action=f"{base}/twilio/acknowledge/{incident_id}",
        method="POST",
        speech_timeout="auto",
        timeout=7,
        action_on_empty_result=True,
        language="en-IN",
        hints="done,reported,acknowledged",
    )
    if audio_filename:
        gather.play(f"{base}/audio/{audio_filename}")
    else:
        gather.say(message, language="en-IN")
    response.append(gather)
    response.say("No acknowledgement was heard. The incident remains active.")
    return str(response)


def say_twiml(text: str) -> str:
    response = VoiceResponse()
    response.say(text, language="en-IN")
    return str(response)
