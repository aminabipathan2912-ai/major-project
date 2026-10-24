from __future__ import annotations

import base64
import logging
import time
import urllib.parse
from datetime import datetime
from pathlib import Path

import feedparser
import httpx
from twilio.rest import Client
from twilio.twiml.voice_response import Gather, VoiceResponse

from .config import Settings

ACKNOWLEDGEMENTS = (
    "done",
    "reported",
    "yes",
    "yeah",
    "yup",
    "ok",
    "okay",
    "acknowledged",
    "confirmed",
    "roger",
    "copy",
    "received",
    "one",
    "1",
)

_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
_MAX_AGE_SEC = 172800  # 48 hours
_FETCH_TIMEOUT_SEC = 4


def fetch_location_context(location: str, event_type: str) -> str:
    """
    Search Google News RSS for recent headlines about *location* + *event_type*.
    Returns a single short sentence suitable for appending to the TTS message,
    or an empty string if nothing relevant or the fetch fails.
    """
    if not location:
        return ""
    kind = "accident" if event_type == "ACCIDENT" else "violence OR assault OR attack"
    candidate_queries = [
        f"{location} {kind}",
        f"{location} (accident OR crash OR collision OR violence OR crime OR incident)",
    ]
    now = time.time()
    for q in candidate_queries:
        query = urllib.parse.quote(q)
        url = _NEWS_RSS.format(query=query)
        try:
            response = httpx.get(url, timeout=_FETCH_TIMEOUT_SEC, follow_redirects=True)
            response.raise_for_status()
            feed = feedparser.parse(response.text)
        except Exception as exc:
            logger.warning("News RSS fetch failed for query=%r: %s", q, exc)
            continue

        for entry in feed.entries[:5]:
            published = entry.get("published_parsed")
            if published is None:
                continue
            age_sec = now - time.mktime(published)
            if age_sec > _MAX_AGE_SEC:
                continue
            title = (entry.get("title") or "").strip()
            if " - " in title:
                title = title.rsplit(" - ", 1)[0].strip()
            if title:
                logger.info("News context found for location=%r: %r", location, title)
                return f"Note: recent news also reports — {title}."

    return ""


def emergency_message(event: dict) -> str:
    kind = "road accident" if event["event_type"] == "ACCIDENT" else "violent incident"
    when = datetime.fromtimestamp(event["timestamp_epoch_s"]).astimezone().strftime("%I:%M %p")
    context = fetch_location_context(event.get("location", ""), event.get("event_type", ""))
    context_part = f" {context}" if context else ""
    return (
        "This is an automated emergency alert. "
        f"A possible {kind} has been detected at {event['location']}. "
        f"Camera ID is {event['camera_id']}. The incident was detected at {when}."
        f"{context_part} "
        "Please say done or press 1 to confirm that the incident has been reported."
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
        input="speech dtmf",
        num_digits=1,
        action=f"{base}/twilio/acknowledge/{incident_id}",
        method="POST",
        speech_timeout="auto",
        timeout=10,
        action_on_empty_result=True,
        language="en-IN",
        hints="done, reported, yes, acknowledged, ok, okay",
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
