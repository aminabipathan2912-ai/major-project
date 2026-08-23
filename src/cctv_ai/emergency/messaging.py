from __future__ import annotations

from datetime import datetime, timezone

from ..core.models import VerifiedEvent


def format_incident_time(epoch_s: float) -> str:
    return datetime.fromtimestamp(epoch_s, tz=timezone.utc).astimezone().strftime("%I:%M %p")


def build_emergency_message(*, event: VerifiedEvent, location: str) -> str:
    kind = "road accident" if event.event_type == "ACCIDENT" else "violent incident"
    when = format_incident_time(event.timestamp_epoch_s)
    return (
        "This is an automated emergency alert. "
        f"A possible {kind} has been detected at {location}. "
        f"Camera ID is {event.camera_id}. "
        f"The incident was detected at {when}. "
        "Please say done to confirm that the incident has been reported."
    )


ACK_PHRASES = (
    "done",
    "it's done",
    "it is done",
    "its done",
    "reported",
    "yes done",
    "yes",
    "acknowledged",
    "confirm",
    "confirmed",
)


def is_acknowledgement(speech: str) -> bool:
    text = (speech or "").lower().strip()
    if not text:
        return False
    return any(phrase in text for phrase in ACK_PHRASES)
