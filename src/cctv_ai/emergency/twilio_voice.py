from __future__ import annotations

from twilio.rest import Client
from twilio.twiml.voice_response import Gather, VoiceResponse


class TwilioVoiceClient:
    def __init__(
        self,
        *,
        account_sid: str,
        auth_token: str,
        from_number: str,
        public_base_url: str,
    ) -> None:
        self._sid = account_sid.strip()
        self._token = auth_token.strip()
        self._from = from_number.strip()
        self._base = public_base_url.rstrip("/")
        self._client = Client(self._sid, self._token) if self.configured else None

    @property
    def configured(self) -> bool:
        return bool(self._sid and self._token and self._from and self._base.startswith("http"))

    def place_call(self, *, to_number: str, incident_id: str, call_id: str) -> str:
        if not self._client or not self.configured:
            raise RuntimeError("Twilio is not configured (SID, token, from-number, PUBLIC_BASE_URL).")
        call = self._client.calls.create(
            to=to_number,
            from_=self._from,
            url=f"{self._base}/twilio/voice/{incident_id}",
            method="POST",
            status_callback=f"{self._base}/twilio/status/{call_id}",
            status_callback_method="POST",
            status_callback_event=["initiated", "ringing", "answered", "completed"],
        )
        return str(call.sid)


def twiml_alert(*, audio_url: str | None, fallback_text: str, incident_id: str, public_base_url: str) -> str:
    response = VoiceResponse()
    gather = Gather(
        input="speech",
        action=f"{public_base_url.rstrip('/')}/twilio/acknowledge/{incident_id}",
        method="POST",
        timeout=5,
        speech_timeout="auto",
        language="en-IN",
    )
    if audio_url:
        gather.play(audio_url)
    else:
        gather.say(fallback_text, language="en-IN")
    response.append(gather)
    response.say("No acknowledgement was heard. The incident remains active.")
    return str(response)


def twiml_say(text: str) -> str:
    response = VoiceResponse()
    response.say(text)
    return str(response)


def twiml_ack_ok() -> str:
    response = VoiceResponse()
    response.say("Acknowledgement received. The accident has been reported.")
    return str(response)


def twiml_ack_retry(*, incident_id: str, public_base_url: str) -> str:
    response = VoiceResponse()
    response.say("Acknowledgement not recognized.")
    gather = Gather(
        input="speech",
        action=f"{public_base_url.rstrip('/')}/twilio/acknowledge/{incident_id}",
        method="POST",
        timeout=5,
        speech_timeout="auto",
        language="en-IN",
    )
    gather.say("Please say done to confirm that the incident has been reported.")
    response.append(gather)
    response.say("The incident remains active.")
    return str(response)
