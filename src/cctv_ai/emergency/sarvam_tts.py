from __future__ import annotations

import base64
from pathlib import Path

import httpx


class SarvamTTSError(RuntimeError):
    pass


class SarvamTTSClient:
    def __init__(
        self,
        *,
        api_key: str,
        url: str,
        language_code: str,
        speaker: str,
        model: str,
    ) -> None:
        self._api_key = api_key.strip()
        self._url = url.rstrip("/")
        self._language_code = language_code
        self._speaker = speaker
        self._model = model

    @property
    def configured(self) -> bool:
        return bool(self._api_key)

    def synthesize_to_file(self, text: str, dest: Path) -> Path:
        if not self.configured:
            raise SarvamTTSError("SARVAM_API_KEY is empty.")

        payload = {
            "text": text,
            "language_code": self._language_code,
            "speaker": self._speaker,
            "model": self._model,
        }
        headers = {
            "api-subscription-key": self._api_key,
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=60.0) as client:
            response = client.post(self._url, json=payload, headers=headers)
        if response.status_code >= 400:
            raise SarvamTTSError(f"Sarvam TTS failed ({response.status_code}): {response.text[:500]}")

        data = response.json()
        chunks = data.get("audios") or data.get("audio")
        if isinstance(chunks, str):
            raw = base64.b64decode(chunks)
        elif isinstance(chunks, list) and chunks:
            raw = b"".join(base64.b64decode(part) for part in chunks)
        else:
            raise SarvamTTSError(f"Unexpected Sarvam TTS payload keys: {list(data)[:12]}")

        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
        return dest
