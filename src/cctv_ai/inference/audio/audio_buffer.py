from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque


@dataclass(frozen=True)
class AudioChunk:
    """
    One encoded audio blob exactly as the browser's MediaRecorder produced it.

    Deliberately *not* decoded: there is no audio model yet, so decoding or
    transcoding would be work with no consumer. `mime_type` carries whatever the
    browser negotiated (typically audio/webm;codecs=opus) so a future classifier
    knows how to decode it.
    """

    data: bytes
    mime_type: str
    received_epoch_s: float


class AudioBuffer:
    """
    Bounded ring of recent audio chunks.

    Bounded is the point: a phone streaming 1 s chunks would otherwise grow
    without limit while nothing consumes them. Oldest chunks are dropped, and
    the drop count is surfaced in `/api/status` so silent data loss is visible.
    """

    def __init__(self, maxlen: int = 8) -> None:
        if maxlen <= 0:
            raise ValueError("maxlen must be > 0")
        self._maxlen = maxlen
        self._lock = threading.Lock()
        self._chunks: Deque[AudioChunk] = deque(maxlen=maxlen)
        self._received = 0
        self._dropped = 0

    def append(self, data: bytes, mime_type: str = "") -> None:
        chunk = AudioChunk(
            data=data, mime_type=mime_type, received_epoch_s=time.time()
        )
        with self._lock:
            if len(self._chunks) == self._maxlen:
                self._dropped += 1
            self._chunks.append(chunk)
            self._received += 1

    def snapshot(self, n: int | None = None) -> list[AudioChunk]:
        with self._lock:
            chunks = list(self._chunks)
        return chunks if n is None else chunks[-n:]

    def clear(self) -> None:
        with self._lock:
            self._chunks.clear()

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "buffered": len(self._chunks),
                "received": self._received,
                "dropped": self._dropped,
                "maxlen": self._maxlen,
            }

    def __len__(self) -> int:
        with self._lock:
            return len(self._chunks)


__all__ = ["AudioChunk", "AudioBuffer"]
