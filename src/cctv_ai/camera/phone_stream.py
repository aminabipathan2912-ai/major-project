from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..inference.audio.audio_buffer import AudioBuffer
from .frame_buffer import FrameBuffer
from .frame_scaler import resize_max_width, resize_shorter_side

# 1-byte kind prefix on binary WebSocket messages. Text frames carry JSON
# control messages. Keeping the framing this trivial avoids a new dependency.
KIND_VIDEO = 0x01
KIND_AUDIO = 0x02


@dataclass
class PhoneSourceStatus:
    source_type: str = "phone"
    source: str = "phone"
    running: bool = False
    last_error: str = ""
    last_frame_timestamp_epoch_s: float | None = None
    frames_decoded: int = 0
    frames_ingested: int = 0
    frames_dropped: int = 0
    audio_chunks: int = 0
    sessions: int = 0
    rejected_messages: int = 0
    audio: dict = field(default_factory=dict)


class PhoneStreamSource:
    """
    Push-mode media source fed by a phone browser over a WebSocket.

    The browser encodes JPEG frames itself (cheap uplink, one `imdecode` on our
    side) and sends MediaRecorder audio blobs. Both are bounded:

    * video frames go into an `asyncio.Queue(maxsize=PHONE_FRAME_QUEUE_MAX)`
      with **drop-oldest** semantics — for detection a fresh frame beats a
      complete history, and drops are counted rather than hidden.
    * audio goes into a bounded `AudioBuffer` ring.
    * oversized messages are rejected outright, never buffered.

    This makes "the browser cannot overwhelm the backend" true regardless of
    what the client does, which is the property client-side backpressure alone
    cannot guarantee.

    Only `PHONE_MAX_SESSIONS` (default 1) stream may be connected: two phones
    feeding one `FrameBuffer` would interleave two unrelated scenes into a
    single clip and corrupt inference.
    """

    def __init__(
        self,
        *,
        frame_buffer: FrameBuffer,
        audio_buffer: AudioBuffer,
        inference_frame_size: int = 256,
        ingest_resize_exact: bool = True,
        preview_max_width: int = 640,
        frame_queue_max: int = 4,
        max_message_bytes: int = 1 << 20,
        max_sessions: int = 1,
    ) -> None:
        self._frame_buffer = frame_buffer
        self._audio_buffer = audio_buffer
        self._inference_frame_size = int(inference_frame_size)
        self._ingest_resize_exact = bool(ingest_resize_exact)
        self._preview_max_width = int(preview_max_width)
        self._frame_queue_max = max(1, int(frame_queue_max))
        self._max_message_bytes = int(max_message_bytes)
        self._max_sessions = max(1, int(max_sessions))

        self._queue: asyncio.Queue[tuple[bytes, float]] = asyncio.Queue(
            maxsize=self._frame_queue_max
        )
        self._decode_task: asyncio.Task | None = None
        self._sessions = 0
        self._running = False
        self._status = PhoneSourceStatus()

    # ------------------------------------------------------------------
    # MediaSource protocol
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._running = True
        self._status.running = True
        self._status.last_error = ""
        if self._decode_task is None or self._decode_task.done():
            self._decode_task = asyncio.create_task(self._decode_loop())

    def stop(self) -> None:
        self._running = False
        self._status.running = False
        if self._decode_task is not None:
            self._decode_task.cancel()
            self._decode_task = None
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    @property
    def status(self) -> PhoneSourceStatus:
        self._status.sessions = self._sessions
        self._status.audio = self._audio_buffer.stats()
        return self._status

    @property
    def has_session(self) -> bool:
        return self._sessions > 0

    # ------------------------------------------------------------------
    # Session accounting
    # ------------------------------------------------------------------

    def try_acquire_session(self) -> bool:
        """Reserve the single ingest slot. False means refuse the connection."""
        if self._sessions >= self._max_sessions:
            return False
        self._sessions += 1
        return True

    def release_session(self) -> None:
        self._sessions = max(0, self._sessions - 1)

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def submit_video(self, jpeg: bytes) -> bool:
        """
        Offer one encoded frame. Returns False if it was rejected or dropped.

        Never blocks and never grows: a full queue evicts the *oldest* pending
        frame, because by the time the decoder catches up that frame is stale
        and the newest one is what detection should see.
        """
        if not jpeg or len(jpeg) > self._max_message_bytes:
            self._status.rejected_messages += 1
            return False

        item = (jpeg, time.time())
        try:
            self._queue.put_nowait(item)
            return True
        except asyncio.QueueFull:
            try:
                self._queue.get_nowait()
                self._status.frames_dropped += 1
            except asyncio.QueueEmpty:
                pass
            try:
                self._queue.put_nowait(item)
                return True
            except asyncio.QueueFull:
                self._status.frames_dropped += 1
                return False

    def submit_audio(self, data: bytes, mime_type: str = "") -> bool:
        if not data or len(data) > self._max_message_bytes:
            self._status.rejected_messages += 1
            return False
        self._audio_buffer.append(data, mime_type)
        self._status.audio_chunks += 1
        return True

    async def _decode_loop(self) -> None:
        while True:
            jpeg, ts = await self._queue.get()
            try:
                frame = await asyncio.to_thread(self._decode_and_scale, jpeg)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._status.last_error = f"decode failed: {exc}"
                continue
            if frame is None:
                self._status.last_error = "decode failed: not a valid image"
                continue

            small, preview = frame
            self._frame_buffer.set_preview(preview)
            self._frame_buffer.append(frame_bgr=small, timestamp_epoch_s=ts)
            self._status.frames_decoded += 1
            self._status.frames_ingested += 1
            self._status.last_frame_timestamp_epoch_s = ts
            self._status.last_error = ""

    def _decode_and_scale(self, jpeg: bytes):
        """JPEG decode + both downscales. Runs in a thread — real CPU work."""
        buf = np.frombuffer(jpeg, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            return None
        small = resize_shorter_side(
            frame, self._inference_frame_size, exact=self._ingest_resize_exact
        )
        preview = resize_max_width(frame, self._preview_max_width)
        return small, preview


__all__ = ["PhoneStreamSource", "PhoneSourceStatus", "KIND_VIDEO", "KIND_AUDIO"]
