from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np


@dataclass(frozen=True)
class BufferedFrame:
    frame_bgr: np.ndarray
    timestamp_epoch_s: float


class FrameBuffer:
    """
    Thread-safe in-memory ring buffer for recent frames.

    Camera capture runs in a background thread; inference tasks run in asyncio.

    Two separate stores, because the two consumers want different things:

    * the ring holds inference-sized frames (downscaled once at ingest). Holding
      full-resolution frames here cost ~199 MB at 1080p for data that was
      immediately downscaled to 224px before every forward pass.
    * a single preview slot holds the most recent frame at preview resolution,
      carrying a monotonic sequence number so the JPEG encoder can skip frames
      it has already encoded.

    The ring is fed at the decimated ingest rate; the preview slot is updated on
    every decoded frame so the browser view stays smooth.
    """

    def __init__(self, maxlen: int = 128):
        if maxlen <= 0:
            raise ValueError("maxlen must be > 0")

        self._maxlen = maxlen
        self._lock = threading.Lock()
        self._frames: Deque[BufferedFrame] = deque(maxlen=maxlen)

        self._preview_lock = threading.Lock()
        self._preview: np.ndarray | None = None
        self._preview_seq = 0

    def append(self, frame_bgr: np.ndarray, timestamp_epoch_s: float | None = None) -> None:
        ts = float(time.time()) if timestamp_epoch_s is None else float(timestamp_epoch_s)
        with self._lock:
            self._frames.append(BufferedFrame(frame_bgr=frame_bgr, timestamp_epoch_s=ts))

    def latest(self) -> BufferedFrame | None:
        with self._lock:
            if not self._frames:
                return None
            return self._frames[-1]

    def snapshot_last_n(self, n: int) -> list[BufferedFrame]:
        if n <= 0:
            return []
        with self._lock:
            n = min(n, len(self._frames))
            if n <= 0:
                return []
            frames = list(self._frames)[-n:]
        return frames

    def clear(self) -> None:
        """Drop retained frames so a new source cannot inherit the previous scene."""
        with self._lock:
            self._frames.clear()
        with self._preview_lock:
            self._preview = None

    # ------------------------------------------------------------------
    # Preview slot
    # ------------------------------------------------------------------

    def set_preview(self, frame_bgr: np.ndarray) -> None:
        with self._preview_lock:
            self._preview = frame_bgr
            self._preview_seq += 1

    def latest_preview(self) -> tuple[np.ndarray, int] | None:
        """Return (frame, seq) or None. `seq` changes only when the frame does."""
        with self._preview_lock:
            if self._preview is None:
                return None
            return self._preview, self._preview_seq

    @property
    def preview_seq(self) -> int:
        with self._preview_lock:
            return self._preview_seq

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)
