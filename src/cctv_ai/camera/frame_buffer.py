from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple

import numpy as np


@dataclass(frozen=True)
class BufferedFrame:
    frame_bgr: np.ndarray
    timestamp_epoch_s: float


class FrameBuffer:
    """
    Thread-safe in-memory ring buffer for recent frames.

    Camera capture runs in a background thread; inference tasks run in asyncio.
    """

    def __init__(self, maxlen: int = 128):
        if maxlen <= 0:
            raise ValueError("maxlen must be > 0")

        self._maxlen = maxlen
        self._lock = threading.Lock()
        self._frames: Deque[BufferedFrame] = deque(maxlen=maxlen)

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

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)
