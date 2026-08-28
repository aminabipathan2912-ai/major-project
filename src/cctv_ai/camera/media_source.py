from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class MediaSource(Protocol):
    """
    Anything that can fill a `FrameBuffer`.

    The whole point of this seam: every source's only job is to push frames into
    the buffer. Inference, temporal verification, incident creation, TTS, and the
    Twilio workflow read from the buffer and know nothing about where the frames
    came from — so a new transport needs zero changes downstream.

        OpenCVCameraWorker   file | webcam | rtsp   pull: cap.read() loop
        PhoneStreamSource    phone                  push: WebSocket
        (WebRTCSource        later)                 push: aiortc track

    `start`/`stop` are synchronous because `OpenCVCameraWorker` predates this
    protocol and drives a thread; a push source implements them as cheap
    state flips and does its real work in whatever task feeds it.
    """

    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...

    @property
    def status(self) -> Any:
        """Source-specific status object. Rendered into `/api/status`."""
        ...


__all__ = ["MediaSource"]
