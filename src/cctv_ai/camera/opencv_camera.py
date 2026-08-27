from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import cv2

from .frame_buffer import FrameBuffer
from .frame_scaler import resize_max_width, resize_shorter_side


@dataclass
class CameraWorkerStatus:
    source_type: str
    source: str
    running: bool
    last_error: str = ""
    last_frame_timestamp_epoch_s: float | None = None
    frames_decoded: int = 0
    frames_ingested: int = 0


class OpenCVCameraWorker:
    """
    Camera capture layer using OpenCV.

    This layer is intentionally independent from AI models. It only:
    - connects (webcam / file / RTSP)
    - continuously pushes frames into FrameBuffer
    - handles reconnection/restart

    Frames are downscaled once here rather than repeatedly downstream. Every
    decoded frame updates the preview slot so the browser view stays smooth,
    while the inference ring buffer is fed at the decimated ingest rate — only
    enough frames to fill a clip window need to be retained.
    """

    def __init__(
        self,
        *,
        source_type: str,
        source: str,
        frame_buffer: FrameBuffer,
        reconnect_backoff_sec: float = 5.0,
        loop_file: bool = True,
        realtime_file: bool = True,
        target_grab_fps: float | None = None,
        cap_buffer_size: int | None = None,
        inference_frame_size: int = 256,
        ingest_resize_exact: bool = True,
        ingest_sample_fps: float = 0.0,
        preview_max_width: int = 640,
    ) -> None:
        self._source_type = source_type
        self._source = source
        self._frame_buffer = frame_buffer
        self._reconnect_backoff_sec = float(reconnect_backoff_sec)
        self._loop_file = loop_file
        self._realtime_file = realtime_file
        self._target_grab_fps = target_grab_fps
        self._cap_buffer_size = cap_buffer_size
        self._inference_frame_size = int(inference_frame_size)
        self._ingest_resize_exact = bool(ingest_resize_exact)
        self._ingest_sample_fps = float(ingest_sample_fps)
        self._preview_max_width = int(preview_max_width)

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._status = CameraWorkerStatus(
            source_type=source_type, source=source, running=False
        )
        self._status_lock = threading.Lock()

    @property
    def status(self) -> CameraWorkerStatus:
        with self._status_lock:
            return self._status

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, timeout_sec: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout_sec)
        self._thread = None

    def restart_with(self, *, source_type: str, source: str) -> None:
        self.stop()
        self._source_type = source_type
        self._source = source
        # A new source must not inherit the previous scene's frames: a clip
        # sampled across the switch would mix two unrelated videos.
        self._frame_buffer.clear()
        self._set_status(
            source_type=source_type,
            source=source,
            running=False,
            last_error="",
            last_frame_timestamp_epoch_s=None,
            frames_decoded=0,
            frames_ingested=0,
        )
        self.start()

    def _set_status(self, **kwargs) -> None:
        with self._status_lock:
            for k, v in kwargs.items():
                setattr(self._status, k, v)

    def _open_capture(self) -> cv2.VideoCapture:
        if self._source_type == "webcam":
            try:
                idx = int(self._source)
            except ValueError:
                # Fallback: let OpenCV parse strings.
                idx = self._source
            return cv2.VideoCapture(idx)
        if self._source_type == "file":
            return cv2.VideoCapture(self._source)
        if self._source_type == "rtsp":
            # For production you may swap in a GStreamer/ffmpeg backend.
            return cv2.VideoCapture(self._source)

        raise ValueError(f"Unsupported source_type: {self._source_type}")

    def _publish(self, frame, ts: float, *, ingest: bool) -> None:
        """Update the preview slot always; feed the inference ring when sampled."""
        self._frame_buffer.set_preview(resize_max_width(frame, self._preview_max_width))
        if ingest:
            self._frame_buffer.append(
                frame_bgr=resize_shorter_side(
                    frame,
                    self._inference_frame_size,
                    exact=self._ingest_resize_exact,
                ),
                timestamp_epoch_s=ts,
            )

    def _run(self) -> None:
        self._set_status(running=True, last_error="")
        last_open_error: str = ""
        decoded = 0
        ingested = 0
        while not self._stop_event.is_set():
            cap = None
            try:
                cap = self._open_capture()
                if self._cap_buffer_size is not None:
                    # Not all builds support CAP_PROP_BUFFERSIZE; ignore failures.
                    try:
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, float(self._cap_buffer_size))
                    except Exception:
                        pass

                if not cap or not cap.isOpened():
                    last_open_error = f"Open failed for {self._source_type}: {self._source}"
                    self._set_status(last_error=last_open_error)
                    time.sleep(self._reconnect_backoff_sec)
                    continue

                grab_fps = self._target_grab_fps
                if self._source_type == "file" and self._realtime_file and not grab_fps:
                    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
                    # Some codecs do not expose FPS; retain a sensible pace
                    # instead of returning to an unbounded decode loop.
                    grab_fps = source_fps if 1.0 <= source_fps <= 120.0 else 25.0
                next_grab_deadline = 0.0
                ingest_period = (
                    1.0 / self._ingest_sample_fps if self._ingest_sample_fps > 0 else 0.0
                )
                next_ingest_deadline = 0.0
                while not self._stop_event.is_set():
                    if grab_fps is not None and grab_fps > 0:
                        now = time.time()
                        if now < next_grab_deadline:
                            time.sleep(next_grab_deadline - now)
                        next_grab_deadline = time.time() + (1.0 / grab_fps)

                    ret, frame = cap.read()
                    if not ret or frame is None:
                        # File ended or connection dropped.
                        break

                    ts = time.time()
                    decoded += 1
                    # ingest_period == 0 disables decimation and keeps every frame.
                    should_ingest = ingest_period <= 0.0 or ts >= next_ingest_deadline
                    if should_ingest:
                        next_ingest_deadline = ts + ingest_period
                        ingested += 1
                    self._publish(frame, ts, ingest=should_ingest)
                    self._set_status(
                        last_frame_timestamp_epoch_s=ts,
                        last_error="",
                        frames_decoded=decoded,
                        frames_ingested=ingested,
                    )

                # Inner loop ended: reopen after backoff.
                try:
                    cap.release()
                except Exception:
                    pass

                # If it's a file and we're not looping, exit gracefully.
                if self._source_type == "file" and not self._loop_file:
                    break

                time.sleep(self._reconnect_backoff_sec)
            except Exception as e:
                last_open_error = f"Camera worker exception: {e}"
                self._set_status(last_error=last_open_error)
                time.sleep(self._reconnect_backoff_sec)
            finally:
                try:
                    if cap is not None:
                        cap.release()
                except Exception:
                    pass

        self._set_status(running=False, last_error=last_open_error)
