from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import cv2

from .frame_buffer import FrameBuffer


@dataclass
class CameraWorkerStatus:
    source_type: str
    source: str
    running: bool
    last_error: str = ""
    last_frame_timestamp_epoch_s: float | None = None


class OpenCVCameraWorker:
    """
    Camera capture layer using OpenCV.

    This layer is intentionally independent from AI models. It only:
    - connects (webcam / file / RTSP)
    - continuously pushes frames into FrameBuffer
    - handles reconnection/restart
    """

    def __init__(
        self,
        *,
        source_type: str,
        source: str,
        frame_buffer: FrameBuffer,
        reconnect_backoff_sec: float = 5.0,
        loop_file: bool = True,
        target_grab_fps: float | None = None,
        cap_buffer_size: int | None = None,
    ) -> None:
        self._source_type = source_type
        self._source = source
        self._frame_buffer = frame_buffer
        self._reconnect_backoff_sec = float(reconnect_backoff_sec)
        self._loop_file = loop_file
        self._target_grab_fps = target_grab_fps
        self._cap_buffer_size = cap_buffer_size

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
        self._set_status(
            source_type=source_type,
            source=source,
            running=False,
            last_error="",
            last_frame_timestamp_epoch_s=None,
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

    def _run(self) -> None:
        self._set_status(running=True, last_error="")
        last_open_error: str = ""
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

                next_grab_deadline = 0.0
                while not self._stop_event.is_set():
                    if self._target_grab_fps is not None and self._target_grab_fps > 0:
                        now = time.time()
                        if now < next_grab_deadline:
                            time.sleep(next_grab_deadline - now)
                        next_grab_deadline = time.time() + (1.0 / self._target_grab_fps)

                    ret, frame = cap.read()
                    if not ret or frame is None:
                        # File ended or connection dropped.
                        break

                    ts = time.time()
                    self._frame_buffer.append(frame_bgr=frame, timestamp_epoch_s=ts)
                    self._set_status(last_frame_timestamp_epoch_s=ts, last_error="")

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
