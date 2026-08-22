from __future__ import annotations

from .frame_buffer import FrameBuffer
from .opencv_camera import OpenCVCameraWorker


def create_camera_worker(*, source_type: str, source: str, frame_buffer: FrameBuffer) -> OpenCVCameraWorker:
    return OpenCVCameraWorker(
        source_type=source_type,
        source=source,
        frame_buffer=frame_buffer,
    )

