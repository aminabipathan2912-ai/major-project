from __future__ import annotations

from .frame_buffer import FrameBuffer
from .opencv_camera import OpenCVCameraWorker


def create_camera_worker(
    *,
    source_type: str,
    source: str,
    frame_buffer: FrameBuffer,
    loop_file: bool = False,
    realtime_file: bool = True,
    inference_frame_size: int = 256,
    ingest_resize_exact: bool = True,
    ingest_sample_fps: float = 0.0,
    preview_max_width: int = 640,
) -> OpenCVCameraWorker:
    return OpenCVCameraWorker(
        source_type=source_type,
        source=source,
        frame_buffer=frame_buffer,
        loop_file=loop_file,
        realtime_file=realtime_file,
        inference_frame_size=inference_frame_size,
        ingest_resize_exact=ingest_resize_exact,
        ingest_sample_fps=ingest_sample_fps,
        preview_max_width=preview_max_width,
    )
