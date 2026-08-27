from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..config import get_settings
from ..core.pipeline import PipelineService
from ..emergency.emergency_service import RemoteEmergencyProvider

STATIC_DIR = Path(__file__).resolve().parent / "static"
UPLOAD_CHUNK_BYTES = 1 << 20


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


class PreviewEncoder:
    """
    Encodes the latest preview frame to JPEG at most once, for all clients.

    Previously each MJPEG connection ran its own encode loop on a fixed timer,
    so two open tabs meant two full JPEG pipelines over identical frames, and
    `/api/frame.jpg` encoded synchronously on the event loop. Here a single task
    encodes only when the source frame's sequence number changes, off the event
    loop, and every consumer reads the same bytes.
    """

    def __init__(self, pipeline: PipelineService, *, fps: float, quality: int) -> None:
        self._pipeline = pipeline
        self._interval = 1.0 / fps if fps > 0 else 0.1
        self._params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        self._jpeg: bytes | None = None
        self._source_seq = -1
        self._revision = 0
        self._condition = asyncio.Condition()
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def _run(self) -> None:
        while True:
            try:
                latest = self._pipeline.get_latest_preview()
                if latest is not None:
                    frame, seq = latest
                    if seq != self._source_seq:
                        ok, buf = await asyncio.to_thread(
                            cv2.imencode, ".jpg", frame, self._params
                        )
                        if ok:
                            async with self._condition:
                                self._jpeg = buf.tobytes()
                                self._source_seq = seq
                                self._revision += 1
                                self._condition.notify_all()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # never let the preview kill the app
                print(f"[PREVIEW] encode error: {exc}")
            await asyncio.sleep(self._interval)

    def current(self) -> bytes | None:
        return self._jpeg

    async def wait_for_next(self, last_revision: int) -> tuple[bytes, int]:
        """Block until a frame newer than `last_revision` has been encoded."""
        async with self._condition:
            await self._condition.wait_for(
                lambda: self._jpeg is not None and self._revision != last_revision
            )
            assert self._jpeg is not None
            return self._jpeg, self._revision


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    pipeline = PipelineService(settings)
    app.state.pipeline = pipeline
    await pipeline.start()
    preview = PreviewEncoder(
        pipeline,
        fps=settings.PREVIEW_FPS,
        quality=settings.PREVIEW_JPEG_QUALITY,
    )
    app.state.preview = preview
    await preview.start()
    try:
        yield
    finally:
        await preview.stop()
        await pipeline.stop()


def create_app() -> FastAPI:
    app = FastAPI(
        title="CCTV Accident/Violence Inference Demo",
        lifespan=lifespan,
    )
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", response_class=FileResponse)
    async def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/status")
    async def status():
        pipeline: PipelineService = app.state.pipeline
        st = pipeline.get_status()
        payload = _to_jsonable(st)
        payload["playback"] = {
            "file_available": pipeline.resolve_video_path() is not None,
            "source_type": pipeline.get_source_info()["source_type"],
        }
        provider = pipeline.emergency_provider
        if isinstance(provider, RemoteEmergencyProvider):
            payload["latest_incident"] = await provider.latest_incident()
        return JSONResponse(content=payload)

    @app.post("/api/upload")
    async def upload_video(file: UploadFile = File(...)):
        pipeline: PipelineService = app.state.pipeline
        suffix = Path(file.filename or "clip.mp4").suffix.lower()
        if suffix not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
            return JSONResponse({"ok": False, "error": "Use mp4, avi, mov, mkv, or webm."}, status_code=400)

        upload_dir = Path(pipeline.settings.VIDEO_UPLOAD_DIR)
        await asyncio.to_thread(upload_dir.mkdir, parents=True, exist_ok=True)
        dest = upload_dir / f"{uuid.uuid4().hex}{suffix}"

        # Stream in bounded chunks. Reading the whole clip into memory and then
        # writing it synchronously spiked RSS by the file size and stalled the
        # event loop for the duration of the write.
        written = 0
        handle = await asyncio.to_thread(dest.open, "wb")
        try:
            while True:
                chunk = await file.read(UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                await asyncio.to_thread(handle.write, chunk)
                written += len(chunk)
        finally:
            await asyncio.to_thread(handle.close)

        if written == 0:
            await asyncio.to_thread(dest.unlink, True)
            return JSONResponse({"ok": False, "error": "Empty file."}, status_code=400)

        await pipeline.switch_to_uploaded_file(str(dest))
        return JSONResponse(
            {
                "ok": True,
                "path": str(dest),
                "bytes": written,
                "playback_url": "/api/video",
            }
        )

    @app.get("/api/video")
    async def video_file():
        pipeline: PipelineService = app.state.pipeline
        path = pipeline.resolve_video_path()
        if not path:
            return Response(status_code=404)
        return FileResponse(path, media_type="video/mp4", filename=Path(path).name)

    @app.get("/api/frame.jpg")
    async def frame_jpg():
        # Served from the shared encoder's cache; no encoding happens on the
        # request path.
        jpeg = app.state.preview.current()
        if jpeg is None:
            return Response(status_code=204)
        return Response(content=jpeg, media_type="image/jpeg")

    @app.get("/api/stream.mjpg")
    async def stream_mjpg():
        preview: PreviewEncoder = app.state.preview

        async def generate():
            last_revision = -1
            while True:
                jpeg, last_revision = await preview.wait_for_next(last_revision)
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"

        return StreamingResponse(
            generate(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.websocket("/ws/events")
    async def ws_events(ws: WebSocket):
        await ws.accept()
        pipeline: PipelineService = app.state.pipeline
        q = pipeline.subscribe_verified_events()
        try:
            while True:
                event = await q.get()
                await ws.send_text(json.dumps(_to_jsonable(event)))
        except WebSocketDisconnect:
            return
        finally:
            pipeline.unsubscribe_verified_events(q)

    return app


__all__ = ["create_app"]
