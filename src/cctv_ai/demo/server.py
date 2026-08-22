from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..config import get_settings
from ..core.pipeline import PipelineService

STATIC_DIR = Path(__file__).resolve().parent / "static"


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    pipeline = PipelineService(settings)
    app.state.pipeline = pipeline
    await pipeline.start()
    try:
        yield
    finally:
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
        return JSONResponse(content=payload)

    @app.get("/api/video")
    async def video_file():
        pipeline: PipelineService = app.state.pipeline
        path = pipeline.resolve_video_path()
        if not path:
            return Response(status_code=404)
        return FileResponse(
            path,
            media_type="video/mp4",
            filename=Path(path).name,
        )

    @app.get("/api/frame.jpg")
    async def frame_jpg():
        pipeline: PipelineService = app.state.pipeline
        frame_bgr: np.ndarray | None = pipeline.get_latest_frame_bgr()
        if frame_bgr is None:
            return Response(status_code=204)
        ok, jpg = cv2.imencode(".jpg", frame_bgr)
        if not ok:
            return Response(status_code=500)
        return Response(content=jpg.tobytes(), media_type="image/jpeg")

    @app.get("/api/stream.mjpg")
    async def stream_mjpg():
        pipeline: PipelineService = app.state.pipeline

        async def generate():
            while True:
                frame = pipeline.get_latest_frame_bgr()
                if frame is not None:
                    ok, jpg = await asyncio.to_thread(
                        cv2.imencode,
                        ".jpg",
                        frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 80],
                    )
                    if ok:
                        payload = jpg.tobytes()
                        yield (
                            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + payload + b"\r\n"
                        )
                await asyncio.sleep(0.08)

        return StreamingResponse(
            generate(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.websocket("/ws/events")
    async def ws_events(ws: WebSocket):
        await ws.accept()
        pipeline: PipelineService = app.state.pipeline
        q = pipeline.verified_events_queue
        try:
            while True:
                event = await q.get()
                await ws.send_text(json.dumps(_to_jsonable(event)))
        except WebSocketDisconnect:
            return

    return app


__all__ = ["create_app"]
