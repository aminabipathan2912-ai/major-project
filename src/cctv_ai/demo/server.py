from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..config import get_settings
from ..core.pipeline import PipelineService
from ..emergency.emergency_service import EmergencyEscalationService
from ..emergency.messaging import is_acknowledgement
from ..emergency.twilio_voice import twiml_ack_ok, twiml_ack_retry, twiml_alert, twiml_say

STATIC_DIR = Path(__file__).resolve().parent / "static"
RETRY_CALL_STATUSES = {"failed", "busy", "no-answer", "canceled"}


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


def _xml(content: str) -> Response:
    return Response(content=content, media_type="application/xml")


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
        latest = pipeline.incident_store.latest_incident()
        if latest:
            latest = dict(latest)
            latest["calls"] = pipeline.incident_store.calls_for_incident(latest["id"])
        payload["latest_incident"] = latest
        return JSONResponse(content=payload)

    @app.get("/api/incidents")
    async def incidents():
        pipeline: PipelineService = app.state.pipeline
        return JSONResponse(content=pipeline.incident_store.list_incidents())

    @app.get("/audio/{filename}")
    async def incident_audio(filename: str):
        pipeline: PipelineService = app.state.pipeline
        audio_dir = Path(pipeline.settings.INCIDENT_AUDIO_DIR).resolve()
        path = (audio_dir / filename).resolve()
        try:
            path.relative_to(audio_dir)
        except ValueError:
            return Response(status_code=404)
        if not path.is_file():
            return Response(status_code=404)
        media = "audio/mpeg" if path.suffix.lower() == ".mp3" else "audio/wav"
        return FileResponse(path, media_type=media)

    @app.get("/api/video")
    async def video_file():
        pipeline: PipelineService = app.state.pipeline
        path = pipeline.resolve_video_path()
        if not path:
            return Response(status_code=404)
        return FileResponse(path, media_type="video/mp4", filename=Path(path).name)

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
                        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + payload + b"\r\n"
                await asyncio.sleep(0.08)

        return StreamingResponse(
            generate(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.api_route("/twilio/voice/{incident_id}", methods=["GET", "POST"])
    async def twilio_voice(incident_id: str):
        pipeline: PipelineService = app.state.pipeline
        incident = pipeline.incident_store.get_incident(incident_id)
        if not incident:
            return _xml(twiml_say("Incident not found."))
        settings = pipeline.settings
        audio_url = None
        if incident.get("audio_filename") and settings.PUBLIC_BASE_URL:
            audio_url = f"{settings.PUBLIC_BASE_URL.rstrip('/')}/audio/{incident['audio_filename']}"
        xml = twiml_alert(
            audio_url=audio_url,
            fallback_text=incident.get("message_text") or "Automated emergency alert. Please say done.",
            incident_id=incident_id,
            public_base_url=settings.PUBLIC_BASE_URL or "",
        )
        return _xml(xml)

    @app.post("/twilio/acknowledge/{incident_id}")
    async def twilio_acknowledge(incident_id: str, request: Request):
        pipeline: PipelineService = app.state.pipeline
        form = await request.form()
        speech = str(form.get("SpeechResult") or "").lower().strip()
        incident = pipeline.incident_store.get_incident(incident_id)
        if not incident:
            return _xml(twiml_ack_retry(incident_id=incident_id, public_base_url=pipeline.settings.PUBLIC_BASE_URL))

        if is_acknowledgement(speech):
            pipeline.incident_store.update_incident(
                incident_id,
                status="REPORTED",
                speech_result=speech,
                acknowledged_at_epoch_s=time.time(),
            )
            return _xml(twiml_ack_ok())

        pipeline.incident_store.update_incident(
            incident_id,
            status="AWAITING_ACKNOWLEDGEMENT",
            speech_result=speech,
        )
        return _xml(
            twiml_ack_retry(
                incident_id=incident_id,
                public_base_url=pipeline.settings.PUBLIC_BASE_URL,
            )
        )

    @app.post("/twilio/status/{call_id}")
    async def twilio_status(call_id: str, request: Request):
        pipeline: PipelineService = app.state.pipeline
        form = await request.form()
        call_status = str(form.get("CallStatus") or "").lower()
        call_sid = str(form.get("CallSid") or "")
        existing = pipeline.incident_store.get_call(call_id)
        if existing:
            fields: dict[str, Any] = {"status": call_status or "unknown"}
            if call_sid:
                fields["twilio_sid"] = call_sid
            pipeline.incident_store.update_call(call_id, **fields)
        provider = pipeline.emergency_provider
        if call_status in RETRY_CALL_STATUSES and isinstance(provider, EmergencyEscalationService):
            asyncio.create_task(provider.retry_call(call_id))
        return Response(status_code=204)

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
