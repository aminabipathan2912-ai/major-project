from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from ..camera.camera_factory import create_camera_worker
from ..camera.frame_buffer import FrameBuffer
from ..config import Settings
from ..core.models import ModelPrediction, ModelStatus, VerifiedEvent
from ..emergency.emergency_service import EmergencyProvider, create_emergency_provider
from ..event_engine.verifier import TemporalVerificationConfig, TemporalVerifier
from ..inference.loader import create_models


@dataclass
class PipelineStatus:
    camera: dict[str, Any]
    models: dict[str, ModelStatus]
    verification: dict[str, Any]
    last_predictions: dict[str, Any]
    running: bool


class PipelineService:
    """
    CCTV/IP camera -> frame buffering -> model inference -> temporal verification -> emergency hook.

    The camera layer and the model layers are independent by design.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

        self._frame_buffer = FrameBuffer(maxlen=settings.FRAME_BUFFER_MAXLEN)
        self._camera_worker = create_camera_worker(
            source_type=settings.CAMERA_SOURCE_TYPE,
            source=settings.CAMERA_SOURCE,
            frame_buffer=self._frame_buffer,
        )

        self._accident_model, self._violence_model, self._audio_model = create_models(settings)

        self._accident_verifier = TemporalVerifier(
            config=TemporalVerificationConfig(
                confidence_threshold=settings.ACCIDENT_CONFIDENCE_THRESHOLD,
                min_hits=settings.ACCIDENT_MIN_HITS,
                window_sec=settings.ACCIDENT_WINDOW_SEC,
                cooldown_sec=settings.ACCIDENT_COOLDOWN_SEC,
                positive_label="ACCIDENT",
            )
        )
        self._violence_verifier = TemporalVerifier(
            config=TemporalVerificationConfig(
                confidence_threshold=settings.VIOLENCE_CONFIDENCE_THRESHOLD,
                min_hits=settings.VIOLENCE_MIN_HITS,
                window_sec=settings.VIOLENCE_WINDOW_SEC,
                cooldown_sec=settings.VIOLENCE_COOLDOWN_SEC,
                positive_label="VIOLENCE",
            )
        )

        self._emergency_provider: EmergencyProvider = create_emergency_provider(settings)

        self._verified_events: asyncio.Queue[VerifiedEvent] = asyncio.Queue()
        self._running = False
        self._last_predictions: dict[str, dict[str, Any] | None] = {
            "accident": None,
            "violence": None,
        }

        self._task_accident: asyncio.Task | None = None
        self._task_violence: asyncio.Task | None = None
        self._shutdown_event = asyncio.Event()

    @property
    def verified_events_queue(self) -> asyncio.Queue[VerifiedEvent]:
        return self._verified_events

    def get_latest_frame_bgr(self):
        """
        Demo helper only.

        Returns the most recent frame from the camera as a BGR numpy array,
        or `None` if the camera hasn't produced frames yet.
        """
        latest = self._frame_buffer.latest()
        return latest.frame_bgr if latest is not None else None

    def get_source_info(self) -> dict[str, Any]:
        return {
            "source_type": self._settings.CAMERA_SOURCE_TYPE,
            "source": self._settings.CAMERA_SOURCE,
            "camera_id": self._settings.CAMERA_ID,
        }

    def resolve_video_path(self) -> str | None:
        if self._settings.CAMERA_SOURCE_TYPE != "file":
            return None
        from pathlib import Path

        path = Path(self._settings.CAMERA_SOURCE)
        if not path.is_absolute():
            path = Path.cwd() / path
        if path.exists() and path.is_file():
            return str(path)
        return None

    def get_status(self) -> PipelineStatus:
        cam_status = self._camera_worker.status
        models = {
            "accident": self._accident_model.status(),
            "violence": self._violence_model.status(),
            "audio": self._audio_model.status(),
        }
        return PipelineStatus(
            camera={
                "source_type": cam_status.source_type,
                "source": cam_status.source,
                "running": cam_status.running,
                "last_error": cam_status.last_error,
                "last_frame_timestamp_epoch_s": cam_status.last_frame_timestamp_epoch_s,
            },
            models=models,
            verification={
                "accident_last_verified_epoch_s": self._accident_verifier.last_verified_epoch_s,
                "violence_last_verified_epoch_s": self._violence_verifier.last_verified_epoch_s,
            },
            last_predictions=self._last_predictions,
            running=self._running,
        )

    async def start(self) -> None:
        if self._running:
            return
        self._running = True

        self._camera_worker.start()

        self._task_accident = asyncio.create_task(self._inference_loop(model=self._accident_model, verifier=self._accident_verifier))
        self._task_violence = asyncio.create_task(self._inference_loop(model=self._violence_model, verifier=self._violence_verifier))

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        self._shutdown_event.set()

        if self._task_accident:
            self._task_accident.cancel()
        if self._task_violence:
            self._task_violence.cancel()

        self._camera_worker.stop()

        # Let canceled tasks exit cleanly.
        await asyncio.sleep(0)

    async def _inference_loop(self, *, model, verifier) -> None:
        """
        Periodically:
        - sample a temporal clip from FrameBuffer
        - run model inference
        - update temporal verifier
        - if verified, call EmergencyProvider
        """
        interval_s = max(0.001, self._settings.INFERENCE_INTERVAL_MS / 1000.0)
        last_model_loaded_log = 0.0

        while self._running and not self._shutdown_event.is_set():
            status: ModelStatus = model.status()
            if not status.loaded:
                # Avoid noisy logs: log at most every 15s per model.
                now = time.time()
                if now - last_model_loaded_log > 15:
                    last_model_loaded_log = now
                    print(f"[INFERENCE][{status.model_name}] model not loaded: {status.reason}")
                await asyncio.sleep(interval_s)
                continue

            clip_frames = self._frame_buffer.snapshot_last_n(self._settings.CLIP_FRAME_COUNT)
            if len(clip_frames) < 2:
                await asyncio.sleep(interval_s)
                continue

            frames_bgr = [bf.frame_bgr for bf in clip_frames]
            timestamps = [bf.timestamp_epoch_s for bf in clip_frames]
            clip_start_time = timestamps[0]

            from ..inference.base import ClipInput  # local import to avoid cycles

            clip_input = ClipInput(
                frames_bgr=frames_bgr,
                frame_timestamps_epoch_s=timestamps,
                clip_start_time_epoch_s=clip_start_time,
                camera_id=self._settings.CAMERA_ID,
            )

            try:
                prediction: ModelPrediction | None = await asyncio.to_thread(model.predict, clip_input)
            except Exception as e:
                # Never crash the pipeline due to one model.
                print(f"[INFERENCE][{status.model_name}] prediction error: {e}")
                await asyncio.sleep(interval_s)
                continue

            if prediction is not None:
                self._last_predictions[status.model_name] = {
                    "label": prediction.predicted_label,
                    "confidence": round(float(prediction.confidence), 4),
                    "timestamp_epoch_s": prediction.timestamp_epoch_s,
                    "camera_id": prediction.camera_id,
                }
                try:
                    verified: VerifiedEvent | None = verifier.update(prediction)
                except Exception as e:
                    print(f"[EVENT_VERIFICATION][{status.model_name}] verifier error: {e}")
                    verified = None

                if verified is not None:
                    # Push to demo listeners first
                    await self._verified_events.put(verified)
                    # Then perform emergency provider action (voice/notification reserved)
                    try:
                        await self._emergency_provider.on_verified_emergency(verified)
                    except Exception as e:
                        print(f"[EMERGENCY] provider error: {e}")

            await asyncio.sleep(interval_s)

