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
    sampling: dict[str, Any] | None = None


class PipelineService:
    """
    CCTV/IP camera -> frame buffering -> model inference -> temporal verification -> emergency hook.

    The camera layer and the model layers are independent by design.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

        self._active_source_type = settings.CAMERA_SOURCE_TYPE
        self._active_source = settings.CAMERA_SOURCE

        self._frame_buffer = FrameBuffer(maxlen=settings.FRAME_BUFFER_MAXLEN)
        self._camera_worker = create_camera_worker(
            source_type=self._active_source_type,
            source=self._active_source,
            frame_buffer=self._frame_buffer,
            loop_file=settings.FILE_LOOP,
            realtime_file=settings.FILE_REALTIME,
            inference_frame_size=settings.INFERENCE_FRAME_SIZE,
            ingest_resize_exact=settings.INGEST_RESIZE_EXACT,
            ingest_sample_fps=settings.INGEST_SAMPLE_FPS,
            preview_max_width=settings.PREVIEW_MAX_WIDTH,
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

        self._event_subscribers: set[asyncio.Queue[VerifiedEvent]] = set()
        # A recorded clip is one review session, not a live camera. Once an
        # incident type has escalated from that clip, do not call again for a
        # later overlapping window in the same video.
        self._file_escalated_event_types: set[str] = set()
        self._running = False
        self._last_predictions: dict[str, dict[str, Any] | None] = {
            "accident": None,
            "violence": None,
        }

        self._inference_task: asyncio.Task | None = None
        self._shutdown_event = asyncio.Event()

    def subscribe_verified_events(self) -> asyncio.Queue[VerifiedEvent]:
        queue: asyncio.Queue[VerifiedEvent] = asyncio.Queue()
        self._event_subscribers.add(queue)
        return queue

    def unsubscribe_verified_events(self, queue: asyncio.Queue[VerifiedEvent]) -> None:
        self._event_subscribers.discard(queue)

    async def _broadcast_verified_event(self, event: VerifiedEvent) -> None:
        for queue in tuple(self._event_subscribers):
            await queue.put(event)

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def emergency_provider(self) -> EmergencyProvider:
        return self._emergency_provider

    def get_latest_frame_bgr(self):
        """
        Demo helper only.

        Returns the most recent preview frame as a BGR numpy array, or `None` if
        the camera hasn't produced frames yet. This reads the preview slot, which
        is updated on every decoded frame and held at preview resolution — the
        inference ring buffer is decimated and downscaled, so it is the wrong
        source for a smooth browser view.
        """
        latest = self._frame_buffer.latest_preview()
        return latest[0] if latest is not None else None

    def get_latest_preview(self) -> tuple[Any, int] | None:
        """
        Returns (frame_bgr, seq) or None.

        `seq` only changes when the frame does, letting the JPEG encoder skip
        work instead of re-encoding an identical frame for every client poll.
        """
        return self._frame_buffer.latest_preview()

    def get_source_info(self) -> dict[str, Any]:
        return {
            "source_type": self._active_source_type,
            "source": self._active_source,
            "camera_id": self._settings.CAMERA_ID,
        }

    def resolve_video_path(self) -> str | None:
        if self._active_source_type != "file":
            return None
        from pathlib import Path

        path = Path(self._active_source)
        if not path.is_absolute():
            path = Path.cwd() / path
        if path.exists() and path.is_file():
            return str(path)
        return None

    async def switch_to_uploaded_file(self, path: str) -> None:
        """
        Point the pipeline at a newly uploaded clip.

        `restart_with` joins the capture thread with a 5s timeout, so it must not
        run on the event loop — a slow `cap.read()` would stall every other
        request, including the verified-event WebSocket.
        """
        self._active_source_type = "file"
        self._active_source = path
        self._file_escalated_event_types.clear()
        await asyncio.to_thread(
            self._camera_worker.restart_with, source_type="file", source=path
        )

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
                "frames_decoded": cam_status.frames_decoded,
                "frames_ingested": cam_status.frames_ingested,
                "buffered_frames": len(self._frame_buffer),
            },
            models=models,
            verification={
                "accident_last_verified_epoch_s": self._accident_verifier.last_verified_epoch_s,
                "violence_last_verified_epoch_s": self._violence_verifier.last_verified_epoch_s,
            },
            last_predictions=self._last_predictions,
            running=self._running,
            sampling={
                "clip_frame_count": self._settings.CLIP_FRAME_COUNT,
                "accident_window_sec": self._model_window_sec("accident"),
                "violence_window_sec": self._model_window_sec("violence"),
            },
        )

    def _model_window_sec(self, name: str) -> float:
        """Clip window for a model. 0 == last-N (unchanged) behaviour."""
        if name == "accident":
            return max(0.0, float(self._settings.ACCIDENT_CLIP_WINDOW_SEC))
        if name == "violence":
            return max(0.0, float(self._settings.VIOLENCE_CLIP_WINDOW_SEC))
        return 0.0

    async def start(self) -> None:
        if self._running:
            return
        self._running = True

        if self._active_source_type in ("webcam", "rtsp"):
            self._camera_worker.start()
        else:
            print("[CAMERA] waiting for an uploaded video on the demo page")

        await self._emergency_provider.start()
        self._inference_task = asyncio.create_task(self._inference_loop())

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        self._shutdown_event.set()

        if self._inference_task:
            self._inference_task.cancel()

        self._camera_worker.stop()
        await self._emergency_provider.aclose()

        # Let canceled tasks exit cleanly.
        await asyncio.sleep(0)

    async def _inference_loop(self) -> None:
        """
        One loop for both video models. Each cycle:
        - snapshot a single temporal clip from FrameBuffer
        - sample + preprocess it once, shared across models with the same frame
          count (accident + violence), instead of each repeating that work
        - run every loaded model's forward pass on that clip, sequentially
        - feed each prediction into its verifier; escalate verified events

        Both models judging the *same* snapshot is deliberate: the previous two
        independent loops drifted apart and scored different clips, which made
        their confidences incomparable and doubled the sampling + preprocess
        cost. Running the forwards one after another keeps peak activation
        memory the same as the old lock-serialized path.

        Per-model clip windows (ACCIDENT_/VIOLENCE_CLIP_WINDOW_SEC) break the
        sharing only when the two models are configured differently: then each
        samples its own clip and runs its own preprocess. With both at the
        default 0 the shared single-snapshot path above is used unchanged.
        """
        interval_s = max(0.001, self._settings.INFERENCE_INTERVAL_MS / 1000.0)
        targets = (
            ("accident", self._accident_model, self._accident_verifier),
            ("violence", self._violence_model, self._violence_verifier),
        )
        last_not_loaded_log: dict[str, float] = {}

        while self._running and not self._shutdown_event.is_set():
            loaded = []
            for name, model, verifier in targets:
                st = model.status()
                if st.loaded:
                    loaded.append((name, model, verifier))
                else:
                    now = time.time()
                    if now - last_not_loaded_log.get(name, 0.0) > 15:
                        last_not_loaded_log[name] = now
                        print(f"[INFERENCE][{name}] model not loaded: {st.reason}")

            if not loaded:
                await asyncio.sleep(interval_s)
                continue

            windows = {name: self._model_window_sec(name) for name, _, _ in loaded}

            if len(set(windows.values())) == 1:
                await self._run_shared(loaded, next(iter(windows.values())))
            else:
                for name, model, verifier in loaded:
                    clip_input = self._snapshot_clip(windows[name])
                    if clip_input is None:
                        continue
                    try:
                        prediction = await asyncio.to_thread(model.predict, clip_input)
                    except Exception as e:
                        print(f"[INFERENCE][{name}] prediction error: {e}")
                        continue
                    if prediction is not None:
                        await self._handle_prediction(name, prediction, verifier)

            await asyncio.sleep(interval_s)

    def _snapshot_clip(self, window_sec: float):
        """Build a ClipInput from the buffer, or None if too few frames yet."""
        from ..inference.base import ClipInput  # local import to avoid cycles

        frames = self._frame_buffer.snapshot_window(
            window_sec, self._settings.CLIP_FRAME_COUNT
        )
        if len(frames) < 2:
            return None
        timestamps = [bf.timestamp_epoch_s for bf in frames]
        return ClipInput(
            frames_bgr=[bf.frame_bgr for bf in frames],
            frame_timestamps_epoch_s=timestamps,
            clip_start_time_epoch_s=timestamps[0],
            camera_id=self._settings.CAMERA_ID,
        )

    async def _run_shared(self, loaded, window_sec: float) -> None:
        """One snapshot, one preprocess, shared across every loaded model."""
        clip_input = self._snapshot_clip(window_sec)
        if clip_input is None:
            return

        # Shared sample + preprocess: only when >1 model is loaded and they
        # agree on frame count. Any failure falls back to per-model predict().
        shared_batch = None
        frame_counts = {m.clip_num_frames for _, m, _ in loaded}
        if len(loaded) > 1 and len(frame_counts) == 1 and None not in frame_counts:
            try:
                shared_batch = await asyncio.to_thread(
                    loaded[0][1].preprocess_clip, clip_input
                )
            except Exception as e:
                print(f"[INFERENCE] shared preprocess failed, per-model path: {e}")
                shared_batch = None

        for name, model, verifier in loaded:
            try:
                if shared_batch is not None:
                    prediction = await asyncio.to_thread(
                        model.predict_preprocessed, shared_batch, clip_input
                    )
                else:
                    prediction = await asyncio.to_thread(model.predict, clip_input)
            except Exception as e:
                # Never crash the pipeline due to one model.
                print(f"[INFERENCE][{name}] prediction error: {e}")
                continue

            if prediction is not None:
                await self._handle_prediction(name, prediction, verifier)

    async def _handle_prediction(
        self, name: str, prediction: ModelPrediction, verifier
    ) -> None:
        """Record the prediction, run its verifier, escalate a verified event."""
        self._last_predictions[name] = {
            "label": prediction.predicted_label,
            "confidence": round(float(prediction.confidence), 4),
            "timestamp_epoch_s": prediction.timestamp_epoch_s,
            "camera_id": prediction.camera_id,
        }
        try:
            verified: VerifiedEvent | None = verifier.update(prediction)
        except Exception as e:
            print(f"[EVENT_VERIFICATION][{name}] verifier error: {e}")
            return

        if verified is None:
            return

        if (
            self._active_source_type == "file"
            and verified.event_type in self._file_escalated_event_types
        ):
            # Later windows from the same recording are expected to overlap.
            # Suppress a second call for the same incident type without
            # changing live-camera behavior.
            return

        # Show a verified event immediately. Escalation may wait on a remote
        # service, but must not delay the event feed.
        await self._broadcast_verified_event(verified)
        try:
            await self._emergency_provider.on_verified_emergency(verified)
        except Exception as e:
            print(f"[EMERGENCY] provider error: {e}")
        if self._active_source_type == "file":
            self._file_escalated_event_types.add(verified.event_type)
