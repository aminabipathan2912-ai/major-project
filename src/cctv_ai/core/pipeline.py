from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from ..camera.camera_factory import create_camera_worker
from ..camera.frame_buffer import FrameBuffer
from ..camera.phone_stream import PhoneStreamSource
from ..config import Settings
from ..inference.audio.audio_buffer import AudioBuffer
from ..core.models import ModelPrediction, ModelStatus, VerifiedEvent
from ..emergency.emergency_service import EmergencyProvider, create_emergency_provider
from ..event_engine.fusion import FusionEngine, ModalityEvidence, build_fusion_config
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
    phone: dict[str, Any] | None = None
    fusion: dict[str, Any] | None = None


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

        # Phone browser source. Constructed eagerly (cheap: no threads, no
        # sockets until a browser connects) so /api/status can always report it.
        self._audio_buffer = AudioBuffer(maxlen=settings.PHONE_AUDIO_BUFFER_MAX)
        self._phone_source = PhoneStreamSource(
            frame_buffer=self._frame_buffer,
            audio_buffer=self._audio_buffer,
            inference_frame_size=settings.INFERENCE_FRAME_SIZE,
            ingest_resize_exact=settings.INGEST_RESIZE_EXACT,
            preview_max_width=settings.PREVIEW_MAX_WIDTH,
            frame_queue_max=settings.PHONE_FRAME_QUEUE_MAX,
            max_message_bytes=settings.PHONE_MAX_MESSAGE_BYTES,
            max_sessions=settings.PHONE_MAX_SESSIONS,
        )
        # What to return to when the phone disconnects.
        self._prev_source_type = self._active_source_type
        self._prev_source = self._active_source

        (
            self._accident_model,
            self._violence_model,
            self._audio_model,
            self._text_model,
        ) = create_models(settings)

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
        # Audio escalates independently too. positive_label="SCREAM" makes the
        # verifier emit a VIOLENCE event (its ACCIDENT-or-VIOLENCE rule) with
        # verified_label "SCREAM", so a lone scream reaches the backend as a
        # type it accepts while still being traceable to the audio modality.
        self._audio_verifier = TemporalVerifier(
            config=TemporalVerificationConfig(
                confidence_threshold=settings.AUDIO_CONFIDENCE_THRESHOLD,
                min_hits=settings.AUDIO_MIN_HITS,
                window_sec=settings.AUDIO_WINDOW_SEC,
                cooldown_sec=settings.AUDIO_COOLDOWN_SEC,
                positive_label="SCREAM",
            )
        )

        self._fusion = FusionEngine(config=build_fusion_config(settings))

        self._emergency_provider: EmergencyProvider = create_emergency_provider(settings)

        self._event_subscribers: set[asyncio.Queue[VerifiedEvent]] = set()
        # A recorded clip is one review session, not a live camera. Once an
        # incident type has escalated from that clip, do not call again for a
        # later overlapping window in the same video.
        self._file_escalated_event_types: set[str] = set()
        # Last time each event type was escalated by ANY path (a video verifier,
        # the audio verifier, or fusion). Guards against the independent path and
        # the fusion path both calling out for one incident.
        self._last_escalation_epoch_s: dict[str, float] = {}
        self._running = False
        self._last_predictions: dict[str, dict[str, Any] | None] = {
            "accident": None,
            "violence": None,
            "audio": None,
            "text": None,
        }

        self._inference_task: asyncio.Task | None = None
        self._audio_task: asyncio.Task | None = None
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
        self._prev_source_type = "file"
        self._prev_source = path
        self._file_escalated_event_types.clear()
        self._last_escalation_epoch_s.clear()
        self._fusion.reset()
        await asyncio.to_thread(
            self._camera_worker.restart_with, source_type="file", source=path
        )

    # ------------------------------------------------------------------
    # Phone browser source
    # ------------------------------------------------------------------

    @property
    def phone_source(self) -> PhoneStreamSource:
        return self._phone_source

    async def switch_to_phone(self) -> None:
        """
        Hand the pipeline over to the phone stream.

        The OpenCV worker is stopped (in a thread — it joins its capture thread)
        so the two sources can never interleave frames from two different scenes
        into a single clip.
        """
        if self._active_source_type == "phone":
            return
        self._prev_source_type = self._active_source_type
        self._prev_source = self._active_source
        await asyncio.to_thread(self._camera_worker.stop)
        self._frame_buffer.clear()
        self._audio_buffer.clear()
        self._fusion.reset()
        self._active_source_type = "phone"
        self._active_source = "phone"
        self._file_escalated_event_types.clear()
        self._last_escalation_epoch_s.clear()
        self._phone_source.start()
        print("[PHONE] live source active")

    async def revert_from_phone(self) -> None:
        """
        Return to whatever source was active before the phone connected.

        A finished file is deliberately *not* restarted: replaying it would
        re-detect the same incident and escalate a second time. The existing
        `_file_escalated_event_types` suppression is preserved by leaving the
        recorded clip's escalation state alone.
        """
        if self._active_source_type != "phone":
            return
        self._phone_source.stop()
        self._frame_buffer.clear()
        self._audio_buffer.clear()
        self._fusion.reset()
        self._active_source_type = self._prev_source_type
        self._active_source = self._prev_source
        if self._prev_source_type in ("webcam", "rtsp"):
            await asyncio.to_thread(
                self._camera_worker.restart_with,
                source_type=self._prev_source_type,
                source=self._prev_source,
            )
        print(f"[PHONE] disconnected; source reverted to {self._prev_source_type}")

    def get_status(self) -> PipelineStatus:
        if self._active_source_type == "phone":
            src_status = self._phone_source.status
        else:
            src_status = self._camera_worker.status
        models = {
            "accident": self._accident_model.status(),
            "violence": self._violence_model.status(),
            "audio": self._audio_model.status(),
            "text": self._text_model.status(),
        }
        phone_status = self._phone_source.status
        return PipelineStatus(
            camera={
                "source_type": src_status.source_type,
                "source": src_status.source,
                "running": src_status.running,
                "last_error": src_status.last_error,
                "last_frame_timestamp_epoch_s": src_status.last_frame_timestamp_epoch_s,
                "frames_decoded": src_status.frames_decoded,
                "frames_ingested": src_status.frames_ingested,
                "buffered_frames": len(self._frame_buffer),
            },
            phone={
                "connected": self._phone_source.has_session,
                "active": self._active_source_type == "phone",
                "sessions": phone_status.sessions,
                "frames_dropped": phone_status.frames_dropped,
                "rejected_messages": phone_status.rejected_messages,
                "audio_chunks": phone_status.audio_chunks,
                "audio": phone_status.audio,
            },
            models=models,
            verification={
                "accident_last_verified_epoch_s": self._accident_verifier.last_verified_epoch_s,
                "violence_last_verified_epoch_s": self._violence_verifier.last_verified_epoch_s,
                "audio_last_verified_epoch_s": self._audio_verifier.last_verified_epoch_s,
            },
            last_predictions=self._last_predictions,
            running=self._running,
            sampling={
                "clip_frame_count": self._settings.CLIP_FRAME_COUNT,
                "accident_window_sec": self._model_window_sec("accident"),
                "violence_window_sec": self._model_window_sec("violence"),
            },
            fusion=self._fusion.status(),
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
        self._audio_task = asyncio.create_task(self._audio_loop())

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        self._shutdown_event.set()

        if self._inference_task:
            self._inference_task.cancel()
        if self._audio_task:
            self._audio_task.cancel()

        self._phone_source.stop()
        self._camera_worker.stop()
        await self._emergency_provider.aclose()

        # Let canceled tasks exit cleanly.
        await asyncio.sleep(0)

    def phone_client_config(self) -> dict[str, Any]:
        """Capture knobs the /phone page reads, so sampling stays env-driven."""
        return {
            "send_fps": self._settings.PHONE_SEND_FPS,
            "frame_max_width": self._settings.PHONE_FRAME_MAX_WIDTH,
            "jpeg_quality": self._settings.PHONE_JPEG_QUALITY,
            "audio_chunk_ms": self._settings.PHONE_AUDIO_CHUNK_MS,
            "max_message_bytes": self._settings.PHONE_MAX_MESSAGE_BYTES,
        }

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

    async def _audio_loop(self) -> None:
        """
        Classify recent phone audio on its own cadence.

        Separate from the video loop because audio arrives on a different clock
        and its clip length is set by the model, not by the frame buffer. Exits
        immediately when no audio model is loaded, so an untrained deployment
        pays nothing.
        """
        interval_s = max(
            0.05, self._settings.AUDIO_INFERENCE_INTERVAL_MS / 1000.0
        )
        while self._running and not self._shutdown_event.is_set():
            await asyncio.sleep(interval_s)
            if not self._audio_model.status().loaded:
                continue

            chunks = self._audio_buffer.snapshot()
            if not chunks:
                continue

            try:
                prediction = await asyncio.to_thread(
                    self._audio_model.predict_audio,
                    chunks,
                    camera_id=self._settings.CAMERA_ID,
                )
            except Exception as e:
                print(f"[INFERENCE][audio] prediction error: {e}")
                continue
            if prediction is None:
                continue

            # Skip a silent room: running a classifier on near-silence invites
            # exactly the out-of-distribution guessing seen with video.
            dbfs = prediction.metadata.get("rms_dbfs")
            if dbfs is not None and dbfs < self._settings.AUDIO_MIN_DBFS:
                self._last_predictions["audio"] = {
                    "label": "SILENCE",
                    "confidence": 0.0,
                    "timestamp_epoch_s": prediction.timestamp_epoch_s,
                    "camera_id": prediction.camera_id,
                }
                continue

            await self._handle_prediction("audio", prediction, self._audio_verifier)

    async def submit_text(self, text: str) -> dict[str, Any]:
        """
        Classify one incoming message (helpline transcript, social post).

        Returns a small result dict for the HTTP caller. The prediction flows
        into fusion like any other modality, so text corroborates video/audio
        rather than escalating on its own.
        """
        status = self._text_model.status()
        if not status.loaded:
            return {"ok": False, "error": status.reason}

        prediction = await asyncio.to_thread(
            self._text_model.predict_message, text, camera_id=self._settings.CAMERA_ID
        )
        if prediction is None:
            return {"ok": False, "error": "No prediction (empty text?)."}

        await self._handle_prediction("text", prediction, None)
        return {
            "ok": True,
            "label": prediction.predicted_label,
            "confidence": round(float(prediction.confidence), 4),
            "metadata": prediction.metadata,
        }

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

    # Which event types each modality can speak to. A video model speaks only to
    # its own class. Audio speaks to VIOLENCE only: a scream is a fight signal,
    # and letting it also corroborate ACCIDENT let a weak out-of-domain accident
    # reading (~0.6 on an indoor scene) ride the audio score into a call. Text is
    # explicit language, not an acoustic guess, so it can speak to both.
    _MODALITY_SUPPORTS = {
        "accident": frozenset({"ACCIDENT"}),
        "violence": frozenset({"VIOLENCE"}),
        "audio": frozenset({"VIOLENCE"}),
        "text": frozenset({"ACCIDENT", "VIOLENCE"}),
    }
    _MODALITY_KEYS = {"accident": "video_accident", "violence": "video_violence"}

    async def _handle_prediction(
        self, name: str, prediction: ModelPrediction, verifier
    ) -> None:
        """
        Record the prediction, run its verifier, then decide escalation.

        A modality with a verifier (video accident, video violence, audio
        scream) escalates on its own as soon as its verifier confirms — fusion
        does not hold it back. When fusion is enabled it runs in addition: a
        corroborated combination can escalate too, and the dashboard panel
        explains which modalities agreed. `_escalate` enforces one cooldown per
        event type across both paths, so an incident seen by a single modality
        and then corroborated does not call twice.
        """
        self._last_predictions[name] = {
            "label": prediction.predicted_label,
            "confidence": round(float(prediction.confidence), 4),
            "timestamp_epoch_s": prediction.timestamp_epoch_s,
            "camera_id": prediction.camera_id,
        }

        self._observe_for_fusion(name, prediction)

        # Audio and text have no temporal verifier of their own: they are
        # corroborating evidence, not independent triggers. Passing verifier=None
        # is how a modality opts out of escalating on its own.
        verified: VerifiedEvent | None = None
        if verifier is not None:
            try:
                verified = verifier.update(prediction)
            except Exception as e:
                print(f"[EVENT_VERIFICATION][{name}] verifier error: {e}")
                verified = None

        if verified is not None:
            # Surface what the single modality concluded, and act on it. Fusion
            # being enabled does not hold this back — it only adds a second way
            # to escalate. The per-event-type cooldown in `_escalate` keeps a
            # later corroboration from calling again for the same incident.
            await self._broadcast_verified_event(verified)
            await self._escalate(verified)

        if self._fusion.config.enabled:
            await self._evaluate_fusion(prediction.camera_id)

    def _observe_for_fusion(self, name: str, prediction: ModelPrediction) -> None:
        """Feed one modality's reading into the fusion engine."""
        supports = self._MODALITY_SUPPORTS.get(name)
        if not supports:
            return
        # Confidence is reported for the *positive* class only. A NORMAL
        # prediction is positive evidence of nothing, so it contributes 0 and
        # actively dilutes the weighted mean.
        floor = {
            "audio": self._settings.AUDIO_CONFIDENCE_THRESHOLD,
            "text": self._settings.TEXT_CONFIDENCE_THRESHOLD,
        }.get(name, 0.0)
        positive = (
            prediction.predicted_label not in ("NORMAL", "OTHER", "SILENCE")
            and float(prediction.confidence) >= floor
        )
        # A text model returns a confident label even for a sentence made
        # entirely of unknown words. Require it to have recognised enough of the
        # message before its opinion counts as evidence.
        if positive and name == "text":
            coverage = prediction.metadata.get("vocab_coverage")
            if (
                coverage is not None
                and coverage < self._settings.TEXT_MIN_VOCAB_COVERAGE
            ):
                positive = False
        conf = float(prediction.confidence) if positive else 0.0
        if positive:
            supports = supports & {prediction.predicted_label} or supports
        self._fusion.observe(
            ModalityEvidence(
                modality=self._MODALITY_KEYS.get(name, name),
                confidence=conf,
                timestamp_epoch_s=prediction.timestamp_epoch_s,
                supports=frozenset(supports),
                label=prediction.predicted_label,
            )
        )

    async def _evaluate_fusion(self, camera_id: str) -> None:
        for event_type in ("ACCIDENT", "VIOLENCE"):
            try:
                fused = self._fusion.evaluate(event_type, camera_id=camera_id)
            except Exception as e:
                print(f"[FUSION] evaluate error: {e}")
                continue
            if fused is not None:
                print(
                    f"[FUSION] {event_type} corroborated by "
                    f"{fused.details['supporting_modalities']} "
                    f"score={fused.details['fused_score']}"
                )
                await self._broadcast_verified_event(fused)
                await self._escalate(fused)

    async def _escalate(self, verified: VerifiedEvent) -> None:
        """Send a verified event to the emergency provider, with dedup guards."""
        if (
            self._active_source_type == "file"
            and verified.event_type in self._file_escalated_event_types
        ):
            # Later windows from the same recording are expected to overlap.
            # Suppress a second call for the same incident type without
            # changing live-camera behavior.
            return

        # One cooldown per event type across every escalation path. A video
        # verifier and fusion (or the audio verifier and a video verifier) can
        # each confirm the same incident seconds apart; only the first calls.
        now = time.time()
        last = self._last_escalation_epoch_s.get(verified.event_type)
        cooldown = float(self._settings.ESCALATION_COOLDOWN_SEC)
        if last is not None and (now - last) < cooldown:
            print(
                f"[EMERGENCY] {verified.event_type} within escalation cooldown "
                f"({now - last:.1f}s < {cooldown:.0f}s); not re-sent"
            )
            return
        self._last_escalation_epoch_s[verified.event_type] = now

        try:
            await self._emergency_provider.on_verified_emergency(verified)
        except Exception as e:
            print(f"[EMERGENCY] provider error: {e}")
        if self._active_source_type == "file":
            self._file_escalated_event_types.add(verified.event_type)
