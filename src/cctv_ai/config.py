from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # "phone" is not a startup source: the pipeline switches to it when a phone
    # browser connects to /ws/ingest, and reverts when it disconnects.
    CAMERA_SOURCE_TYPE: Literal["webcam", "rtsp", "file", "phone"] = "file"
    CAMERA_SOURCE: str = "tests/fixtures/sample.mp4"
    CAMERA_ID: str = "camera-1"
    INCIDENT_LOCATION: str = "Main Road, test site"
    # Uploaded/test clips should behave like a camera feed: play once at the
    # source FPS instead of racing through and repeatedly retriggering alerts.
    FILE_LOOP: bool = False
    FILE_REALTIME: bool = True

    INFERENCE_INTERVAL_MS: int = 1000
    CLIP_FRAME_COUNT: int = 8
    # 32 comfortably retains the 8-frame inference clip without keeping a
    # large number of full-resolution OpenCV frames in RAM.
    FRAME_BUFFER_MAXLEN: int = 32

    # Frames are downscaled once at ingest to the shorter side the training
    # transform resizes to anyway, so the ring buffer holds ~20x less at 1080p
    # without changing the tensor the models receive. See
    # scripts/check_preprocess_equivalence.py.
    INFERENCE_FRAME_SIZE: int = 256
    # True uses PIL BICUBIC, which is bit-identical to the training transform
    # (EfficientNet_B0_Weights.DEFAULT.transforms() resizes with BICUBIC).
    # False uses cv2 INTER_AREA: faster, but only enable it after the
    # equivalence script reports an acceptable difference on your own clips.
    INGEST_RESIZE_EXACT: bool = True
    # Frames per second pushed into the inference ring. 0 disables decimation
    # and keeps every decoded frame, which reproduces the pre-optimization clip
    # exactly: snapshot_last_n(8) then spans the same fraction of a second it
    # does today. A non-zero value widens that span (fewer near-duplicate
    # frames per clip) and is a Phase 3 / B7 decision, not a Phase 1 one.
    INGEST_SAMPLE_FPS: float = 0.0

    # Browser preview only. No model ever sees these pixels, so they are
    # encoded once per new frame and shared by every connected client.
    PREVIEW_FPS: float = 10.0
    PREVIEW_MAX_WIDTH: int = 640
    PREVIEW_JPEG_QUALITY: int = 80

    # Torch intra-op threads. The default fans out to every core, which
    # oversubscribes once two models plus the event loop share the box. 0 keeps
    # torch's default. Applied once at model-load time.
    TORCH_NUM_THREADS: int = 0

    # cv2 batched clip preprocessing instead of per-frame PIL. Frames leave the
    # ring already at INFERENCE_FRAME_SIZE, so this is a centre-crop + normalise
    # on a single stacked array. Off by default (byte-identical PIL path); turn
    # on only after scripts/check_preprocess_equivalence.py reports the fast
    # path is within tolerance on your clips.
    CLIP_PREPROCESS_FAST: bool = False

    # Per-model clip sampling window. 0 (default) keeps the old behaviour: the
    # clip is the last CLIP_FRAME_COUNT consecutive frames (~0.3 s tail of the
    # interval). A positive value spreads the same CLIP_FRAME_COUNT frames
    # evenly across the trailing N seconds instead, which matches how the models
    # were trained (frames sampled across a whole clip) and covers the whole
    # inference interval. Detection is byte-identical until a value is set.
    # The two models can differ; when they do, each samples its own clip and
    # the shared-tensor fast path is skipped for that cycle. A window wider than
    # the buffer holds (FRAME_BUFFER_MAXLEN frames of history) is clamped to
    # what is available.
    ACCIDENT_CLIP_WINDOW_SEC: float = 0.0
    VIOLENCE_CLIP_WINDOW_SEC: float = 0.0

    # Phone browser live source (/phone -> WS /ws/ingest).
    # The first four are echoed to the browser, which encodes frames itself so
    # the uplink stays small and the server only pays one imdecode per frame.
    PHONE_SEND_FPS: float = 5.0
    PHONE_FRAME_MAX_WIDTH: int = 480
    PHONE_JPEG_QUALITY: float = 0.7
    PHONE_AUDIO_CHUNK_MS: int = 1000
    # Server-side bounds. These hold regardless of what the client does.
    PHONE_FRAME_QUEUE_MAX: int = 4
    PHONE_AUDIO_BUFFER_MAX: int = 8
    PHONE_MAX_MESSAGE_BYTES: int = 1048576
    # Two streams into one FrameBuffer would interleave two scenes into a single
    # clip and corrupt inference, so the second connection is refused cleanly.
    PHONE_MAX_SESSIONS: int = 1

    ACCIDENT_CONFIDENCE_THRESHOLD: float = 0.6
    ACCIDENT_MIN_HITS: int = 3
    ACCIDENT_WINDOW_SEC: float = 3.0
    ACCIDENT_COOLDOWN_SEC: float = 30.0

    VIOLENCE_CONFIDENCE_THRESHOLD: float = 0.6
    VIOLENCE_MIN_HITS: int = 2
    VIOLENCE_WINDOW_SEC: float = 3.0
    VIOLENCE_COOLDOWN_SEC: float = 20.0

    # Decision-level multimodal fusion.
    #
    # Fusion is ADDITIVE, not a gate. Each modality that has its own verifier
    # (video accident, video violence, audio scream) still escalates on its own.
    # With fusion enabled, a corroborated combination can ALSO escalate — and the
    # dashboard panel explains which modalities agreed. It never suppresses an
    # independent single-modality escalation.
    FUSION_ENABLED: bool = False
    # How long a modality's reading stays usable as corroboration.
    FUSION_WINDOW_SEC: float = 5.0
    # Independent modalities that must agree before escalation.
    FUSION_MIN_MODALITIES: int = 2
    # Confidence one modality needs before it counts as "supporting".
    FUSION_SUPPORT_THRESHOLD: float = 0.6
    # Weighted-mean score the combined evidence must reach.
    FUSION_THRESHOLD: float = 0.6
    FUSION_COOLDOWN_SEC: float = 30.0
    # Minimum gap between two escalations of the SAME event type, across every
    # path (a video verifier, the audio verifier, or fusion). Stops the
    # independent path and the fusion path both calling for one incident.
    ESCALATION_COOLDOWN_SEC: float = 30.0
    # Relative trust per modality. Text is weighted lower: it carries context
    # rather than direct observation.
    FUSION_WEIGHT_VIDEO: float = 1.0
    FUSION_WEIGHT_AUDIO: float = 1.0
    FUSION_WEIGHT_TEXT: float = 0.5

    ACCIDENT_MODEL_WEIGHTS_PATH: str = "models/accident_best.pt"
    VIOLENCE_MODEL_WEIGHTS_PATH: str = "models/violence_best.pt"
    # Audio and text weights. Empty (the default) means the modality reports
    # "not loaded" and produces no predictions — it never guesses.
    AUDIO_MODEL_WEIGHTS_PATH: str = ""
    TEXT_MODEL_WEIGHTS_PATH: str = ""

    # How often buffered phone audio is classified, and the positive label the
    # audio model emits. Audio is evidence of distress generally, so it
    # corroborates both accident and violence rather than defining its own type.
    AUDIO_INFERENCE_INTERVAL_MS: int = 1000
    AUDIO_CONFIDENCE_THRESHOLD: float = 0.6
    # Below this loudness a chunk is treated as silence and skipped, so the
    # model is not run on (and cannot hallucinate from) an empty room.
    AUDIO_MIN_DBFS: float = -55.0
    # Audio escalates on its own, like the video models: a scream must persist
    # across AUDIO_MIN_HITS classifications inside AUDIO_WINDOW_SEC before it
    # calls out, then holds off for AUDIO_COOLDOWN_SEC. A lone scream is reported
    # as a VIOLENCE event (the label the hosted backend accepts); verified_label
    # stays "SCREAM" so the origin is visible.
    AUDIO_MIN_HITS: int = 2
    AUDIO_WINDOW_SEC: float = 6.0
    AUDIO_COOLDOWN_SEC: float = 30.0

    TEXT_CONFIDENCE_THRESHOLD: float = 0.6
    # Fraction of a message's tokens the model must actually know before its
    # prediction is trusted as evidence. A softmax over an all-unknown sentence
    # still returns a confident label — the text equivalent of the video models
    # confidently classifying a scene they were never trained on.
    TEXT_MIN_VOCAB_COVERAGE: float = 0.5

    # remote posts verified events to the lightweight hosted alert backend.
    EMERGENCY_MODE: Literal["log-only", "remote"] = "log-only"
    ALERT_BACKEND_URL: str = ""
    ALERT_BACKEND_TOKEN: str = ""
    # How often the background task refreshes the active incident's status from
    # the hosted backend while a call is in progress. /api/status only ever
    # reads the cached result, so this never blocks the dashboard.
    INCIDENT_POLL_INTERVAL_SEC: float = 3.0

    VIDEO_UPLOAD_DIR: str = "data/uploads"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached: constructing Settings re-reads and re-parses .env every time."""
    return Settings()
