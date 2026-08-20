"""
ml/video/model.py
Video safety analysis model for the Live Multimodal Monitoring System.

Architecture
------------
VideoSafetyModel
├── Strategy 1 — HuggingFaceFireDetector
│     Model : EdBianchi/vit-fire-detection (HuggingFace)
│     Type  : ViT image classifier (pretrained + fine-tuned)
│     Labels: fire → FIRE | non_fire → NO_EVENT
│     Deps  : transformers, torch
│
└── Strategy 2 — ColorBasedFireDetector (fallback, always available)
      Approach : HSV pixel-ratio analysis
      Deps     : opencv-python only

Phase 2 supported events
------------------------
  FIRE       — detected via ViT model (primary) or HSV analysis (fallback)
  NO_EVENT   — default when no safety event is detected

PLANNED (future phases with appropriate models/datasets):
  VIOLENCE   — Phase 2 ext: RWF-2000 + action recognition model
  ACCIDENT   — Phase 2 ext: UCF-Crime anomaly detector
  CROWD_ANOMALY — crowd density/counting model

Honest model notes
------------------
- The HuggingFace model is a binary fire classifier, NOT a multi-class detector.
- Claims of VIOLENCE/ACCIDENT/CROWD_ANOMALY detection are NOT made in Phase 2.
- The colour-based fallback has higher false-positive rate (sunsets, red objects).
- "AI-generated detection. Human verification required."
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ml.base import BaseModalityModel, ModalityPredictionData
from ml.video.preprocessing import load_and_preprocess

logger = logging.getLogger(__name__)


# ================================================================== #
# Strategy 1 — HSV colour-based fire detector (always available)
# ================================================================== #
class ColorBasedFireDetector:
    """
    Rule-based fire/smoke detector using HSV colour analysis.
    Requires only opencv-python. No model weights to download.

    Detection principle
    -------------------
    Fire pixels typically fall in:
      - Orange-red zone: H ∈ [0°, 25°],  S > 100, V > 100
      - Bright yellow  : H ∈ [25°, 35°], S > 100, V > 200

    Confidence is derived from the fraction of fire-coloured pixels
    using a logistic-style mapping so small patches → low confidence
    and large fire areas → high (but capped) confidence.

    Limitations
    -----------
    - False positives: sunsets, red/orange objects, warm lighting
    - Should be combined with other modalities for reliability
    """

    # HSV colour ranges for fire detection
    _RANGES = [
        # core flame: orange-red
        (np.array([0,   100, 100], dtype=np.uint8),
         np.array([25,  255, 255], dtype=np.uint8)),
        # high flame: bright yellow
        (np.array([25,  100, 200], dtype=np.uint8),
         np.array([35,  255, 255], dtype=np.uint8)),
    ]
    _SCALE = 20   # pixels-to-confidence scaling factor

    def analyze_frame(self, frame_bgr: np.ndarray) -> Dict[str, float]:
        """
        Analyse one BGR frame.
        Returns dict: {"FIRE": float, "NO_EVENT": float}.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return {"FIRE": 0.0, "NO_EVENT": 1.0}

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        total_px = frame_bgr.shape[0] * frame_bgr.shape[1]

        fire_px = 0
        for lo, hi in self._RANGES:
            mask = cv2.inRange(hsv, lo, hi)
            fire_px += int(cv2.countNonZero(mask))

        ratio = fire_px / max(total_px, 1)
        # Logistic-style: ratio=0.05 → ~50%, ratio=0.15 → ~75%, ratio=0.25 → ~83%
        fire_conf = round(1.0 - 1.0 / (1.0 + ratio * self._SCALE), 4)
        fire_conf = min(fire_conf, 0.92)   # cap — colour alone is not conclusive

        return {"FIRE": fire_conf, "NO_EVENT": round(1.0 - fire_conf, 4)}

    def analyze_frames(self, frames: List[np.ndarray]) -> Dict[str, float]:
        """Aggregate across multiple frames — take max fire confidence."""
        if not frames:
            return {"FIRE": 0.0, "NO_EVENT": 1.0}
        per_frame = [self.analyze_frame(f) for f in frames]
        fire_max = max(s["FIRE"] for s in per_frame)
        return {"FIRE": round(fire_max, 4), "NO_EVENT": round(1.0 - fire_max, 4)}


# ================================================================== #
# Strategy 2 — HuggingFace ViT fire detector (optional, requires torch)
# ================================================================== #
class HuggingFaceFireDetector:
    """
    ViT-based binary fire classifier from HuggingFace.

    Model card
    ----------
    ID      : EdBianchi/vit-fire-detection
    Type    : ViT image classification (pretrained + fine-tuned)
    Classes : fire, non_fire
    Task    : binary classification

    Class mapping (project labels)
    -------------------------------
    fire     → FIRE
    non_fire → NO_EVENT

    Dataset note
    ------------
    Fine-tuned on a fire/non-fire image dataset.
    Accurately detects visible fire in still images.
    Does NOT detect smoke without visible flame.
    Does NOT detect VIOLENCE, ACCIDENT, or CROWD events.
    """

    MODEL_ID = "EdBianchi/vit-fire-detection"
    _LABEL_MAP = {
        "fire":     "FIRE",
        "non_fire": "NO_EVENT",
        # normalise whitespace / case variants
        "fire ":    "FIRE",
        "non fire": "NO_EVENT",
    }

    def __init__(self) -> None:
        self._pipe = None

    def load(self) -> bool:
        """Download and cache the HuggingFace model. Returns True on success."""
        try:
            from transformers import pipeline as hf_pipeline
            logger.info("Loading HuggingFace model '%s' …", self.MODEL_ID)
            self._pipe = hf_pipeline(
                "image-classification",
                model=self.MODEL_ID,
                device=-1,   # -1 = CPU; 0 = first GPU
            )
            logger.info("HuggingFace fire model loaded ✓")
            return True
        except ImportError:
            logger.warning(
                "transformers/torch not installed. "
                "Install with: pip install -r requirements-ml.txt"
            )
            return False
        except Exception as exc:
            logger.warning(
                "Failed to load '%s': %s. Falling back to colour-based detector.",
                self.MODEL_ID, exc,
            )
            return False

    @property
    def is_loaded(self) -> bool:
        return self._pipe is not None

    def analyze_image(self, pil_image) -> Dict[str, float]:
        """
        Run inference on a single PIL Image.
        Returns {"FIRE": float, "NO_EVENT": float}.
        Returns {} if model not loaded or inference fails.
        """
        if not self.is_loaded:
            return {}
        try:
            results = self._pipe(pil_image, top_k=None)
            scores: Dict[str, float] = {}
            for r in results:
                raw_label = r["label"].lower().strip().replace(" ", "_")
                event = self._LABEL_MAP.get(raw_label, "NO_EVENT")
                # Keep highest score per event (handles label variants)
                scores[event] = max(scores.get(event, 0.0), round(r["score"], 4))
            return scores
        except Exception as exc:
            logger.warning("HuggingFace inference failed: %s", exc)
            return {}

    def analyze_frames(self, pil_images: list) -> Dict[str, float]:
        """Aggregate across multiple PIL images — take max fire confidence."""
        if not pil_images:
            return {}
        per_frame = [self.analyze_image(img) for img in pil_images]
        valid = [s for s in per_frame if s]
        if not valid:
            return {}
        fire_max = max(s.get("FIRE", 0.0) for s in valid)
        no_event = 1.0 - fire_max
        return {"FIRE": round(fire_max, 4), "NO_EVENT": round(no_event, 4)}


# ================================================================== #
# Main VideoSafetyModel — implements BaseModalityModel
# ================================================================== #
class VideoSafetyModel(BaseModalityModel):
    """
    Public-safety video analysis model.

    Phase 2 supports: FIRE detection
    Future phases  : VIOLENCE, ACCIDENT, CROWD_ANOMALY (separate models)

    Detection pipeline
    ------------------
    raw_bytes → preprocess → [ViT model or colour analysis] → ModalityPrediction

    Threshold: confidence < 0.55 → event mapped to NO_EVENT
    """

    modality = "video"
    model_name = "VideoSafetyModel"
    model_version = "1.0.0"
    supported_events = ["FIRE", "NO_EVENT"]
    # VIOLENCE / ACCIDENT / CROWD_ANOMALY: PLANNED — separate models required

    _CONF_THRESHOLD = 0.55   # below this: report NO_EVENT

    def __init__(self) -> None:
        self._hf = HuggingFaceFireDetector()
        self._color = ColorBasedFireDetector()
        self._loaded = False
        self._backend = "uninitialised"

    # ---------------------------------------------------------------- #
    # Lifecycle
    # ---------------------------------------------------------------- #
    def load(self) -> None:
        """
        Attempt to load HuggingFace ViT model.
        Always succeeds — falls back to colour-based detector if needed.
        """
        hf_ok = self._hf.load()
        self._backend = "HuggingFace-ViT" if hf_ok else "colour-analysis"
        if hf_ok:
            self.model_name = f"vit-fire-detection ({self._hf.MODEL_ID})"
        else:
            self.model_name = "ColorBasedFireDetector"
        self._loaded = True
        logger.info(
            "VideoSafetyModel ready | backend=%s | events=%s",
            self._backend, self.supported_events,
        )

    @property
    def is_ready(self) -> bool:
        return self._loaded

    @property
    def backend(self) -> str:
        return self._backend

    # ---------------------------------------------------------------- #
    # BaseModalityModel interface
    # ---------------------------------------------------------------- #
    def preprocess(self, raw_input: dict) -> dict:
        """
        raw_input: {"bytes": bytes, "content_type": str}
        Returns: {"frames": List[ndarray], "pil_images": List[PIL.Image]}
        """
        data = raw_input.get("bytes", b"")
        ct = raw_input.get("content_type", "image/jpeg")
        frames, pil_images = load_and_preprocess(data, ct)
        return {"frames": frames, "pil_images": pil_images}

    def predict(self, preprocessed: dict) -> ModalityPredictionData:
        """
        Run inference on preprocessed frames.
        Returns a ModalityPredictionData with standardised fields.
        """
        frames: List[np.ndarray] = preprocessed.get("frames", [])
        pil_images: list = preprocessed.get("pil_images", [])
        n_frames = len(frames)

        if not frames:
            return self._no_event_result(reason="no_frames_extracted")

        # ---- Choose inference strategy -------------------------------- #
        if self._hf.is_loaded:
            scores = self._hf.analyze_frames(pil_images)
            if not scores:          # HF failed on all frames → colour fallback
                scores = self._color.analyze_frames(frames)
                used_backend = "colour-analysis (HF fallback)"
            else:
                used_backend = "HuggingFace-ViT"
        else:
            scores = self._color.analyze_frames(frames)
            used_backend = "colour-analysis"

        # ---- Map to project event ------------------------------------ #
        fire_conf = scores.get("FIRE", 0.0)

        if fire_conf >= self._CONF_THRESHOLD:
            event = "FIRE"
            confidence = fire_conf
            status = "active"
        else:
            event = "NO_EVENT"
            confidence = scores.get("NO_EVENT", 1.0 - fire_conf)
            status = "no_event"

        # ---- Build evidence ------------------------------------------ #
        evidence = [{
            "type": "video_analysis",
            "backend": used_backend,
            "frames_analyzed": n_frames,
            "fire_confidence": fire_conf,
            "fire_pixels_detected": fire_conf > 0.1,
        }]

        return ModalityPredictionData(
            modality="video",
            event=event,
            confidence=round(confidence, 4),
            timestamp=datetime.now(timezone.utc),
            evidence=evidence,
            status=status,
            raw_scores={k: round(v, 4) for k, v in scores.items()},
            model_name=self.model_name,
            model_version=self.model_version,
        )

    # ---------------------------------------------------------------- #
    # Helpers
    # ---------------------------------------------------------------- #
    def _no_event_result(self, reason: str = "") -> ModalityPredictionData:
        return ModalityPredictionData(
            modality="video",
            event="NO_EVENT",
            confidence=1.0,
            timestamp=datetime.now(timezone.utc),
            evidence=[{"reason": reason}],
            status="no_event",
            raw_scores={"FIRE": 0.0, "NO_EVENT": 1.0},
            model_name=self.model_name,
            model_version=self.model_version,
        )

    def __repr__(self) -> str:
        return (
            f"<VideoSafetyModel backend={self._backend} "
            f"ready={self._loaded} events={self.supported_events}>"
        )


# ================================================================== #
# Module-level singleton + loader
# ================================================================== #
_video_model: Optional[VideoSafetyModel] = None
_load_attempted = False


def get_video_model() -> VideoSafetyModel:
    """
    Return the module-level VideoSafetyModel singleton.
    Loads the model on first call (lazy initialisation).
    Thread-safe for single-process uvicorn workers.
    """
    global _video_model, _load_attempted
    if not _load_attempted:
        _load_attempted = True
        _video_model = VideoSafetyModel()
        _video_model.load()
    return _video_model
