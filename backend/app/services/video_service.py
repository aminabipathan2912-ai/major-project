"""
backend/app/services/video_service.py
Video inference service — wraps the VideoSafetyModel and handles
async execution, result conversion, and WebSocket broadcasting.

Design rules (from project spec)
---------------------------------
- ML inference must NEVER be called directly inside route functions.
- All inference goes through service classes.
- Services call the model, convert to API schema, and broadcast via WebSocket.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.schemas.common import ModalityPrediction, Modality

logger = logging.getLogger(__name__)

# Thread pool for CPU-bound inference (keeps the async event loop free)
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="video_inference")


def _get_ml_path() -> Path:
    """Return the ml/ directory (3 levels above backend/app/services/)."""
    return Path(__file__).resolve().parents[3] / "ml"


def _ensure_ml_on_path() -> None:
    """Add the repo root to sys.path so 'ml' package is importable."""
    repo_root = str(Path(__file__).resolve().parents[3])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


class VideoService:
    """
    Async service that runs VideoSafetyModel inference in a thread pool.

    Usage (inside a FastAPI route)
    --------------------------------
    svc = VideoService()
    prediction = await svc.analyze(file_bytes, content_type)
    """

    def __init__(self) -> None:
        _ensure_ml_on_path()
        # Import lazily so missing ML deps don't break Phase 1 startup
        try:
            from ml.video.model import get_video_model
            self._model = get_video_model()
        except ImportError as exc:
            logger.error("Failed to import video model: %s", exc)
            self._model = None

    # ---------------------------------------------------------------- #
    # Public API
    # ---------------------------------------------------------------- #
    async def analyze(
        self,
        file_bytes: bytes,
        content_type: str,
    ) -> ModalityPrediction:
        """
        Run video inference asynchronously.

        Parameters
        ----------
        file_bytes   : raw bytes of uploaded image or video file
        content_type : MIME type (e.g. "image/jpeg", "video/mp4")

        Returns
        -------
        ModalityPrediction (Pydantic schema) ready for the API response
        and WebSocket broadcast.
        """
        if self._model is None or not self._model.is_ready:
            logger.warning("VideoSafetyModel not ready — returning default NO_EVENT.")
            return self._default_prediction()

        loop = asyncio.get_event_loop()
        raw_input = {"bytes": file_bytes, "content_type": content_type}

        # Run CPU-bound inference in thread pool
        result = await loop.run_in_executor(
            _executor,
            self._model.predict_raw,
            raw_input,
        )

        prediction = self._to_schema(result)

        # Broadcast to connected WebSocket clients
        await self._broadcast(prediction)

        return prediction

    def model_info(self) -> dict:
        """Return a summary of the loaded model for the /models/status endpoint."""
        if self._model is None:
            return {"status": "UNAVAILABLE", "reason": "Model failed to import"}
        return {
            "name": self._model.model_name,
            "version": self._model.model_version,
            "backend": getattr(self._model, "backend", "unknown"),
            "supported_events": self._model.supported_events,
            "status": "LOADED" if self._model.is_ready else "LOADING",
        }

    # ---------------------------------------------------------------- #
    # Private helpers
    # ---------------------------------------------------------------- #
    @staticmethod
    def _to_schema(result) -> ModalityPrediction:
        """Convert ModalityPredictionData (ml layer) → ModalityPrediction (API schema)."""
        return ModalityPrediction(
            modality=Modality.VIDEO,
            event=result.event,
            confidence=result.confidence,
            timestamp=result.timestamp,
            evidence=result.evidence,
            status=result.status,
            raw_scores=result.raw_scores,
            model_name=result.model_name,
            model_version=result.model_version,
        )

    @staticmethod
    def _default_prediction() -> ModalityPrediction:
        return ModalityPrediction(
            modality=Modality.VIDEO,
            event="NO_EVENT",
            confidence=0.0,
            timestamp=datetime.now(timezone.utc),
            evidence=[{"reason": "model_not_ready"}],
            status="error",
            raw_scores={},
            model_name="VideoSafetyModel",
            model_version="1.0.0",
        )

    @staticmethod
    async def _broadcast(prediction: ModalityPrediction) -> None:
        """Push prediction to WebSocket dashboard if clients are connected."""
        try:
            from app.api.websocket import manager
            if manager.connection_count > 0:
                await manager.broadcast({
                    "type": "prediction",
                    "modality": "video",
                    "event": prediction.event,
                    "confidence": prediction.confidence,
                    "status": prediction.status,
                    "timestamp": prediction.timestamp.isoformat(),
                    "evidence": prediction.evidence,
                    "raw_scores": prediction.raw_scores,
                })
        except Exception as exc:
            logger.debug("WebSocket broadcast skipped: %s", exc)
