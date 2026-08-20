"""
backend/app/api/routes/text.py
Text / NLP analysis endpoint — Phase 4: disaster tweet classification.

Uses DisasterTextModel (DistilBERT + keyword fallback) from ml/text/model.py.
"""

from __future__ import annotations

import asyncio
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.schemas.common import ModalityPrediction, Modality
from app.core.logging import get_logger

router = APIRouter(prefix="/text", tags=["text"])
logger = get_logger(__name__)

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="text_inference")


def _ensure_ml_on_path() -> None:
    repo_root = str(Path(__file__).resolve().parents[4])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


class TextAnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000,
                      description="Text to classify (tweet, call transcript, news snippet)")
    source: str = Field(default="manual",
                        description="Source type: 'manual', 'social_media', 'emergency_call', 'news'")


class TextService:
    """Async wrapper around DisasterTextModel."""

    def __init__(self) -> None:
        _ensure_ml_on_path()
        try:
            from ml.text.model import get_text_model
            self._model = get_text_model()
        except ImportError as exc:
            logger.warning("ml.text.model not importable: %s — keyword fallback active", exc)
            self._model = None

    async def classify(self, text: str, source: str) -> ModalityPrediction:
        if self._model is None or not self._model.is_ready:
            return self._keyword_fallback(text, source)

        loop = asyncio.get_event_loop()
        raw_input = {"text": text, "source": source}

        result = await loop.run_in_executor(
            _executor,
            self._model.predict_raw,
            raw_input,
        )

        return ModalityPrediction(
            modality=Modality.TEXT,
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
    def _keyword_fallback(text: str, source: str) -> ModalityPrediction:
        """Inline keyword fallback when ml module unavailable."""
        import re
        from datetime import datetime, timezone

        HIGH_KW = {
            'earthquake','tsunami','tornado','hurricane','cyclone',
            'flood','flooding','wildfire','explosion','crash',
            'accident','emergency','disaster','sos','killed','injured',
            'trapped','evacuate','fatalities','casualties','derailment',
        }
        MEDIUM_KW = {
            'fire','smoke','burning','damage','danger','alert',
            'rescue','ambulance','police','storm','help',
        }
        words = set(re.findall(r"\b\w+\b", text.lower()))
        score = sum(0.35 for w in words & HIGH_KW) + sum(0.18 for w in words & MEDIUM_KW)
        disaster_conf = round(min(0.92, score), 4)

        event = "DISASTER" if disaster_conf >= 0.40 else "NON_EMERGENCY"
        return ModalityPrediction(
            modality=Modality.TEXT,
            event=event,
            confidence=disaster_conf if event == "DISASTER" else round(1.0 - disaster_conf, 4),
            timestamp=datetime.now(timezone.utc),
            evidence=[{"text_length": len(text), "source": source, "backend": "keyword-fallback"}],
            status="active" if event == "DISASTER" else "no_event",
            raw_scores={"DISASTER": disaster_conf, "NON_EMERGENCY": round(1.0 - disaster_conf, 4)},
            model_name="DisasterTextModel (keyword-fallback)",
            model_version="1.0.0",
        )


@router.post(
    "/analyze",
    response_model=ModalityPrediction,
    summary="Classify text for disaster/emergency content",
    description=(
        "Accepts social-media posts, emergency transcripts, or news snippets. "
        "Returns disaster probability using DistilBERT (or keyword fallback). "
        "Dataset: Kaggle NLP2 — Twitter Disaster Tweets. "
        "⚠️ AI-generated classification. Human verification required."
    ),
)
async def analyze_text(request: TextAnalyzeRequest) -> ModalityPrediction:
    """Phase 4 NLP endpoint — disaster tweet classification."""
    svc = TextService()
    result = await svc.classify(request.text, request.source)
    logger.info(
        "Text analysis: event=%s conf=%.3f source=%s len=%d",
        result.event, result.confidence, request.source, len(request.text),
    )
    return result
