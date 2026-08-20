"""
backend/app/api/routes/audio.py
Audio analysis endpoint — Phase 1 stub.
Full ML pipeline implemented in Phase 3.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, File, UploadFile, HTTPException

from app.schemas.common import ModalityPrediction, Modality

router = APIRouter(prefix="/audio", tags=["audio"])


@router.post("/analyze", response_model=ModalityPrediction)
async def analyze_audio(file: UploadFile = File(...)):
    """
    Accept an audio chunk and return a modality prediction.
    Phase 1: returns a mock response. Full audio pipeline in Phase 3.
    """
    if not file.content_type or not file.content_type.startswith("audio/"):
        raise HTTPException(
            status_code=422,
            detail=f"File must be audio. Received content-type: {file.content_type}",
        )

    return ModalityPrediction(
        modality=Modality.AUDIO,
        event="no_event",
        confidence=0.0,
        timestamp=datetime.now(timezone.utc),
        evidence=[],
        status="no_event",
        raw_scores={},
        model_name="YAMNet-emergency-sounds",
        model_version="1.0.0",
    )
