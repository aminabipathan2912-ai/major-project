"""
backend/app/api/routes/video.py
Video analysis endpoint — Phase 2: real CV inference.

Supported uploads
-----------------
  Images : image/jpeg, image/png, image/bmp, image/webp
  Videos : video/mp4, video/avi, video/quicktime (frame sampling)

Processing pipeline (see VideoService / VideoSafetyModel for full detail)
--------------------------------------------------------------------------
  Upload → VideoService.analyze() → VideoSafetyModel.predict_raw()
         → ModalityPrediction → (optional WebSocket broadcast)

Responsible-AI note
-------------------
  All responses include the disclaimer:
  "AI-generated detection. Human verification required."
"""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas.common import ModalityPrediction
from app.services.video_service import VideoService

router = APIRouter(prefix="/video", tags=["video"])

# Accepted MIME types
_ACCEPTED_TYPES = {
    "image/jpeg", "image/jpg", "image/png",
    "image/bmp", "image/webp", "image/tiff",
    "video/mp4", "video/avi", "video/quicktime",
    "video/x-msvideo", "video/webm",
}

_MAX_UPLOAD_MB = 50
_MAX_BYTES = _MAX_UPLOAD_MB * 1024 * 1024


@router.post(
    "/analyze",
    response_model=ModalityPrediction,
    summary="Analyse a video frame or clip for public-safety events",
    description=(
        "Upload an image (JPEG/PNG) or short video clip (MP4/AVI). "
        "The system extracts up to 8 frames, runs fire detection, "
        "and returns a standardised ModalityPrediction. "
        "⚠️ AI-generated detection. Human verification required."
    ),
)
async def analyze_video(
    file: UploadFile = File(
        ...,
        description="Image file (JPEG/PNG) or video clip (MP4/AVI/MOV). Max 50 MB.",
    ),
) -> ModalityPrediction:
    """
    Phase 2 video analysis endpoint.

    Returns a ModalityPrediction with:
    - event    : FIRE | NO_EVENT
    - confidence: 0.0–1.0 (fire probability)
    - evidence : frames analyzed, backend used, fire pixel stats
    - raw_scores: full per-class softmax/heuristic scores
    """
    # ---- Validate content type --------------------------------------- #
    ct = (file.content_type or "").lower()
    if ct and ct not in _ACCEPTED_TYPES and not any(
        ct.startswith(p) for p in ("image/", "video/")
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unsupported file type: '{ct}'. "
                f"Accepted: image/jpeg, image/png, image/bmp, "
                f"video/mp4, video/avi, video/quicktime"
            ),
        )

    # ---- Read file --------------------------------------------------- #
    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")
    if len(file_bytes) > _MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed: {_MAX_UPLOAD_MB} MB.",
        )

    # ---- Run inference (delegated to VideoService) ------------------- #
    svc = VideoService()
    return await svc.analyze(file_bytes, ct or "image/jpeg")
