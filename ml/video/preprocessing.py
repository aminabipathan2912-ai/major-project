"""
ml/video/preprocessing.py
Frame extraction and preprocessing for the video safety pipeline.

Responsibilities:
  - Load image bytes → BGR numpy array
  - Extract evenly-spaced frames from a video clip
  - Resize and normalise frames for model input
  - Convert BGR frames to PIL Images (for HuggingFace pipelines)
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Constants
# ------------------------------------------------------------------ #
MAX_VIDEO_FRAMES = 8          # maximum frames sampled from a video
FRAME_SIZE: Tuple[int, int] = (224, 224)  # (width, height) — standard ViT input

IMAGE_MIME_TYPES = {
    "image/jpeg", "image/jpg", "image/png",
    "image/bmp", "image/webp", "image/tiff",
}
VIDEO_MIME_TYPES = {
    "video/mp4", "video/avi", "video/quicktime",
    "video/x-msvideo", "video/webm", "video/x-matroska",
}


# ------------------------------------------------------------------ #
# Image loader
# ------------------------------------------------------------------ #
def load_image_from_bytes(data: bytes) -> Optional[np.ndarray]:
    """
    Decode image bytes into a BGR numpy array.
    Returns None if decoding fails.
    """
    try:
        arr = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            logger.warning("cv2.imdecode returned None — invalid image bytes")
        return frame
    except Exception as exc:
        logger.error("Failed to load image from bytes: %s", exc)
        return None


# ------------------------------------------------------------------ #
# Video frame extractor
# ------------------------------------------------------------------ #
def extract_frames_from_video_bytes(
    data: bytes,
    max_frames: int = MAX_VIDEO_FRAMES,
) -> List[np.ndarray]:
    """
    Write video bytes to a temporary file, open with OpenCV,
    and extract up to `max_frames` evenly-spaced frames.

    Returns a list of BGR numpy arrays. Empty list on failure.
    """
    frames: List[np.ndarray] = []
    tmp_path: Optional[str] = None

    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            logger.error("OpenCV could not open video: %s", tmp_path)
            return frames

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            logger.warning("Video has 0 frames or unknown frame count.")
            cap.release()
            return frames

        n = min(max_frames, total)
        indices = np.linspace(0, total - 1, n, dtype=int)

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(frame)

        cap.release()
        logger.debug("Extracted %d frames from video (total=%d)", len(frames), total)

    except Exception as exc:
        logger.error("Frame extraction failed: %s", exc)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return frames


# ------------------------------------------------------------------ #
# Frame preprocessor
# ------------------------------------------------------------------ #
def preprocess_frame(
    frame_bgr: np.ndarray,
    target_size: Tuple[int, int] = FRAME_SIZE,
) -> np.ndarray:
    """
    Resize a BGR frame to target_size using area interpolation.
    Returns the resized frame (still BGR numpy array).
    """
    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError("preprocess_frame received an empty/None frame")
    resized = cv2.resize(frame_bgr, target_size, interpolation=cv2.INTER_AREA)
    return resized


def frame_to_pil(frame_bgr: np.ndarray) -> Image.Image:
    """Convert a BGR numpy array to an RGB PIL Image (for HuggingFace pipelines)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


# ------------------------------------------------------------------ #
# Unified input dispatcher
# ------------------------------------------------------------------ #
def load_and_preprocess(
    raw_bytes: bytes,
    content_type: str,
    max_frames: int = MAX_VIDEO_FRAMES,
    target_size: Tuple[int, int] = FRAME_SIZE,
) -> Tuple[List[np.ndarray], List[Image.Image]]:
    """
    Given raw file bytes and MIME type, return a list of preprocessed
    BGR frames and their corresponding PIL Images.

    Returns ([], []) if nothing can be extracted.
    """
    ct = (content_type or "").lower()

    # ---- Determine input type ---------------------------------------- #
    if any(v in ct for v in ("video", ".mp4", ".avi", ".mov")):
        raw_frames = extract_frames_from_video_bytes(raw_bytes, max_frames)
    else:
        # Treat as image (default)
        frame = load_image_from_bytes(raw_bytes)
        raw_frames = [frame] if frame is not None else []

    if not raw_frames:
        logger.warning("load_and_preprocess: no frames available for content_type=%s", content_type)
        return [], []

    prep_frames: List[np.ndarray] = []
    pil_images: List[Image.Image] = []

    for f in raw_frames:
        try:
            pf = preprocess_frame(f, target_size)
            prep_frames.append(pf)
            pil_images.append(frame_to_pil(pf))
        except Exception as exc:
            logger.warning("Skipping frame due to preprocessing error: %s", exc)

    return prep_frames, pil_images
