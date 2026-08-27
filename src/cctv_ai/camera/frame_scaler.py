from __future__ import annotations

import cv2
import numpy as np

# torchvision's EfficientNet_B0_Weights.DEFAULT.transforms() is
# ImageClassification(resize_size=256, crop_size=224, interpolation=BICUBIC).
# Its Resize step scales the *shorter* side to 256 preserving aspect ratio, then
# CenterCrop takes 224. Downscaling frames to that same shorter side at ingest
# lets the ring buffer hold a fraction of the memory without changing what the
# models see: torchvision's Resize becomes a no-op because PIL's Image.resize
# short-circuits to a copy when the requested size already matches. The ingest
# downscale must use the *same* interpolation (BICUBIC) or the two-step resize
# lands on different pixels than the one-step training transform — verified with
# scripts/check_preprocess_equivalence.py (BILINEAR: max abs diff ~0.51;
# BICUBIC: bit-exact).


def resized_output_size(width: int, height: int, target_short: int) -> tuple[int, int]:
    """
    Replicate torchvision's `_compute_resized_output_size` for an int size.

    Returns (width, height). The integer truncation matters: it is what makes the
    downstream Resize a genuine no-op rather than an off-by-one rescale.
    """
    short, long_ = (width, height) if width <= height else (height, width)
    new_short = target_short
    new_long = int(target_short * long_ / short)
    return (new_short, new_long) if width <= height else (new_long, new_short)


def resize_shorter_side(
    frame_bgr: np.ndarray, target_short: int, *, exact: bool = True
) -> np.ndarray:
    """
    Scale `frame_bgr` so its shorter side is `target_short`, preserving aspect ratio.

    Never upscales: a frame already at or below the target is returned unchanged,
    which keeps small sources (phone uplink, low-res fixtures) untouched.

    `exact=True` uses PIL BICUBIC, matching the training-time transform bit for
    bit. `exact=False` uses cv2 INTER_AREA, which is faster but not identical;
    `scripts/check_preprocess_equivalence.py` reports the actual difference so the
    trade can be made on measured numbers rather than assumption.
    """
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        return frame_bgr

    height, width = frame_bgr.shape[:2]
    if min(width, height) <= target_short:
        return frame_bgr

    new_width, new_height = resized_output_size(width, height, target_short)

    if not exact:
        return cv2.resize(frame_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)

    from PIL import Image

    # Round-trip through RGB because PIL is channel-order aware. The two extra
    # channel reorders happen at ingest rate (INGEST_SAMPLE_FPS), not per
    # inference, so they cost far less than they save.
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = Image.fromarray(rgb).resize((new_width, new_height), Image.BICUBIC)
    return cv2.cvtColor(np.asarray(resized), cv2.COLOR_RGB2BGR)


def resize_max_width(frame_bgr: np.ndarray, max_width: int) -> np.ndarray:
    """
    Scale down for the browser preview only. Never upscales.

    Preview quality is cosmetic, so this always uses cv2 INTER_AREA; no model
    ever sees these pixels.
    """
    if frame_bgr.ndim < 2:
        return frame_bgr

    height, width = frame_bgr.shape[:2]
    if width <= max_width or width == 0:
        return frame_bgr

    new_height = max(1, int(round(height * max_width / width)))
    return cv2.resize(frame_bgr, (max_width, new_height), interpolation=cv2.INTER_AREA)


__all__ = ["resize_shorter_side", "resize_max_width", "resized_output_size"]
