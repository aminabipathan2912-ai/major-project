#!/usr/bin/env python3
"""
Prove that downscaling frames at ingest does not change what the models see.

Phase 1 shrinks the frame ring buffer by storing frames at the shorter side the
training transform resizes to anyway (256) instead of full resolution. That is
only safe if the tensor handed to EfficientNet is unchanged, because the trained
weights in models/*.pt were fitted against
`EfficientNet_B0_Weights.DEFAULT.transforms()` applied to full frames.

This compares three preprocessing paths on real decoded frames:

  A  full-res            -> PIL -> transforms()      (today's behaviour, the reference)
  B  ingest exact  (PIL) -> PIL -> transforms()      (Phase 1 default)
  C  ingest fast   (cv2) -> PIL -> transforms()      (opt-in, INGEST_RESIZE_EXACT=false)

B must be bit-exact against A or the script fails. C is reported so the speed/
fidelity trade can be judged on measured numbers.

    python scripts/check_preprocess_equivalence.py [video_path] [--frames N]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision.models import EfficientNet_B0_Weights  # noqa: E402

from cctv_ai.camera.frame_scaler import resize_shorter_side  # noqa: E402

PREPROCESS = EfficientNet_B0_Weights.DEFAULT.transforms()
TARGET_SHORT = 256


def preprocess(frame_bgr: np.ndarray) -> torch.Tensor:
    """Exactly the app's current inference path (clip_classifier.predict_clip)."""
    rgb = frame_bgr[:, :, ::-1]
    return PREPROCESS(Image.fromarray(rgb))


def decode_frames(path: Path, count: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {path}")
    frames: list[np.ndarray] = []
    while len(frames) < count:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise SystemExit(f"No frames decoded from {path}")
    return frames


def find_default_clip() -> Path:
    candidates = [
        *sorted((ROOT / "data" / "uploads").glob("*.mp4")),
        *sorted((ROOT / "tests" / "fixtures").glob("*.mp4")),
    ]
    configured = os.getenv("CAMERA_SOURCE")
    if configured and Path(configured).is_file():
        return Path(configured)
    if not candidates:
        raise SystemExit(
            "No clip found. Pass a video path, or drop one in data/uploads/."
        )
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", nargs="?", default=None)
    parser.add_argument("--frames", type=int, default=8)
    args = parser.parse_args()

    clip = Path(args.video) if args.video else find_default_clip()
    frames = decode_frames(clip, args.frames)
    h, w = frames[0].shape[:2]

    print(f"clip            : {clip}")
    print(f"frames decoded  : {len(frames)}  ({w}x{h})")
    print(f"target short    : {TARGET_SHORT}")
    print()

    if min(w, h) <= TARGET_SHORT:
        print(
            f"NOTE: this clip's shorter side ({min(w, h)}) is already <= {TARGET_SHORT}, so"
        )
        print("      resize_shorter_side is a pass-through and the comparison is trivially")
        print("      exact. Re-run against a 720p/1080p clip to exercise the real path.")
        print()

    worst_exact = 0.0
    worst_fast = 0.0
    sum_fast = 0.0
    exact_bitwise = True

    for frame in frames:
        ref = preprocess(frame)
        got_exact = preprocess(resize_shorter_side(frame, TARGET_SHORT, exact=True))
        got_fast = preprocess(resize_shorter_side(frame, TARGET_SHORT, exact=False))

        if got_exact.shape != ref.shape or got_fast.shape != ref.shape:
            print(f"FAIL: shape mismatch ref={tuple(ref.shape)} "
                  f"exact={tuple(got_exact.shape)} fast={tuple(got_fast.shape)}")
            return 1

        d_exact = (got_exact - ref).abs()
        d_fast = (got_fast - ref).abs()
        worst_exact = max(worst_exact, float(d_exact.max()))
        worst_fast = max(worst_fast, float(d_fast.max()))
        sum_fast += float(d_fast.mean())
        if not torch.equal(got_exact, ref):
            exact_bitwise = False

    mean_fast = sum_fast / len(frames)

    # Normalised tensor values sit roughly in [-2.2, 2.7]; state the tolerance in
    # those units rather than pretending a raw number is self-explanatory.
    print("path B  ingest exact (PIL BICUBIC)")
    print(f"  bit-exact vs today : {exact_bitwise}")
    print(f"  max abs diff       : {worst_exact:.3e}")
    print()
    print("path C  ingest fast (cv2 INTER_AREA)")
    print(f"  max abs diff       : {worst_fast:.3e}")
    print(f"  mean abs diff      : {mean_fast:.3e}")
    print()

    small = resize_shorter_side(frames[0], TARGET_SHORT, exact=True)
    est_full = w * h * 3
    est_small = small.shape[0] * small.shape[1] * 3
    buf = int(os.getenv("FRAME_BUFFER_MAXLEN", "32"))
    print(f"ring buffer @ maxlen={buf}")
    print(f"  before : {buf * est_full / 1e6:8.1f} MB  ({w}x{h})")
    print(f"  after  : {buf * est_small / 1e6:8.1f} MB  "
          f"({small.shape[1]}x{small.shape[0]})")
    print(f"  saving : {(1 - est_small / est_full) * 100:8.1f} %")
    print()

    if not exact_bitwise:
        print("RESULT: FAIL — exact mode is not bit-identical. Do not ship B3 as-is.")
        return 1

    print("RESULT: PASS — ingest downscale does not change the model input.")
    print("        models/*.pt remain valid; no retraining implied.")
    if worst_fast > 0:
        print(f"        Fast mode differs by up to {worst_fast:.3e}; it stays opt-in")
        print("        via INGEST_RESIZE_EXACT=false.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
