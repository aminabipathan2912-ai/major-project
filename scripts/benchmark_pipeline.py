#!/usr/bin/env python3
"""
Baseline harness for the optimization work. Run this BEFORE and AFTER each phase.

Measures the things the plan claims to improve, so every claim is a measured
number rather than an assertion:

  * per-frame ingest downscale cost (Phase 1)
  * preprocessing cost for one 8-frame clip
  * accident / violence forward pass cost
  * the full per-cycle cost, both as it runs today (preprocess twice, once per
    model) and as Phase 2 would run it (preprocess once, shared)
  * process RSS, and the ring buffer's memory footprint at both frame sizes

Models are optional: if models/*.pt are missing the forward-pass timings are
skipped and everything else still reports.

    python scripts/benchmark_pipeline.py [video_path] [--clips N] [--warmup N]
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from cctv_ai.camera.frame_scaler import resize_shorter_side  # noqa: E402
from cctv_ai.config import get_settings  # noqa: E402


def rss_mb() -> float:
    """Current resident set size in MB, without adding a psutil dependency."""
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return float("nan")


def timed(fn, repeats: int, warmup: int) -> tuple[float, float]:
    """Return (median_ms, stdev_ms). Median resists the odd scheduler hiccup."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    stdev = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return statistics.median(samples), stdev


def decode_frames(path: Path, count: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    frames: list[np.ndarray] = []
    while len(frames) < count:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise SystemExit(f"No frames decoded from {path}")
    print(f"source fps      : {fps:.2f}")
    return frames


def find_default_clip() -> Path:
    configured = os.getenv("CAMERA_SOURCE")
    if configured and Path(configured).is_file():
        return Path(configured)
    candidates = [
        *sorted((ROOT / "data" / "uploads").glob("*.mp4")),
        *sorted((ROOT / "tests" / "fixtures").glob("*.mp4")),
    ]
    if not candidates:
        raise SystemExit("No clip found. Pass a video path, or drop one in data/uploads/.")
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", nargs="?", default=None)
    parser.add_argument("--clips", type=int, default=10, help="timed repeats")
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    settings = get_settings()
    clip_path = Path(args.video) if args.video else find_default_clip()
    n_frames = max(2, int(settings.CLIP_FRAME_COUNT))

    print("=" * 66)
    print("PIPELINE BASELINE")
    print("=" * 66)
    print(f"clip            : {clip_path}")

    frames = decode_frames(clip_path, n_frames)
    height, width = frames[0].shape[:2]
    target_short = int(getattr(settings, "INFERENCE_FRAME_SIZE", 256))
    exact = bool(getattr(settings, "INGEST_RESIZE_EXACT", True))

    print(f"frames per clip : {len(frames)} ({width}x{height})")
    print(f"ingest target   : shorter side {target_short}, exact={exact}")
    print(f"rss at start    : {rss_mb():.1f} MB")
    print()

    small = [resize_shorter_side(f, target_short, exact=exact) for f in frames]
    sh, sw = small[0].shape[:2]

    # ---------------- memory ----------------
    maxlen = int(settings.FRAME_BUFFER_MAXLEN)
    per_full = width * height * 3
    per_small = sw * sh * 3
    print("-- ring buffer memory ------------------------------------------")
    print(f"  full-res  {width}x{height}: {maxlen * per_full / 1e6:8.1f} MB "
          f"({maxlen} frames)")
    print(f"  ingested  {sw}x{sh}: {maxlen * per_small / 1e6:8.1f} MB "
          f"({maxlen} frames)")
    if per_full:
        print(f"  reduction : {(1 - per_small / per_full) * 100:8.1f} %")
    print()

    # ---------------- ingest ----------------
    print("-- ingest downscale (per frame) --------------------------------")
    ms, sd = timed(lambda: resize_shorter_side(frames[0], target_short, exact=exact),
                   args.clips, args.warmup)
    ingest_fps = float(getattr(settings, "INGEST_SAMPLE_FPS", 10.0))
    print(f"  resize        : {ms:7.2f} ms  (+/- {sd:.2f})")
    print(f"  at {ingest_fps:g} fps ingest : {ms * ingest_fps:7.2f} ms/s "
          f"({ms * ingest_fps / 10:.1f}% of one core)")
    print()

    # ---------------- preprocessing ----------------
    from PIL import Image
    from torchvision.models import EfficientNet_B0_Weights

    import torch

    preprocess = EfficientNet_B0_Weights.DEFAULT.transforms()

    def build(src: list[np.ndarray]) -> "torch.Tensor":
        tensors = [preprocess(Image.fromarray(f[:, :, ::-1])) for f in src]
        return torch.stack(tensors, dim=0).unsqueeze(0)

    print("-- preprocessing (one 8-frame clip) ----------------------------")
    ms_full, sd_full = timed(lambda: build(frames), args.clips, args.warmup)
    ms_small, sd_small = timed(lambda: build(small), args.clips, args.warmup)
    print(f"  from full-res : {ms_full:7.2f} ms  (+/- {sd_full:.2f})")
    print(f"  from ingested : {ms_small:7.2f} ms  (+/- {sd_small:.2f})")
    if ms_full:
        print(f"  speedup       : {ms_full / max(ms_small, 1e-9):7.2f}x")
    print()

    # ---------------- forward passes ----------------
    from cctv_ai.inference.loader import create_models

    accident, violence, *_ = create_models(settings)
    batch = build(small)

    loaded = {}
    for name, adapter in (("accident", accident), ("violence", violence)):
        status = adapter.status()
        if status.loaded:
            loaded[name] = adapter
        else:
            print(f"-- {name}: NOT LOADED ({status.reason})")

    fwd_ms: dict[str, float] = {}
    if loaded:
        print("-- forward pass (batch of 1 clip) ------------------------------")
        for name, adapter in loaded.items():
            model = adapter._loaded_model  # noqa: SLF001 - benchmark introspection
            net = model.model
            # The adapters put the net on CUDA when it is available, so the batch
            # has to follow it. predict_clip does this via .to(loaded.device).
            device_batch = batch.to(model.device)
            print(f"  {name:9s} device: {model.device}")

            def run(net=net, x=device_batch):
                with torch.inference_mode():
                    net(x)
                if x.device.type == "cuda":
                    # Kernel launches are async; without this we would time the
                    # launch, not the compute.
                    torch.cuda.synchronize()

            ms, sd = timed(run, args.clips, args.warmup)
            fwd_ms[name] = ms
            print(f"  {name:9s} time  : {ms:7.2f} ms  (+/- {sd:.2f})")
        print()

    # ---------------- per-cycle totals ----------------
    print("-- per inference cycle ----------------------------------------")
    total_fwd = sum(fwd_ms.values())
    today = ms_full * 2 + total_fwd
    phase1 = ms_small * 2 + total_fwd
    phase2 = ms_small * 1 + total_fwd
    print(f"  today   (2x preprocess full-res) : {today:8.2f} ms")
    print(f"  phase 1 (2x preprocess ingested) : {phase1:8.2f} ms"
          f"   {'' if not today else f'-> {(1 - phase1 / today) * 100:5.1f}% faster'}")
    print(f"  phase 2 (1x preprocess, shared)  : {phase2:8.2f} ms"
          f"   {'' if not today else f'-> {(1 - phase2 / today) * 100:5.1f}% faster'}")
    if not fwd_ms:
        print("  (forward-pass time excluded: models not loaded)")
    interval_ms = float(settings.INFERENCE_INTERVAL_MS)
    print(f"  budget per cycle                 : {interval_ms:8.2f} ms "
          f"(INFERENCE_INTERVAL_MS)")
    if today > interval_ms:
        print("  WARNING: today's cycle exceeds the interval — inference cannot keep up.")
    print()

    # ---------------- temporal coverage ----------------
    cap = cv2.VideoCapture(str(clip_path))
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    cap.release()
    if src_fps > 0:
        span_last_n = len(frames) / src_fps
        print("-- temporal coverage ------------------------------------------")
        print(f"  last-N clip spans   : {span_last_n:.3f} s of video")
        print(f"  cycle interval      : {interval_ms / 1000:.3f} s")
        print(f"  timeline inspected  : {min(100.0, span_last_n / (interval_ms / 1000) * 100):.1f} %")
        print("  (training sampled 8 frames across the whole clip; see plan B7)")
        print()

    print(f"rss at end      : {rss_mb():.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
