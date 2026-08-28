from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from ..config import Settings
from .accident.model_adapter import AccidentModelAdapter
from .violence.model_adapter import ViolenceModelAdapter
from .audio.model_adapter import AudioModelAdapter
from .text.model_adapter import TextModelAdapter

_torch_tuned = False


def _apply_torch_tuning(settings: Settings) -> None:
    """
    CPU thread count, applied once per process.

    Left at the torch default a 12-core box spawns 12 intra-op threads per
    forward, which oversubscribes when two models and the event loop share the
    machine. `TORCH_NUM_THREADS=0` keeps the default.
    """
    global _torch_tuned
    if _torch_tuned:
        return
    _torch_tuned = True
    n = int(getattr(settings, "TORCH_NUM_THREADS", 0) or 0)
    if n > 0:
        try:
            import torch

            torch.set_num_threads(n)
        except Exception as exc:  # tuning is best-effort, never fatal
            print(f"[LOADER] could not set torch threads: {exc}")


def create_models(settings: Settings):
    """
    Create model adapters.

    Important:
    - The app must NOT download datasets.
    - The app must NOT train models.
    - If a model isn't configured/loaded, it must report clearly and return no predictions.

    The two video checkpoints are independent and each takes a few hundred ms to
    deserialize; loading them on separate threads halves that startup cost, and
    `torch.load` releases the GIL for most of the work.
    """
    _apply_torch_tuning(settings)

    fast = bool(getattr(settings, "CLIP_PREPROCESS_FAST", False))
    with ThreadPoolExecutor(max_workers=2) as pool:
        accident_future = pool.submit(
            AccidentModelAdapter,
            weights_path=settings.ACCIDENT_MODEL_WEIGHTS_PATH,
            preprocess_fast=fast,
        )
        violence_future = pool.submit(
            ViolenceModelAdapter,
            weights_path=settings.VIOLENCE_MODEL_WEIGHTS_PATH,
            preprocess_fast=fast,
        )
        accident = accident_future.result()
        violence = violence_future.result()

    # Small checkpoints (or absent); loading them serially costs nothing.
    audio = AudioModelAdapter(
        weights_path=getattr(settings, "AUDIO_MODEL_WEIGHTS_PATH", "")
    )
    text = TextModelAdapter(
        weights_path=getattr(settings, "TEXT_MODEL_WEIGHTS_PATH", "")
    )

    return accident, violence, audio, text
