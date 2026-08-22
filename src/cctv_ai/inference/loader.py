from __future__ import annotations

from ..config import Settings
from .accident.model_adapter import AccidentModelAdapter
from .violence.model_adapter import ViolenceModelAdapter
from .audio.model_adapter import AudioModelAdapter


def create_models(settings: Settings):
    """
    Create model adapters.

    Important:
    - The app must NOT download datasets.
    - The app must NOT train models.
    - If a model isn't configured/loaded, it must report clearly and return no predictions.
    """

    accident = AccidentModelAdapter(weights_path=settings.ACCIDENT_MODEL_WEIGHTS_PATH)
    violence = ViolenceModelAdapter(weights_path=settings.VIOLENCE_MODEL_WEIGHTS_PATH)
    audio = AudioModelAdapter()

    return accident, violence, audio

