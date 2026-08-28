from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from . import features as feat


@dataclass
class LoadedAudioModel:
    model: Any
    device: Any
    id_to_label: dict[int, str]
    arch: str
    val_acc: float | None
    sample_rate: int
    clip_seconds: float


def build_audio_classifier(num_classes: int = 2):
    """
    EfficientNet-B0 over a log-mel spectrogram treated as an image.

    Same backbone and the same ImageNet normalisation as the video models, so
    the training recipe, checkpoint format, and adapter contract all carry over.
    There is no temporal mean-pool here: one spectrogram already covers the
    whole clip along its time axis.
    """
    import torch.nn as nn
    from torchvision.models import efficientnet_b0

    class AudioSpectrogramClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            backbone = efficientnet_b0(weights=None)
            in_features = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()
            self.backbone = backbone
            self.head = nn.Linear(in_features, num_classes)

        def forward(self, x):
            return self.head(self.backbone(x))

    return AudioSpectrogramClassifier()


def load_audio_checkpoint(weights_path: str, *, positive_label: str = "SCREAM") -> LoadedAudioModel:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        ckpt = torch.load(weights_path, map_location=device, weights_only=True)
    except Exception:
        ckpt = torch.load(weights_path, map_location=device, weights_only=False)

    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError("Checkpoint must be a dict with a 'state_dict' key.")

    label_map = ckpt.get("label_map") or {"NORMAL": 0, positive_label: 1}
    id_to_label = {int(v): str(k) for k, v in label_map.items()}

    net = build_audio_classifier(num_classes=max(len(id_to_label), 2))
    net.load_state_dict(ckpt["state_dict"])
    net.to(device)
    net.eval()

    val_acc = ckpt.get("val_acc")
    return LoadedAudioModel(
        model=net,
        device=device,
        id_to_label=id_to_label,
        arch=str(ckpt.get("arch") or "efficientnet_b0_logmel"),
        val_acc=float(val_acc) if val_acc is not None else None,
        sample_rate=int(ckpt.get("sample_rate") or feat.SAMPLE_RATE),
        clip_seconds=float(ckpt.get("clip_seconds") or feat.CLIP_SECONDS),
    )


def predict_waveform(
    loaded: LoadedAudioModel, wave: np.ndarray
) -> tuple[str, float, dict[str, Any]]:
    """Classify one mono float32 waveform."""
    import torch

    if wave.size == 0:
        raise ValueError("Empty waveform.")

    batch = feat.spectrogram_to_batch(wave, loaded.sample_rate).to(loaded.device)
    with torch.inference_mode():
        logits = loaded.model(batch)
        probs = torch.softmax(logits, dim=1)[0]
        pred_id = int(torch.argmax(probs).item())
        confidence = float(probs[pred_id].item())

    label = loaded.id_to_label.get(pred_id, str(pred_id))
    metadata = {
        "arch": loaded.arch,
        "rms_dbfs": round(feat.rms_dbfs(wave), 2),
        "samples": int(wave.size),
        "class_probabilities": {
            loaded.id_to_label.get(i, str(i)): float(probs[i].item())
            for i in range(len(probs))
        },
    }
    if loaded.val_acc is not None:
        metadata["train_val_acc"] = loaded.val_acc
    return label, confidence, metadata


__all__ = [
    "LoadedAudioModel",
    "build_audio_classifier",
    "load_audio_checkpoint",
    "predict_waveform",
]
