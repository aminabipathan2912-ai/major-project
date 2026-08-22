from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .base import ClipInput


def even_sample_frames(frames: list[np.ndarray], count: int) -> list[np.ndarray]:
    if count <= 0:
        return []
    if not frames:
        return []
    if len(frames) == 1:
        return [frames[0]] * count
    idxs = np.linspace(0, len(frames) - 1, count).astype(int)
    return [frames[int(i)] for i in idxs]


@dataclass
class LoadedClipModel:
    model: Any
    preprocess: Any
    device: Any
    num_frames: int
    img_size: int
    id_to_label: dict[int, str]
    arch: str
    val_acc: float | None


def build_clip_classifier(num_classes: int = 2):
    import torch.nn as nn
    from torchvision.models import efficientnet_b0

    class EfficientNetClipClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            backbone = efficientnet_b0(weights=None)
            in_features = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()
            self.backbone = backbone
            self.head = nn.Linear(in_features, num_classes)

        def forward(self, x):
            b, t, c, h, w = x.shape
            x = x.view(b * t, c, h, w)
            feats = self.backbone(x)
            feats = feats.view(b, t, -1).mean(dim=1)
            return self.head(feats)

    return EfficientNetClipClassifier()


def load_clip_checkpoint(weights_path: str, *, positive_label: str) -> LoadedClipModel:
    import torch
    from torchvision.models import EfficientNet_B0_Weights

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)

    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError("Checkpoint must be a dict with a 'state_dict' key from the training notebook.")

    num_frames = int(ckpt.get("num_frames") or 8)
    img_size = int(ckpt.get("img_size") or 224)
    arch = str(ckpt.get("arch") or "efficientnet_b0_temporal_mean")
    val_acc = ckpt.get("val_acc")
    label_map = ckpt.get("label_map") or {"NORMAL": 0, positive_label: 1}
    id_to_label = {int(v): str(k) for k, v in label_map.items()}

    net = build_clip_classifier(num_classes=max(len(id_to_label), 2))
    net.load_state_dict(ckpt["state_dict"])
    net.to(device)
    net.eval()

    preprocess = EfficientNet_B0_Weights.DEFAULT.transforms()
    return LoadedClipModel(
        model=net,
        preprocess=preprocess,
        device=device,
        num_frames=num_frames,
        img_size=img_size,
        id_to_label=id_to_label,
        arch=arch,
        val_acc=float(val_acc) if val_acc is not None else None,
    )


def predict_clip(loaded: LoadedClipModel, clip: ClipInput) -> tuple[str, float, dict[str, Any]]:
    import torch
    from PIL import Image

    frames = even_sample_frames(clip.frames_bgr, loaded.num_frames)
    if not frames:
        raise ValueError("Clip has no frames.")

    tensors = []
    for frame_bgr in frames:
        rgb = frame_bgr[:, :, ::-1] if frame_bgr.ndim == 3 else frame_bgr
        img = Image.fromarray(rgb)
        tensors.append(loaded.preprocess(img))

    batch = torch.stack(tensors, dim=0).unsqueeze(0).to(loaded.device)
    with torch.inference_mode():
        logits = loaded.model(batch)
        probs = torch.softmax(logits, dim=1)[0]
        pred_id = int(torch.argmax(probs).item())
        confidence = float(probs[pred_id].item())

    label = loaded.id_to_label.get(pred_id, str(pred_id))
    metadata = {
        "arch": loaded.arch,
        "num_frames_used": loaded.num_frames,
        "class_probabilities": {
            loaded.id_to_label.get(i, str(i)): float(probs[i].item()) for i in range(len(probs))
        },
    }
    if loaded.val_acc is not None:
        metadata["train_val_acc"] = loaded.val_acc
    return label, confidence, metadata
