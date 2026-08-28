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
    # The training notebooks save only primitives and tensors, so the safe
    # loader works. Fall back to the unrestricted loader for any checkpoint it
    # rejects, so this never becomes a hard requirement on the weights format.
    try:
        ckpt = torch.load(weights_path, map_location=device, weights_only=True)
    except Exception:
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


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_RESIZE_SIZE = 256
_CROP_SIZE = 224


def _preprocess_pil(frames: list[np.ndarray], preprocess):
    """Per-frame PIL path — identical to how the models were trained."""
    import torch
    from PIL import Image

    tensors = []
    for frame_bgr in frames:
        rgb = frame_bgr[:, :, ::-1] if frame_bgr.ndim == 3 else frame_bgr
        tensors.append(preprocess(Image.fromarray(rgb)))
    return torch.stack(tensors, dim=0)


def _preprocess_fast(frames: list[np.ndarray]):
    """
    Batched cv2/torch path. Only valid when every frame already has its shorter
    side at `_RESIZE_SIZE` (the ring buffer downscales to exactly that at
    ingest), so the training transform's Resize is a no-op and only a
    centre-crop + normalise remain. Returns None to signal "fall back to PIL"
    if any frame is a different size.
    """
    import torch

    cropped = []
    for frame_bgr in frames:
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            return None
        h, w = frame_bgr.shape[:2]
        if min(h, w) != _RESIZE_SIZE or h < _CROP_SIZE or w < _CROP_SIZE:
            return None
        top = int(round((h - _CROP_SIZE) / 2.0))
        left = int(round((w - _CROP_SIZE) / 2.0))
        crop = frame_bgr[top:top + _CROP_SIZE, left:left + _CROP_SIZE, ::-1]  # BGR->RGB
        cropped.append(np.ascontiguousarray(crop))

    batch = torch.from_numpy(np.stack(cropped, axis=0))  # [t, 224, 224, 3] uint8
    batch = batch.permute(0, 3, 1, 2).to(torch.float32).div_(255.0)
    mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
    return batch.sub_(mean).div_(std)


def sample_and_preprocess_clip(
    loaded: LoadedClipModel, clip: ClipInput, *, fast: bool = False
):
    """
    Sample `num_frames` evenly across the clip and apply the training transform.

    Returns a CPU tensor shaped ``[1, num_frames, 3, H, W]``. Device placement is
    left to the caller so a single batch can be shared across models that sit on
    the same device (accident + violence use the same weights enum and frame
    count, so the tensor is identical for both).

    `fast=True` uses a batched cv2/torch path; it silently falls back to the PIL
    path for any clip whose frames are not already at the ingest size.
    """
    import torch

    frames = even_sample_frames(clip.frames_bgr, loaded.num_frames)
    if not frames:
        raise ValueError("Clip has no frames.")

    stacked = _preprocess_fast(frames) if fast else None
    if stacked is None:
        stacked = _preprocess_pil(frames, loaded.preprocess)

    return stacked.unsqueeze(0)


def predict_from_batch(loaded: LoadedClipModel, batch) -> tuple[str, float, dict[str, Any]]:
    """Run one model on an already sampled + preprocessed clip batch."""
    import torch

    with torch.inference_mode():
        logits = loaded.model(batch.to(loaded.device))
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


def predict_clip(loaded: LoadedClipModel, clip: ClipInput) -> tuple[str, float, dict[str, Any]]:
    return predict_from_batch(loaded, sample_and_preprocess_clip(loaded, clip))
