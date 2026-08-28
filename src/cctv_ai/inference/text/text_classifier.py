from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# Text modality: incident-report classification over short messages
# (helpline transcripts, social posts).
#
# A mean-pooled EmbeddingBag + linear head, in pure torch. No transformers,
# no sklearn, no new dependencies. It trains in seconds on a laptop, the whole
# checkpoint is a few MB, and — like the video and audio models — it carries its
# own vocabulary and label map, so the app needs no side files to run it.
#
# Tokenisation lives here and is imported by the training notebook, for the same
# reason the audio front-end is shared: train/inference feature mismatch silently
# destroys a trained model.

MAX_TOKENS = 64
PAD, UNK = 0, 1

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens. URLs, @handles and #tags are normalised first."""
    text = (text or "").lower()
    text = re.sub(r"https?://\S+|www\.\S+", " <url> ", text)
    text = re.sub(r"@\w+", " <user> ", text)
    text = re.sub(r"#(\w+)", r" \1 ", text)
    return _TOKEN_RE.findall(text)[:MAX_TOKENS]


def encode(text: str, vocab: dict[str, int]) -> list[int]:
    ids = [vocab.get(t, UNK) for t in tokenize(text)]
    return ids or [UNK]


@dataclass
class LoadedTextModel:
    model: Any
    device: Any
    vocab: dict[str, int]
    id_to_label: dict[int, str]
    arch: str
    val_acc: float | None


def build_text_classifier(vocab_size: int, num_classes: int, embed_dim: int = 128):
    import torch.nn as nn

    class TextBagClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean",
                                             padding_idx=PAD)
            self.dropout = nn.Dropout(0.3)
            self.head = nn.Sequential(
                nn.Linear(embed_dim, 64), nn.ReLU(), nn.Linear(64, num_classes)
            )

        def forward(self, ids, offsets):
            return self.head(self.dropout(self.embedding(ids, offsets)))

    return TextBagClassifier()


def load_text_checkpoint(weights_path: str) -> LoadedTextModel:
    import torch

    device = torch.device("cpu")  # tiny model; CPU avoids a pointless transfer
    try:
        ckpt = torch.load(weights_path, map_location=device, weights_only=True)
    except Exception:
        ckpt = torch.load(weights_path, map_location=device, weights_only=False)

    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError("Checkpoint must be a dict with a 'state_dict' key.")
    vocab = ckpt.get("vocab")
    if not isinstance(vocab, dict) or not vocab:
        raise ValueError("Checkpoint must carry its 'vocab'.")

    label_map = ckpt.get("label_map") or {"NORMAL": 0, "ACCIDENT": 1, "VIOLENCE": 2}
    id_to_label = {int(v): str(k) for k, v in label_map.items()}
    embed_dim = int(ckpt.get("embed_dim") or 128)

    net = build_text_classifier(len(vocab), max(len(id_to_label), 2), embed_dim)
    net.load_state_dict(ckpt["state_dict"])
    net.to(device)
    net.eval()

    val_acc = ckpt.get("val_acc")
    return LoadedTextModel(
        model=net,
        device=device,
        vocab={str(k): int(v) for k, v in vocab.items()},
        id_to_label=id_to_label,
        arch=str(ckpt.get("arch") or "embeddingbag_mean"),
        val_acc=float(val_acc) if val_acc is not None else None,
    )


def predict_text(loaded: LoadedTextModel, text: str) -> tuple[str, float, dict[str, Any]]:
    import torch

    ids = encode(text, loaded.vocab)
    tensor = torch.tensor(ids, dtype=torch.long, device=loaded.device)
    offsets = torch.tensor([0], dtype=torch.long, device=loaded.device)

    with torch.inference_mode():
        logits = loaded.model(tensor, offsets)
        probs = torch.softmax(logits, dim=1)[0]
        pred_id = int(torch.argmax(probs).item())
        confidence = float(probs[pred_id].item())

    known = sum(1 for t in tokenize(text) if t in loaded.vocab)
    total = max(len(tokenize(text)), 1)
    metadata = {
        "arch": loaded.arch,
        "tokens": total,
        # Low coverage means the message is mostly out-of-vocabulary, which is
        # a strong hint the prediction should not be trusted.
        "vocab_coverage": round(known / total, 3),
        "class_probabilities": {
            loaded.id_to_label.get(i, str(i)): float(probs[i].item())
            for i in range(len(probs))
        },
    }
    if loaded.val_acc is not None:
        metadata["train_val_acc"] = loaded.val_acc
    return loaded.id_to_label.get(pred_id, str(pred_id)), confidence, metadata


__all__ = [
    "MAX_TOKENS", "PAD", "UNK", "tokenize", "encode",
    "LoadedTextModel", "build_text_classifier", "load_text_checkpoint",
    "predict_text",
]
