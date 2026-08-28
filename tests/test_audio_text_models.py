"""
Audio and text modality contracts.

Checkpoints are built inside the tests rather than shipped: an untrained
checkpoint in `models/` would look like a working model to anyone who pointed
config at it, which is exactly the failure these adapters exist to avoid.
"""
import time

import numpy as np
import pytest
import torch

from cctv_ai.inference.audio import features as feat
from cctv_ai.inference.audio.audio_buffer import AudioBuffer, AudioChunk
from cctv_ai.inference.audio.audio_classifier import build_audio_classifier
from cctv_ai.inference.audio.model_adapter import AudioModelAdapter
from cctv_ai.inference.text.model_adapter import TextModelAdapter
from cctv_ai.inference.text.text_classifier import (
    PAD,
    UNK,
    build_text_classifier,
    encode,
    tokenize,
)

# ---------------------------------------------------------------- audio features


def test_mel_filterbank_shape_and_coverage():
    fb = feat.mel_filterbank()
    assert fb.shape == (feat.N_MELS, feat.N_FFT // 2 + 1)
    assert (fb.sum(axis=1) > 0).all(), "every mel band must cover some frequency"


def test_higher_frequency_lands_in_higher_mel_band():
    sr = feat.SAMPLE_RATE
    t = np.arange(sr) / sr
    peaks = []
    for hz in (500, 2000, 6000):
        wave = (0.5 * np.sin(2 * np.pi * hz * t)).astype(np.float32)
        mel = feat.log_mel_spectrogram(feat.fix_length(wave))
        peaks.append(int(mel.mean(dim=1).argmax()))
    assert peaks[0] < peaks[1] < peaks[2]


def test_spectrogram_batch_is_efficientnet_shaped():
    wave = np.zeros(feat.SAMPLE_RATE, dtype=np.float32)
    assert tuple(feat.spectrogram_to_batch(wave).shape) == (1, 3, feat.IMG_SIZE, feat.IMG_SIZE)


def test_fix_length_pads_and_crops():
    assert feat.fix_length(np.zeros(500, np.float32)).shape[0] == feat.SAMPLE_RATE
    assert feat.fix_length(np.zeros(99999, np.float32)).shape[0] == feat.SAMPLE_RATE


def test_loudness_is_preserved_not_normalised():
    """A scream is defined by being loud; the front-end must not erase that."""
    sr = feat.SAMPLE_RATE
    t = np.arange(sr) / sr
    quiet = (0.01 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    loud = (0.9 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    assert feat.rms_dbfs(loud) - feat.rms_dbfs(quiet) > 30


def test_int16_pcm_round_trip():
    sr = feat.SAMPLE_RATE
    t = np.arange(sr) / sr
    pcm = ((0.5 * np.sin(2 * np.pi * 440 * t)) * 32767).astype(np.int16).tobytes()
    wave = feat.pcm_int16_to_float(pcm)
    assert wave.dtype == np.float32 and wave.shape == (sr,)
    assert 0.45 < float(np.abs(wave).max()) < 0.55


# ---------------------------------------------------------------- audio adapter


def _audio_checkpoint(path) -> str:
    net = build_audio_classifier(2)
    torch.save(
        {
            "model_name": "audio",
            "arch": "efficientnet_b0_logmel",
            "label_map": {"NORMAL": 0, "SCREAM": 1},
            "sample_rate": feat.SAMPLE_RATE,
            "clip_seconds": feat.CLIP_SECONDS,
            "val_acc": 0.0,
            "state_dict": net.state_dict(),
        },
        path,
    )
    return str(path)


def _chunk(seconds=1.0, hz=3000, amp=0.8) -> AudioChunk:
    sr = feat.SAMPLE_RATE
    t = np.arange(int(sr * seconds)) / sr
    pcm = ((amp * np.sin(2 * np.pi * hz * t)) * 32767).astype(np.int16).tobytes()
    return AudioChunk(data=pcm, mime_type="audio/pcm", received_epoch_s=time.time())


def test_audio_adapter_reports_missing_weights():
    a = AudioModelAdapter(weights_path="")
    assert a.status().loaded is False
    assert "not loaded" in a.status().reason
    assert a.predict_audio([_chunk()]) is None


def test_audio_adapter_never_guesses_when_unloaded(tmp_path):
    a = AudioModelAdapter(weights_path=str(tmp_path / "nope.pt"))
    assert a.status().loaded is False
    assert a.predict_audio([_chunk()]) is None


def test_audio_adapter_predicts_when_loaded(tmp_path):
    a = AudioModelAdapter(weights_path=_audio_checkpoint(tmp_path / "audio.pt"))
    assert a.status().loaded is True

    pred = a.predict_audio([_chunk()], camera_id="cam-1")
    assert pred is not None
    assert pred.model_name == "audio"
    assert pred.predicted_label in {"NORMAL", "SCREAM"}
    assert set(pred.metadata["class_probabilities"]) == {"NORMAL", "SCREAM"}
    assert pred.metadata["rms_dbfs"] > -20  # loud tone


def test_audio_adapter_concatenates_chunks(tmp_path):
    """A scream spanning a chunk boundary must not be cut in half."""
    a = AudioModelAdapter(weights_path=_audio_checkpoint(tmp_path / "audio.pt"))
    pred = a.predict_audio([_chunk(0.5), _chunk(0.5)], camera_id="c")
    assert pred is not None
    assert pred.metadata["samples"] == feat.SAMPLE_RATE


def test_audio_adapter_ignores_video_clips(tmp_path):
    a = AudioModelAdapter(weights_path=_audio_checkpoint(tmp_path / "audio.pt"))
    assert a.predict(clip=None) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------- audio buffer


def test_audio_buffer_is_bounded_and_counts_drops():
    buf = AudioBuffer(maxlen=3)
    for _ in range(10):
        buf.append(b"\x00\x01", "audio/pcm")
    stats = buf.stats()
    assert stats["buffered"] == 3
    assert stats["received"] == 10
    assert stats["dropped"] == 7


# ---------------------------------------------------------------- text


def test_tokenizer_normalises_urls_handles_and_tags():
    toks = tokenize("Crash on #MGRoad see https://x.com/a @police NOW")
    assert "url" in toks and "user" in toks
    assert "mgroad" in toks
    assert "crash" in toks and "now" in toks


def test_tokenizer_caps_length():
    assert len(tokenize(" ".join(["word"] * 500))) == 64


def test_encode_falls_back_to_unk():
    assert encode("totallyunseenword", {"<pad>": PAD, "<unk>": UNK}) == [UNK]
    assert encode("", {"<pad>": PAD, "<unk>": UNK}) == [UNK]


def _text_checkpoint(path, vocab=None) -> str:
    vocab = vocab or {"<pad>": PAD, "<unk>": UNK, "crash": 2, "car": 3, "sunset": 4}
    net = build_text_classifier(len(vocab), 3, 32)
    torch.save(
        {
            "model_name": "text",
            "arch": "embeddingbag_mean",
            "label_map": {"NORMAL": 0, "ACCIDENT": 1, "VIOLENCE": 2},
            "vocab": vocab,
            "embed_dim": 32,
            "val_acc": 0.0,
            "state_dict": net.state_dict(),
        },
        path,
    )
    return str(path)


def test_text_adapter_reports_missing_weights():
    t = TextModelAdapter(weights_path="")
    assert t.status().loaded is False
    assert t.predict_message("car crash") is None


def test_text_adapter_rejects_checkpoint_without_vocab(tmp_path):
    """The vocabulary must travel with the weights or predictions are nonsense."""
    net = build_text_classifier(5, 3, 32)
    p = tmp_path / "novocab.pt"
    torch.save({"state_dict": net.state_dict(), "label_map": {"NORMAL": 0}}, p)
    t = TextModelAdapter(weights_path=str(p))
    assert t.status().loaded is False
    assert "vocab" in t.status().reason


def test_text_adapter_predicts_and_reports_coverage(tmp_path):
    t = TextModelAdapter(weights_path=_text_checkpoint(tmp_path / "text.pt"))
    assert t.status().loaded is True

    known = t.predict_message("car crash", camera_id="c")
    assert known is not None
    assert known.metadata["vocab_coverage"] == pytest.approx(1.0)

    # Out-of-vocabulary input still yields a confident label — which is exactly
    # why coverage is surfaced and gated on downstream.
    unknown = t.predict_message("zzz qqq wibble", camera_id="c")
    assert unknown is not None
    assert unknown.metadata["vocab_coverage"] == pytest.approx(0.0)


def test_text_adapter_ignores_empty_and_video(tmp_path):
    t = TextModelAdapter(weights_path=_text_checkpoint(tmp_path / "text.pt"))
    assert t.predict_message("   ") is None
    assert t.predict(clip=None) is None  # type: ignore[arg-type]
