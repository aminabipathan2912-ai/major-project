from __future__ import annotations

import math

import numpy as np

# Audio front-end shared by the live app and the training notebook.
#
# IMPORTANT: training and inference must compute features identically, or the
# trained weights are meaningless at runtime — the same class of bug the video
# equivalence check exists to catch. Keep this file the single source of truth
# and import it (or copy it verbatim) into the notebook.
#
# Everything here is pure numpy/torch: no torchaudio, librosa, or ffmpeg. The
# browser sends raw PCM (see phone.js), so nothing needs decoding either.

SAMPLE_RATE = 16000
N_FFT = 400          # 25 ms window
HOP_LENGTH = 160     # 10 ms hop
N_MELS = 64
F_MIN = 20.0
F_MAX = 7600.0
CLIP_SECONDS = 1.0
IMG_SIZE = 224

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(
    sample_rate: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    n_mels: int = N_MELS,
    f_min: float = F_MIN,
    f_max: float = F_MAX,
) -> np.ndarray:
    """Triangular mel filterbank, shape [n_mels, n_fft // 2 + 1]."""
    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0, sample_rate / 2.0, n_freqs)

    mel_points = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(n_mels):
        left, centre, right = hz_points[i], hz_points[i + 1], hz_points[i + 2]
        if right <= left:
            continue
        rising = (fft_freqs - left) / max(centre - left, 1e-9)
        falling = (right - fft_freqs) / max(right - centre, 1e-9)
        fb[i] = np.clip(np.minimum(rising, falling), 0.0, None)
    return fb


_FILTERBANK: np.ndarray | None = None


def _filterbank_cached() -> np.ndarray:
    global _FILTERBANK
    if _FILTERBANK is None:
        _FILTERBANK = mel_filterbank()
    return _FILTERBANK


def pcm_int16_to_float(pcm: bytes | np.ndarray) -> np.ndarray:
    """Browser Int16 PCM -> mono float32 in [-1, 1]."""
    arr = np.frombuffer(pcm, dtype=np.int16) if isinstance(pcm, (bytes, bytearray)) else pcm
    return np.asarray(arr, dtype=np.float32) / 32768.0


def fix_length(wave: np.ndarray, seconds: float = CLIP_SECONDS,
               sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Centre-pad or centre-crop to an exact duration, so every clip is one shape."""
    target = int(round(seconds * sample_rate))
    n = wave.shape[0]
    if n == target:
        return wave
    if n < target:
        pad = target - n
        return np.pad(wave, (pad // 2, pad - pad // 2))
    start = (n - target) // 2
    return wave[start:start + target]


def log_mel_spectrogram(wave: np.ndarray, sample_rate: int = SAMPLE_RATE):
    """
    Mono waveform -> log-mel spectrogram tensor [n_mels, frames].

    Deliberately *not* peak-normalised: loudness is the single most
    discriminative feature of a scream, so normalising it away would throw out
    exactly the signal the classifier needs (this is also why the browser
    disables autoGainControl).
    """
    import torch

    x = torch.as_tensor(np.ascontiguousarray(wave), dtype=torch.float32)
    window = torch.hann_window(N_FFT)
    spec = torch.stft(
        x, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=N_FFT,
        window=window, center=True, pad_mode="reflect",
        normalized=False, onesided=True, return_complex=True,
    )
    power = spec.real.pow(2) + spec.imag.pow(2)          # [freqs, frames]
    fb = torch.as_tensor(_filterbank_cached())            # [n_mels, freqs]
    mel = fb @ power
    return torch.log(mel + 1e-6)


def spectrogram_to_batch(wave: np.ndarray, sample_rate: int = SAMPLE_RATE):
    """
    Waveform -> a batch tensor an EfficientNet-B0 can consume: [1, 3, 224, 224].

    The log-mel is treated as a single-channel image, min-max scaled per clip,
    resized to the network's input size, replicated across RGB, and normalised
    with the ImageNet statistics the pretrained backbone expects. This is what
    lets the audio model reuse the exact video recipe.
    """
    import torch
    import torch.nn.functional as F

    mel = log_mel_spectrogram(fix_length(wave, sample_rate=sample_rate), sample_rate)
    lo, hi = mel.min(), mel.max()
    mel = (mel - lo) / (hi - lo) if (hi - lo) > 1e-6 else torch.zeros_like(mel)

    img = mel.unsqueeze(0).unsqueeze(0)                    # [1,1,mels,frames]
    img = F.interpolate(img, size=(IMG_SIZE, IMG_SIZE), mode="bilinear",
                        align_corners=False)
    img = img.repeat(1, 3, 1, 1)

    mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
    return (img - mean) / std


def rms_dbfs(wave: np.ndarray) -> float:
    """Loudness in dBFS. Reported alongside predictions for explainability."""
    if wave.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(wave, dtype=np.float64))))
    return 20.0 * math.log10(max(rms, 1e-9))


__all__ = [
    "SAMPLE_RATE", "N_FFT", "HOP_LENGTH", "N_MELS", "CLIP_SECONDS", "IMG_SIZE",
    "mel_filterbank", "pcm_int16_to_float", "fix_length",
    "log_mel_spectrogram", "spectrogram_to_batch", "rms_dbfs",
]
