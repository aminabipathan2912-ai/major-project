import numpy as np

from cctv_ai.inference.clip_classifier import even_sample_frames


def test_even_sample_repeats_single_frame():
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    out = even_sample_frames([frame], 8)
    assert len(out) == 8


def test_even_sample_picks_endpoints():
    frames = [np.full((2, 2, 3), i, dtype=np.uint8) for i in range(10)]
    out = even_sample_frames(frames, 5)
    assert len(out) == 5
    assert int(out[0][0, 0, 0]) == 0
    assert int(out[-1][0, 0, 0]) == 9
