"""H3 sampler: first-frame conditioning rows are held, generated rows are stepped."""

import mlx.core as mx
import pytest

from tensorfold.families.h3.config import DiTConfig
from tensorfold.families.h3.sampler import denoise


class FakeDiT:
    """Returns zero velocity and records the row counts it was called with."""

    def __init__(self):
        self.config = DiTConfig(num_layers=1)
        self.calls = []

    def cache_modulation(self, timestep, release=False):
        return 0

    def __call__(self, video, audio, text, timestep, timestep_rows, tags, position_ids, video_rows, audio_rows,
                 text_rows):
        self.calls.append((video.shape[1], audio.shape[1], int(timestep_rows.shape[0])))
        return mx.zeros(video.shape), mx.zeros(audio.shape)


def run(condition=None, keyframes=()):
    dit = FakeDiT()
    text = mx.zeros((1, 3, dit.config.text_dim))
    out = denoise(dit, text, [1, 1, 1], 64, 64, 22, points=3, seed=1, condition=condition, keyframes=keyframes)
    return dit, out


def test_text_to_video_steps_every_video_row():
    dit, out = run()
    assert dit.calls == [(7 * 4, 74, 3 + 74 + 28)] * 2
    assert out.video_rows.shape == (28, 96) and out.audio_rows.shape == (74, 32)


def test_first_frame_rows_are_passed_in_and_not_returned():
    condition = mx.random.normal((4, 96))
    dit, out = run(condition, ("first",))
    assert dit.calls == [(4 + 28, 74, 3 + 4 + 74 + 28)] * 2
    assert out.video_rows.shape == (28, 96)
    assert out.packed.condition_video_rows == 4


def test_condition_rows_must_match_the_keyframes():
    with pytest.raises(ValueError):
        run(mx.zeros((3, 96)), ("first",))
    with pytest.raises(ValueError):
        run(None, ("first",))
