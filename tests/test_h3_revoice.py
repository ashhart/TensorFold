"""H3 re-voice: audio denoised again against held video rows, whole-sequence or from kept keys and values."""

import mlx.core as mx
import numpy as np
import pytest

from tensorfold.families.h3.config import DiTConfig
from tensorfold.families.h3.dit import H3DiT
from tensorfold.families.h3.packing import KEYFRAME_NOISE
from tensorfold.families.h3.sampler import audio_velocity, denoise, held_context, revoice

SMALL = DiTConfig(hidden_size=32, num_layers=2, token_refiner_num_layers=1, num_attention_heads=2,
                  attention_head_dim=16, ffn_hidden_size=48, latents_dim=4, audio_latents_dim=32, text_dim=20,
                  timestep_input_dim=16, time_embed_hidden_size=32, time_embed_dim=24, adaln_out_features=576,
                  final_adaln_out_features=64, rope_inv_freq_len=2)


def small_model(seed=0):
    mx.random.seed(seed)
    model = H3DiT(SMALL)
    mx.eval(model.parameters())
    return model


def clip(model, condition=None):
    text = mx.random.normal((1, 3, SMALL.text_dim))
    keyframes = ("first",) if condition is not None else ()
    return text, denoise(model, text, [1, 1, 1], 64, 64, 22, points=3, seed=1, condition=condition,
                         keyframes=keyframes)


class Recorder:
    """Wraps a model and records what each whole-sequence call was given."""

    def __init__(self, model):
        self.model, self.config, self.calls = model, model.config, []

    def cache_modulation(self, timestep, release=False):
        self.table = np.asarray(timestep)
        return self.model.cache_modulation(timestep, release=release)

    def __call__(self, video, audio, text, timestep, timestep_rows, *layout):
        self.calls.append((np.asarray(video), np.asarray(audio), self.table[np.asarray(timestep_rows)]))
        return self.model(video, audio, text, timestep, timestep_rows, *layout)


def test_exact_revoice_holds_every_row_but_the_audio():
    model = small_model()
    text, latents = clip(model)
    recorder = Recorder(model)
    audio = revoice(recorder, text, latents, points=4, seed=5, exact=True)
    assert audio.shape == latents.audio_rows.shape and len(recorder.calls) == 3
    audio_index = np.asarray(latents.packed.audio_rows.tolist())
    others = np.setdiff1d(np.arange(latents.packed.rows), audio_index)
    for step, (video, _, timesteps) in enumerate(recorder.calls):
        np.testing.assert_array_equal(video, recorder.calls[0][0])            # the picture never moves
        np.testing.assert_allclose(timesteps[others], KEYFRAME_NOISE)          # and sits at the keyframe level
        assert np.unique(timesteps[audio_index]).size == 1
    seen = [float(call[2][audio_index[0]]) for call in recorder.calls]
    assert seen[0] == 0.0 and seen == sorted(seen) and seen[-1] < 1.0          # audio starts from noise
    np.testing.assert_allclose(recorder.calls[0][0][0], KEYFRAME_NOISE * np.asarray(latents.video_rows), atol=0.02)
    assert not np.allclose(np.asarray(audio), np.asarray(latents.audio_rows))


def test_audio_only_pass_equals_the_whole_sequence_pass_it_was_kept_from():
    model = small_model(2)
    text, latents = clip(model)
    packed = latents.packed
    table = mx.array(np.array([0.25, KEYFRAME_NOISE], dtype=np.float32))
    plan = np.ones(packed.rows, dtype=np.int32)
    plan[np.asarray(packed.audio_rows.tolist())] = 0
    plan = mx.array(plan)
    model.cache_modulation(table)
    video = latents.video_rows.astype(mx.float32)
    audio = mx.random.normal(latents.audio_rows.shape)
    _, whole = model(video[None], audio[None], text.astype(mx.bfloat16), table, plan, packed.tags,
                     packed.position_ids, packed.video_rows, packed.audio_rows, packed.text_rows)
    context = held_context(model, video, audio, text.astype(mx.bfloat16), table, plan, packed)
    assert len(context.kv) == SMALL.num_layers
    assert context.kv[0][0].shape[2] == packed.rows - latents.audio_rows.shape[0]
    alone = audio_velocity(model, audio, plan, context)
    np.testing.assert_allclose(np.asarray(alone), np.asarray(whole[0]), atol=2e-2, rtol=2e-2)


def test_kept_revoice_runs_with_a_first_frame_and_is_deterministic():
    model = small_model(3)
    condition = mx.random.normal((4, 16))
    text, latents = clip(model, condition)
    steps = []
    first = revoice(model, text, latents, points=4, seed=9, condition=condition,
                    on_step=lambda i, n, s: steps.append((i, n)))
    second = revoice(model, text, latents, points=4, seed=9, condition=condition)
    assert steps == [(1, 3), (2, 3), (3, 3)] and first.shape == latents.audio_rows.shape
    np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
    assert np.isfinite(np.asarray(first)).all()
    with pytest.raises(ValueError):
        revoice(model, text, latents, points=4)  # the clip was made with a first frame
