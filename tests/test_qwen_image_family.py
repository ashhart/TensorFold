"""Qwen-Image-2.1 family: schedule, rotary layout, prefix reuse, int8 QKV layout, adapter merge, decoder pieces."""

from __future__ import annotations

import json

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mlx import nn  # noqa: E402

from tensorfold.families.qwen_image import config as qconfig  # noqa: E402
from tensorfold.families.qwen_image import dit as qdit  # noqa: E402
from tensorfold.families.qwen_image import lora, sampler, schedule, vae, weights  # noqa: E402

TINY = qconfig.DiTConfig(in_channels=8, out_channels=8, num_layers=2, attention_head_dim=16, num_attention_heads=2,
                         context_in_dim=24, mlp_ratio=3, axes_dims_rope=(4, 6, 6), timestep_dim=32)


def tiny_model(seed: int = 0):
    mx.random.seed(seed)
    model = qdit.QwenImageDiT(TINY)
    mx.eval(model.parameters())
    return model


def test_family_is_registered_without_lanes():
    from tensorfold.families import families, qwen_image

    assert "qwen_image_21" in families()
    assert qwen_image.LANES is False and not hasattr(qwen_image, "load")


def test_pipeline_root_reads_model_index(tmp_path):
    assert qconfig.pipeline_root(tmp_path) is None
    (tmp_path / "model_index.json").write_text(json.dumps({"_class_name": "SomethingElse"}))
    assert qconfig.pipeline_root(tmp_path) is None
    (tmp_path / "model_index.json").write_text(json.dumps({"_class_name": "QwenImage21Pipeline"}))
    assert qconfig.pipeline_root(tmp_path) == tmp_path
    assert qwen_is_checkpoint(tmp_path)


def qwen_is_checkpoint(path) -> bool:
    from tensorfold.families import qwen_image

    return qwen_image.is_checkpoint(path)


def test_latent_grid_needs_multiples_of_16():
    assert qconfig.latent_grid(1344, 768) == (48, 84)
    with pytest.raises(ValueError):
        qconfig.latent_grid(1340, 768)


def test_schedule_runs_from_one_to_the_terminal_then_zero():
    levels = schedule.sigmas(40, 1344, 768)
    assert levels.shape == (41,) and levels[0] == pytest.approx(1.0) and levels[-1] == 0.0
    assert levels[-2] == pytest.approx(schedule.TERMINAL, abs=1e-6)
    assert np.all(np.diff(levels) < 0)
    # larger images shift more of the schedule towards high noise
    assert schedule.sigmas(40, 2048, 2048)[20] > schedule.sigmas(40, 512, 512)[20]


def test_fixed_nodes_take_the_shift_but_not_the_terminal_stretch():
    nodes = (1.0, 0.9375, 0.875, 0.75, 0.5, 0.25)
    levels = schedule.sigmas(6, 1024, 1024, nodes)
    scale = np.exp(schedule.shift_for(1024, 1024))
    assert levels.shape == (7,) and levels[-1] == 0.0
    np.testing.assert_allclose(levels[:-1], scale / (scale + (1 / np.array(nodes) - 1)), rtol=1e-6)
    with pytest.raises(ValueError):
        schedule.sigmas(2, 1024, 1024, (0.5, 0.75))


def test_rotary_text_advances_all_axes_and_image_is_centred():
    cos, sin = qdit.rotary_tables(TINY, text_len=3, rows=2, columns=4)
    assert cos.shape == (3 + 8, 8) and sin.shape == cos.shape
    angles = np.arctan2(np.asarray(sin), np.asarray(cos))
    # the first frequency of each axis is 1, so its angle is the position (wrapped)
    frame, height, width = angles[:, 0], angles[:, 2], angles[:, 5]
    np.testing.assert_allclose(frame[:3], [0, 1, 2], atol=1e-5)
    np.testing.assert_allclose(height[:3], [0, 1, 2], atol=1e-5)
    np.testing.assert_allclose(frame[3:], np.full(8, 3.0), atol=1e-5)
    np.testing.assert_allclose(height[3:], np.repeat([-1, 0], 4), atol=1e-5)
    np.testing.assert_allclose(width[3:], np.tile([-2, -1, 0, 1], 2), atol=1e-5)


def joint_forward(model, latents, sigma, text):
    """The whole [text | image] sequence in one pass with the block-causal mask and per-row modulation."""

    text_len, count = text.shape[1], latents.shape[1]
    rows, columns = 2, count // 2
    cos, sin = qdit.rotary_tables(TINY, text_len, rows, columns)
    rotary = qdit.Rotary(cos, sin)
    temb = model.time_text_embed(mx.array([sigma, 0.0], dtype=mx.float32))
    table = model.modulation[1](nn.silu(temb))
    per_row = mx.concatenate([mx.broadcast_to(table[1], (text_len, table.shape[-1])),
                              mx.broadcast_to(table[0], (count, table.shape[-1]))])[None]
    modulation = tuple(mx.split(per_row, 4, axis=-1))
    index = mx.arange(text_len + count)
    allowed = (index[None, :] <= index[:, None]) | (index[:, None] >= text_len)
    mask = mx.where(allowed, 0.0, -1e9)[None, None]
    x = mx.concatenate([model.txt_in(text), model.img_in(latents)], axis=1)
    for block in model.transformer_blocks:
        q, k, v = block.attn.qkv(block.img_norm1(x) * (1 + modulation[0]), rotary)
        x = block._finish(x, block.attn.mix(q, k, v, mask=mask), modulation)
    scale = model.norm_out.linear(nn.silu(temb))[0]
    return model.proj_out(model.norm_out.norm(x) * (1 + scale))[:, text_len:]


def test_stored_text_prefix_equals_the_joint_block_causal_pass():
    model = tiny_model()
    text = mx.random.normal((1, 5, TINY.context_in_dim))
    latents = mx.random.normal((1, 8, TINY.in_channels))
    prefix = model.prefix(text, 2, 4)
    assert len(prefix.kv) == TINY.num_layers and prefix.kv[0][0].shape == (1, 2, 5, 16)
    for sigma in (1.0, 0.4):
        ours = model(latents, sigma, prefix)
        np.testing.assert_allclose(np.asarray(ours), np.asarray(joint_forward(model, latents, sigma, text)),
                                   atol=2e-4, rtol=2e-4)


def test_forward_rejects_a_wrong_latent_shape():
    model = tiny_model()
    prefix = model.prefix(mx.zeros((1, 3, TINY.context_in_dim)), 2, 4)
    with pytest.raises(ValueError):
        model(mx.zeros((1, 6, TINY.in_channels)), 0.5, prefix)
    with pytest.raises(ValueError):
        model.prefix(mx.zeros((1, 3, 7)), 2, 4)


def test_fused_qkv_layout_gives_the_same_attention_scores():
    heads, dim, width = 2, 8, 12
    mx.random.seed(1)
    to_q, to_k, to_v = (mx.random.normal((heads * dim, width)) for _ in range(3))
    fused, order = weights.fused_qkv_weight(to_q, to_k, to_v, heads, dim)
    assert fused.shape == (3 * heads * dim, width)
    np.testing.assert_array_equal(np.asarray(order), [0, 2, 4, 6, 1, 3, 5, 7])
    x = mx.random.normal((5, width))
    angle = np.random.default_rng(0).normal(size=(5, dim // 2)).astype(np.float32)
    cos, sin = mx.array(np.cos(angle)), mx.array(np.sin(angle))
    # reference: neighbouring pairs
    q = qdit.apply_rotary((x @ to_q.T).reshape(1, 5, heads, dim), cos, sin)
    k = qdit.apply_rotary((x @ to_k.T).reshape(1, 5, heads, dim), cos, sin)
    reference = mx.einsum("brhd,bshd->bhrs", q, k)
    # fused layout: per head [q, k, v], q and k as [even, odd], rotated half against half
    out = (x @ fused.T).reshape(5, heads, 3, dim)

    def rotate_half(t):
        low, high = t[..., : dim // 2], t[..., dim // 2:]
        c, s = cos[:, None, :], sin[:, None, :]
        return mx.concatenate([low * c - high * s, high * c + low * s], axis=-1)

    fq, fk = rotate_half(out[:, :, 0]), rotate_half(out[:, :, 1])
    np.testing.assert_allclose(np.asarray(mx.einsum("rhd,shd->hrs", fq, fk)), np.asarray(reference[0]),
                               atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(np.asarray(out[:, :, 2]), np.asarray((x @ to_v.T).reshape(5, heads, dim)), atol=1e-5)


def test_sampler_is_deterministic_and_returns_channel_first_latents():
    model = tiny_model()
    text = mx.random.normal((1, 4, TINY.context_in_dim))
    seen = []
    first = sampler.denoise(model, text, 64, 32, steps=3, seed=5, on_step=lambda i, n: seen.append((i, n)))
    second = sampler.denoise(model, text, 64, 32, steps=3, seed=5)
    assert first.shape == (1, TINY.in_channels, 2, 4) and seen == [(0, 3), (1, 3), (2, 3)]
    np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
    assert not np.allclose(np.asarray(first), np.asarray(sampler.denoise(model, text, 64, 32, steps=3, seed=6)))


def test_lora_merge_adds_the_scaled_update_in_float32(tmp_path):
    model = tiny_model()
    target = model.transformer_blocks[1].attn.to_out[0]
    before = np.asarray(target.weight.astype(mx.float32))
    down, up = mx.random.normal((4, 32)), mx.random.normal((32, 4))
    path = tmp_path / "adapter.safetensors"
    mx.save_safetensors(str(path), {"transformer.transformer_blocks.1.attn.to_out.0.lora_A.weight": down,
                                    "transformer.transformer_blocks.1.attn.to_out.0.lora_B.weight": up},
                        metadata={"lora_adapter_metadata": json.dumps({"transformer.lora_alpha": 8})})
    report = lora.merge(model, path, strength=0.5)
    assert report["matrices"] == 1 and report["alpha"] == 8
    assert target.weight.dtype == mx.float32
    np.testing.assert_allclose(np.asarray(target.weight), before + 0.5 * (8 / 4) * np.asarray(up @ down), atol=1e-5)
    mx.save_safetensors(str(path), {"transformer.nowhere.lora_A.weight": down,
                                    "transformer.nowhere.lora_B.weight": up})
    with pytest.raises(KeyError):
        lora.merge(model, path)


def reference_shuffle(x: np.ndarray, out_dim: int, frames: int) -> np.ndarray:
    """mflux's channel-first formulation, written out with numpy."""

    batch, in_dim, height, width = x.shape
    repeats = out_dim * frames * 4 // in_dim
    y = np.repeat(x, repeats, axis=1).reshape(batch, out_dim, frames, 2, 2, 1, height, width)
    y = y.transpose(0, 1, 5, 2, 6, 3, 7, 4).reshape(batch, out_dim, frames, height * 2, width * 2)
    return y[:, :, frames - 1]


@pytest.mark.parametrize(("in_dim", "out_dim", "frames"), [(16, 16, 2), (16, 8, 2), (8, 4, 1)])
def test_shuffle_shortcut_matches_the_channel_first_reference(in_dim, out_dim, frames):
    x = np.random.default_rng(2).normal(size=(1, in_dim, 3, 5)).astype(np.float32)
    ours = vae.shuffle_shortcut(mx.array(x.transpose(0, 2, 3, 1)), out_dim, frames)
    np.testing.assert_array_equal(np.asarray(ours).transpose(0, 3, 1, 2), reference_shuffle(x, out_dim, frames))


def test_decoder_upsamples_sixteen_times_and_returns_rgb():
    cfg = {"z_dim": 4, "decoder_base_dim": 8, "dim_mult": [1, 2, 4, 8, 8], "num_res_blocks": 1, "out_channels": 4,
           "temporal_downsample": [False, True, True, True], "is_residual": True, "patch_size": None,
           "latents_mean": [0.0] * 4, "latents_std": [1.0] * 4}
    mx.random.seed(3)
    decoder = vae.ImageDecoder(cfg)
    names = [name for name, _ in __import__("mlx.utils", fromlist=["tree_flatten"]).tree_flatten(decoder.parameters())]
    assert "decoder.up_blocks.0.upsampler.resample.1.weight" in names and not any("mean" in n for n in names)
    image = decoder.decode(mx.random.normal((1, 4, 2, 3)))
    assert image.shape == (1, 32, 48, 3)
    values = np.asarray(image)
    assert values.min() >= 0.0 and values.max() <= 1.0
    with pytest.raises(ValueError):
        decoder.decode(mx.zeros((1, 5, 2, 3)))
