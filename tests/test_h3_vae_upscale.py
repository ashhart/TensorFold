"""H3 video decoder with a packed 2x head: frame size, pixel order and loading a replacement decoder."""

from __future__ import annotations

import json

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from tensorfold.families.h3 import vae_video  # noqa: E402

SMALL = dict(latent_channels=4, layers=1, heads=2, head_dim=8, register_tokens=1, spatial_ratio=2, temporal_ratio=4,
             clip_length=17, token_drop=3, tile=8, tile_overlap=2, latents_mean=(0.0,) * 4, latents_std=(1.0,) * 4)


def test_upscale_follows_the_packed_channel_count():
    assert vae_video.DecoderConfig(**SMALL).upscale == 1
    assert vae_video.DecoderConfig(**SMALL, out_channels=12).upscale == 2
    assert vae_video.DecoderConfig(**SMALL, out_channels=27).upscale == 3


def test_packed_frames_are_shuffled_into_twice_the_size():
    decoder = vae_video.VideoDecoder(vae_video.DecoderConfig(**SMALL, out_channels=12))
    packed = np.random.default_rng(0).uniform(-1, 1, size=(1, 12, 3, 2, 5)).astype(np.float32)
    decoder.decode = lambda z: mx.array(packed)
    frames = decoder.frames(mx.zeros((1, 4, 1, 1, 1)))
    assert frames.shape == (3, 4, 10, 3) and frames.dtype == np.uint8
    mean, std = np.array(vae_video.PIXEL_MEAN, np.float32), np.array(vae_video.PIXEL_STD, np.float32)
    for colour, i, j, frame, y, x in [(0, 0, 0, 0, 0, 0), (1, 1, 0, 2, 1, 4), (2, 0, 1, 1, 0, 3), (2, 1, 1, 2, 1, 2)]:
        value = packed[0, colour * 4 + i * 2 + j, frame, y, x] * std[colour] + mean[colour]
        expected = int(np.clip(value, 0, 1) * 255.0 + 0.5)
        assert abs(int(frames[frame, y * 2 + i, x * 2 + j, colour]) - expected) <= 1


def test_plain_decoder_frames_keep_their_size():
    decoder = vae_video.VideoDecoder(vae_video.DecoderConfig(**SMALL))
    decoder.decode = lambda z: mx.zeros((1, 3, 2, 4, 6))
    assert decoder.frames(mx.zeros((1, 4, 1, 1, 1))).shape == (2, 4, 6, 3)


def write_pipeline(tmp_path, tensors):
    (tmp_path / "model_index.json").write_text(json.dumps({"_class_name": "MiniMaxH3Pipeline"}))
    (tmp_path / "transformer").mkdir()
    source = tmp_path / "video_vae" / "source"
    source.mkdir(parents=True)
    (tmp_path / "video_vae" / "config.json").write_text(json.dumps({
        "vae_clip_length": 17, "vae_token_drop": 3, "vae_tile_size": 8, "vae_tile_overlap_min": 2,
        "latents_mean": [0.0] * 4, "latents_std": [1.0] * 4}))
    (source / "config.json").write_text(json.dumps({
        "z_channels": 4, "out_ch": 3, "space_down": [2], "time_down": [4],
        "vit_decoder_kwargs": {"num_layers": 1, "heads": 2, "dim_head": 8, "rope_theta": 100.0,
                               "rope_dim_ratio": 0.75}}))
    mx.save_safetensors(str(source / "model.safetensors"), tensors)


def test_replacement_decoder_head_is_loaded_and_the_rest_kept(tmp_path):
    from mlx.utils import tree_flatten

    mx.random.seed(4)
    stock = vae_video.VideoDecoder(vae_video.DecoderConfig(**{**SMALL, "register_tokens": 4}))
    tensors = {name: mx.random.normal(value.shape) for name, value in tree_flatten(stock.parameters())}
    tensors["post_quant_conv.weight"] = tensors.pop("post_quant.weight").reshape(4, 4)
    tensors["post_quant_conv.bias"] = tensors.pop("post_quant.bias")
    write_pipeline(tmp_path, tensors)
    block = 4 * 2 * 2
    head = mx.random.normal((12 * block, 16))
    replacement = tmp_path / "x2.safetensors"
    mx.save_safetensors(str(replacement), {"decoder.proj_out.weight": head.astype(mx.float16),
                                           "decoder.proj_out.bias": mx.zeros((12 * block,), dtype=mx.float16),
                                           "decoder.norm_out.weight": mx.full((16,), 2.0, dtype=mx.float16)})
    plain = vae_video.load_video_decoder(tmp_path, int8=False)
    doubled = vae_video.load_video_decoder(tmp_path, int8=False, upscale_decoder=replacement)
    assert plain.config.upscale == 1 and doubled.config.upscale == 2
    assert doubled.decoder.proj_out.weight.shape == (12 * block, 16)
    assert doubled.decoder.proj_out.weight.dtype == mx.float32
    np.testing.assert_allclose(np.asarray(doubled.decoder.norm_out.weight), 2.0)
    np.testing.assert_array_equal(np.asarray(doubled.decoder.x_embedder.weight),
                                  np.asarray(plain.decoder.x_embedder.weight))
    bad = tmp_path / "bad.safetensors"
    mx.save_safetensors(str(bad), {"decoder.proj_out.weight": mx.zeros((5 * block, 16))})
    with pytest.raises(ValueError):
        vae_video.load_video_decoder(tmp_path, int8=False, upscale_decoder=bad)
