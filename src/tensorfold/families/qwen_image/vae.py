"""The Qwen-Image-2.1 image decoder on MLX: 64-channel latents to RGB(A) at 16x, channels last."""

# Adapted from mflux (MIT, https://github.com/mflux-community/mflux, revision add5164), whose qwen21 VAE follows
# Qwen and Hugging Face's AutoencoderKLQwenImage21 (Apache-2.0). Only the single-image decode path is here: the
# checkpoint's per-frame `time_conv` weights and its encoder are not loaded.

from __future__ import annotations

import json

import mlx.core as mx
import numpy as np
from mlx import nn

from .config import pipeline_root


class ChannelNorm(nn.Module):
    """Unit-length channels scaled by sqrt(channels) and a learned gain."""

    def __init__(self, channels: int):
        super().__init__()
        self.gamma = mx.ones((channels,))
        self.scale = float(channels) ** 0.5

    def __call__(self, x: mx.array) -> mx.array:
        value = x.astype(mx.float32)
        norm = mx.maximum(mx.sqrt(mx.sum(value * value, axis=-1, keepdims=True)), 1e-12)
        return (value / norm * self.scale * self.gamma.astype(mx.float32)).astype(x.dtype)


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm1 = ChannelNorm(in_dim)
        self.conv1 = nn.Conv2d(in_dim, out_dim, 3, padding=1)
        self.norm2 = ChannelNorm(out_dim)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, padding=1)
        if in_dim != out_dim:
            self.conv_shortcut = nn.Conv2d(in_dim, out_dim, 1)

    def __call__(self, x: mx.array) -> mx.array:
        shortcut = self.conv_shortcut(x) if "conv_shortcut" in self else x
        x = self.conv1(nn.silu(self.norm1(x)))
        return self.conv2(nn.silu(self.norm2(x))) + shortcut


class AttentionBlock(nn.Module):
    """Single-head attention over every position; the 1x1 convolutions are projections."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = ChannelNorm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def __call__(self, x: mx.array) -> mx.array:
        batch, height, width, channels = x.shape
        qkv = self.to_qkv(self.norm(x)).reshape(batch, 1, height * width, 3 * channels)
        q, k, v = mx.split(qkv, 3, axis=-1)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=channels**-0.5)
        return x + self.proj(out.reshape(batch, height, width, channels))


class MidBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.resnets = [ResidualBlock(dim, dim), ResidualBlock(dim, dim)]
        self.attentions = [AttentionBlock(dim)]

    def __call__(self, x: mx.array) -> mx.array:
        return self.resnets[1](self.attentions[0](self.resnets[0](x)))


class Upsampler(nn.Module):
    """Nearest 2x then a 3x3 convolution; `resample.1` is the checkpoint's name for it."""

    def __init__(self, dim: int):
        super().__init__()
        self.resample = [nn.Identity(), nn.Conv2d(dim, dim, 3, padding=1)]

    def __call__(self, x: mx.array) -> mx.array:
        return self.resample[1](mx.repeat(mx.repeat(x, 2, axis=1), 2, axis=2))


def shuffle_shortcut(x: mx.array, out_dim: int, frames: int) -> mx.array:
    """The parameter-free 2x shortcut: channels repeated, then moved into space (last frame of ``frames``).

    Output channel ``o`` at offset (i, j) of each 2x2 cell reads input channel
    ``(((o * frames + frames - 1) * 2 + i) * 2 + j) // repeats``.
    """

    batch, height, width, in_dim = x.shape
    factor = frames * 4
    if out_dim * factor % in_dim:
        raise ValueError(f"{out_dim} x {factor} is not a multiple of {in_dim}")
    repeats = out_dim * factor // in_dim
    outputs = np.arange(out_dim)[None, :]
    cell = np.arange(4)[:, None]                       # i * 2 + j
    source = ((outputs * frames + frames - 1) * 4 + cell) // repeats
    picked = x[..., mx.array(source.reshape(-1))].reshape(batch, height, width, 2, 2, out_dim)
    return picked.transpose(0, 1, 3, 2, 4, 5).reshape(batch, height * 2, width * 2, out_dim)


class UpBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, blocks: int, up: bool, temporal: bool):
        super().__init__()
        self.out_dim = out_dim
        self.frames = 2 if temporal else 1
        self.resnets = [ResidualBlock(in_dim if i == 0 else out_dim, out_dim) for i in range(blocks + 1)]
        if up:
            self.upsampler = Upsampler(out_dim)

    def __call__(self, x: mx.array) -> mx.array:
        original = x
        for block in self.resnets:
            x = block(x)
        if "upsampler" in self:
            x = self.upsampler(x) + shuffle_shortcut(original, self.out_dim, self.frames)
        return x


class Decoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        mult = config["dim_mult"]
        dims = [config["decoder_base_dim"] * factor for factor in [mult[-1], *reversed(mult)]]
        temporal = list(reversed(config["temporal_downsample"]))
        self.conv_in = nn.Conv2d(config["z_dim"], dims[0], 3, padding=1)
        self.mid_block = MidBlock(dims[0])
        self.up_blocks = [
            UpBlock(a, b, config["num_res_blocks"], i < len(dims) - 2, temporal[i] if i < len(temporal) else False)
            for i, (a, b) in enumerate(zip(dims[:-1], dims[1:]))
        ]
        self.norm_out = ChannelNorm(dims[-1])
        self.conv_out = nn.Conv2d(dims[-1], config["out_channels"], 3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.mid_block(self.conv_in(x))
        for block in self.up_blocks:
            x = block(x)
            mx.eval(x)
        return self.conv_out(nn.silu(self.norm_out(x)))


class ImageDecoder(nn.Module):
    """`post_quant_conv` and the decoder; parameter names equal the checkpoint's."""

    def __init__(self, config: dict):
        super().__init__()
        if not config.get("is_residual") or config.get("patch_size") is not None:
            raise ValueError("only the residual, unpatched Qwen-Image-2.1 VAE is supported")
        self.channels = config["z_dim"]
        self._mean = mx.array(config["latents_mean"], dtype=mx.float32)
        self._std = mx.array(config["latents_std"], dtype=mx.float32)
        self.post_quant_conv = nn.Conv2d(self.channels, self.channels, 1)
        self.decoder = Decoder(config)

    def decode(self, latents: mx.array) -> mx.array:
        """(1, rows * 16, columns * 16, 3) float32 in [0, 1] from latents (1, channels, rows, columns)."""

        if latents.ndim != 4 or latents.shape[1] != self.channels:
            raise ValueError(f"latents must be (batch, {self.channels}, rows, columns), got {tuple(latents.shape)}")
        dtype = self.post_quant_conv.weight.dtype
        x = latents.transpose(0, 2, 3, 1).astype(mx.float32) * self._std + self._mean
        image = self.decoder(self.post_quant_conv(x.astype(dtype)))[..., :3].astype(mx.float32)
        return mx.clip(image / 2 + 0.5, 0, 1)


def load_decoder(model_dir, dtype=None) -> ImageDecoder:
    """The image decoder from a Qwen-Image-2.1 pipeline folder, float32 unless ``dtype`` says otherwise."""

    from mlx.utils import tree_flatten, tree_unflatten

    root = pipeline_root(model_dir)
    if root is None:
        raise FileNotFoundError(f"{model_dir} is not a Qwen-Image-2.1 pipeline folder")
    with open(root / "vae" / "config.json") as handle:
        config = json.load(handle)
    config.setdefault("temporal_downsample", config.get("temperal_downsample"))
    model = ImageDecoder(config)
    expected = {name: value.shape for name, value in tree_flatten(model.parameters())}
    found = {}
    for name, tensor in mx.load(str(root / "vae" / "diffusion_pytorch_model.safetensors")).items():
        if name.startswith(("encoder.", "quant_conv.")) or ".time_conv." in name:
            continue
        if name not in expected:
            raise KeyError(f"the VAE tensor {name} has no place in the decoder")
        if name.endswith(".gamma"):
            tensor = tensor.reshape(-1)
        elif tensor.ndim == 4:
            tensor = tensor.transpose(0, 2, 3, 1)  # (out, in, kh, kw) -> (out, kh, kw, in)
        if tuple(tensor.shape) != tuple(expected[name]):
            raise ValueError(f"{name} is {tuple(tensor.shape)}, the decoder expects {tuple(expected[name])}")
        found[name] = tensor if dtype is None else tensor.astype(dtype)
    missing = sorted(expected.keys() - found.keys())
    if missing:
        raise KeyError(f"{len(missing)} decoder parameters are not in the checkpoint, e.g. {missing[:3]}")
    model.update(tree_unflatten(list(found.items())))
    mx.eval(model.parameters())
    return model
