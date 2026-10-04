"""The Qwen-Image-2.1 transformer on MLX: one block stack over a causal text prefix and the image tokens."""

# Adapted from mflux (MIT, https://github.com/mflux-community/mflux, revision add5164), whose qwen21 transformer
# follows Qwen and Hugging Face's QwenImage21Transformer2DModel (Apache-2.0). The text prefix is run once per
# prompt and its keys and values reused on every step, as there.

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import numpy as np
from mlx import nn

from .config import DiTConfig

TIME_FACTOR = 1000.0


def timestep_embedding(timesteps: mx.array, dim: int, max_period: float = 10000.0) -> mx.array:
    """Sinusoidal embedding of noise levels in [0, 1], cosine half first."""

    half = dim // 2
    freqs = mx.array(np.exp(-math.log(max_period) * np.arange(half, dtype=np.float32) / half))
    angles = timesteps.astype(mx.float32)[:, None] * TIME_FACTOR * freqs[None, :]
    return mx.concatenate([mx.cos(angles), mx.sin(angles)], axis=-1)


def rotary_tables(config: DiTConfig, text_len: int, rows: int, columns: int) -> tuple[mx.array, mx.array]:
    """cos and sin, (text_len + rows * columns, head_dim / 2), over (frame, height, width).

    Text tokens advance one position on all three axes. The image sits at frame ``text_len`` on a height and
    width grid centred on zero, so its spatial positions do not depend on the prompt length.
    """

    count = rows * columns
    frame = np.concatenate([np.arange(text_len), np.full(count, text_len)]).astype(np.float32)
    height = np.concatenate([np.arange(text_len), np.repeat(np.arange(-(rows - rows // 2), rows // 2), columns)])
    width = np.concatenate([np.arange(text_len), np.tile(np.arange(-(columns - columns // 2), columns // 2), rows)])
    angles = []
    for axis, dim in zip((frame, height.astype(np.float32), width.astype(np.float32)), config.axes_dims_rope):
        omega = 1.0 / (config.rope_theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        angles.append(np.outer(axis, omega))
    angles = np.concatenate(angles, axis=-1)
    return mx.array(np.cos(angles)), mx.array(np.sin(angles))


def apply_rotary(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Rotate channel pairs (2k, 2k + 1) by angle k. x: (batch, rows, heads, dim); tables: (rows, dim / 2)."""

    pairs = x.astype(mx.float32).reshape(*x.shape[:-1], -1, 2)
    real, imag = pairs[..., 0], pairs[..., 1]
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    out = mx.stack([real * cos - imag * sin, real * sin + imag * cos], axis=-1)
    return out.reshape(x.shape).astype(x.dtype)


@dataclass
class Rotary:
    """Rotary tables for a run of rows: pair form for the float path, doubled for the fused int8 kernel."""

    cos: mx.array
    sin: mx.array

    def __post_init__(self):
        self.cos_half = mx.concatenate([self.cos, self.cos], axis=-1)
        self.sin_half = mx.concatenate([self.sin, self.sin], axis=-1)
        mx.eval(self.cos_half, self.sin_half)


class Attention(nn.Module):
    """Self-attention with per-head RMSNorm on q and k; the caller decides which keys and values are visible."""

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.heads = config.num_attention_heads
        self.head_dim = config.attention_head_dim
        self.scale = self.head_dim**-0.5
        width = config.hidden_size
        self.to_q = nn.Linear(width, width, bias=False)
        self.to_k = nn.Linear(width, width, bias=False)
        self.to_v = nn.Linear(width, width, bias=False)
        self.to_out = [nn.Linear(width, width, bias=False)]
        self.norm_q = nn.RMSNorm(config.attention_head_dim, eps=config.eps)
        self.norm_k = nn.RMSNorm(config.attention_head_dim, eps=config.eps)

    def qkv(self, x: mx.array, rotary: Rotary) -> tuple[mx.array, mx.array, mx.array]:
        """q, k, v as (batch, heads, rows, head_dim), q and k normalised and rotated."""

        fused = getattr(self, "fused_qkv", None)
        if fused is not None:
            # int8 projection with the q/k norm and rotation inside the kernel; its q and k channels are
            # reordered within each head, which a dot product between them does not see
            return fused(x, rotary.cos_half, rotary.sin_half)
        shape = (*x.shape[:-1], self.heads, self.head_dim)
        q = apply_rotary(self.norm_q(self.to_q(x).astype(x.dtype).reshape(shape)), rotary.cos, rotary.sin)
        k = apply_rotary(self.norm_k(self.to_k(x).astype(x.dtype).reshape(shape)), rotary.cos, rotary.sin)
        v = self.to_v(x).astype(x.dtype).reshape(shape)
        return q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3)

    def mix(self, q: mx.array, k: mx.array, v: mx.array, mask=None) -> mx.array:
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        batch, _, rows, _ = out.shape
        out = out.transpose(0, 2, 1, 3).reshape(batch, rows, self.heads * self.head_dim)
        return self.to_out[0](out.astype(q.dtype)).astype(q.dtype)


class FeedForward(nn.Module):
    """SwiGLU with separate gate and value projections."""

    def __init__(self, config: DiTConfig):
        super().__init__()
        width = config.hidden_size * config.mlp_ratio
        self.proj = nn.Linear(config.hidden_size, width, bias=False)
        self.gate_layer = nn.Linear(config.hidden_size, width, bias=False)
        self.out = nn.Linear(width, config.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        hidden = nn.silu(self.gate_layer(x).astype(x.dtype)) * self.proj(x).astype(x.dtype)
        return self.out(hidden).astype(x.dtype)


class Block(nn.Module):
    """Pre-norm attention and SwiGLU, each scaled and gated by the modulation shared across blocks."""

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.img_norm1 = nn.LayerNorm(config.hidden_size, eps=config.eps, affine=False)
        self.attn = Attention(config)
        self.img_norm2 = nn.LayerNorm(config.hidden_size, eps=config.eps, affine=False)
        self.img_mlp = FeedForward(config)

    def _finish(self, x: mx.array, mixed: mx.array, modulation) -> mx.array:
        _, gate1, scale2, gate2 = modulation
        x = x + (mx.tanh(gate1) * mixed).astype(x.dtype)
        return x + (mx.tanh(gate2) * self.img_mlp(self.img_norm2(x) * (1 + scale2))).astype(x.dtype)

    def text(self, x: mx.array, modulation, rotary: Rotary) -> tuple[mx.array, mx.array, mx.array]:
        """The block over the text prefix alone (causal); also returns its keys and values for the image rows."""

        q, k, v = self.attn.qkv(self.img_norm1(x) * (1 + modulation[0]), rotary)
        return self._finish(x, self.attn.mix(q, k, v, mask="causal"), modulation), k, v

    def image(self, x: mx.array, text_k: mx.array, text_v: mx.array, modulation, rotary: Rotary) -> mx.array:
        """The block over the image rows, which see the stored text keys and values and each other."""

        q, k, v = self.attn.qkv(self.img_norm1(x) * (1 + modulation[0]), rotary)
        mixed = self.attn.mix(q, mx.concatenate([text_k, k], axis=2), mx.concatenate([text_v, v], axis=2))
        return self._finish(x, mixed, modulation)


class ZeroCenterRMSNorm(nn.Module):
    """RMSNorm whose stored weight is the offset from one."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = mx.zeros((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        value = x.astype(mx.float32)
        value = value * mx.rsqrt(mx.mean(value * value, axis=-1, keepdims=True) + self.eps)
        return (value * (self.weight.astype(mx.float32) + 1)).astype(x.dtype)


class TextProjection(nn.Module):
    def __init__(self, config: DiTConfig):
        super().__init__()
        self.text_norm = ZeroCenterRMSNorm(config.context_in_dim, config.eps)
        self.in_layer = nn.Linear(config.context_in_dim, config.hidden_size, bias=False)
        self.out_layer = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.out_layer(nn.gelu_approx(self.in_layer(self.text_norm(x))))


class TimestepEmbedder(nn.Module):
    def __init__(self, config: DiTConfig):
        super().__init__()
        self.linear_1 = nn.Linear(config.timestep_dim, config.hidden_size, bias=False)
        self.linear_2 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def __call__(self, sinusoid: mx.array) -> mx.array:
        return self.linear_2(nn.silu(self.linear_1(sinusoid)))


class TimeEmbed(nn.Module):
    def __init__(self, config: DiTConfig):
        super().__init__()
        self.dim = config.timestep_dim
        self.timestep_embedder = TimestepEmbedder(config)

    def __call__(self, timesteps: mx.array) -> mx.array:
        sinusoid = timestep_embedding(timesteps, self.dim)
        return self.timestep_embedder(sinusoid.astype(self.timestep_embedder.linear_1.weight.dtype))


class OutputNorm(nn.Module):
    """Final scale-only adaptive norm."""

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.linear = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.eps, affine=False)


@dataclass
class Prefix:
    """A prompt run through every block once: per-block text keys and values, and the image rows' rotary."""

    kv: list[tuple[mx.array, mx.array]]
    text_len: int
    grid: tuple[int, int]
    rotary: Rotary


class QwenImageDiT(nn.Module):
    """Qwen-Image-2.1 transformer; parameter names equal the checkpoint's tensor names."""

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config
        hidden = config.hidden_size
        self.time_text_embed = TimeEmbed(config)
        self.txt_in = TextProjection(config)
        self.img_in = nn.Linear(config.in_channels, hidden, bias=False)
        # one modulation for every block: [attention scale, attention gate, MLP scale, MLP gate]
        self.modulation = [nn.SiLU(), nn.Linear(hidden, 4 * hidden, bias=False)]
        self.transformer_blocks = [Block(config) for _ in range(config.num_layers)]
        self.norm_out = OutputNorm(config)
        self.proj_out = nn.Linear(hidden, config.out_channels, bias=False)

    @property
    def dtype(self):
        return self.img_in.weight.dtype

    def _modulation(self, temb: mx.array, row: int) -> tuple[mx.array, ...]:
        table = self.modulation[1](nn.silu(temb))[row].astype(self.dtype)
        return tuple(mx.split(table, 4, axis=-1))

    def prefix(self, text: mx.array, rows: int, columns: int) -> Prefix:
        """Run the prompt through the stack once for an image of ``rows`` x ``columns`` latent tokens.

        Text tokens take the modulation of noise level zero and see only earlier text, so their stream, and
        each block's keys and values over them, are the same on every denoising step.
        """

        if text.ndim != 3 or text.shape[0] != 1 or text.shape[-1] != self.config.context_in_dim:
            raise ValueError(f"text must be (1, tokens, {self.config.context_in_dim}), got {tuple(text.shape)}")
        text_len = text.shape[1]
        cos, sin = rotary_tables(self.config, text_len, rows, columns)
        text_rotary = Rotary(cos[:text_len], sin[:text_len])
        modulation = self._modulation(self.time_text_embed(mx.zeros((1,))), 0)
        x = self.txt_in(text.astype(self.txt_in.in_layer.weight.dtype)).astype(self.dtype)
        kv = []
        for block in self.transformer_blocks:
            x, k, v = block.text(x, modulation, text_rotary)
            kv.append((k, v))
            mx.eval(x, k, v)
        return Prefix(kv, text_len, (rows, columns), Rotary(cos[text_len:], sin[text_len:]))

    def __call__(self, latents: mx.array, sigma: float, prefix: Prefix) -> mx.array:
        """Velocity (1, rows * columns, out_channels) for image latents at noise level ``sigma``."""

        rows, columns = prefix.grid
        if latents.shape != (1, rows * columns, self.config.in_channels):
            raise ValueError(f"latents must be (1, {rows * columns}, {self.config.in_channels}), got "
                             f"{tuple(latents.shape)}")
        temb = self.time_text_embed(mx.array([sigma], dtype=mx.float32))
        modulation = self._modulation(temb, 0)
        x = self.img_in(latents.astype(self.dtype)).astype(self.dtype)
        for block, (text_k, text_v) in zip(self.transformer_blocks, prefix.kv, strict=True):
            x = block.image(x, text_k, text_v, modulation, prefix.rotary)
        scale = self.norm_out.linear(nn.silu(temb))[0].astype(self.dtype)
        return self.proj_out((self.norm_out.norm(x) * (1 + scale)).astype(self.dtype))
