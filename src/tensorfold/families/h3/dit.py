"""The H3 diffusion transformer on MLX: one block stack over a packed video, audio and text sequence."""

# Adapted from minimax-h3-mlx (Apache-2.0, https://github.com/mrbizarro/minimax-h3-mlx, revision 7919020):
# the module tree and forward follow its `dit.py`, reduced to the released full-precision checkpoint.

from __future__ import annotations

import math

import mlx.core as mx
from mlx import nn

from .config import DiTConfig

MODALITIES = 3  # AdaLN keeps one parameter row per (timestep, modality): 0 video, 1 text, 2 audio


def timestep_embedding(timesteps: mx.array, dim: int, max_period: float = 10000.0) -> mx.array:
    """Sinusoidal embedding of unscaled timesteps in [0, 1], cosine half first (diffusers `flip_sin_to_cos`)."""

    half = dim // 2
    freqs = mx.exp(-math.log(max_period) * mx.arange(half, dtype=mx.float32) / half)
    angles = timesteps.astype(mx.float32)[:, None] * freqs[None, :]
    return mx.concatenate([mx.cos(angles), mx.sin(angles)], axis=-1)


class TimestepEmbedder(nn.Module):
    """Timestep MLP shared by every AdaLN projection; float32 in the released checkpoint and kept so."""

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.proj_in = nn.Linear(config.timestep_input_dim, config.time_embed_hidden_size, bias=True)
        self.proj_out = nn.Linear(config.time_embed_hidden_size, config.time_embed_dim, bias=True)

    def __call__(self, sinusoid: mx.array) -> mx.array:
        return self.proj_out(nn.silu(self.proj_in(sinusoid)))


def rotary_tables(config: DiTConfig, position_ids: mx.array) -> tuple[mx.array, mx.array]:
    """cos and sin, (rows, 2 * 3 * inv_freq_len), from (rows, 3) positions over (t, h, w)."""

    n = config.rope_inv_freq_len
    inv_freq = 1.0 / (config.rope_theta ** (mx.arange(0, 2 * n, 2, dtype=mx.float32) / (2 * n)))
    freqs = position_ids.astype(mx.float32)[..., None] * inv_freq.reshape(1, 1, -1)
    freqs = mx.concatenate([freqs[:, 0], freqs[:, 1], freqs[:, 2]], axis=-1)
    freqs = mx.concatenate([freqs, freqs], axis=-1)
    return mx.cos(freqs), mx.sin(freqs)


def apply_rotary(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Rotate the leading channels of every head (rotate-half), pass the rest through. x: (B, heads, rows, dim)."""

    width = cos.shape[-1]
    turned, kept = x[..., :width], x[..., width:]
    cos = cos.astype(x.dtype)[None, None]
    sin = sin.astype(x.dtype)[None, None]
    half = width // 2
    out = turned * cos + mx.concatenate([-turned[..., half:], turned[..., :half]], axis=-1) * sin
    return out if kept.shape[-1] == 0 else mx.concatenate([out, kept], axis=-1)


class Attention(nn.Module):
    """Full self-attention over the packed sequence with per-head RMSNorm on q and k."""

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.heads = config.num_attention_heads
        self.head_dim = config.attention_head_dim
        self.scale = self.head_dim**-0.5
        self.qkv_proj = nn.Linear(config.hidden_size, 3 * config.inner_dim, bias=False)
        self.q_norm = nn.RMSNorm(config.attention_head_dim, eps=config.qk_norm_eps)
        self.k_norm = nn.RMSNorm(config.attention_head_dim, eps=config.qk_norm_eps)
        self.out_proj = nn.Linear(config.inner_dim, config.hidden_size, bias=False)

    def qkv(self, x: mx.array, rotary=None) -> tuple[mx.array, mx.array, mx.array]:
        fused = getattr(self, "fused_qkv", None)
        if fused is not None and rotary is not None:
            return fused(x, *rotary)  # int8 projection with the q/k norm and rotation inside the kernel
        batch, rows, _ = x.shape
        # checkpoint rows are interleaved per head: [h0: q, k, v][h1: q, k, v]...
        qkv = self.qkv_proj(x).astype(x.dtype).reshape(batch, rows, self.heads, 3, self.head_dim)
        q = self.q_norm(qkv[:, :, :, 0]).transpose(0, 2, 1, 3)
        k = self.k_norm(qkv[:, :, :, 1]).transpose(0, 2, 1, 3)
        v = qkv[:, :, :, 2].transpose(0, 2, 1, 3)
        if rotary is not None:
            q, k = apply_rotary(q, *rotary), apply_rotary(k, *rotary)
        return q, k, v

    def mix(self, q: mx.array, k: mx.array, v: mx.array) -> mx.array:
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=None)
        batch, _, rows, _ = out.shape
        return out.transpose(0, 2, 1, 3).reshape(batch, rows, self.heads * self.head_dim)

    def __call__(self, x: mx.array, rotary=None) -> mx.array:
        return self.out_proj(self.mix(*self.qkv(x, rotary)).astype(x.dtype)).astype(x.dtype)


class FeedForward(nn.Module):
    """SwiGLU; ``fc1`` is the fused [gate; value] projection."""

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.width = config.ffn_hidden_size
        self.fc1 = nn.Linear(config.hidden_size, 2 * config.ffn_hidden_size, bias=False)
        self.fc2 = nn.Linear(config.ffn_hidden_size, config.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        fused = self.fc1(x).astype(x.dtype)
        return self.fc2((nn.silu(fused[..., : self.width]) * fused[..., self.width :]).astype(x.dtype)).astype(x.dtype)


class Modulation(nn.Module):
    """One block's AdaLN projection: timestep embedding to six (timesteps * MODALITIES, hidden) tables."""

    def __init__(self, config: DiTConfig, features: int):
        super().__init__()
        self.hidden = config.hidden_size
        self.linear = nn.Linear(config.time_embed_dim, features, bias=True)

    def __call__(self, temb: mx.array) -> mx.array:
        return self.linear(nn.silu(temb).astype(self.linear.weight.dtype))

    def tables(self, temb: mx.array) -> tuple[mx.array, ...]:
        flat = self(temb).reshape(-1, 6 * self.hidden).astype(mx.bfloat16)
        return tuple(flat[..., i * self.hidden : (i + 1) * self.hidden] for i in range(6))


class RefinerBlock(nn.Module):
    """Plain pre-norm block on the text rows before packing: no AdaLN, no rotary."""

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = Attention(config)
        self.norm2 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = FeedForward(config)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.norm1(x)).astype(x.dtype)
        return x + self.mlp(self.norm2(x)).astype(x.dtype)


class TokenRefiner(nn.Module):
    def __init__(self, config: DiTConfig):
        super().__init__()
        self.blocks = [RefinerBlock(config) for _ in range(config.token_refiner_num_layers)]
        self.final_norm = nn.RMSNorm(config.hidden_size, eps=config.final_norm_eps)

    def __call__(self, x: mx.array) -> mx.array:
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)


class Block(nn.Module):
    """Pre-norm attention and SwiGLU, each modulated by AdaLN rows chosen per sequence row."""

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = Attention(config)
        self.norm2 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = FeedForward(config)
        self.adaln_proj = Modulation(config, config.adaln_out_features)

    def __call__(self, x: mx.array, tables: tuple[mx.array, ...], rows: mx.array, rotary) -> mx.array:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = tables
        h = self.norm1(x) * (1.0 + scale_a[rows]) + shift_a[rows]
        x = x + (gate_a[rows] * self.attn(h, rotary)).astype(x.dtype)
        h = self.norm2(x) * (1.0 + scale_m[rows]) + shift_m[rows]
        return x + (gate_m[rows] * self.mlp(h)).astype(x.dtype)


class FinalLayer(nn.Module):
    """Shared modulated norm and the two output heads; its modulation has one row per timestep."""

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.hidden = config.hidden_size
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.final_norm_eps)
        self.adaln_proj = Modulation(config, config.final_adaln_out_features)
        patch = config.latents_dim * config.patch_size[0] * config.patch_size[1] * config.patch_size[2]
        self.video_out = nn.Linear(config.hidden_size, patch, bias=True)
        self.audio_out = nn.Linear(config.hidden_size, config.audio_latents_dim, bias=True)

    def norm_out(self, x: mx.array, table: mx.array, timestep_rows: mx.array) -> mx.array:
        shift, scale = table[..., : self.hidden], table[..., self.hidden :]
        return self.norm(x) * (1.0 + scale[timestep_rows]) + shift[timestep_rows]


class H3DiT(nn.Module):
    """MiniMax H3 joint video and audio transformer; parameter names equal the checkpoint's tensor names."""

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config
        patch = config.latents_dim * config.patch_size[0] * config.patch_size[1] * config.patch_size[2]
        self.video_patch_proj = nn.Linear(patch, config.hidden_size, bias=True)
        self.audio_patch_proj = nn.Linear(config.audio_latents_dim, config.hidden_size, bias=True)
        self.condition_proj = nn.Linear(config.text_dim, config.hidden_size, bias=True)
        self.time_embedder = TimestepEmbedder(config)
        self.token_refiner = TokenRefiner(config)
        self.blocks = [Block(config) for _ in range(config.num_layers)]
        self.final_layer = FinalLayer(config)
        self._modulation = None

    def cache_modulation(self, timestep: mx.array, release: bool = False) -> int:
        """Project every block's AdaLN tables for the run's timestep table once; returns the bytes cached.

        The tables depend only on the schedule, so a run needs them computed once. ``release`` then drops
        the projection weights (about 24 GiB), after which only this timestep table can be served.
        """

        temb = self.time_embedder(timestep_embedding(timestep, self.config.timestep_input_dim))
        tables = []
        for block in self.blocks:
            tables.append(block.adaln_proj.tables(temb))
            mx.eval(*tables[-1])
        final = self.final_layer.adaln_proj(temb).astype(mx.bfloat16)
        mx.eval(final)
        self._modulation = (timestep, tables, final)
        if release:
            for block in self.blocks:
                block.adaln_proj.linear = None
            self.final_layer.adaln_proj.linear = None
            mx.clear_cache()
        return sum(t.nbytes for group in tables for t in group) + final.nbytes

    def modulation(self, timestep: mx.array):
        """(per-block tables, final table) for ``timestep``, from the cache when it was built for the same table."""

        cached = self._modulation
        if cached is not None and (cached[0] is timestep or (cached[0].shape == timestep.shape
                                                              and bool(mx.all(cached[0] == timestep).item()))):
            return cached[1], cached[2]
        if self.final_layer.adaln_proj.linear is None:
            raise ValueError("the AdaLN projections were released; this model only serves its cached timesteps")
        temb = self.time_embedder(timestep_embedding(timestep, self.config.timestep_input_dim))
        return ([block.adaln_proj.tables(temb) for block in self.blocks],
                self.final_layer.adaln_proj(temb).astype(mx.bfloat16))

    def pack(self, video, audio, text, timestep, timestep_rows, tags, position_ids, video_rows, audio_rows,
             text_rows):
        """(x, adaln rows, rotary): everything the block stack needs before its first block."""

        rows = position_ids.shape[0]
        if position_ids.shape != (rows, 3) or tags.shape != (rows,) or timestep_rows.shape != (rows,):
            raise ValueError("position_ids must be (rows, 3) with tags and timestep_rows of (rows,)")
        rotary = rotary_tables(self.config, position_ids)
        # the packed sequence is bfloat16 whatever the projection weights are (adapters can leave them float32)
        text = self.condition_proj(text.astype(self.condition_proj.weight.dtype)).astype(mx.bfloat16)
        text = self.token_refiner(text).astype(mx.bfloat16)
        x = mx.zeros((text.shape[0], rows, text.shape[-1]), dtype=mx.bfloat16)
        x[:, text_rows] = text
        x[:, video_rows] = self.video_patch_proj(video.astype(mx.float32)).astype(mx.bfloat16)
        x[:, audio_rows] = self.audio_patch_proj(audio.astype(mx.float32)).astype(mx.bfloat16)
        # padding rows carry tag -1 and must not index backwards; they never reach an output
        return x, timestep_rows * MODALITIES + mx.maximum(tags, 0), rotary

    def __call__(self, video, audio, text, timestep, timestep_rows, tags, position_ids, video_rows, audio_rows,
                 text_rows) -> tuple[mx.array, mx.array]:
        """Video and audio velocity for one packed sequence, in the order of ``video_rows`` and ``audio_rows``.

        ``timestep`` holds the distinct noise levels present and ``timestep_rows`` indexes it per sequence row;
        ``tags`` is the modality per row.
        """

        x, rows, rotary = self.pack(video, audio, text, timestep, timestep_rows, tags, position_ids, video_rows,
                                    audio_rows, text_rows)
        tables, final = self.modulation(timestep)
        for block, table in zip(self.blocks, tables, strict=True):
            x = block(x, table, rows, rotary)
        x = self.final_layer.norm_out(x, final, timestep_rows).astype(mx.float32)
        return self.final_layer.video_out(x)[:, video_rows], self.final_layer.audio_out(x)[:, audio_rows]
