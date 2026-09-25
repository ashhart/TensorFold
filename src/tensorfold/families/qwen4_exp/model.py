"""Qwen3.8 Flash Next (model_type ``qwen4_exp``): TensorFold's own forward pass.

Text only, written from the checkpoint layout:

- four residual streams joined by hyper-connections (each block reads a learned
  mix of the streams and writes back to each with its own gate);
- 36 Gated DeltaNet layers and 12 Qwen sparse-attention (QSA) layers, whose
  block indexer picks 512 blocks of 4 keys once the context passes 2,048 tokens;
- a 512-expert MoE (top 10) with a gated shared expert in every layer;
- a hashed 2- and 3-gram embedding (PLE, 128 row shards) added before layer 1.

Weights are the MLX 4-bit checkpoint as shipped (group size 32). The n-gram
hashes are computed on the host from token ids the caller already holds, so a
decode step never waits on the GPU to find its embedding rows.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.gated_delta import gated_delta_update
from mlx_lm.models.switch_layers import SwitchGLU

MODEL_TYPE = "qwen4_exp"

_MASK64 = (1 << 64) - 1
_GOLDEN = 0x9E3779B97F4A7C15
_MIX1 = 0xBF58476D1CE4E5B9
_MIX2 = 0x94D049BB133111EB
_PRIME = 10007


@dataclass
class Config:
    hidden_size: int
    num_hidden_layers: int
    layer_types: list[str]
    vocab_size: int
    rms_norm_eps: float
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rope_theta: float
    rotary_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    output_gate_type: str
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    hc_count: int
    hc_lowrank: int
    indexer_n_heads: int
    indexer_head_dim: int
    indexer_budget: int
    indexer_compress_ratio: int
    ple_layer_ids: list[int]          # one-indexed decoder layers that get the n-gram embedding
    ple_embed_dim: int
    ple_conv_kernel_size: int
    ngram_size: int
    heads_per_ngram: int
    ngram_vocab_size_base: int
    ngram_vocab_divisor: int
    ngram_shards: int
    seed: int
    ple_eos: int
    group_size: int
    bits: int

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "Config":
        t = dict(config.get("text_config") or config)
        rope = dict(t.get("rope_parameters") or {})
        head_dim = int(t.get("head_dim") or t["hidden_size"] // t["num_attention_heads"])
        partial = float(rope.get("partial_rotary_factor", t.get("partial_rotary_factor", 0.25)))
        eos = t.get("eos_token_id")
        quant = config.get("quantization") or config.get("quantization_config") or {}
        layer_types = [
            "linear_attention" if kind == "linear_attention" else "sparse_attention"
            for kind in t["layer_types"]
        ]
        return cls(
            hidden_size=int(t["hidden_size"]),
            num_hidden_layers=int(t["num_hidden_layers"]),
            layer_types=layer_types,
            vocab_size=int(t["vocab_size"]),
            rms_norm_eps=float(t["rms_norm_eps"]),
            num_attention_heads=int(t["num_attention_heads"]),
            num_key_value_heads=int(t["num_key_value_heads"]),
            head_dim=head_dim,
            rope_theta=float(rope.get("rope_theta", 10_000_000)),
            rotary_dim=int(head_dim * partial),
            linear_num_key_heads=int(t["linear_num_key_heads"]),
            linear_num_value_heads=int(t["linear_num_value_heads"]),
            linear_key_head_dim=int(t["linear_key_head_dim"]),
            linear_value_head_dim=int(t["linear_value_head_dim"]),
            linear_conv_kernel_dim=int(t["linear_conv_kernel_dim"]),
            output_gate_type=str(t.get("output_gate_type") or "sigmoid"),
            num_experts=int(t["num_experts"]),
            num_experts_per_tok=int(t["num_experts_per_tok"]),
            moe_intermediate_size=int(t["moe_intermediate_size"]),
            shared_expert_intermediate_size=int(t["shared_expert_intermediate_size"]),
            hc_count=int(t.get("hc_count", 4)),
            hc_lowrank=int(t.get("hc_lowrank", 320)),
            indexer_n_heads=int(t.get("indexer_n_heads", 4)),
            indexer_head_dim=int(t.get("indexer_head_dim", 128)),
            indexer_budget=int(t.get("indexer_budget", 2048)),
            indexer_compress_ratio=int(t.get("indexer_compress_ratio", 4)),
            ple_layer_ids=sorted({int(i) for i in t.get("ple_layer_ids") or []}),
            ple_embed_dim=int(t.get("ple_embed_dim") or t["hidden_size"]),
            ple_conv_kernel_size=int(t.get("ple_conv_kernel_size", 4)),
            ngram_size=int(t.get("ngram_size", 3)),
            heads_per_ngram=int(t.get("heads_per_ngram", 8)),
            ngram_vocab_size_base=int(t.get("ngram_vocab_size_base", 20_000_000)),
            ngram_vocab_divisor=int(t.get("make_ngram_vocab_size_divisible_by", 128)),
            ngram_shards=int(t.get("split_ngram_parts", 128)),
            seed=int(t.get("seed", 1234)),
            ple_eos=int(eos[0] if isinstance(eos, list) else eos) if eos is not None else 0,
            group_size=int(quant.get("group_size", 32)),
            bits=int(quant.get("bits", 4)),
        )


# -- norms -----------------------------------------------------------------
def _derived(module: nn.Module, name: str, make: Any) -> mx.array:
    """A constant derived from a loaded weight, kept outside the parameter tree."""

    cache = module.__dict__.setdefault("_derived", {})
    value = cache.get(name)
    if value is None:
        value = make()
        mx.eval(value)
        cache[name] = value
    return value


class CenteredRMSNorm(nn.Module):
    """RMSNorm whose weight is stored centred on zero: scale by (1 + w), in float32.

    ``group`` normalizes each run of ``group`` features on its own (one per residual stream).
    """

    def __init__(self, dims: int, eps: float, group: int | None = None) -> None:
        super().__init__()
        self.eps = eps
        self.group = group
        self.weight = mx.zeros((dims,))

    def __call__(self, x: mx.array) -> mx.array:
        scale = _derived(self, "scale", lambda: 1.0 + self.weight.astype(mx.float32))
        y = x.astype(mx.float32)
        if self.group is not None:
            y = y.reshape(*y.shape[:-1], -1, self.group)
            scale = scale.reshape(-1, self.group)
        y = y * mx.rsqrt(mx.mean(mx.square(y), axis=-1, keepdims=True) + self.eps)
        return (y * scale).reshape(x.shape).astype(x.dtype)


class GatedRMSNorm(nn.Module):
    """RMSNorm of the recurrence output times a sigmoid (or SiLU) of the gate projection."""

    def __init__(self, dims: int, eps: float, activation: str) -> None:
        super().__init__()
        self.eps = eps
        self.activation = activation
        self.weight = mx.ones((dims,))

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        y = mx.fast.rms_norm(x, self.weight, self.eps).astype(mx.float32)
        g = gate.astype(mx.float32)
        g = mx.sigmoid(g) if self.activation == "sigmoid" else nn.silu(g)
        return (y * g).astype(x.dtype)


# -- hyper-connections -------------------------------------------------------
class HyperConnection(nn.Module):
    """Read a block's input as a gated mix of the residual streams; say how much each stream takes back."""

    def __init__(self, cfg: Config, combine: bool = True) -> None:
        super().__init__()
        self.streams = cfg.hc_count
        self.dims = cfg.hidden_size
        wide = cfg.hc_count * cfg.hidden_size
        self.hc_norm = CenteredRMSNorm(wide, cfg.rms_norm_eps, group=cfg.hidden_size)
        self.input_mix_weight_down = nn.Linear(wide, cfg.hc_lowrank, bias=False)
        self.input_mix_weight_up = nn.Linear(cfg.hc_lowrank, wide, bias=False)
        if combine:
            self.block_inject_weight = nn.Linear(wide, cfg.hc_count, bias=False)

    def __call__(self, h: mx.array) -> Any:
        normed = self.hc_norm(h)
        mix = nn.silu(self.input_mix_weight_down(normed) / self.streams)
        mix = mx.sigmoid(self.input_mix_weight_up(mix))
        shape = (*h.shape[:-1], self.streams, self.dims)
        mixed = mx.mean(mix.reshape(shape) * normed.reshape(shape), axis=-2)
        if "block_inject_weight" not in self:
            return mixed
        inject = 2 * mx.sigmoid(self.block_inject_weight(normed) / self.streams)
        return mixed, inject


def _write_back(h: mx.array, branch: mx.array, inject: mx.array) -> mx.array:
    """Add the block's output to every residual stream, scaled by that stream's gate."""

    return h + (branch[..., None, :] * inject[..., None]).reshape(h.shape)


# -- caches ------------------------------------------------------------------
class LinearCache:
    """A DeltaNet layer: conv tail and recurrent state (plus the n-gram conv tail and token history on PLE layers)."""

    # rows of the last call on the PLE layer (history before it, its tokens, its conv input): not stored
    ple_rollback: Any = None
    transient = ("ple_rollback",)

    def __init__(self) -> None:
        self.conv: mx.array | None = None
        self.ssm: mx.array | None = None
        self.ple_conv: mx.array | None = None
        self.history: np.ndarray | None = None   # reassigned, never written in place (copies share it)
        self.offset = 0

    @property
    def state(self) -> list[mx.array]:
        return [a for a in (self.conv, self.ssm, self.ple_conv) if a is not None]


class AttentionCache:
    """A sparse-attention layer: keys, values, the indexer's raw keys, and its pooled block keys."""

    step = 256

    def __init__(self) -> None:
        self.keys: Any = None
        self.values: Any = None
        self.index_keys: Any = None
        self.pooled: Any = None     # normalized, rotated keys of blocks [0, pooled.shape[1])
        self.offset = 0

    def trim(self, n: int, ratio: int = 4) -> int:
        """Forget the last ``n`` positions (and pooled blocks no longer complete)."""

        n = min(self.offset, n)
        self.offset -= n
        if self.pooled is not None and self.pooled.shape[1] > self.offset // ratio:
            blocks = self.offset // ratio
            self.pooled = self.pooled[:, :blocks] if blocks else None
        return n

    @property
    def state(self) -> list[mx.array]:
        if self.keys is None:
            return []
        n = self.offset
        out = [self.keys[:, :, :n], self.values[:, :, :n], self.index_keys[:, :n]]
        return out + ([self.pooled] if self.pooled is not None else [])

    def update(self, keys: mx.array, values: mx.array, index_keys: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        prev, length = self.offset, keys.shape[2]
        end = prev + length
        if self.keys is None or end > self.keys.shape[2]:
            cap = ((end + self.step - 1) // self.step) * self.step
            batch, heads, _, dim = keys.shape

            def grow(old: mx.array | None, shape: tuple[int, ...], dtype: Any, axis: int) -> mx.array:
                fresh = mx.zeros(shape, dtype)
                if old is None or prev == 0:
                    return fresh
                kept = old[:, :, :prev] if axis == 2 else old[:, :prev]
                pad = list(shape)
                pad[axis] = cap - prev
                return mx.concatenate([kept, mx.zeros(tuple(pad), dtype)], axis=axis)

            self.keys = grow(self.keys, (batch, heads, cap, dim), keys.dtype, 2)
            self.values = grow(self.values, (batch, heads, cap, values.shape[3]), values.dtype, 2)
            self.index_keys = grow(self.index_keys, (batch, cap, index_keys.shape[2]), index_keys.dtype, 1)
        self.keys[:, :, prev:end] = keys
        self.values[:, :, prev:end] = values
        self.index_keys[:, prev:end] = index_keys
        self.offset = end
        return self.keys[:, :, :end], self.values[:, :, :end], self.index_keys[:, :end]


# -- Gated DeltaNet ------------------------------------------------------------
class GatedDeltaNet(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.nk, self.nv = cfg.linear_num_key_heads, cfg.linear_num_value_heads
        self.dk, self.dv = cfg.linear_key_head_dim, cfg.linear_value_head_dim
        self.key_dim, self.value_dim = self.nk * self.dk, self.nv * self.dv
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.kernel = cfg.linear_conv_kernel_dim
        d = cfg.hidden_size
        self.in_proj_qkv = nn.Linear(d, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(d, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(d, self.nv, bias=False)
        self.in_proj_a = nn.Linear(d, self.nv, bias=False)
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, kernel_size=self.kernel,
                                groups=self.conv_dim, bias=False)
        self.dt_bias = mx.ones((self.nv,))
        self.A_log = mx.zeros((self.nv,))
        self.norm = GatedRMSNorm(self.dv, cfg.rms_norm_eps, cfg.output_gate_type)
        self.out_proj = nn.Linear(self.value_dim, d, bias=False)

    def __call__(self, x: mx.array, cache: LinearCache) -> mx.array:
        batch, length, _ = x.shape
        qkv = self.in_proj_qkv(x)
        z = self.in_proj_z(x).reshape(batch, length, self.nv, self.dv)
        b = self.in_proj_b(x)
        a = self.in_proj_a(x)
        tail = cache.conv if cache.conv is not None else mx.zeros((batch, self.kernel - 1, self.conv_dim), x.dtype)
        conv_in = mx.concatenate([tail, qkv], axis=1)
        cache.conv = conv_in[:, -(self.kernel - 1):]
        conv = nn.silu(self.conv1d(conv_in))
        q, k, v = mx.split(conv, [self.key_dim, 2 * self.key_dim], axis=-1)
        q = q.reshape(batch, length, self.nk, self.dk)
        k = k.reshape(batch, length, self.nk, self.dk)
        v = v.reshape(batch, length, self.nv, self.dv)
        # L2 normalization with the epsilon inside the sum (as the reference model), then 1/sqrt(d) on queries
        q = q * mx.rsqrt(mx.sum(mx.square(q), axis=-1, keepdims=True) + 1e-6) * (self.dk ** -0.5)
        k = k * mx.rsqrt(mx.sum(mx.square(k), axis=-1, keepdims=True) + 1e-6)
        out, cache.ssm = gated_delta_update(q, k, v, a, b, self.A_log, self.dt_bias, cache.ssm)
        cache.offset += length
        out = self.norm(out, z)
        return self.out_proj(out.reshape(batch, length, -1))


# -- sparse attention ----------------------------------------------------------
class Indexer(nn.Module):
    """Scores 4-key blocks for each query; past 2,048 keys a query attends to its best 512 blocks and its tail."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.heads = cfg.indexer_n_heads
        self.dims = cfg.indexer_head_dim
        self.ratio = cfg.indexer_compress_ratio
        self.top_blocks = cfg.indexer_budget // cfg.indexer_compress_ratio
        self.rotary_dim = cfg.rotary_dim
        self.base = cfg.rope_theta
        self.index_qk_proj = nn.Linear(cfg.hidden_size, (self.heads + 1) * self.dims, bias=False)
        self.q_layernorm = CenteredRMSNorm(self.dims, cfg.rms_norm_eps)
        self.k_layernorm = CenteredRMSNorm(self.dims, cfg.rms_norm_eps)

    def project(self, x: mx.array) -> tuple[mx.array, mx.array]:
        batch, length, _ = x.shape
        qk = self.index_qk_proj(x).reshape(batch, length, self.heads + 1, self.dims)
        return qk[:, :, : self.heads], qk[:, :, self.heads]

    def _pool(self, raw: mx.array, start: int, stop: int) -> mx.array:
        """Mean of each block's raw keys, normalized, rotated to the block's first position."""

        batch = raw.shape[0]
        blocks = raw[:, start * self.ratio: stop * self.ratio].reshape(batch, stop - start, self.ratio, self.dims)
        pooled = mx.mean(blocks.astype(mx.float32), axis=-2).astype(raw.dtype)
        pooled = self.k_layernorm(pooled)
        return mx.fast.rope(pooled[:, None], self.rotary_dim, traditional=False, base=self.base,
                            scale=float(self.ratio), offset=start)[:, 0]

    def select(self, query: mx.array, raw: mx.array, cache: AttentionCache, past: int) -> mx.array | None:
        """Keys each query may read, [B, 1, L, keys] (bool), or None while the context is short (causal)."""

        batch, length = query.shape[0], query.shape[1]
        keys = past + length
        blocks = keys // self.ratio
        if blocks <= self.top_blocks:
            return None
        done = 0 if cache.pooled is None else cache.pooled.shape[1]
        if blocks > done:
            fresh = self._pool(raw, done, blocks)
            cache.pooled = fresh if cache.pooled is None else mx.concatenate([cache.pooled, fresh], axis=1)
        pooled = cache.pooled[:, :blocks]
        q = self.q_layernorm(query).transpose(0, 2, 1, 3)
        q = mx.fast.rope(q, self.rotary_dim, traditional=False, base=self.base, scale=1.0, offset=past)
        # float32 scores: which blocks win is a discrete choice and rounding flips the ones at the cut
        scores = q.astype(mx.float32) @ pooled.astype(mx.float32)[:, None].transpose(0, 1, 3, 2)
        scores = mx.sum(mx.maximum(scores, 0), axis=1) / math.sqrt(self.dims)          # [B, L, blocks]
        ends = past + mx.arange(length) + 1
        complete = ends // self.ratio
        valid = mx.arange(blocks)[None, None, :] < complete[None, :, None]
        scores = mx.where(valid, scores, -mx.inf)
        chosen = mx.argpartition(scores, kth=-self.top_blocks, axis=-1)[..., -self.top_blocks:]
        hits = mx.put_along_axis(mx.zeros((batch, length, blocks), dtype=mx.bool_), chosen,
                                 mx.array(True), axis=-1)
        picked = mx.repeat(hits, self.ratio, axis=-1)
        if blocks * self.ratio < keys:
            picked = mx.concatenate(
                [picked, mx.zeros((batch, length, keys - blocks * self.ratio), dtype=mx.bool_)], axis=-1)
        index = mx.arange(keys)
        tail = (index[None, None, :] >= (complete * self.ratio)[None, :, None]) & (index[None, None, :] < ends[None, :, None])
        causal = index[None, None, :] < ends[None, :, None]
        sparse = complete > self.top_blocks
        return mx.where(sparse[None, :, None], picked | tail, causal)[:, None]


class SparseAttention(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.heads, self.kv_heads, self.dims = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        self.scale = self.dims ** -0.5
        self.rotary_dim, self.base = cfg.rotary_dim, cfg.rope_theta
        d = cfg.hidden_size
        self.q_proj = nn.Linear(d, self.heads * self.dims * 2, bias=False)
        self.k_proj = nn.Linear(d, self.kv_heads * self.dims, bias=False)
        self.v_proj = nn.Linear(d, self.kv_heads * self.dims, bias=False)
        self.o_proj = nn.Linear(self.heads * self.dims, d, bias=False)
        self.q_norm = CenteredRMSNorm(self.dims, cfg.rms_norm_eps)
        self.k_norm = CenteredRMSNorm(self.dims, cfg.rms_norm_eps)
        self.indexer = Indexer(cfg)

    def __call__(self, x: mx.array, cache: AttentionCache) -> mx.array:
        batch, length, _ = x.shape
        past = cache.offset
        q = self.q_proj(x).reshape(batch, length, self.heads, 2 * self.dims)
        queries, gate = q[..., : self.dims], q[..., self.dims:]
        gate = gate.reshape(batch, length, self.heads * self.dims)
        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(self.k_proj(x).reshape(batch, length, self.kv_heads, self.dims)).transpose(0, 2, 1, 3)
        values = self.v_proj(x).reshape(batch, length, self.kv_heads, self.dims).transpose(0, 2, 1, 3)
        queries = mx.fast.rope(queries, self.rotary_dim, traditional=False, base=self.base, scale=1.0, offset=past)
        keys = mx.fast.rope(keys, self.rotary_dim, traditional=False, base=self.base, scale=1.0, offset=past)
        index_query, index_key = self.indexer.project(x)
        keys, values, raw = cache.update(keys, values, index_key)
        mask = self.indexer.select(index_query, raw, cache, past)
        if mask is None and length > 1:
            mask = "causal"
        out = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(out * mx.sigmoid(gate))


# -- MoE -----------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, dims: int, hidden: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dims, hidden, bias=False)
        self.up_proj = nn.Linear(dims, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dims, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class SparseMoE(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        d = cfg.hidden_size
        self.top_k = cfg.num_experts_per_tok
        self.gate = nn.Linear(d, cfg.num_experts, bias=False)          # stays bf16 in the checkpoint
        self.switch_mlp = SwitchGLU(d, cfg.moe_intermediate_size, cfg.num_experts)
        self.shared_expert = MLP(d, cfg.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(d, 1, bias=False)

    def route(self, x: mx.array) -> tuple[mx.array, mx.array]:
        probs = mx.softmax(self.gate(x), axis=-1, precise=True)
        experts = mx.argpartition(probs, kth=-self.top_k, axis=-1)[..., -self.top_k:]
        weights = mx.take_along_axis(probs, experts, axis=-1)
        return experts, weights / weights.sum(axis=-1, keepdims=True)

    def __call__(self, x: mx.array) -> mx.array:
        experts, weights = self.route(x)
        routed = (self.switch_mlp(x, experts) * weights[..., None]).sum(axis=-2)
        shared = self.shared_expert(x) * mx.sigmoid(self.shared_expert_gate(x))
        return routed + shared


# -- n-gram embedding (PLE) ------------------------------------------------------
def _splitmix64(value: int) -> int:
    value = (value + _GOLDEN) & _MASK64
    value = ((value ^ (value >> 30)) * _MIX1) & _MASK64
    value = ((value ^ (value >> 27)) * _MIX2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for divisor in range(3, math.isqrt(value) + 1, 2):
        if value % divisor == 0:
            return False
    return True


def _nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


def layer_multipliers(vocab: int, ngram: int, ple_index: int, seed: int) -> np.ndarray:
    half = max(1, (((1 << 63) - 1) // max(vocab, 1)) // 2)
    base = seed + _PRIME * ple_index
    return np.array([2 * (_splitmix64((base + _GOLDEN * (i + 1)) & _MASK64) % half) + 1 for i in range(ngram)],
                    dtype=np.int64)


class NGramEmbedding(nn.Module):
    """2- and 3-gram ids hashed into 16 heads of prime-sized tables, looked up in 128 row shards."""

    def __init__(self, cfg: Config, ple_index: int) -> None:
        super().__init__()
        self.n = cfg.ngram_size
        self.context = cfg.ngram_size - 1
        self.per_ngram = cfg.heads_per_ngram
        self.heads = self.context * self.per_ngram
        self.eos = cfg.ple_eos
        sizes, offsets, total = [], [], 0
        for head in range(self.heads):
            size = _nth_prime_after(cfg.ngram_vocab_size_base - 1, ple_index * self.heads + head + 1)
            sizes.append(size)
            offsets.append(total)
            total += size
        self.head_sizes = np.array(sizes, dtype=np.int64)
        self.head_offsets = np.array(offsets, dtype=np.int64)
        self.multipliers = layer_multipliers(cfg.vocab_size, cfg.ngram_size, ple_index, cfg.seed)
        rows = math.ceil(total / cfg.ngram_vocab_divisor) * cfg.ngram_vocab_divisor
        base, extra = divmod(rows, cfg.ngram_shards)
        shard_rows = [base + (1 if i < extra else 0) for i in range(cfg.ngram_shards)]
        self.shard_starts = [0]
        for count in shard_rows:
            self.shard_starts.append(self.shard_starts[-1] + count)
        self.dims = cfg.ple_embed_dim // self.heads
        self.shards = [nn.Embedding(count, self.dims) for count in shard_rows]

    def ids(self, history: np.ndarray, tokens: np.ndarray) -> np.ndarray:
        """Row ids [B, L, heads] for ``tokens`` [B, L] after ``history`` [B, n-1] (EOS resets the n-grams)."""

        seq = np.concatenate([history, tokens], axis=1).astype(np.int64)
        batch, width = seq.shape
        pos = np.arange(width)
        eos_at = np.where(seq == self.eos, pos[None], -1)
        before = np.concatenate([np.full((batch, 1), -1), np.maximum.accumulate(eos_at, axis=1)[:, :-1]], axis=1)
        in_segment = pos[None] - (before + 1)
        shifted = []
        for shift in range(self.n):
            source = pos - shift
            taken = np.take_along_axis(seq, np.broadcast_to(np.maximum(source, 0)[None], seq.shape), axis=1)
            shifted.append(np.where((in_segment >= shift) & (source[None] >= 0), taken, self.eos))
        blocks = []
        for ngram in range(2, self.n + 1):
            first = (ngram - 2) * self.per_ngram
            mixed = shifted[0] * self.multipliers[0]
            for p in range(1, ngram):
                mixed = np.bitwise_xor(mixed, shifted[p] * self.multipliers[p])
            sizes = self.head_sizes[first:first + self.per_ngram]
            blocks.append(mixed[..., None] % sizes + self.head_offsets[first:first + self.per_ngram])
        return np.concatenate(blocks, axis=-1)[:, -tokens.shape[1]:]

    def __call__(self, ids: np.ndarray) -> mx.array:
        flat = ids.reshape(-1)
        shard = np.searchsorted(np.asarray(self.shard_starts), flat, side="right") - 1
        parts, order = [], []
        for s in np.unique(shard):
            where = np.nonzero(shard == s)[0]
            local = mx.array((flat[where] - self.shard_starts[int(s)]).astype(np.int32))
            parts.append(self.shards[int(s)](local))
            order.append(where)
        rows = mx.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
        inverse = np.empty(len(flat), dtype=np.int32)
        inverse[np.concatenate(order)] = np.arange(len(flat), dtype=np.int32)
        rows = rows[mx.array(inverse)]
        return rows.reshape(*ids.shape[:-1], self.heads * self.dims)


class PLELayer(nn.Module):
    """Adds a gated n-gram embedding to every residual stream, then a dilated short conv over it."""

    def __init__(self, cfg: Config, ple_index: int) -> None:
        super().__init__()
        self.streams, self.dims = cfg.hc_count, cfg.hidden_size
        wide = cfg.hc_count * cfg.hidden_size
        self.ple_embedding = NGramEmbedding(cfg, ple_index)
        self.key_proj = nn.Linear(cfg.ple_embed_dim, wide, bias=False)
        self.value_proj = nn.Linear(cfg.ple_embed_dim, cfg.hidden_size, bias=False)
        self.norm_key = CenteredRMSNorm(wide, cfg.rms_norm_eps, group=cfg.hidden_size)
        self.norm_query = CenteredRMSNorm(wide, cfg.rms_norm_eps, group=cfg.hidden_size)
        self.norm_conv = CenteredRMSNorm(wide, cfg.rms_norm_eps, group=cfg.hidden_size)
        self.dilation = cfg.ngram_size
        self.tail = (cfg.ple_conv_kernel_size - 1) * self.dilation
        self.conv1d = nn.Conv1d(wide, wide, kernel_size=cfg.ple_conv_kernel_size, dilation=self.dilation,
                                groups=wide, bias=False)

    def __call__(self, h: mx.array, tokens: np.ndarray, cache: LinearCache) -> mx.array:
        batch, length, _ = h.shape
        history = cache.history
        if history is None:
            history = np.full((batch, self.ple_embedding.context), self.ple_embedding.eos, dtype=np.int64)
        ids = self.ple_embedding.ids(history, tokens)
        cache.history = np.concatenate([history, tokens.astype(np.int64)], axis=1)[:, -self.ple_embedding.context:]
        emb = self.ple_embedding(ids)
        shape = (batch, length, self.streams, self.dims)
        keys = self.norm_key(self.key_proj(emb)).reshape(shape)
        values = self.value_proj(emb)
        queries = self.norm_query(h).reshape(shape)
        gate = mx.sum(keys * queries, axis=-1, keepdims=True) / math.sqrt(self.dims)
        gate = mx.sign(gate) * mx.sqrt(mx.maximum(mx.abs(gate), 1e-6))
        gated = (mx.sigmoid(gate) * values[..., None, :]).reshape(h.shape)
        normed = self.norm_conv(gated)
        tail = cache.ple_conv if cache.ple_conv is not None else mx.zeros((batch, self.tail, h.shape[-1]), h.dtype)
        conv_in = mx.concatenate([tail, normed], axis=1)
        cache.ple_conv = conv_in[:, -self.tail:]
        # what keeping only the first rows of this call needs: the history before it, its tokens, the conv window
        cache.ple_rollback = (history, tokens.astype(np.int64), conv_in)
        return gated + nn.silu(self.conv1d(conv_in))


# -- model -----------------------------------------------------------------------
class DecoderLayer(nn.Module):
    def __init__(self, cfg: Config, index: int) -> None:
        super().__init__()
        self.is_linear = cfg.layer_types[index] == "linear_attention"
        if self.is_linear:
            self.linear_attn = GatedDeltaNet(cfg)
        else:
            self.self_attn = SparseAttention(cfg)
        self.mlp = SparseMoE(cfg)
        if index + 1 in cfg.ple_layer_ids:
            self.ple = PLELayer(cfg, cfg.ple_layer_ids.index(index + 1))
        self.attn_hyper_connection = HyperConnection(cfg)
        self.mlp_hyper_connection = HyperConnection(cfg)

    def __call__(self, h: mx.array, tokens: np.ndarray, cache: Any) -> mx.array:
        if "ple" in self:
            h = h + self.ple(h, tokens, cache)
        mixed, inject = self.attn_hyper_connection(h)
        branch = self.linear_attn(mixed, cache) if self.is_linear else self.self_attn(mixed, cache)
        h = _write_back(h, branch, inject)
        mixed, inject = self.mlp_hyper_connection(h)
        return _write_back(h, self.mlp(mixed), inject)


class Body(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = [DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers)]
        self.hyper_connection_mixer = HyperConnection(cfg, combine=False)


class Qwen4Exp(nn.Module):
    # layers per slice handed to the GPU while Python builds the next slice (same graph, same bits)
    pipeline_layers = 4
    # decode steps of up to this many rows go through ``decode.FusedDecode`` when it is attached
    fused_rows = 16

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.args = cfg
        self.model = Body(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    @property
    def layers(self) -> list[DecoderLayer]:
        return self.model.layers

    def make_cache(self) -> list[Any]:
        return [LinearCache() if layer.is_linear else AttentionCache() for layer in self.layers]

    def hidden(self, inputs: Any, cache: list[Any]) -> mx.array:
        """The mixed hidden state [B, L, D] after the last layer."""

        tokens = np.asarray(inputs, dtype=np.int64)
        if tokens.ndim == 1:
            tokens = tokens[None]
        fused = self.__dict__.get("fused")
        if fused is not None and tokens.shape[0] == 1 and tokens.shape[1] <= self.fused_rows:
            return fused(tokens, cache)
        h = self.model.embed_tokens(mx.array(tokens.astype(np.int32)))
        h = mx.tile(h, (1, 1, self.args.hc_count))
        for i, (layer, layer_cache) in enumerate(zip(self.layers, cache)):
            h = layer(h, tokens, layer_cache)
            if self.pipeline_layers and (i + 1) % self.pipeline_layers == 0:
                mx.async_eval(h)
        self.__dict__["last_streams"] = h[0]          # [L, S*D]: the residual streams before the final mixer
        return self.model.hyper_connection_mixer(h)

    def head(self, hidden: mx.array) -> mx.array:
        return self.lm_head(hidden)

    def __call__(self, inputs: Any, cache: list[Any]) -> mx.array:
        return self.head(self.hidden(inputs, cache))


# -- loading ---------------------------------------------------------------------
# hashing constants the checkpoint ships; each must equal what NGramEmbedding derives
_PLE_CONSTANTS = {
    "layer_multipliers": "multipliers",
    "ngram_heads_vocab_sizes": "head_sizes",
    "ngram_heads_offsets": "head_offsets",
}


def sanitize(weights: dict[str, mx.array]) -> tuple[dict[str, mx.array], dict[str, mx.array]]:
    """Checkpoint names -> this module's; the n-gram hashing constants come back separately (not weights)."""

    out: dict[str, mx.array] = {}
    extras: dict[str, mx.array] = {}
    for name, value in weights.items():
        if not name.startswith("language_model.") or ".mtp." in name or name.startswith("language_model.mtp"):
            continue
        key = name[len("language_model."):]
        if key.rsplit(".", 1)[-1] in _PLE_CONSTANTS:
            extras[key] = value
            continue
        key = key.replace("ngram_embedding.shard_", "shards.")
        out[key] = value
    return out, extras


def norms_stored_around_one(weights: dict[str, mx.array]) -> bool:
    """Whether the checkpoint's centred norms hold gamma (around 1) instead of gamma - 1 (around 0).

    Decided from the attention hyper-connection norms, which sit near their centre in every layer
    (the reference implementation's rule: 90% of their means above 0.5 and a median in [0.75, 1.5]).
    """

    anchors = [value for key, value in weights.items()
               if key.startswith("model.layers.") and key.endswith(".attn_hyper_connection.hc_norm.weight")]
    if len(anchors) < 8:
        return False
    means = np.array([float(mx.mean(a.astype(mx.float32)).item()) for a in anchors])
    around_one = (means > 0.5).mean() >= 0.9 and 0.75 <= float(np.median(means)) <= 1.5
    around_zero = (means > 0.5).mean() <= 0.1 and -0.5 <= float(np.median(means)) <= 0.25
    if not (around_one or around_zero):
        raise ValueError(f"cannot tell how the norm weights are stored (median mean {np.median(means):.3f})")
    return around_one


def load(model_dir: Path, *, lazy: bool = False) -> tuple[Qwen4Exp, Any]:
    from mlx_lm.utils import load_tokenizer

    config = json.loads((Path(model_dir) / "config.json").read_text())
    cfg = Config.from_dict(config)
    model = Qwen4Exp(cfg)
    weights: dict[str, mx.array] = {}
    for path in sorted(Path(model_dir).glob("model*.safetensors")):
        weights.update(mx.load(str(path)))
    weights, extras = sanitize(weights)

    def quantized(path: str, module: nn.Module) -> bool:
        return hasattr(module, "to_quantized") and f"{path}.scales" in weights

    nn.quantize(model, group_size=cfg.group_size, bits=cfg.bits, class_predicate=quantized)
    if norms_stored_around_one(weights):
        # early MLX conversions store these norms' gamma itself (around 1), not gamma - 1
        for path, module in model.named_modules():
            if isinstance(module, CenteredRMSNorm) and f"{path}.weight" in weights:
                weights[f"{path}.weight"] = weights[f"{path}.weight"].astype(mx.float32) - 1.0
    model.load_weights(list(weights.items()), strict=True)
    for key, value in extras.items():
        embedding = model.layers[int(key.split(".")[2])].ple.ple_embedding
        derived = getattr(embedding, _PLE_CONSTANTS[key.rsplit(".", 1)[-1]])
        shipped = np.array(value).astype(np.int64)
        if not np.array_equal(shipped, derived):
            raise ValueError(f"{key}: checkpoint {shipped} != derived {derived}")
    if not lazy:
        mx.eval(model.parameters())
        import os

        if os.environ.get("TF_FLASH_FUSED", "1") != "0":
            from tensorfold.families.qwen4_exp.decode import FusedDecode

            # kept out of the module tree (a plain attribute), so parameters() stays the checkpoint's
            model.__dict__["fused"] = FusedDecode(model)
    eos = config.get("eos_token_id")
    tokenizer = load_tokenizer(Path(model_dir), eos_token_ids=eos if isinstance(eos, list) else None)
    return model, tokenizer
