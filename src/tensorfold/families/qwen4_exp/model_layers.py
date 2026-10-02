"""Configuration, caches, attention, and feed-forward layers for Qwen3.8 Flash Next."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.gated_delta import gated_delta_update
from mlx_lm.models.switch_layers import SwitchGLU

from tensorfold.kernels.qwen.flash_next.v1 import prefill, prefill_mm


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
    """RMSNorm scales by float32 (1 + w), normalizing each residual stream separately when ``group`` is set."""

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


def _queue(value: mx.array, previous: mx.array | None) -> None:
    """Queue ``value`` then wait for ``previous`` to bound live temporaries allocated when MLX queues an op."""

    mx.async_eval(value)
    if previous is not None:
        mx.eval(previous)


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
        self.history: Any = None   # host int64, or uint32 on the GPU after a GPU window; never written in place
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
        qkv, z, b, a = prefill_mm.deltanet_in(self, x)
        tail = cache.conv if cache.conv is not None else mx.zeros((batch, self.kernel - 1, self.conv_dim), x.dtype)
        conv_in = mx.concatenate([tail, qkv], axis=1)
        cache.conv = mx.contiguous(conv_in[:, -(self.kernel - 1):])     # a copy: a view would keep the chunk
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
        return prefill_mm.linear(self.out_proj, out.reshape(batch, length, -1))


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
        qk = prefill_mm.linear(self.index_qk_proj, x).reshape(batch, length, self.heads + 1, self.dims)
        return qk[:, :, : self.heads], qk[:, :, self.heads]

    def _pool(self, raw: mx.array, start: int, stop: int) -> mx.array:
        """Mean of each block's raw keys, normalized, rotated to the block's first position."""

        batch = raw.shape[0]
        blocks = raw[:, start * self.ratio: stop * self.ratio].reshape(batch, stop - start, self.ratio, self.dims)
        pooled = mx.mean(blocks.astype(mx.float32), axis=-2).astype(raw.dtype)
        pooled = self.k_layernorm(pooled)
        return mx.fast.rope(pooled[:, None], self.rotary_dim, traditional=False, base=self.base,
                            scale=float(self.ratio), offset=start)[:, 0]

    def pool(self, raw: mx.array, cache: AttentionCache, blocks: int) -> mx.array:
        """Pooled keys of blocks [0, blocks), pooling the ones the cache does not hold yet."""

        done = 0 if cache.pooled is None else cache.pooled.shape[1]
        if blocks > done:
            fresh = self._pool(raw, done, blocks)
            cache.pooled = fresh if cache.pooled is None else mx.concatenate([cache.pooled, fresh], axis=1)
        return cache.pooled[:, :blocks]

    def rotated(self, query: mx.array, past: int) -> mx.array:
        """Normed, rotated queries [B, heads, L, dims]."""

        q = self.q_layernorm(query).transpose(0, 2, 1, 3)
        return mx.fast.rope(q, self.rotary_dim, traditional=False, base=self.base, scale=1.0, offset=past)

    def block_scores(self, query: mx.array, raw: mx.array, cache: AttentionCache, past: int) -> mx.array:
        """Every row's score of every complete block, [L, blocks] fp32 (batch 1), as ``select`` scores them."""

        blocks = (past + query.shape[1]) // self.ratio
        pooled = self.pool(raw, cache, blocks)[0].astype(mx.float32).T
        q = self.rotated(query, past)[0].astype(mx.float32)
        scores = mx.maximum(q[0] @ pooled, 0)
        for h in range(1, self.heads):
            scores = scores + mx.maximum(q[h] @ pooled, 0)
        return scores / math.sqrt(self.dims)

    def select(self, query: mx.array, raw: mx.array, cache: AttentionCache, past: int) -> mx.array | None:
        """Keys each query may read, [B, 1, L, keys] (bool), or None while the context is short (causal)."""

        batch, length = query.shape[0], query.shape[1]
        keys = past + length
        blocks = keys // self.ratio
        if blocks <= self.top_blocks:
            return None
        pooled = self.pool(raw, cache, blocks)
        q = self.rotated(query, past)
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
    split_keys = 8192
    split_rows = 256

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
        linear = prefill_mm.linear
        q = linear(self.q_proj, x).reshape(batch, length, self.heads, 2 * self.dims)
        queries, gate = q[..., : self.dims], q[..., self.dims:]
        gate = gate.reshape(batch, length, self.heads * self.dims)
        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(linear(self.k_proj, x).reshape(batch, length, self.kv_heads, self.dims)).transpose(0, 2, 1, 3)
        values = linear(self.v_proj, x).reshape(batch, length, self.kv_heads, self.dims).transpose(0, 2, 1, 3)
        queries = mx.fast.rope(queries, self.rotary_dim, traditional=False, base=self.base, scale=1.0, offset=past)
        keys = mx.fast.rope(keys, self.rotary_dim, traditional=False, base=self.base, scale=1.0, offset=past)
        index_query, index_key = self.indexer.project(x)
        keys, values, raw = cache.update(keys, values, index_key)
        if self.__dict__.get("kernel_select") and batch == 1 and prefill.through_kernels(self, past + length):
            out = prefill.selected(self, queries, index_query, raw, cache, past)
            return linear(self.o_proj, out * mx.sigmoid(gate))
        # Queue at most two query chunks, each using keys through its last row, to bound materialized scores.
        step = self.split_rows if past + length > self.split_keys else length
        outs = []
        for begin in range(0, length, step):
            end = min(length, begin + step)
            mask = self.indexer.select(index_query[:, begin:end], raw, cache, past + begin)
            if mask is None and end - begin > 1:
                mask = "causal"
            visible = past + end
            part = mx.fast.scaled_dot_product_attention(queries[:, :, begin:end], keys[:, :, :visible],
                                                        values[:, :, :visible], scale=self.scale, mask=mask)
            if step < length:
                _queue(part, outs[-1] if outs else None)
            outs.append(part)
        out = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=2)
        out = out.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return linear(self.o_proj, out * mx.sigmoid(gate))


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
        if "streamer" in self.__dict__:                                 # routed experts from the slot pool
            from tensorfold.families.qwen4_exp import stream

            return stream.moe_chunk(self, x)
        if prefill_mm.moe_applies(self, x):
            return prefill_mm.moe(self, x)
        experts, weights = self.route(x)
        routed = (self.switch_mlp(x, experts) * weights[..., None]).sum(axis=-2)
        shared = self.shared_expert(x) * mx.sigmoid(self.shared_expert_gate(x))
        return routed + shared
