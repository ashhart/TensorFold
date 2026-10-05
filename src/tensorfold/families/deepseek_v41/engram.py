"""DeepSeek-V4.1's engram conditional memory: the n-gram hash (primes, rng multipliers), the signed-sqrt gate."""

from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
import numpy as np

from tensorfold.families.deepseek_v41.config import Config
from tensorfold.families.glm5_next.linear import Q


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    return all(n % d for d in range(3, math.isqrt(n) + 1, 2))


def find_next_prime(start: int, seen: set[int]) -> int:
    """The smallest prime above ``start`` that has not been handed out yet."""
    candidate = start + 1
    while not is_prime(candidate) or candidate in seen:
        candidate += 1
    return candidate


def engram_layout(cfg: Config) -> dict[int, tuple[list[int], list[int], np.ndarray]]:
    """Per engram layer: (flat primes [24], offsets = prime prefix sums, multipliers [order] odd int64).

    The primes are the smallest unused above ``engram_vocab_size - 1``, drawn per (layer, n-gram, head)
    in order; the multipliers come from ``default_rng(10007 * layer)`` bounded by
    ``((2^63 - 1) // compressed_vocab) // 2`` -- both exactly the official ``engram.py``'s derivation.
    """
    layout: dict[int, tuple[list[int], list[int], np.ndarray]] = {}
    seen: set[int] = set()
    vocab = cfg.engram_compressed_vocab_size or cfg.engram_vocab_size
    for layer, expected in zip(cfg.engram_layer_ids, cfg.engram_num_embeddings):
        primes: list[int] = []
        for _ in range(cfg.engram_ngram - 1):
            current = cfg.engram_vocab_size - 1
            for _ in range(cfg.engram_n_heads):
                current = find_next_prime(current, seen)
                seen.add(current)
                primes.append(current)
        offsets = np.cumsum([0, *primes[:-1]]).tolist()
        rng = np.random.default_rng(10007 * layer)
        bound = max(1, ((2 ** 63 - 1) // vocab) // 2)
        mult = rng.integers(0, bound, size=cfg.engram_ngram, dtype=np.int64) * 2 + 1
        if expected and sum(primes) != expected:
            raise ValueError(f"deepseek_v41: layer {layer}'s engram primes sum to {sum(primes)}, the checkpoint "
                             f"holds {expected}: the compressed-vocab map or engram_vocab_size does not match")
        layout[layer] = (primes, offsets, mult)
    return layout


class EngramHash:
    """Maps each position to 24 hash ids per engram layer from the last 4 tokens' compressed ids."""

    def __init__(self, cfg: Config, token_map: list[int], pad: int) -> None:
        self.cfg = cfg
        self.token_map = token_map
        self.pad = int(pad)
        self.layout = engram_layout(cfg)
        self.order = cfg.engram_ngram
        self.heads = cfg.engram_n_heads

    def push(self, token: int) -> int:
        return self.token_map[token]

    def ids(self, compressed: list[int], layer: int) -> list[int]:
        """The layer's 24 row ids for a window of ``order`` compressed ids (newest first; pad beyond history)."""
        primes, offsets, mult = self.layout[layer]
        tokens = [compressed[-1 - i] if i < len(compressed) else self.pad for i in range(self.order)]
        rolling = int(tokens[0]) * int(mult[0])
        ids: list[int] = []
        for i in range(1, self.order):
            rolling ^= int(tokens[i]) * int(mult[i])
            for head in range(self.heads):
                j = (i - 1) * self.heads + head
                ids.append(rolling % primes[j] + int(offsets[j]))
        return ids


class EngramHistory:
    """The compressed ids of the tokens a stream has committed, trimmed to the accepted prefix on rollback."""

    def __init__(self) -> None:
        self.ids: list[int] = []

    def push(self, compressed: list[int]) -> None:
        self.ids.extend(int(t) for t in compressed)

    def trim(self, count: int) -> None:
        if count:
            self.ids = self.ids[:-count] if count < len(self.ids) else []

    def trim_to(self, length: int) -> None:
        self.ids = self.ids[:int(length)]

    def window(self) -> list[int]:
        return self.ids[-4:]


class Engram:
    """One engram module: 24 gathered table rows a token, a low-rank KV projection, the signed-sqrt gate."""

    def __init__(self, embed: Q, wkv: Q, q_weight: mx.array, k_weight: mx.array, cfg: Config, layer: int) -> None:
        self.layer = layer
        self.embed = embed
        self.wkv = wkv
        self.weight = (q_weight.astype(mx.float32) * k_weight.astype(mx.float32))     # only ever used as a product
        self.eps = cfg.rms_norm_eps
        self.dim = int(cfg.hidden_size)
        self.streams = int(cfg.hc_mult)
        self.head_dim = int(cfg.engram_head_dim)

    def values(self, ids: list[int]) -> mx.array:
        """The 24 rows [n, head_dim] dequantized from the table (a gathered per-row affine read, order kept)."""
        e = self.embed
        rows = mx.array(sorted(set(ids)), dtype=mx.uint32)
        got = mx.dequantize(e.weight[rows], e.scales[rows], e.biases[rows], group_size=e.group, bits=e.bits)
        where = {int(r): i for i, r in enumerate(rows.tolist())}
        return got[mx.array([where[i] for i in ids])]

    def forward(self, h: mx.array, ids: list[list[int]], rows_exact: bool = False) -> mx.array:
        """h [R, 4, D], ids one 24-row list per row: per-stream gated read, fp32 inside, out in h's dtype.

        ``rows_exact``: a decode window's rows take the wkv projection one row a call (a batched
        2-D quantized matmul changes its bits with the row count on the GPU; one-row calls do not).
        """
        from tensorfold.families.deepseek_v41.dense import dense

        rows = len(ids)
        flat = [i for row in ids for i in row]
        values = self.values(flat).reshape(rows, len(flat) // rows, self.head_dim).reshape(rows, -1)
        kv = dense(values, self.wkv, rows_exact).astype(mx.float32)      # [R, (4 + 1) * D]
        key = kv[:, :self.streams * self.dim].reshape(rows, self.streams, self.dim)
        value = kv[:, self.streams * self.dim:]
        x = h.astype(mx.float32)
        rstd = (mx.rsqrt(mx.mean(x * x, axis=-1) + self.eps)
                * mx.rsqrt(mx.mean(key * key, axis=-1) + self.eps))       # [R, 4]
        dot = mx.sum(x * key * self.weight[None], axis=-1) * rstd * (self.dim ** -0.5)
        gate = mx.sigmoid(mx.where(dot < 0, -1.0, 1.0) * mx.sqrt(mx.maximum(mx.abs(dot), 1e-6)))
        return (x + gate[..., None] * value[:, None, :]).astype(h.dtype)
