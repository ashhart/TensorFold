"""Attention over a verify window computed exactly as serial decoding computes it.

MLX runs a single query through its vector attention kernel and several
queries through a different (full) kernel whose summation order differs, so
the attention output of window row t differs in its last bits from the serial
step at the same position, and a near-tie can flip the next token: the lane
engine's drafted output diverged from its own serial decode at characters
591, 1,439 and 1,664 on three prompts even with 9-row windows (2026-09-23).

Here each query of a short window goes through the one-query kernel over
exactly the keys the serial step at its position would see (the cache
through that position), so window row t is bit-identical to that step.
Consecutive queries share one causal call where MLX provably runs the same
kernel with the same key partition as their one-query calls (up to 5 queries
for 24 query heads over 4 key heads): at 20k keys an 8-row window's attention
drops from 1.70 to 1.14 ms a layer.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

EXACT_MAX_QUERIES = 16
_STOCK: Any = None

# MLX 0.31.2 runs its vector kernel for up to 8 queries while queries x (query heads per
# key head) <= 32, and within it a key's slot depends only on its index: a masked tail
# key is skipped. So a causal call over a few consecutive queries gives each query the
# bits of its own one-query call, unless the two calls' key counts fall on different
# sides of a point where MLX changes variant (one pass from 1024 keys on the M5 Max,
# two-pass blocks at 1025, 8193, 32769, 65537; 4096/16384/65536 on other chips).
# Checked bit for bit against one-query calls at 23 key counts around those points.
_REGIME_POINTS = (1024, 1025, 4096, 8193, 16384, 32769, 65536, 65537)


def _one(queries: mx.array, keys: mx.array, values: mx.array, scale: float, mask: Any, t: int, T: int,
         L: int) -> mx.array:
    n = L - T + t + 1
    m = mask[..., t:t + 1, :n] if isinstance(mask, mx.array) else None
    return mx.fast.scaled_dot_product_attention(
        queries[:, :, t:t + 1], keys[:, :, :n], values[:, :, :n], scale=scale, mask=m)


def exact_sdpa(queries: mx.array, keys: mx.array, values: mx.array, cache: Any, scale: float,
               mask: Any, sinks: Any = None) -> mx.array:
    T = int(queries.shape[2])
    if T < 2 or T > EXACT_MAX_QUERIES or sinks is not None or hasattr(cache, "bits"):
        return _STOCK(queries, keys, values, cache, scale, mask, sinks)
    L = int(keys.shape[2])
    heads, kv_heads = int(queries.shape[1]), int(keys.shape[1])
    group = 1
    if not isinstance(mask, mx.array) and heads % kv_heads == 0:
        group = max(1, min(8, 32 // (heads // kv_heads)))
    outs = []
    t = 0
    while t < T:
        g = min(group, T - t)
        first, last = L - T + t + 1, L - T + t + g
        if g > 1 and not any(first < p <= last for p in _REGIME_POINTS):
            outs.append(mx.fast.scaled_dot_product_attention(
                queries[:, :, t:t + g], keys[:, :, :last], values[:, :, :last], scale=scale, mask="causal"))
        else:
            outs.extend(_one(queries, keys, values, scale, mask, j, T, L) for j in range(t, t + g))
        t += g
    return mx.concatenate(outs, axis=2)


def install() -> None:
    """Route the Qwen3-Next / Qwen3.5 full-attention call through ``exact_sdpa``. Idempotent."""

    global _STOCK
    import mlx_lm.models.qwen3_next as qn

    if _STOCK is None:
        _STOCK = qn.scaled_dot_product_attention
    qn.scaled_dot_product_attention = exact_sdpa


__all__ = ["EXACT_MAX_QUERIES", "exact_sdpa", "install"]
