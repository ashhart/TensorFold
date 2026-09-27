"""Match serial attention bits by grouping queries only when their one-query calls share MLX's kernel variant and key partition."""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from tensorfold.kernels.qwen.dense.v1 import prompt_attention

EXACT_MAX_QUERIES = 16
# Group queries with identical one-query arithmetic; disable when MLX dispatch differs.
GROUP_QUERIES = True
_STOCK: Any = None

# MLX 0.31.2 skips masked tail keys without changing key slots; grouping is exact only within one kernel regime.
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
    if T > EXACT_MAX_QUERIES and int(queries.shape[-1]) == 256 \
            and isinstance(mask, str) and mask == "causal" and sinks is None \
            and not hasattr(cache, "bits"):
        return prompt_attention.attend(queries, keys, values, scale)      # prompt chunks: bounded memory
    if T < 2 or T > EXACT_MAX_QUERIES or sinks is not None or hasattr(cache, "bits"):
        return _STOCK(queries, keys, values, cache, scale, mask, sinks)
    L = int(keys.shape[2])
    heads, kv_heads = int(queries.shape[1]), int(keys.shape[1])
    group = 1
    if GROUP_QUERIES and not isinstance(mask, mx.array) and heads % kv_heads == 0:
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
