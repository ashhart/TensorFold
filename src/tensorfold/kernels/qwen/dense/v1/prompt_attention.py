"""Bound fallback prompt attention to two causal query parts in flight."""

from __future__ import annotations

import mlx.core as mx

ROWS = 128
FROM_KEYS = 4096


def attend(queries: mx.array, keys: mx.array, values: mx.array, scale: float) -> mx.array:
    """Causal attention of the last ``queries.shape[2]`` of ``keys.shape[2]`` positions."""

    rows, total = int(queries.shape[2]), int(keys.shape[2])
    if total <= FROM_KEYS or rows <= ROWS:
        return mx.fast.scaled_dot_product_attention(queries, keys, values, scale=scale, mask="causal")
    outs: list[mx.array] = []
    begin = 0
    while begin < rows:
        end = min(rows, begin + ROWS)
        # A short tail would select vector attention and change the full prompt's bits.
        if 0 < rows - end <= 16:
            end = rows
        visible = total - rows + end
        part = mx.fast.scaled_dot_product_attention(queries[:, :, begin:end], keys[:, :, :visible],
                                                    values[:, :, :visible], scale=scale, mask="causal")
        mx.async_eval(part)
        if outs:
            mx.eval(outs[-1])
        outs.append(part)
        begin = end
    return mx.concatenate(outs, axis=2)


__all__ = ["FROM_KEYS", "ROWS", "attend"]
