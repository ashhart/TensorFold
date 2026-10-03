"""Decode attention over each row's own keys: Gemma's kernels with the simdgroup split fixed by the layer's shape."""

from __future__ import annotations

import mlx.core as mx

from tensorfold.kernels.gemma.v1 import attention as gemma
from tensorfold.kernels.gemma.v1.attention import Rows
from tensorfold.kernels.inputs import MIN_ELEMENTS

MAX_THREADS = 1024


def split_for(heads: int, kv_heads: int, dims: int) -> int:
    """Simdgroups a query head: Gemma's for the head size, halved until a threadgroup fits."""

    group = heads // kv_heads
    split = gemma.SHAPES.get(dims, gemma.OTHER)[1]
    while split > 1 and 32 * group * split > MAX_THREADS:
        split //= 2
    if 32 * group * split > MAX_THREADS:
        raise ValueError(f"attend: {group} query heads a key head do not fit a threadgroup")
    return split


def attend(q: mx.array, keys: mx.array, values: mx.array, rows: Rows, new_keys: mx.array, new_values: mx.array,
           scale: float = 1.0) -> mx.array:
    """q [R, H, D] over its rows' new keys [Hk, R, D] and the earlier ones in [1, Hk, CAP, D] buffers: [R, H, D]."""

    count, heads, dims = (int(s) for s in q.shape)
    kv_heads = int(keys.shape[1])
    if dims % 32 or heads % kv_heads or rows.count != count:
        raise ValueError(f"attend: head dim a multiple of 32, heads a multiple of key heads, {count} rows")
    chunk, _, block = gemma.SHAPES.get(dims, gemma.OTHER)
    split = split_for(heads, kv_heads, dims)
    group = heads // kv_heads
    consts = (("D", dims), ("G", group), ("HK", kv_heads), ("CK", chunk), ("S", split), ("BLK", block),
              ("SCALE", float(scale)))
    slots = heads * count * rows.chunks
    pm, pl, po = gemma._partial(consts, inputs=[q, keys, values, new_keys, new_values, rows.positions, rows.lows,
                                                rows.meta],
                                grid=(32 * group * split * rows.chunks, kv_heads, count),
                                threadgroup=(32 * group * split, 1, 1),
                                output_shapes=[(max(slots, MIN_ELEMENTS),), (max(slots, MIN_ELEMENTS),),
                                               (slots, dims)],
                                output_dtypes=[mx.float32, mx.float32, mx.float32])
    return gemma._merge((("D", dims), ("H", heads)), inputs=[pm, pl, po, rows.meta],
                        grid=(32, heads, count), threadgroup=(32, 1, 1),
                        output_shapes=[(count, heads, dims)], output_dtypes=[mx.bfloat16])[0]


__all__ = ["Rows", "attend", "split_for"]
