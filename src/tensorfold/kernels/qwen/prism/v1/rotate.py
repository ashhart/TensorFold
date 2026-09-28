"""Prism's rotated basis on Metal: signs, then normalized 1024-point Walsh-Hadamard blocks, each row on its own."""

from __future__ import annotations

import hashlib
from typing import Any

import mlx.core as mx

BLOCK = 1024              # hadamard.json's block_size; each threadgroup owns one block of one row
THREADS = 512             # two elements a thread, one butterfly pair a thread per stage

# Sylvester order, stage h pairs (i, i + h): a row's bits depend on its own values only, whatever the row count.
_BUTTERFLY = r"""
  for (uint h = 1; h < 1024; h <<= 1) {
    const uint i = (t / h) * (2 * h) + (t % h);
    const float a = buf[i], b = buf[i + h];
    buf[i] = a + b;
    buf[i + h] = a - b;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
"""

_HEAD = r"""
  const uint t = thread_position_in_threadgroup.x;
  const uint blk = threadgroup_position_in_grid.x;
  const uint r = threadgroup_position_in_grid.y;
  threadgroup float buf[1024];
"""

# x * signs, then the transform, then the 1/32 normalization (exact) and one rounding to bf16
_ROTATE = _HEAD + r"""
  const size_t base = size_t(r) * K + blk * 1024;
  for (uint e = t; e < 1024; e += 512) buf[e] = float(X[base + e]) * SG[blk * 1024 + e];
  threadgroup_barrier(mem_flags::mem_threadgroup);
""" + _BUTTERFLY + r"""
  for (uint e = t; e < 1024; e += 512) OUT[base + e] = bfloat(buf[e] * 0.03125f);
"""

# the inverse for the embedding: dequantize the 2-bit row (exact in fp32), transform, normalize, then the signs
_EMBED = _HEAD + r"""
  const size_t row = size_t(IDS[r]);
  for (uint e = t; e < 1024; e += 512) {
    const uint k = blk * 1024 + e;
    const uint q = (W[row * (K / 16) + k / 16] >> (2 * (k % 16))) & 3u;
    const size_t g = row * (K / G) + k / G;
    buf[e] = float(SC[g]) * float(q) + float(BI[g]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
""" + _BUTTERFLY + r"""
  const size_t base = size_t(r) * K + blk * 1024;
  for (uint e = t; e < 1024; e += 512) OUT[base + e] = bfloat((buf[e] * 0.03125f) * SG[blk * 1024 + e]);
"""

# one simdgroup per (row, output): lane-strided fma over K in order, then simd_sum; no dependence on the row count
_DENSE = r"""
  const uint lane = thread_index_in_simdgroup;
  const uint n = threadgroup_position_in_grid.x * 8 + simdgroup_index_in_threadgroup;
  const uint r = threadgroup_position_in_grid.y;
  if (n >= uint(N)) return;
  const size_t xb = size_t(r) * K, wb = size_t(n) * K;
  float acc = 0.0f;
  for (int k = int(lane); k < K; k += 32) acc = fma(float(X[xb + k]), WT[wb + k], acc);
  acc = simd_sum(acc);
  if (lane == 0) OUT[size_t(r) * N + n] = bfloat(acc);
"""

_kernels: dict[tuple[str, tuple[tuple[str, int], ...]], Any] = {}


def _kernel(kind: str, consts: tuple[tuple[str, int], ...]) -> Any:
    """One compiled kernel per kind and shape constants, the constants written into the source."""

    key = (kind, consts)
    if key not in _kernels:
        body, inputs = {"rotate": (_ROTATE, ["X", "SG"]), "embed": (_EMBED, ["IDS", "W", "SC", "BI", "SG"]),
                        "dense": (_DENSE, ["X", "WT"])}[kind]
        source = "".join(f"  constexpr int {k} = {v};\n" for k, v in consts) + body
        name = f"prism_{kind}_{hashlib.sha256(source.encode()).hexdigest()[:16]}"
        _kernels[key] = mx.fast.metal_kernel(name=name, input_names=inputs, output_names=["OUT"], source=source)
    return _kernels[key]


def rotate_rows(x: mx.array, signs: mx.array) -> mx.array:
    """x [..., K] in the rotated basis as bf16: bf16(H(x * signs) / 32) over each 1024-block."""

    k = int(x.shape[-1])
    if k % BLOCK:
        raise ValueError(f"the rotated basis takes widths in whole {BLOCK}-blocks, not {k}")
    rows = x.size // k
    out = _kernel("rotate", (("K", k),))(inputs=[x.reshape(rows, k), signs], grid=(THREADS * (k // BLOCK), rows, 1),
                                         threadgroup=(THREADS, 1, 1), output_shapes=[(rows, k)],
                                         output_dtypes=[mx.bfloat16])[0]
    return out.reshape(*x.shape)


def embed_rows(ids: mx.array, weight: mx.array, scales: mx.array, biases: mx.array, signs: mx.array,
               group: int) -> mx.array:
    """Rows ``ids`` of a 2-bit embedding back in the model's basis as bf16 [..., K]: signs * H(dequantized) / 32."""

    k = int(weight.shape[-1]) * 16
    flat = ids.reshape(-1).astype(mx.uint32)
    rows = int(flat.size)
    out = _kernel("embed", (("K", k), ("G", int(group))))(
        inputs=[flat, weight, scales, biases, signs], grid=(THREADS * (k // BLOCK), rows, 1),
        threadgroup=(THREADS, 1, 1), output_shapes=[(rows, k)], output_dtypes=[mx.bfloat16])[0]
    return out.reshape(*ids.shape, k)


def dense_rows(x: mx.array, weight: mx.array) -> mx.array:
    """x [..., K] @ weight.T for an fp32 weight [N, K] as bf16 [..., N], each row's sums in a fixed order."""

    n, k = (int(d) for d in weight.shape)
    rows = x.size // k
    out = _kernel("dense", (("K", k), ("N", n)))(
        inputs=[x.reshape(rows, k), weight], grid=(256 * -(-n // 8), rows, 1), threadgroup=(256, 1, 1),
        output_shapes=[(rows, n)], output_dtypes=[mx.bfloat16])[0]
    return out.reshape(*x.shape[:-1], n)


def sources() -> dict[str, str]:
    """Every kernel body, for the snapshot fingerprint."""

    return {"rotate": _ROTATE, "embed": _EMBED, "dense": _DENSE}


__all__ = ["BLOCK", "dense_rows", "embed_rows", "rotate_rows", "sources"]
