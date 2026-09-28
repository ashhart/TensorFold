"""Sparse attention for prompt chunks: the decode's block selection, then grouped-query attention."""

from __future__ import annotations

import math
from functools import partial
from typing import Any

import mlx.core as mx

from tensorfold.kernels.qwen.flash_next.v1 import attention, base
from tensorfold.kernels.qwen.flash_next.v1.base import consts, ints

DENSE_KEYS = 4096                             # up to this many keys MLX's dense attention is cheaper for a chunk
PARTS = 4                                     # parts a row's key list is cut into (a chunk's rows fill the GPU)
TK = 16                                       # keys a tile

_ATTN_GQA_PARTS = r"""
  // Threadgroup (kvh, r, part): KV head kvh's G query heads, HS a simdgroup (each key and value read from threadgroup
  // memory serves HS heads), over one part of row r's key list. Each head's sums run as with one head a simdgroup.
  constexpr int G = H / KVH;
  constexpr int PER = D / 32;                 // 8 output dims a lane
  constexpr int HALF = D / 2;
  constexpr int KP = D + 8;                   // padded key rows, 16-byte aligned
  constexpr int VEC = D / 8;                  // 16-byte vectors a row
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const int t = int(thread_position_in_threadgroup.x);
  const int nt = int(threads_per_threadgroup.x);
  const int kvh = int(threadgroup_position_in_grid.x);
  const int r = int(threadgroup_position_in_grid.y);
  const int part = int(threadgroup_position_in_grid.z);
  const int n = NK[r];
  const int lo = int((long(part) * n) / P), hi = int((long(part + 1) * n) / P);
  const bool sparse = SPARSE[r] != 0;
  const size_t cap = size_t(Kc_shape[2]);
  const device uint4* kbase = (const device uint4*)(Kc + size_t(kvh) * cap * D);
  const device uint4* vbase = (const device uint4*)(Vc + size_t(kvh) * cap * D);
  const auto ids = IDS + size_t(r) * IDS_shape[1];
  threadgroup float4 qs[G][D / 4];
  threadgroup uint4 ks[TK][KP / 8];
  threadgroup uint4 vs[TK][VEC];
  for (int j = 0; j < HS; j++) {
    const int g = int(sg) * HS + j;
    const device bfloat* qp = Q + (size_t(r) * H + kvh * G + g) * D;
    for (int i = int(lane); i < D / 4; i += 32) {
      const float s0 = SCALE[0];
      qs[g][i] = float4(s0 * float(qp[4 * i]), s0 * float(qp[4 * i + 1]), s0 * float(qp[4 * i + 2]), s0 * float(qp[4 * i + 3]));
    }
  }
  float o[HS][PER];
  float m[HS], l[HS];
  for (int j = 0; j < HS; j++) {
    for (int i = 0; i < PER; i++) o[j][i] = 0.0f;
    m[j] = -INFINITY; l[j] = 0.0f;
  }
  const int kt = int(lane) / 2, hf = int(lane) & 1;   // lanes 2k and 2k + 1 score the tile's key k, half each
  for (int base = lo; base < hi; base += TK) {
    const int cnt = metal::min(TK, hi - base);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int e = t; e < cnt * VEC; e += nt) {
      const int k = e / VEC, c = e - k * VEC;
      const int jj = base + k;
      const size_t row = size_t(sparse ? ids[jj] : jj) * VEC;
      ks[k][c] = kbase[row + c];
      vs[k][c] = vbase[row + c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int kk = metal::min(kt, cnt - 1);
    const threadgroup bfloat4* kr = (const threadgroup bfloat4*)(&ks[kk][0]) + hf * (HALF / 4);
    float a[HS];
    for (int j = 0; j < HS; j++) a[j] = 0.0f;
    for (int i = 0; i < HALF / 4; i++) {
      const float4 kv = float4(kr[i]);
      for (int j = 0; j < HS; j++) {
        const float4 qv = qs[int(sg) * HS + j][hf * (HALF / 4) + i];
        a[j] = fma(qv.x, kv.x, a[j]); a[j] = fma(qv.y, kv.y, a[j]);
        a[j] = fma(qv.z, kv.z, a[j]); a[j] = fma(qv.w, kv.w, a[j]);
      }
    }
    const bool live = kt < cnt;
    float e[HS];
    for (int j = 0; j < HS; j++) {
      const float aj = a[j] + simd_shuffle_xor(a[j], ushort(1));
      const float sc = live ? aj : -INFINITY;
      const float mn = metal::max(m[j], simd_max(sc));
      const float f = metal::exp(m[j] - mn);
      e[j] = (live && hf == 0) ? metal::exp(sc - mn) : 0.0f;
      l[j] = fma(l[j], f, simd_sum(e[j]));
      for (int i = 0; i < PER; i++) o[j][i] *= f;
      m[j] = mn;
    }
    for (int k = 0; k < cnt; k++) {
      const threadgroup bfloat4* vr = (const threadgroup bfloat4*)(&vs[k][lane]);
      const float4 v0 = float4(vr[0]), v1 = float4(vr[1]);
      for (int j = 0; j < HS; j++) {
        const float ek = simd_shuffle(e[j], ushort(2 * k));
        o[j][0] = fma(ek, v0.x, o[j][0]); o[j][1] = fma(ek, v0.y, o[j][1]);
        o[j][2] = fma(ek, v0.z, o[j][2]); o[j][3] = fma(ek, v0.w, o[j][3]);
        o[j][4] = fma(ek, v1.x, o[j][4]); o[j][5] = fma(ek, v1.y, o[j][5]);
        o[j][6] = fma(ek, v1.z, o[j][6]); o[j][7] = fma(ek, v1.w, o[j][7]);
      }
    }
  }
  for (int j = 0; j < HS; j++) {
    const int h = kvh * G + int(sg) * HS + j;
    const size_t at = (size_t(r) * H + h) * P + part;
    for (int i = 0; i < PER; i++) PO[at * D + int(lane) * PER + i] = o[j][i];
    if (lane == 0) { PM[at * 2] = m[j]; PM[at * 2 + 1] = l[j]; }
  }
"""


@partial(mx.compile, shapeless=True)
def _relu_sum(root: mx.array, *heads: mx.array) -> mx.array:
    """Indexer.block_scores' head sum in one kernel: ReLUs added in head order, then divided by sqrt(dims)."""

    total = mx.maximum(heads[0], 0)
    for s in heads[1:]:
        total = total + mx.maximum(s, 0)
    return total / root


def block_scores(ix: Any, query: mx.array, raw: mx.array, cache: Any, past: int) -> mx.array:
    """Indexer.block_scores with the heads in one batched fp32 matmul and the ReLU sum in one kernel, [L, blocks]."""

    if int(query.shape[1]) == 1:        # one row is a vector product, which the batched matmul sums in another order
        return ix.block_scores(query, raw, cache, past)
    blocks = (past + int(query.shape[1])) // ix.ratio
    pooled = ix.pool(raw, cache, blocks)[0].astype(mx.float32).T
    q = ix.rotated(query, past)[0].astype(mx.float32)
    root = consts.get(("root", ix.dims))
    if root is None:
        root = consts[("root", ix.dims)] = mx.array(math.sqrt(ix.dims), dtype=mx.float32)
    scores = q @ pooled                                                 # [heads, L, blocks]
    return _relu_sum(root, *(scores[h] for h in range(scores.shape[0])))


def heads_a_simdgroup() -> tuple[int, ...]:
    """Query heads a simdgroup scores: two on M1-M4 (7% faster, same bits), one on tensor-unit GPUs (not measured)."""

    from tensorfold.kernels.qwen.flash_next.v1 import prefill_mm

    return (1,) if prefill_mm._tensor_units() else (2, 1)


def gqa_supported(heads: int, kv_heads: int, dims: int) -> bool:
    """Its layouts: 256-dim heads, a KV head's query heads in one threadgroup (at most 32)."""

    return dims == 256 and kv_heads > 0 and heads % kv_heads == 0 and heads // kv_heads <= 32


def attention_rows_gqa(q: mx.array, keys: mx.array, values: mx.array, counts: list[int], ids: mx.array | None,
                       sparse: list[bool], scale: float, *, parts: int = 4) -> mx.array:
    """attention.attention_rows' result for many rows, [R, H, D] bf16; its sums run in another order."""

    rows, heads, dims = q.shape
    kv_heads = int(keys.shape[1])
    group = heads // kv_heads
    if not gqa_supported(heads, kv_heads, dims):
        raise ValueError("attention_rows_gqa: unsupported head layout")
    if ids is None:
        ids = mx.zeros((max(rows, 8), 1), dtype=mx.int32)
    scale_arr = consts.get(("scale", scale))
    if scale_arr is None:
        scale_arr = consts[("scale", scale)] = mx.array([scale], dtype=mx.float32)
    per = next(h for h in heads_a_simdgroup() if group % h == 0)
    first = base.kernel("q4_attn_gqa_parts", _ATTN_GQA_PARTS, ["Q", "Kc", "Vc", "IDS", "NK", "SPARSE", "SCALE"],
                        ["PO", "PM"])
    po, pm = first(inputs=[q, keys, values, ids, ints(counts), ints([int(bool(x)) for x in sparse]), scale_arr],
                   template=[("H", heads), ("KVH", kv_heads), ("D", dims), ("P", parts), ("TK", TK), ("HS", per)],
                   grid=(32 * (group // per) * kv_heads, rows, parts), threadgroup=(32 * (group // per), 1, 1),
                   output_shapes=[(rows, heads, parts, dims), (rows, heads, parts, 2)],
                   output_dtypes=[mx.float32, mx.float32])
    merge = base.kernel("q4_attn_merge", attention._ATTN_MERGE, ["PO", "PM"], ["OUT"])
    return merge(inputs=[po, pm], template=[("H", heads), ("D", dims), ("P", parts)],
                 grid=(dims, heads, rows), threadgroup=(dims, 1, 1),
                 output_shapes=[(rows, heads, dims)], output_dtypes=[mx.bfloat16])[0]


def through_kernels(attn: Any, keys: int) -> bool:
    """Whether a prompt chunk ending at ``keys`` attends here: a choice by its place alone, so resumes match."""

    return keys > DENSE_KEYS and keys // attn.indexer.ratio > attn.indexer.top_blocks


def selected(attn: Any, queries: mx.array, index_query: mx.array, raw: mx.array, cache: Any, past: int) -> mx.array:
    """A chunk's attention [1, L, H * D]: rows past the budget read their selected blocks and tail."""

    ix = attn.indexer
    length = queries.shape[2]
    ends = list(range(past + 1, past + length + 1))
    complete = [e // ix.ratio for e in ends]
    ids = attention.select_blocks(block_scores(ix, index_query, raw, cache, past), complete, ends, top=ix.top_blocks)
    sparse = [c > ix.top_blocks for c in complete]
    counts = [ix.ratio * (ix.top_blocks - c) + e if s else e for e, c, s in zip(ends, complete, sparse)]
    attend = attention_rows_gqa if gqa_supported(attn.heads, attn.kv_heads, attn.dims) else attention.attention_rows
    out = attend(queries[0].transpose(1, 0, 2), cache.keys, cache.values, counts, ids, sparse, attn.scale, parts=PARTS)
    return out.reshape(1, length, -1)
