"""Sparse attention for prompt chunks: the decode's block selection, then grouped-query attention."""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from tensorfold.kernels.qwen.flash_next.v1 import attention, base
from tensorfold.kernels.qwen.flash_next.v1.base import consts, ints

DENSE_KEYS = 4096                             # up to this many keys MLX's dense attention is cheaper for a chunk
PARTS = 4                                     # parts a row's key list is cut into (a chunk's rows fill the GPU)
TK = 16                                       # keys a tile

_ATTN_GQA_PARTS = r"""
  // Threadgroup (kvh, r, part): KV head kvh's G query heads (a simdgroup each) over one part of row r's key list.
  constexpr int G = H / KVH;
  constexpr int PER = D / 32;                 // 8 output dims a lane
  constexpr int HALF = D / 2;
  constexpr int KP = D + 8;                   // padded key rows, 16-byte aligned
  constexpr int VEC = D / 8;                  // 16-byte vectors a row
  const uint lane = thread_index_in_simdgroup;
  const uint g = simdgroup_index_in_threadgroup;
  const int t = int(thread_position_in_threadgroup.x);
  const int nt = int(threads_per_threadgroup.x);
  const int kvh = int(threadgroup_position_in_grid.x);
  const int r = int(threadgroup_position_in_grid.y);
  const int part = int(threadgroup_position_in_grid.z);
  const int h = kvh * G + int(g);
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
  const device bfloat* qp = Q + (size_t(r) * H + h) * D;
  for (int i = int(lane); i < D / 4; i += 32) {
    const float s0 = SCALE[0];
    qs[g][i] = float4(s0 * float(qp[4 * i]), s0 * float(qp[4 * i + 1]), s0 * float(qp[4 * i + 2]), s0 * float(qp[4 * i + 3]));
  }
  float o[PER];
  for (int i = 0; i < PER; i++) o[i] = 0.0f;
  float m = -INFINITY, l = 0.0f;
  const int kt = int(lane) / 2, hf = int(lane) & 1;   // lanes 2k and 2k + 1 score the tile's key k, half each
  for (int base = lo; base < hi; base += TK) {
    const int cnt = metal::min(TK, hi - base);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int e = t; e < cnt * VEC; e += nt) {
      const int k = e / VEC, c = e - k * VEC;
      const int j = base + k;
      const size_t row = size_t(sparse ? ids[j] : j) * VEC;
      ks[k][c] = kbase[row + c];
      vs[k][c] = vbase[row + c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int kk = metal::min(kt, cnt - 1);
    const threadgroup bfloat4* kr = (const threadgroup bfloat4*)(&ks[kk][0]) + hf * (HALF / 4);
    const threadgroup float4* qr = &qs[g][hf * (HALF / 4)];
    float a = 0.0f;
    for (int i = 0; i < HALF / 4; i++) {
      const float4 kv = float4(kr[i]);
      const float4 qv = qr[i];
      a = fma(qv.x, kv.x, a); a = fma(qv.y, kv.y, a); a = fma(qv.z, kv.z, a); a = fma(qv.w, kv.w, a);
    }
    a += simd_shuffle_xor(a, ushort(1));
    const bool live = kt < cnt;
    const float sc = live ? a : -INFINITY;
    const float mn = metal::max(m, simd_max(sc));
    const float f = metal::exp(m - mn);
    const float e = (live && hf == 0) ? metal::exp(sc - mn) : 0.0f;
    l = fma(l, f, simd_sum(e));
    for (int i = 0; i < PER; i++) o[i] *= f;
    for (int k = 0; k < cnt; k++) {
      const float ek = simd_shuffle(e, ushort(2 * k));
      const threadgroup bfloat4* vr = (const threadgroup bfloat4*)(&vs[k][lane]);
      const float4 v0 = float4(vr[0]), v1 = float4(vr[1]);
      o[0] = fma(ek, v0.x, o[0]); o[1] = fma(ek, v0.y, o[1]); o[2] = fma(ek, v0.z, o[2]); o[3] = fma(ek, v0.w, o[3]);
      o[4] = fma(ek, v1.x, o[4]); o[5] = fma(ek, v1.y, o[5]); o[6] = fma(ek, v1.z, o[6]); o[7] = fma(ek, v1.w, o[7]);
    }
    m = mn;
  }
  const size_t at = (size_t(r) * H + h) * P + part;
  for (int i = 0; i < PER; i++) PO[at * D + int(lane) * PER + i] = o[i];
  if (lane == 0) { PM[at * 2] = m; PM[at * 2 + 1] = l; }
"""


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
    first = base.kernel("q4_attn_gqa_parts", _ATTN_GQA_PARTS, ["Q", "Kc", "Vc", "IDS", "NK", "SPARSE", "SCALE"],
                        ["PO", "PM"])
    po, pm = first(inputs=[q, keys, values, ids, ints(counts), ints([int(bool(x)) for x in sparse]), scale_arr],
                   template=[("H", heads), ("KVH", kv_heads), ("D", dims), ("P", parts), ("TK", TK)],
                   grid=(32 * group * kv_heads, rows, parts), threadgroup=(32 * group, 1, 1),
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
    ids = attention.select_blocks(ix.block_scores(index_query, raw, cache, past), complete, ends, top=ix.top_blocks)
    sparse = [c > ix.top_blocks for c in complete]
    counts = [ix.ratio * (ix.top_blocks - c) + e if s else e for e, c, s in zip(ends, complete, sparse)]
    attend = attention_rows_gqa if gqa_supported(attn.heads, attn.kv_heads, attn.dims) else attention.attention_rows
    out = attend(queries[0].transpose(1, 0, 2), cache.keys, cache.values, counts, ids, sparse, attn.scale, parts=PARTS)
    return out.reshape(1, length, -1)
