"""Top-k over a vocabulary row, fast and deterministic: radix select on bf16 keys.

MLX's argpartition over [16, 248320] takes 2-3 ms and ran twice a round (exact
sampling's candidates, DFlash2's lattice). A bf16 logit is a 16-bit key: two
256-bin histogram passes find the k-th largest key T, one pass collects every
element above T and the ties at T with the smallest token ids, and one
simdgroup sorts the k survivors by (value desc, id asc). One threadgroup per
row; each row's result depends only on that row.
"""

from __future__ import annotations

import hashlib
from typing import Any

import mlx.core as mx

MAX_K = 64

_SOURCE = r"""
  // one threadgroup (TPG threads) per row of X [R, V] bf16
  const uint row = threadgroup_position_in_grid.x;
  const uint tid = thread_position_in_threadgroup.x;
  const int Vn = dims[0], K = dims[1];
  const device ushort* x = (const device ushort*)X + (int64_t)row * Vn;
  threadgroup atomic_uint hist[256];
  threadgroup uint found[4];                 // [0] high byte, [1] threshold key, [2] count above, [3] ties needed
  threadgroup atomic_uint n_sel;
  threadgroup atomic_uint n_tie;
  threadgroup uint sel_key[MAXK];
  threadgroup uint sel_idx[MAXK];
  threadgroup uint tie_idx[MAXT];
  // sortable key: larger float -> larger unsigned
  #define SKEY(b) ((b & 0x8000u) ? (~b & 0xFFFFu) : (b | 0x8000u))

  // pass 1: histogram of the key's high byte
  for (uint i = tid; i < 256; i += TPG) atomic_store_explicit(&hist[i], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = tid; i < Vn; i += TPG) {
    const uint key = SKEY(uint(x[i]));
    atomic_fetch_add_explicit(&hist[key >> 8], 1u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    uint above = 0; int b = 255;
    for (; b >= 0; --b) {
      const uint c = atomic_load_explicit(&hist[b], memory_order_relaxed);
      if (above + c >= uint(K)) break;
      above += c;
    }
    found[0] = uint(max(b, 0)); found[2] = above;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint hi = found[0];
  // pass 2: histogram of the low byte inside that high byte
  for (uint i = tid; i < 256; i += TPG) atomic_store_explicit(&hist[i], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = tid; i < Vn; i += TPG) {
    const uint key = SKEY(uint(x[i]));
    if ((key >> 8) == hi) atomic_fetch_add_explicit(&hist[key & 0xFFu], 1u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    uint above = found[2]; int b = 255;
    for (; b >= 0; --b) {
      const uint c = atomic_load_explicit(&hist[b], memory_order_relaxed);
      if (above + c >= uint(K)) break;
      above += c;
    }
    found[1] = (hi << 8) | uint(max(b, 0));
    found[2] = above;                        // elements strictly above the threshold key
    found[3] = uint(K) - above;              // ties at the threshold to keep (smallest ids)
    atomic_store_explicit(&n_sel, 0u, memory_order_relaxed);
    atomic_store_explicit(&n_tie, 0u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint T = found[1];
  // pass 3: collect everything above T, and ties at T
  for (int i = tid; i < Vn; i += TPG) {
    const uint key = SKEY(uint(x[i]));
    if (key > T) {
      const uint s = atomic_fetch_add_explicit(&n_sel, 1u, memory_order_relaxed);
      if (s < MAXK) { sel_key[s] = key; sel_idx[s] = uint(i); }
    } else if (key == T) {
      const uint s = atomic_fetch_add_explicit(&n_tie, 1u, memory_order_relaxed);
      if (s < MAXT) tie_idx[s] = uint(i);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    // the smallest ids among the ties (selection by repeated minimum: ties are few)
    const uint above = found[2];
    const uint need = found[3];
    const uint ties = min(atomic_load_explicit(&n_tie, memory_order_relaxed), uint(MAXT));
    for (uint j = 0; j < need; ++j) {
      uint best = 0xFFFFFFFFu, at = 0;
      for (uint t = 0; t < ties; ++t) if (tie_idx[t] < best) { best = tie_idx[t]; at = t; }
      tie_idx[at] = 0xFFFFFFFFu;
      sel_key[above + j] = T; sel_idx[above + j] = best;
    }
    // sort the K survivors: key desc, id asc (insertion sort, K <= MAXK)
    for (int a = 1; a < K; ++a) {
      const uint kk = sel_key[a], ii = sel_idx[a];
      int b = a - 1;
      while (b >= 0 && (sel_key[b] < kk || (sel_key[b] == kk && sel_idx[b] > ii))) {
        sel_key[b + 1] = sel_key[b]; sel_idx[b + 1] = sel_idx[b]; --b;
      }
      sel_key[b + 1] = kk; sel_idx[b + 1] = ii;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int j = tid; j < K; j += TPG) {
    const uint idx = sel_idx[j];
    IDX[row * K + j] = int(idx);
    VAL[row * K + j] = float(X[(int64_t)row * Vn + idx]);
  }
"""

_kernels: dict[str, Any] = {}


def _kernel() -> Any:
    if "topk" not in _kernels:
        digest = hashlib.sha256(_SOURCE.encode()).hexdigest()[:16]
        _kernels["topk"] = mx.fast.metal_kernel(name=f"radix_topk_{digest}", input_names=["X", "dims"],
                                                output_names=["IDX", "VAL"], source=_SOURCE)
    return _kernels["topk"]


def topk_rows(x: mx.array, k: int) -> tuple[mx.array, mx.array]:
    """(indices [R, k] int32, values [R, k] float32) of each row's k largest, value desc then id asc."""

    if x.dtype != mx.bfloat16:
        x = x.astype(mx.bfloat16)
    x2 = mx.contiguous(x.reshape(-1, int(x.shape[-1])))
    rows, vocab = int(x2.shape[0]), int(x2.shape[1])
    k = int(k)
    if not 1 <= k <= MAX_K or k > vocab:
        raise ValueError(f"topk_rows: k must be in [1, {MAX_K}] and at most the row length")
    tpg = 1024
    idx, val = _kernel()(
        inputs=[x2, mx.array([vocab, k], dtype=mx.int32)],
        template=[("TPG", tpg), ("MAXK", MAX_K), ("MAXT", 2048)],
        grid=(rows * tpg, 1, 1), threadgroup=(tpg, 1, 1),
        output_shapes=[(rows, k), (rows, k)], output_dtypes=[mx.int32, mx.float32])
    return idx, val


__all__ = ["MAX_K", "topk_rows"]
