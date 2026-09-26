"""Attention for verify windows and draft trees without tensor units: every row gets the bits of the one-row step at
its position.

A window row's attention runs over the committed keys [0, P) and then its own path: the key at position P + i is
window row path[i] (a chain's row t has path 0..t; a tree node's path runs through its ancestors). The keys are cut
into chunks of CK positions counted from 0, and a chunk's keys over SPLIT simdgroups by position (interleaved); per
simdgroup and query head, an online softmax over its keys in ascending position (lane l holding dimensions
[8 l, 8 l + 8)); the simdgroups' partials merge in simdgroup order, then the chunks' in chunk order. Which rows ride in the window, and where the window starts, change nothing a row computes: a serial step
at the same position (a one-row window, its own row at P) runs the same chunks over the same key values in the same
order. So serial decoding through this kernel is the reference drafted windows and trees reproduce, as
``lane_attention`` is on the M5.
"""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

import mlx.core as mx

CK = 128           # keys a chunk (fixed: part of the arithmetic)
SPLIT = 4          # simdgroups a query head's chunk is split over, keys interleaved (fixed: part of the arithmetic)
BLK = 4            # keys a simdgroup scores before one softmax update (fixed: part of the arithmetic)

_PARTIAL = r"""
  // threadgroup (chunk c, kv head h): simdgroup (g, s) = (sg / SPLIT, sg % SPLIT) takes query head h G + g and the
  // chunk's keys at positions k0 + s, k0 + s + SPLIT, ... up to the row's own position, an online softmax over
  // them in that order (lane l: dimensions [DPL l, DPL l + DPL)); the SPLIT partials of a head then merge in
  // simdgroup order. One window row at a time. A row's key at position P + i is window row path[i].
  const uint lane = thread_index_in_simdgroup;
  const uint sgi = simdgroup_index_in_threadgroup;
  const int g = int(sgi) / SPLIT, s = int(sgi) % SPLIT;
  const int c = int(threadgroup_position_in_grid.y);
  const int h = int(threadgroup_position_in_grid.z);
  const int P = dims[0], W = dims[1], CAP = dims[2], NCH = dims[3], MAXD = dims[4];
  constexpr int DPL = D / 32;
  const int qh = h * G + g;
  threadgroup float sm[G * SPLIT], sl[G * SPLIT];
  threadgroup float so[G * SPLIT][D];
  for (int w = 0; w < W; w++) {
    const int last = P + depth[w];                       // this row's own position
    const int k0 = c * CK;
    const int k1 = min(k0 + CK, last + 1);
    float q[DPL], o[DPL];
    for (int i = 0; i < DPL; i++) {
      q[i] = float(Q[((qh * W) + w) * D + int(lane) * DPL + i]);
      o[i] = 0.0f;
    }
    float m = -INFINITY, l = 0.0f;
    // this simdgroup's keys in blocks of BLK: the block's scores first (independent), then one update
    for (int base = k0 + s; base < k1; base += SPLIT * BLK) {
      float sc[BLK];
      int rows[BLK];
      float bm = -INFINITY;
      for (int j = 0; j < BLK; j++) {
        const int pos = base + j * SPLIT;
        rows[j] = pos < k1 ? (pos < P ? pos : P + path[w * MAXD + (pos - P)]) : -1;
        float d = 0.0f;
        if (rows[j] >= 0) {
          const device bfloat* kr = K + (size_t(h) * CAP + rows[j]) * D + int(lane) * DPL;
          for (int i = 0; i < DPL; i++) d = fma(q[i], float(kr[i]), d);
        }
        sc[j] = simd_sum(d) * scale[0];
        if (rows[j] >= 0) bm = metal::max(bm, sc[j]);
      }
      const float mn = metal::max(m, bm);
      const float a = metal::exp(m - mn);
      l *= a;
      for (int i = 0; i < DPL; i++) o[i] *= a;
      for (int j = 0; j < BLK; j++) {
        if (rows[j] < 0) continue;
        const float b = metal::exp(sc[j] - mn);
        l += b;
        const device bfloat* vr = V + (size_t(h) * CAP + rows[j]) * D + int(lane) * DPL;
        for (int i = 0; i < DPL; i++) o[i] = fma(b, float(vr[i]), o[i]);
      }
      m = mn;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) { sm[sgi] = m; sl[sgi] = l; }
    for (int i = 0; i < DPL; i++) so[sgi][int(lane) * DPL + i] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (s == 0) {
      float mx_ = -INFINITY;
      for (int t = 0; t < SPLIT; t++) mx_ = metal::max(mx_, sm[g * SPLIT + t]);
      float lsum = 0.0f, acc[DPL];
      for (int i = 0; i < DPL; i++) acc[i] = 0.0f;
      for (int t = 0; t < SPLIT; t++) {
        const float e = sl[g * SPLIT + t] > 0.0f ? metal::exp(sm[g * SPLIT + t] - mx_) : 0.0f;
        lsum = fma(sl[g * SPLIT + t], e, lsum);
        for (int i = 0; i < DPL; i++) acc[i] = fma(so[g * SPLIT + t][int(lane) * DPL + i], e, acc[i]);
      }
      const int slot = (qh * W + w) * NCH + c;
      if (lane == 0) { PM[slot] = mx_; PL[slot] = lsum; }
      for (int i = 0; i < DPL; i++) PO[size_t(slot) * D + int(lane) * DPL + i] = acc[i];
    }
  }
"""

_MERGE = r"""
  // one simdgroup per (query head, row): the chunks' partials in chunk order
  const uint lane = thread_index_in_simdgroup;
  const int qh = int(threadgroup_position_in_grid.y);
  const int w = int(threadgroup_position_in_grid.z);
  const int W = dims[1], NCH = dims[3];
  constexpr int DPL = D / 32;
  const int base = (qh * W + w) * NCH;
  float mx_ = -INFINITY;
  for (int c = 0; c < NCH; c++) mx_ = metal::max(mx_, PM[base + c]);
  float lsum = 0.0f, acc[DPL];
  for (int i = 0; i < DPL; i++) acc[i] = 0.0f;
  for (int c = 0; c < NCH; c++) {
    const float e = PL[base + c] > 0.0f ? metal::exp(PM[base + c] - mx_) : 0.0f;
    lsum = fma(PL[base + c], e, lsum);
    for (int i = 0; i < DPL; i++) acc[i] = fma(PO[size_t(base + c) * D + int(lane) * DPL + i], e, acc[i]);
  }
  for (int i = 0; i < DPL; i++) OUT[((qh * W) + w) * D + int(lane) * DPL + i] = bfloat(acc[i] / lsum);
"""

_kernels: dict[str, Any] = {}


def sources() -> dict[str, str]:
    return {"partial": _PARTIAL, "merge": _MERGE, "chunk": f"{CK}/{SPLIT}/{BLK}"}


def _kernel(name: str) -> Any:
    if name not in _kernels:
        source, inputs, outputs = {
            "partial": (_PARTIAL, ["Q", "K", "V", "depth", "path", "scale", "dims"], ["PM", "PL", "PO"]),
            "merge": (_MERGE, ["PM", "PL", "PO", "dims"], ["OUT"]),
        }[name]
        digest = hashlib.sha256((source + f"{CK}/{SPLIT}/{BLK}").encode()).hexdigest()[:16]
        _kernels[name] = mx.fast.metal_kernel(name=f"row_attention_{name}_{digest}", input_names=inputs,
                                              output_names=outputs, source=source)
    return _kernels[name]


_consts: dict[Any, mx.array] = {}


def _const(key: Any, make: Any) -> mx.array:
    if key not in _consts:
        _consts[key] = make()
        if len(_consts) > 4096:
            _consts.clear()
            _consts[key] = make()
    return _consts[key]


def paths_of(parents: Sequence[int]) -> tuple[list[int], list[list[int]]]:
    """(depths, paths) of a window's rows: row w's path lists the window rows at positions P, P + 1, ... P + depth."""

    depths: list[int] = []
    paths: list[list[int]] = []
    for row, parent in enumerate(parents):
        path = [row] if parent < 0 else paths[parent] + [row]
        paths.append(path)
        depths.append(len(path) - 1)
    return depths, paths


def row_sdpa(queries: mx.array, keys: mx.array, values: mx.array, scale: float, start: int,
             parents: Sequence[int]) -> mx.array:
    """queries [1, H, W, D] of a window whose rows sit at ``start`` + their depth; keys and values [1, HKV, cap, D]
    (a KV cache's whole buffers: row-contiguous, ``cap`` at least start + W) hold the committed keys at
    [0, start) and the window's rows at start + row. -> [1, H, W, D]."""

    _, H, W, D = (int(s) for s in queries.shape)
    HKV, CAP = int(keys.shape[1]), int(keys.shape[2])
    if D % 32 or H % HKV or len(parents) != W:
        raise ValueError(f"row_sdpa: head dim a multiple of 32, heads a multiple of kv heads, {W} parents")
    G = H // HKV
    depths, paths = paths_of(parents)
    maxd = max(depths) + 1
    nch = -(-(int(start) + maxd) // CK)
    if CAP < int(start) + W:
        raise ValueError(f"row_sdpa: the buffers hold {CAP} positions, the window reaches {int(start) + W}")
    dims = mx.array([int(start), W, CAP, nch, maxd], dtype=mx.int32)
    depth_a = _const(("depth", tuple(parents)), lambda: mx.array(depths, dtype=mx.int32))
    path_a = _const(("path", tuple(parents)), lambda: mx.array(
        [p + [0] * (maxd - len(p)) for p in paths], dtype=mx.int32).reshape(-1))
    scale_a = _const(("scale", float(scale)), lambda: mx.array([float(scale)], dtype=mx.float32))
    if G * SPLIT * 32 > 1024:
        raise ValueError(f"row_sdpa: {G} query heads a kv head need {G * SPLIT * 32} threads a threadgroup")
    q = mx.contiguous(queries)
    pm, pl, po = _kernel("partial")(
        inputs=[q, mx.contiguous(keys), mx.contiguous(values), depth_a, path_a, scale_a, dims],
        template=[("D", D), ("G", G), ("CK", CK), ("SPLIT", SPLIT), ("BLK", BLK)],
        grid=(32 * G * SPLIT, nch, HKV), threadgroup=(32 * G * SPLIT, 1, 1),
        output_shapes=[(H * W * nch,), (H * W * nch,), (H * W * nch, D)],
        output_dtypes=[mx.float32, mx.float32, mx.float32])
    return _kernel("merge")(
        inputs=[pm, pl, po, dims], template=[("D", D)],
        grid=(32, H, W), threadgroup=(32, 1, 1),
        output_shapes=[(1, H, W, D)], output_dtypes=[queries.dtype])[0]


__all__ = ["CK", "paths_of", "row_sdpa", "sources"]
