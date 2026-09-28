"""DeepSeek-V4-Flash's MoE on decode rows in five kernels: route + group, mxfp4 gate/up + SwiGLU, down, combine."""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from tensorfold.kernels import inputs
from tensorfold.kernels.deepseek.v4.rows import _HEADER_FP4, picks_fp4
from tensorfold.kernels.glm.flash.v1 import kernels as GK

MAX_ROWS = 16

_HEADER = _HEADER_FP4 + r"""
inline float log1p_accurate(float y) {
  const float u = 1.0f + y;
  return u == 1.0f ? y : metal::precise::log(u) * y / (u - 1.0f);
}
inline float sqrtsoftplus(float l) {
  return metal::precise::sqrt(metal::max(l, 0.0f) + log1p_accurate(metal::precise::exp(-metal::abs(l))));
}
"""

# Simdgroup r routes row r (top 6 by score + bias, or its table row, ascending); thread e then lists expert e's picks
_ROUTE = r"""
  const uint lane = thread_index_in_simdgroup;
  const uint g = simdgroup_index_in_threadgroup;
  const int e = int(thread_position_in_threadgroup.x);
  const int R = int(LOGITS_shape[0]);
  constexpr int PER = (NE + 31) / 32;
  threadgroup int picks[MAXR * TOPK];
  threadgroup int offs[32];
  if (int(g) < R) {
    const int r = int(g);
    float c[PER], sc[PER];
    for (int j = 0; j < PER; j++) {
      const int id = j * 32 + int(lane);
      sc[j] = id < NE ? sqrtsoftplus(LOGITS[r * NE + id]) : 0.0f;
      c[j] = id < NE ? sc[j] + BIAS[id] : -INFINITY;
    }
    int ids[TOPK];
    float w[TOPK];
    for (int k = 0; k < TOPK; k++) {
      if (HASHED) {
        const int id = TABLE[size_t(TOKENS[r]) * TOPK + k];
        float mine = 0.0f;
        for (int j = 0; j < PER; j++) if (j == id / 32) mine = sc[j];
        ids[k] = id;
        w[k] = simd_shuffle(mine, ushort(id % 32));
        continue;
      }
      float best = -INFINITY, bsc = 0.0f;
      int bid = NE;
      for (int j = 0; j < PER; j++) {
        const int id = j * 32 + int(lane);
        if (id < NE && (c[j] > best || (c[j] == best && id < bid))) { best = c[j]; bid = id; bsc = sc[j]; }
      }
      for (int off = 16; off > 0; off /= 2) {
        const float ob = simd_shuffle_xor(best, off);
        const int oi = simd_shuffle_xor(bid, off);
        const float os = simd_shuffle_xor(bsc, off);
        if (ob > best || (ob == best && oi < bid)) { best = ob; bid = oi; bsc = os; }
      }
      ids[k] = bid;
      w[k] = bsc;
      for (int j = 0; j < PER; j++) if (j * 32 + int(lane) == bid) c[j] = -INFINITY;
    }
    for (int a = 1; a < TOPK; a++)
      for (int b = a; b > 0 && ids[b - 1] > ids[b]; b--) {
        const int ti = ids[b]; ids[b] = ids[b - 1]; ids[b - 1] = ti;
        const float tw = w[b]; w[b] = w[b - 1]; w[b - 1] = tw;
      }
    if (lane == 0) {
      float total = w[0];
      for (int k = 1; k < TOPK; k++) total = total + w[k];
      for (int k = 0; k < TOPK; k++) {
        picks[r * TOPK + k] = ids[k];
        WTS[r * TOPK + k] = (w[k] / (total + 1e-20f)) * SCALE[0];
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  int members[MAXR];
  int count = 0;
  if (e < NE)
    for (int p = 0; p < R * TOPK; p++)
      if (picks[p] == e) members[count++] = p;
  const int used = count > 0 ? 1 : 0;
  const int before = simd_prefix_exclusive_sum(used);
  if (lane == 31) offs[g] = before + used;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  int base = 0;
  for (int q = 0; q < int(g); q++) base += offs[q];
  if (used) {
    const int u = base + before;
    UIDS[u] = e;
    for (int j = 0; j < MAXR; j++) UMEM[u * MAXR + j] = j < count ? members[j] : -1;
  }
  if (e == int(NT) - 1) UCOUNT[0] = base + before + used;
"""

# Simdgroup m runs pick m of expert u: gate and up rows, then w * silu(g) * u (clamped) in fp32, rounded once
_GATEUP = r"""
  const uint lane = thread_index_in_simdgroup;
  const int m = int(simdgroup_index_in_threadgroup);
  const int u = int(threadgroup_position_in_grid.z);
  if (u >= UCOUNT[0]) return;
  const int pick = UMEM[u * MAXR + m];
  if (pick < 0) return;
  const size_t e = size_t(UIDS[u]);
  const int row0 = int(threadgroup_position_in_grid.y) * RPS;
  constexpr int KB = K / 2;
  constexpr int KG = K / 32;
  const device uint8_t* wg = (const device uint8_t*)GW + (e * N + row0) * KB + lane * 8;
  const device uint8_t* wu = (const device uint8_t*)UW + (e * N + row0) * KB + lane * 8;
  const device uint8_t* sg = GS + (e * N + row0) * KG + lane / 2;
  const device uint8_t* su = US + (e * N + row0) * KG + lane / 2;
  const device bfloat* x = X + size_t(pick / TOPK) * K + lane * 16;
  float ag[RPS], au[RPS];
  for (int j = 0; j < RPS; j++) { ag[j] = 0.0f; au[j] = 0.0f; }
  for (int k0 = 0; k0 < K; k0 += 512) {
    float xt[16];
    for (int i = 0; i < 16; i++) xt[i] = float(x[i]);
    for (int j = 0; j < RPS; j++) {
      ag[j] += fp4dot16(wg + j * KB, xt, e8m0(sg[j * KG]));
      au[j] += fp4dot16(wu + j * KB, xt, e8m0(su[j * KG]));
    }
    wg += 256; wu += 256; sg += 16; su += 16; x += 512;
  }
  const float wt = WTS[pick];
  const float lim = LIM[0];
  for (int j = 0; j < RPS; j++) {
    const float gs = simd_sum(ag[j]);
    const float us = simd_sum(au[j]);
    if (lane == 0) {
      float gb = float(bfloat(gs)), ub = float(bfloat(us));
      if (lim > 0.0f) { ub = metal::clamp(ub, -lim, lim); gb = metal::min(gb, lim); }
      ACT[size_t(pick) * N + row0 + j] = bfloat(wt * (gb / (1.0f + metal::precise::exp(-gb))) * ub);
    }
  }
"""

# Row r's routed outputs summed in fp32 in slot order, plus the shared expert's, rounded once
_COMBINE = r"""
  const uint gid = thread_position_in_grid.x;
  const int r = int(gid / uint(D)), d = int(gid % uint(D));
  if (r >= int(YS_shape[0])) return;
  const device bfloat* y = Y + size_t(r) * TOPK * D + d;
  float acc = 0.0f;
  for (int k = 0; k < TOPK; k++) acc += float(y[size_t(k) * D]);
  OUT[size_t(r) * D + d] = bfloat(acc + float(YS[size_t(r) * D + d]));
"""


def _kernel(name: str, source: str, ins: list[str], outs: list[str]) -> Any:
    return GK._kernel(name, source, ins, outs, _HEADER)


def fits(moe: Any, rows: int) -> bool:
    k = int(moe.gate.weight.shape[-1]) * 8
    return (GK.metal() and 1 <= rows <= MAX_ROWS and k % 512 == 0 and int(moe.gate.weight.shape[-2]) % 4 == 0
            and int(moe.down.weight.shape[-1]) * 8 % 512 == 0 and moe.top * MAX_ROWS <= 1024)


def route(logits: mx.array, moe: Any, tokens: mx.array) -> tuple[mx.array, ...]:
    """Weights [R, K] (ascending experts) and the picks grouped by expert: (wts, uids, umem, ucount)."""

    rows, experts = logits.shape
    threads = max(32 * MAX_ROWS, -(-experts // 32) * 32)
    hashed = moe.table is not None
    kernel = _kernel("ds4_moe_route", _ROUTE, ["LOGITS", "BIAS", "TABLE", "TOKENS", "SCALE"],
                     ["WTS", "UIDS", "UMEM", "UCOUNT"])
    return tuple(kernel(
        inputs=[logits, moe.zero_bias if hashed else moe.bias, moe.table if hashed else moe.no_table,
                tokens if tokens.dtype == mx.uint32 else tokens.astype(mx.uint32), moe.scale_arr],
        template=[("NE", experts), ("TOPK", moe.top), ("MAXR", MAX_ROWS), ("NT", threads), ("HASHED", int(hashed))],
        grid=(threads, 1, 1), threadgroup=(threads, 1, 1),
        output_shapes=[(rows, moe.top), (max(rows * moe.top, inputs.MIN_ELEMENTS),), (rows * moe.top, MAX_ROWS),
                       (inputs.MIN_ELEMENTS,)],
        output_dtypes=[mx.float32, mx.int32, mx.int32, mx.int32]))


def routed(x: mx.array, moe: Any, wts: mx.array, uids: mx.array, umem: mx.array, ucount: mx.array,
           rps: int = 4) -> mx.array:
    """The window's picks through their experts: [R * K, D] bf16, each expert's weights read once."""

    rows, dims = x.shape
    picks = rows * moe.top
    inter = int(moe.gate.weight.shape[-2])
    gateup = _kernel("ds4_moe_gateup", _GATEUP, ["X", "GW", "GS", "UW", "US", "WTS", "UIDS", "UMEM", "UCOUNT", "LIM"],
                     ["ACT"])
    act = gateup(inputs=[mx.contiguous(x), moe.gate.weight, moe.gate.scales, moe.up.weight, moe.up.scales, wts, uids,
                         umem, ucount, moe.limit_arr],
                 template=[("K", dims), ("N", inter), ("RPS", rps), ("TOPK", moe.top), ("MAXR", MAX_ROWS)],
                 grid=(32 * rows, inter // rps, picks), threadgroup=(32 * rows, 1, 1),
                 output_shapes=[(picks, inter)], output_dtypes=[mx.bfloat16])[0]
    return picks_fp4(act, rows, moe.top, (uids, umem, ucount), moe.down, per_pick=True,
                     rows_per_simdgroup=rps).reshape(picks, dims)


def combine(y: mx.array, shared: mx.array, top: int) -> mx.array:
    rows, dims = shared.shape
    kernel = _kernel("ds4_moe_combine", _COMBINE, ["Y", "YS"], ["OUT"])
    return kernel(inputs=[y, shared], template=[("D", dims), ("TOPK", top)], grid=(rows * dims, 1, 1),
                  threadgroup=(256, 1, 1), output_shapes=[(rows, dims)], output_dtypes=[mx.bfloat16])[0]
