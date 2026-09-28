"""Gemma 4's MoE block for any number of rows: router, top 8 of 128, experts; each (row, slot) its own simdgroups."""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from tensorfold.kernels.gemma.v1.base import Kernel
from tensorfold.kernels.inputs import MIN_ELEMENTS

# one simdgroup per row: top K by score (ties to the lower id), softmax over the K, times the expert's scale, as bf16
_ROUTE = r"""
  const uint lane = thread_index_in_simdgroup;
  const uint r = threadgroup_position_in_grid.x;
  float sc[NE / 32];
  for (int j = 0; j < NE / 32; j++) sc[j] = float(G[int(r) * NE + int(lane) + 32 * j]);
  float picked[K];
  int ids[K];
  for (int k = 0; k < K; k++) {
    float best = -INFINITY;
    int best_e = 1 << 20;
    for (int j = 0; j < NE / 32; j++) {
      if (sc[j] > best) { best = sc[j]; best_e = int(lane) + 32 * j; }
    }
    const float top = simd_max(best);
    const int winner = simd_min(best == top ? best_e : (1 << 20));
    for (int j = 0; j < NE / 32; j++) {
      if (int(lane) + 32 * j == winner) sc[j] = -INFINITY;
    }
    picked[k] = top;
    ids[k] = winner;
  }
  if (lane == 0) {
    float total = 0.0f;
    for (int k = 0; k < K; k++) total += metal::exp(picked[k] - picked[0]);
    for (int k = 0; k < K; k++) {
      const float p = float(bfloat(metal::exp(picked[k] - picked[0]) / total));
      IDX[int(r) * K + k] = uint(ids[k]);
      WT[int(r) * K + k] = bfloat(p * float(PES[ids[k]]));
    }
  }
"""

# threadgroup (b, r): simdgroup g takes outputs RPS (SG b + g) .. of row r, lane l inputs 8 l .. of each 256-input step
_ROUTER = r"""
  const uint lane = thread_index_in_simdgroup;
  const int r = int(threadgroup_position_in_grid.y);
  const int row0 = (int(threadgroup_position_in_grid.x) * SG + int(simdgroup_index_in_threadgroup)) * RPS;
  constexpr int KG = K / GS;
  const device uint8_t* w = (const device uint8_t*)W + size_t(row0) * K + lane * 8;
  const device bfloat* sc = S + size_t(row0) * KG + (lane * 8) / GS;
  const device bfloat* bi = B + size_t(row0) * KG + (lane * 8) / GS;
  const device bfloat* x = X + size_t(r) * K + lane * 8;
  float acc[RPS];
  for (int j = 0; j < RPS; j++) acc[j] = 0.0f;
  for (int k0 = 0; k0 < K; k0 += 256) {
    float xt[8];
    float sum = 0.0f;
    for (int i = 0; i < 8; i++) { xt[i] = float(x[i]); sum += xt[i]; }
    for (int j = 0; j < RPS; j++) {
      float a = 0.0f;
      for (int i = 0; i < 8; i++) a = fma(xt[i], float(w[j * K + i]), a);
      acc[j] += float(sc[j * KG]) * a + float(bi[j * KG]) * sum;
    }
    w += 256; sc += 256 / GS; bi += 256 / GS; x += 256;
  }
  for (int j = 0; j < RPS; j++) {
    const float total = simd_sum(acc[j]);
    if (lane == 0) OUT[size_t(r) * N + row0 + j] = bfloat(total);
  }
"""

# MLX's 4-bit qmv inner loop: 16 inputs a lane, pre-divided by 1, 16, 256, 4096 so masked nibbles need no shift
_QDOT_HEADER = r"""
inline float load16(const device bfloat* x, thread float* xt) {
  float sum = 0.0f;
  for (int i = 0; i < 16; i += 4) {
    const bfloat a = x[i], b = x[i + 1], c = x[i + 2], d = x[i + 3];
    sum += float(bfloat(float(bfloat(float(bfloat(float(a) + float(b))) + float(c))) + float(d)));
    xt[i] = float(a); xt[i + 1] = float(b) / 16.0f; xt[i + 2] = float(c) / 256.0f; xt[i + 3] = float(d) / 4096.0f;
  }
  return sum;
}
inline float qdot16(const device uint8_t* w, const thread float* xt, float scale, float bias, float sum) {
  const device uint16_t* ws = (const device uint16_t*)w;
  float accum = 0.0f;
  for (int i = 0; i < 4; i++)
    accum += xt[4 * i] * float(ws[i] & 0x000f) + xt[4 * i + 1] * float(ws[i] & 0x00f0) +
             xt[4 * i + 2] * float(ws[i] & 0x0f00) + xt[4 * i + 3] * float(ws[i] & 0xf000);
  return scale * accum + sum * bias;
}
// mlx.nn.gelu_approx in fp32
inline float gelu_tanh(float x) {
  return 0.5f * x * (1.0f + metal::precise::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}
"""

# threadgroup (b, p): rows RPS g .. of slot p's gate and up (p = row TOPK + k), then bf16(gelu(bf16(gate)) * bf16(up))
_EXPERT_GATEUP = r"""
  const uint lane = thread_index_in_simdgroup;
  const uint g = simdgroup_index_in_threadgroup;
  const int p = int(threadgroup_position_in_grid.z);
  const int r = p / TOPK;
  const size_t e = size_t(IDX[p]);
  const int row0 = int(threadgroup_position_in_grid.y) * (SG * RPS) + int(g) * RPS;
  constexpr int KB = K / 2;
  constexpr int KG = K / GS;
  const device uint8_t* gw = (const device uint8_t*)GW + (e * N + row0) * KB + lane * 8;
  const device uint8_t* uw = (const device uint8_t*)UW + (e * N + row0) * KB + lane * 8;
  const device bfloat* gs = GSC + (e * N + row0) * KG + lane / (GS / 16);
  const device bfloat* gb = GBI + (e * N + row0) * KG + lane / (GS / 16);
  const device bfloat* us = USC + (e * N + row0) * KG + lane / (GS / 16);
  const device bfloat* ub = UBI + (e * N + row0) * KG + lane / (GS / 16);
  const device bfloat* x = X + r * K + lane * 16;
  float xt[16];
  float ag[RPS], au[RPS];
  for (int row = 0; row < RPS; row++) { ag[row] = 0.0f; au[row] = 0.0f; }
  for (int k0 = 0; k0 < K; k0 += 512) {
    if (k0 + int(lane) * 16 < K) {
      const float sum = load16(x, xt);
      for (int row = 0; row < RPS; row++) {
        ag[row] += qdot16(gw + row * KB, xt, float(gs[row * KG]), float(gb[row * KG]), sum);
        au[row] += qdot16(uw + row * KB, xt, float(us[row * KG]), float(ub[row * KG]), sum);
      }
    }
    gw += 256; uw += 256; gs += 512 / GS; gb += 512 / GS; us += 512 / GS; ub += 512 / GS; x += 512;
  }
  for (int row = 0; row < RPS; row++) {
    const float gv = simd_sum(ag[row]), uv = simd_sum(au[row]);
    if (lane == 0) ACT[p * N + row0 + row] = bfloat(gelu_tanh(float(bfloat(gv))) * float(bfloat(uv)));
  }
"""

# threadgroup (b, r): simdgroup k the row's k-th expert for dims 8 b ..; out = bf16(sum_k bf16(w_k y_k)), slots in order
_EXPERT_DOWN = r"""
  const uint lane = thread_index_in_simdgroup;
  const int k = int(simdgroup_index_in_threadgroup);
  const int r = int(threadgroup_position_in_grid.z);
  const int d0 = int(threadgroup_position_in_grid.y) * 8;
  constexpr int KB = NI / 2;
  constexpr int KG = NI / GS;
  constexpr int NC = NI / 16;
  threadgroup float ys[TOPK][8];
  const size_t e = size_t(IDX[r * TOPK + k]);
  const device bfloat* x = ACT + (r * TOPK + k) * NI;
  float xa[16], xb[16];
  const float sa = load16(x + lane * 16, xa);
  const bool second = int(lane) < NC - 32;
  const float sb = second ? load16(x + (32 + lane) * 16, xb) : 0.0f;
  // the 8 rows' loads before the first simd_sum, so their reads are in flight together
  float acc[8];
  #pragma unroll
  for (int row = 0; row < 8; row++) {
    const size_t at = e * D + d0 + row;
    const device uint8_t* w = (const device uint8_t*)DW + at * KB;
    acc[row] = qdot16(w + lane * 8, xa, float(DSC[at * KG + lane / (GS / 16)]),
                      float(DBI[at * KG + lane / (GS / 16)]), sa);
  }
  if (second) {
    #pragma unroll
    for (int row = 0; row < 8; row++) {
      const size_t at = e * D + d0 + row;
      const device uint8_t* w = (const device uint8_t*)DW + at * KB;
      acc[row] += qdot16(w + (32 + lane) * 8, xb, float(DSC[at * KG + (32 + lane) / (GS / 16)]),
                         float(DBI[at * KG + (32 + lane) / (GS / 16)]), sb);
    }
  }
  #pragma unroll
  for (int row = 0; row < 8; row++) {
    const float s = simd_sum(acc[row]);
    if (lane == 0) ys[k][row] = float(bfloat(s));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (k == 0 && lane < 8) {
    float routed = 0.0f;
    for (int kk = 0; kk < TOPK; kk++) routed += float(bfloat(ys[kk][lane] * float(WT[r * TOPK + kk])));
    OUT[r * D + d0 + int(lane)] = bfloat(routed);
  }
"""

_route = Kernel("gemma_route", _ROUTE, ["G", "PES"], ["IDX", "WT"])
_router = Kernel("gemma_router", _ROUTER, ["X", "W", "S", "B"], ["OUT"])
_gateup = Kernel("gemma_expert_gateup", _EXPERT_GATEUP, ["X", "IDX", "GW", "GSC", "GBI", "UW", "USC", "UBI"], ["ACT"],
                 header=_QDOT_HEADER)
_down = Kernel("gemma_expert_down", _EXPERT_DOWN, ["ACT", "IDX", "WT", "DW", "DSC", "DBI"], ["OUT"],
               header=_QDOT_HEADER)


def check_q4(linear: Any) -> None:
    if getattr(linear, "bits", None) != 4 or getattr(linear, "group_size", None) not in (32, 64, 128):
        raise ValueError("the Gemma expert kernels read 4-bit weights in groups of 32, 64 or 128")


def router_logits(x: mx.array, proj: Any, *, simdgroups: int = 4, rows_per_simdgroup: int = 1) -> mx.array:
    """x [R, K] bf16 through the router's 8-bit linear -> [R, E] bf16."""

    if getattr(proj, "bits", None) != 8 or getattr(proj, "group_size", None) not in (32, 64, 128):
        raise ValueError("router_logits reads 8-bit weights in groups of 32, 64 or 128")
    rows, dims = x.shape
    experts = int(proj.weight.shape[0])
    block = simdgroups * rows_per_simdgroup
    if dims % 256 or experts % block:
        raise ValueError(f"router_logits: needs K a multiple of 256 and E a multiple of {block}")
    consts = (("K", dims), ("N", experts), ("GS", int(proj.group_size)), ("SG", simdgroups),
              ("RPS", rows_per_simdgroup))
    return _router(consts, inputs=[x, proj.weight, proj.scales, proj.biases],
                   grid=((experts // block) * 32 * simdgroups, rows, 1), threadgroup=(32 * simdgroups, 1, 1),
                   output_shapes=[(rows, experts)], output_dtypes=[mx.bfloat16])[0]


def route(scores: mx.array, expert_scale: mx.array, top_k: int) -> tuple[mx.array, mx.array]:
    """Scores [R, E] -> ids (best first) and bf16 weights, row r's at [r K, r K + K), padded to MIN_ELEMENTS."""

    rows, experts = scores.shape
    if experts % 32:
        raise ValueError("route: the expert count must be a multiple of 32")
    size = max(rows * top_k, MIN_ELEMENTS)
    return _route((("NE", experts), ("K", top_k)), inputs=[scores, expert_scale],
                  grid=(32 * rows, 1, 1), threadgroup=(32, 1, 1),
                  output_shapes=[(size,), (size,)], output_dtypes=[mx.uint32, mx.bfloat16])


def expert_gateup(x: mx.array, ids: mx.array, top_k: int, gate: Any, up: Any, *, simdgroups: int = 2,
                  rows_per_simdgroup: int = 4) -> mx.array:
    """x [R, K] and ``route``'s ids -> bf16(gelu(x W_gate^T) * (x W_up^T)) for every (row, slot): [R * TOPK, N]."""

    check_q4(gate)
    rows, dims = x.shape
    width = gate.weight.shape[1]
    per_tg = simdgroups * rows_per_simdgroup
    if width % per_tg or dims % 16:
        raise ValueError("expert_gateup: expert width must split into threadgroups, inputs into chunks of 16")
    consts = (("K", dims), ("N", width), ("TOPK", top_k), ("GS", gate.group_size), ("SG", simdgroups),
              ("RPS", rows_per_simdgroup))
    return _gateup(consts, inputs=[x, ids, gate.weight, gate.scales, gate.biases, up.weight, up.scales, up.biases],
                   grid=(32 * simdgroups, width // per_tg, rows * top_k), threadgroup=(32 * simdgroups, 1, 1),
                   output_shapes=[(rows * top_k, width)], output_dtypes=[mx.bfloat16])[0]


def expert_down(act: mx.array, ids: mx.array, weights: mx.array, top_k: int, down: Any) -> mx.array:
    """act [R * TOPK, NI] and ``route``'s ids and weights -> bf16(sum_k w_k * (act_k W_down^T)): [R, D]."""

    check_q4(down)
    rows = int(act.shape[0]) // top_k
    inner = act.shape[-1]
    dims = down.weight.shape[1]
    if inner % 16 or not 32 <= inner // 16 <= 64 or dims % 8:
        raise ValueError("expert_down: the expert width must be 512 to 1,024 in chunks of 16")
    consts = (("NI", inner), ("D", dims), ("TOPK", top_k), ("GS", down.group_size))
    return _down(consts, inputs=[act, ids, weights, down.weight, down.scales, down.biases],
                 grid=(32 * top_k, dims // 8, rows), threadgroup=(32 * top_k, 1, 1),
                 output_shapes=[(rows, dims)], output_dtypes=[mx.bfloat16])[0]


__all__ = ["check_q4", "expert_down", "expert_gateup", "route", "router_logits"]
