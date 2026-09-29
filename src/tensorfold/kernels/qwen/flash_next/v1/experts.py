"""The MoE: router logits, then each (row, slot)'s expert gate/up and down projections."""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from tensorfold.kernels.qwen.flash_next.v1.base import LANE_CODES, MAX_ROWS, QDOT_HEADER, by_rows, count, kernel

_ROUTER = r"""
  // one simdgroup per expert: lane l reads 8 consecutive inputs at a time, 256 apart; rows in order
  const uint lane = thread_index_in_simdgroup;
  const int e = int(threadgroup_position_in_grid.x) * (T / 32) + int(simdgroup_index_in_threadgroup);
  const int R = rows[0];
  if (e >= NE) return;
  const device bfloat* w = GW + size_t(e) * D;
  float acc[MAXR];
  for (int r = 0; r < MAXR; r++) acc[r] = 0.0f;
  for (int c = 8 * int(lane); c < D; c += 256) {
    float wv[8];
    for (int j = 0; j < 8; j++) wv[j] = float(w[c + j]);
    for (int r = 0; r < MAXR; r++) {
      if (r >= R) break;
      const device bfloat* xr = X + r * D + c;
      float a = acc[r];
      for (int j = 0; j < 8; j++) a = fma(float(xr[j]), wv[j], a);
      acc[r] = a;
    }
  }
  for (int r = 0; r < MAXR; r++) {
    if (r >= R) break;
    const float total = simd_sum(acc[r]);
    if (lane == 0) OUT[r * NE + e] = OUT_T(total);
  }
"""

_ROUTER_SPLIT = r"""
  // Two experts a threadgroup, four simdgroups an expert: simdgroup q of an expert takes inputs q D / 4 .. (lane l:
  // 4 consecutive inputs every 128), fp32 in order, simd_sum; the quarters add in order. Rows in order.
  const uint lane = thread_index_in_simdgroup;
  const int sg = int(simdgroup_index_in_threadgroup);
  const int e = int(threadgroup_position_in_grid.x) * 2 + sg / 4, q = sg % 4;
  const int R = rows[0];
  constexpr int QD = D / 4;
  threadgroup float part[MAXR][8];
  const int ee = min(e, NE - 1);
  const device bfloat* w = GW + size_t(ee) * D + q * QD + 4 * int(lane);
  float wv[QD / 32];
  for (int i = 0; i < QD / 128; i++)
    for (int j = 0; j < 4; j++) wv[4 * i + j] = float(w[128 * i + j]);
  for (int r = 0; r < R; r++) {
    const device bfloat* xr = X + r * D + q * QD + 4 * int(lane);
    float a = 0.0f;
    for (int i = 0; i < QD / 128; i++)
      for (int j = 0; j < 4; j++) a = fma(float(xr[128 * i + j]), wv[4 * i + j], a);
    a = simd_sum(a);
    if (lane == 0) part[r][sg] = a;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int t = int(thread_position_in_threadgroup.x);
  if (t < 2 * R) {
    const int r = t / 2, k = t % 2, e2 = int(threadgroup_position_in_grid.x) * 2 + k;
    const float sum = ((part[r][4 * k] + part[r][4 * k + 1]) + part[r][4 * k + 2]) + part[r][4 * k + 3];
    if (e2 < NE) OUT[r * NE + e2] = OUT_T(sum);
  }
"""

_EXPERT_GATEUP = r"""
  // Threadgroup (b, p): 2 simdgroups, rows 8 b + 4 g .. + 3 of slot p (p = r * SLOTS + k; slot k < TOPK is the
  // row's k-th expert by router logit, each simdgroup finding it itself), gate and up,
  // over K in steps of 512 (MLX's qmv_fast loop); then bf16(SiLU(bf16(gate)) * bf16(up)).
  // A slot past TOPK (SHARED = 1) is the shared expert, from its own matrices. Threadgroup (0, p)'s first
  // simdgroup also writes the slot's expert to PICK and, for the last routed slot (whose selection rounds give the
  // top-k logits), the weights exp(l_k - l_0) / their sum (fp32, bf16-rounded) to WTS: expert_down reads them.
  const uint lane = thread_index_in_simdgroup;
  const uint g = simdgroup_index_in_threadgroup;
  const int p = int(threadgroup_position_in_grid.z);
  constexpr int SLOTS = TOPK + SHARED;
  const int r = p / SLOTS, slot = p % SLOTS;
  const bool shared = slot == TOPK;
  float picked[TOPK];
  const size_t e = shared ? 0 : size_t(simd_topk<NE>(LOGITS + r * NL, slot, lane, picked));
  if (!shared && threadgroup_position_in_grid.y == 0 && g == 0 && lane == 0) {
    PICK[r * TOPK + slot] = uint32_t(e);
    if (slot == TOPK - 1) {
      float total = 0.0f;
      float ex[TOPK];
      for (int kk = 0; kk < TOPK; kk++) { ex[kk] = metal::exp(picked[kk] - picked[0]); total += ex[kk]; }
      for (int kk = 0; kk < TOPK; kk++) WTS[r * TOPK + kk] = float(bfloat(ex[kk] / total));
    }
  }
  const int row0 = int(threadgroup_position_in_grid.y) * (SG * RPS) + int(g) * RPS;
  constexpr int KB = K / 2;                         // bytes a row
  constexpr int KG = K / 32;                        // groups a row
  const device uint32_t* GWp = shared ? SGW : GW;
  const device uint32_t* UWp = shared ? SUW : UW;
  const device bfloat* GSp = shared ? SGS : GS;
  const device bfloat* GBp = shared ? SGB : GB;
  const device bfloat* USp = shared ? SUS : US;
  const device bfloat* UBp = shared ? SUB : UB;
  const device uint8_t* gw = (const device uint8_t*)GWp + (e * N + row0) * KB + lane * 8;
  const device uint8_t* uw = (const device uint8_t*)UWp + (e * N + row0) * KB + lane * 8;
  const device bfloat* gs = GSp + (e * N + row0) * KG + lane / 2;
  const device bfloat* gb = GBp + (e * N + row0) * KG + lane / 2;
  const device bfloat* us = USp + (e * N + row0) * KG + lane / 2;
  const device bfloat* ub = UBp + (e * N + row0) * KG + lane / 2;
  const device bfloat* x = X + r * K + lane * 16;
  float xt[16];
  float ag[RPS], au[RPS];
  for (int row = 0; row < RPS; row++) { ag[row] = 0.0f; au[row] = 0.0f; }
  for (int k0 = 0; k0 < K; k0 += 512) {
    const float sum = load16(x, xt);
    for (int row = 0; row < RPS; row++) {
      ag[row] += qdot16(gw + row * KB, xt, float(gs[row * KG]), float(gb[row * KG]), sum);
      au[row] += qdot16(uw + row * KB, xt, float(us[row * KG]), float(ub[row * KG]), sum);
    }
    gw += 256; uw += 256; gs += 16; gb += 16; us += 16; ub += 16; x += 512;
  }
  for (int row = 0; row < RPS; row++) {
    const float gv = simd_sum(ag[row]), uv = simd_sum(au[row]);
    if (lane == 0) ACT[p * N + row0 + row] = bfloat(bsilu(float(bfloat(gv))) * float(bfloat(uv)));
  }
"""

_EXPERT_DOWN_Y = r"""
  // Threadgroup (b, z): SG simdgroups, each one (row, slot) pair (z SG + g in row-major order; slot TOPK: the shared
  // expert; routed slots' experts from expert_gateup's PICK) for model dims 8 b .. 8 b + 7: lane l reads 16-input
  // chunks l and (l < NC - 32) 32 + l of the activation; Y[row][slot][d] = bf16(sum), expert_down's sums. The
  // combine (weights, shared gate) is the next hc_norm's "grouped" write-back, expert_down's arithmetic.
  const uint lane = thread_index_in_simdgroup;
  const int R = rows[0];
  const int pair = int(threadgroup_position_in_grid.z) * SG + int(simdgroup_index_in_threadgroup);
  constexpr int SLOTS = TOPK + 1;
  if (pair >= R * SLOTS) return;
  const int r = pair / SLOTS, k = pair % SLOTS;
  const int d0 = int(threadgroup_position_in_grid.y) * 8;
  constexpr int KB = NI / 2;
  constexpr int KG = NI / 32;
  constexpr int NC = NI / 16;
  const bool shared = k == TOPK;
  const size_t e = shared ? 0 : size_t(PICK[r * TOPK + k]);
  const device uint32_t* DWp = shared ? SDW : DW;
  const device bfloat* DSp = shared ? SDS : DS;
  const device bfloat* DBp = shared ? SDB : DB;
  const device bfloat* x = ACT + (r * SLOTS + k) * NI;
  float xa[16], xb[16];
  const float sa = load16(x + lane * 16, xa);
  const bool second = int(lane) < NC - 32;
  const float sb = second ? load16(x + (32 + lane) * 16, xb) : 0.0f;
  for (int row = 0; row < 8; row++) {
    const size_t at = e * D + d0 + row;
    const device uint8_t* w = (const device uint8_t*)DWp + at * KB;
    float acc = qdot16(w + lane * 8, xa, float(DSp[at * KG + lane / 2]), float(DBp[at * KG + lane / 2]), sa);
    if (second)
      acc += qdot16(w + (32 + lane) * 8, xb, float(DSp[at * KG + (32 + lane) / 2]), float(DBp[at * KG + (32 + lane) / 2]), sb);
    acc = simd_sum(acc);
    if (lane == 0) Y[(r * SLOTS + k) * D + d0 + row] = bfloat(acc);
  }
"""


# Any MLX affine width: a lane reads VPT codes a step (MLX's qmv_fast widths) from its rows' bit streams.
_AFFINE_EXPERTS = LANE_CODES + r"""
// RPS gate and up rows from ``first`` over K inputs: per lane step, a dot of VPT codes, then scale and bias
template <int BITS, int GSZ, int K, int RPS>
inline void gateup_rows(const device uint32_t* GWp, const device uint32_t* UWp, const device bfloat* GSp,
                        const device bfloat* GBp, const device bfloat* USp, const device bfloat* UBp, size_t first,
                        const device bfloat* x, uint lane, thread float* ag, thread float* au) {
  constexpr int VPT = lane_values(BITS), RB = K * BITS / 8, KG = K / GSZ;
  const device uint8_t* gw = (const device uint8_t*)GWp + first * RB;
  const device uint8_t* uw = (const device uint8_t*)UWp + first * RB;
  for (int v0 = int(lane) * VPT; v0 < K; v0 += 32 * VPT) {
    float xv[VPT], sum = 0.0f;
    for (int i = 0; i < VPT; i++) { xv[i] = float(x[v0 + i]); sum += xv[i]; }
    for (int row = 0; row < RPS; row++) {
      float qg[VPT], qu[VPT];
      lane_codes<BITS, VPT>(gw + row * RB, v0, qg);
      lane_codes<BITS, VPT>(uw + row * RB, v0, qu);
      float dg = 0.0f, du = 0.0f;
      for (int i = 0; i < VPT; i++) { dg = fma(qg[i], xv[i], dg); du = fma(qu[i], xv[i], du); }
      const size_t at = (first + row) * KG + v0 / GSZ;
      ag[row] += fma(float(GSp[at]), dg, float(GBp[at]) * sum);
      au[row] += fma(float(USp[at]), du, float(UBp[at]) * sum);
    }
  }
}
// 8 down rows from ``first`` over NI inputs, the lane's input chunks read once
template <int BITS, int GSZ, int NI>
inline void down_rows(const device uint32_t* W, const device bfloat* S, const device bfloat* B, size_t first,
                      const device bfloat* x, uint lane, thread float* out) {
  constexpr int VPT = lane_values(BITS), NC = (NI / VPT + 31) / 32, RB = NI * BITS / 8, KG = NI / GSZ;
  float xv[NC][VPT], sums[NC];
  for (int c = 0; c < NC; c++) {
    const int v0 = (c * 32 + int(lane)) * VPT;
    sums[c] = 0.0f;
    for (int i = 0; i < VPT; i++) { xv[c][i] = v0 < NI ? float(x[v0 + i]) : 0.0f; sums[c] += xv[c][i]; }
  }
  for (int row = 0; row < 8; row++) {
    const device uint8_t* w = (const device uint8_t*)W + (first + row) * RB;
    float acc = 0.0f;
    for (int c = 0; c < NC; c++) {
      const int v0 = (c * 32 + int(lane)) * VPT;
      if (v0 < NI) {
        float q[VPT];
        lane_codes<BITS, VPT>(w, v0, q);
        float d = 0.0f;
        for (int i = 0; i < VPT; i++) d = fma(q[i], xv[c][i], d);
        const size_t at = (first + row) * KG + v0 / GSZ;
        acc += fma(float(S[at]), d, float(B[at]) * sums[c]);
      }
    }
    out[row] = acc;
  }
}
"""

_EXPERT_GATEUP_Q = r"""
  // expert_gateup's slots and picks for any width: routed experts WB-bit in groups of WG, the shared one SWB / SWG
  const uint lane = thread_index_in_simdgroup;
  const uint g = simdgroup_index_in_threadgroup;
  const int p = int(threadgroup_position_in_grid.z);
  constexpr int SLOTS = TOPK + SHARED;
  const int r = p / SLOTS, slot = p % SLOTS;
  const bool shared = slot == TOPK;
  float picked[TOPK];
  const size_t e = shared ? 0 : size_t(simd_topk<NE>(LOGITS + r * NL, slot, lane, picked));
  if (!shared && threadgroup_position_in_grid.y == 0 && g == 0 && lane == 0) {
    PICK[r * TOPK + slot] = uint32_t(e);
    if (slot == TOPK - 1) {
      float total = 0.0f;
      float ex[TOPK];
      for (int kk = 0; kk < TOPK; kk++) { ex[kk] = metal::exp(picked[kk] - picked[0]); total += ex[kk]; }
      for (int kk = 0; kk < TOPK; kk++) WTS[r * TOPK + kk] = float(bfloat(ex[kk] / total));
    }
  }
  const int row0 = int(threadgroup_position_in_grid.y) * (SG * RPS) + int(g) * RPS;
  float ag[RPS], au[RPS];
  for (int row = 0; row < RPS; row++) { ag[row] = 0.0f; au[row] = 0.0f; }
  if (shared) gateup_rows<SWB, SWG, K, RPS>(SGW, SUW, SGS, SGB, SUS, SUB, size_t(row0), X + r * K, lane, ag, au);
  else gateup_rows<WB, WG, K, RPS>(GW, UW, GS, GB, US, UB, e * N + row0, X + r * K, lane, ag, au);
  for (int row = 0; row < RPS; row++) {
    const float gv = simd_sum(ag[row]), uv = simd_sum(au[row]);
    if (lane == 0) ACT[p * N + row0 + row] = bfloat(bsilu(float(bfloat(gv))) * float(bfloat(uv)));
  }
"""

_EXPERT_DOWN_Y_Q = r"""
  // expert_down_y for any width: simdgroup (row, slot) pair, dims 8 b .. 8 b + 7, one fp32 sum a dim
  const uint lane = thread_index_in_simdgroup;
  const int R = rows[0];
  const int pair = int(threadgroup_position_in_grid.z) * SG + int(simdgroup_index_in_threadgroup);
  constexpr int SLOTS = TOPK + 1;
  if (pair >= R * SLOTS) return;
  const int r = pair / SLOTS, k = pair % SLOTS;
  const int d0 = int(threadgroup_position_in_grid.y) * 8;
  const bool shared = k == TOPK;
  const device bfloat* x = ACT + (r * SLOTS + k) * NI;
  float out[8];
  if (shared) down_rows<SWB, SWG, NI>(SDW, SDS, SDB, size_t(d0), x, lane, out);
  else down_rows<WB, WG, NI>(DW, DS, DB, size_t(PICK[r * TOPK + k]) * D + d0, x, lane, out);
  for (int row = 0; row < 8; row++) {
    const float v = simd_sum(out[row]);
    if (lane == 0) Y[(r * SLOTS + k) * D + d0 + row] = bfloat(v);
  }
"""


def _format(*linears: Any) -> tuple[int, int]:
    """The one (bits, group size) of linears that a kernel reads together."""

    found = {(int(getattr(l, "bits", 4)), int(getattr(l, "group_size", 32))) for l in linears}
    if len(found) != 1:
        raise ValueError(f"expert projections read together need one format, got {sorted(found)}")
    return found.pop()


def router(x: mx.array, gate_weight: mx.array, *, threads: int = 256, dtype: Any = mx.float32,
           split: bool = False) -> mx.array:
    """x [R, D] @ gate_weight.T -> logits [R, E], a row's bits the same at any R; ``split``: 4 simdgroups an expert."""

    rows, dims = x.shape
    experts = int(gate_weight.shape[0])
    out_t = "float" if dtype == mx.float32 else "bfloat"
    if split and dims % 512 == 0:
        run = kernel(f"q4_router_split_{out_t}", lambda: _ROUTER_SPLIT.replace("OUT_T", out_t), ["X", "GW", "rows"],
                     ["OUT"])
        return run(inputs=[x, gate_weight, count(rows)], template=[("D", dims), ("NE", experts), ("MAXR", MAX_ROWS)],
                   grid=(-(-experts // 2) * 256, 1, 1), threadgroup=(256, 1, 1),
                   output_shapes=[(rows, experts)], output_dtypes=[dtype])[0]
    run = kernel(f"q4_router_{out_t}", lambda: _ROUTER.replace("OUT_T", out_t), ["X", "GW", "rows"], ["OUT"])
    return run(inputs=[x, gate_weight, count(rows)],
                  template=[("D", dims), ("NE", experts), ("T", threads), ("MAXR", MAX_ROWS)],
                  grid=(-(-experts // (threads // 32)) * threads, 1, 1), threadgroup=(threads, 1, 1),
                  output_shapes=[(rows, experts)], output_dtypes=[dtype])[0]

def expert_gateup(x: mx.array, logits: mx.array, top_k: int, experts: int, gate: Any, up: Any,
                  shared: tuple[Any, Any] | None = None, *, rows_per_simdgroup: int = 4, simdgroups: int = 2
                  ) -> mx.array:
    """Return each row's top-k expert activations, picks and weights, with the optional shared expert in the last slot."""

    rows, dims = x.shape
    width = int(gate.weight.shape[1])
    if dims % 512 or width % 8:
        raise ValueError("expert_gateup: needs K % 512 == 0 and N % 8 == 0")
    extra = 1 if shared is not None else 0
    sg, su = shared if shared is not None else (gate, up)
    wb, wg = _format(gate, up)
    swb, swg = _format(sg, su)
    names = ["X", "LOGITS", "GW", "GS", "GB", "UW", "US", "UB", "SGW", "SGS", "SGB", "SUW", "SUS", "SUB"]
    if (wb, wg, swb, swg) == (4, 32, 4, 32):
        run, formats = kernel(*by_rows("q4_expert_gateup", _EXPERT_GATEUP, rows), names, ["ACT", "PICK", "WTS"]), []
    else:
        run = kernel("qa_expert_gateup", _EXPERT_GATEUP_Q, names, ["ACT", "PICK", "WTS"],
                     header=QDOT_HEADER + _AFFINE_EXPERTS)
        formats = [("WB", wb), ("WG", wg), ("SWB", swb), ("SWG", swg)]
    return tuple(run(inputs=[x, logits, gate.weight, gate.scales, gate.biases, up.weight, up.scales, up.biases,
                                sg.weight, sg.scales, sg.biases, su.weight, su.scales, su.biases],
                        template=[("K", dims), ("N", width), ("TOPK", top_k), ("SHARED", extra), ("NE", experts),
                                  ("NL", int(logits.shape[-1])), ("RPS", rows_per_simdgroup), ("SG", simdgroups),
                                  *formats],
                        grid=(32 * simdgroups, width // (rows_per_simdgroup * simdgroups), rows * (top_k + extra)),
                        threadgroup=(32 * simdgroups, 1, 1),
                        output_shapes=[(rows, top_k + extra, width), (rows, top_k), (rows, top_k)],
                        output_dtypes=[mx.bfloat16, mx.uint32, mx.float32]))

def expert_down_y(act: mx.array, picks: mx.array, down: Any, shared: Any, *, simdgroups: int = 2) -> mx.array:
    """Down-project each row's picked experts and final shared slot to bf16 [R, k + 1, D] for grouped hc_norm write-back."""

    rows, slots, width = act.shape
    top_k = int(picks.shape[-1])
    dims = int(down.weight.shape[1])
    wb, wg = _format(down)
    swb, swg = _format(shared)
    names = ["ACT", "PICK", "DW", "DS", "DB", "SDW", "SDS", "SDB", "rows"]
    if (wb, wg, swb, swg) == (4, 32, 4, 32):
        run, formats = kernel(*by_rows("q4_expert_down_y", _EXPERT_DOWN_Y, rows), names, ["Y"]), []
    else:
        run = kernel("qa_expert_down_y", _EXPERT_DOWN_Y_Q, names, ["Y"], header=QDOT_HEADER + _AFFINE_EXPERTS)
        formats = [("WB", wb), ("WG", wg), ("SWB", swb), ("SWG", swg)]
    return run(inputs=[act, picks, down.weight, down.scales, down.biases, shared.weight, shared.scales,
                          shared.biases, count(rows)],
                  template=[("NI", width), ("D", dims), ("TOPK", top_k), ("SG", simdgroups), *formats],
                  grid=(32 * simdgroups, dims // 8, -(-rows * slots // simdgroups)),
                  threadgroup=(32 * simdgroups, 1, 1),
                  output_shapes=[(rows, slots, dims)], output_dtypes=[mx.bfloat16])[0]
