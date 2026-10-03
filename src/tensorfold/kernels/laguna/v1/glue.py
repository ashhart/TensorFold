"""Laguna's kernels around its matmuls, every row its own simdgroup or threadgroup."""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from tensorfold.kernels.gemma.v1 import moe as gemma_moe
from tensorfold.kernels.gemma.v1.base import Kernel
from tensorfold.kernels.inputs import MIN_ELEMENTS

# simdgroup (head slot, row) of the stacked q|k|v: lane l holds elements l, l + 32, ... of the head
_QKV_PREP = r"""
  const uint lane = thread_index_in_simdgroup;
  const int slot = int(threadgroup_position_in_grid.x);
  const int r = int(threadgroup_position_in_grid.y);
  const int R = int(threadgroups_per_grid.y);
  constexpr int PER = DH / 32;
  constexpr int PR = RD / 32;                          // rotated elements a lane: pairs (i, i + PR / 2)
  const int kind = slot < NQ ? 0 : (slot < NQ + NK ? 1 : 2);
  const int head = kind == 0 ? slot : (kind == 1 ? slot - NQ : slot - NQ - NK);
  float xv[PER];
  for (int i = 0; i < PER; i++) xv[i] = float(QKV[r * W + slot * DH + int(lane) + 32 * i]);
  if (kind == 2) {
    for (int i = 0; i < PER; i++) V[(head * R + r) * DH + int(lane) + 32 * i] = bfloat(xv[i]);
    return;
  }
  float ss = 0.0f;
  for (int i = 0; i < PER; i++) ss = fma(xv[i], xv[i], ss);
  ss = simd_sum(ss);
  const float inv = metal::precise::rsqrt(ss / float(DH) + eps[0]);
  const device bfloat* w = kind == 0 ? QW : KW;
  float y[PER];
  for (int i = 0; i < PER; i++) y[i] = float(bfloat(float(w[int(lane) + 32 * i]) * float(bfloat(xv[i] * inv))));
  for (int i = 0; i < PR; i++) y[i] = float(bfloat(y[i] * MS));
  const float pos = float(POS[r]);
  for (int i = 0; i < PR / 2; i++) {
    const float theta = pos * INVF[int(lane) + 32 * i];
    const float c = metal::fast::cos(theta), s = metal::fast::sin(theta);
    const float x1 = y[i], x2 = y[i + PR / 2];
    y[i] = x1 * c - x2 * s;
    y[i + PR / 2] = x1 * s + x2 * c;
  }
  if (kind == 0)
    for (int i = 0; i < PER; i++) Q[(r * NQ + head) * DH + int(lane) + 32 * i] = bfloat(y[i]);
  else
    for (int i = 0; i < PER; i++) K[(head * R + r) * DH + int(lane) + 32 * i] = bfloat(y[i]);
"""

# threadgroup (block, row): simdgroup g outputs RPS (SG block + g) .. of the row; lane l inputs 8 l .. of each 256
_ROUTER = r"""
  const uint lane = thread_index_in_simdgroup;
  const int r = int(threadgroup_position_in_grid.y);
  const int row0 = (int(threadgroup_position_in_grid.x) * SG + int(simdgroup_index_in_threadgroup)) * RPS;
  const device bfloat* w = W + size_t(row0) * K + lane * 8;
  const device bfloat* x = X + size_t(r) * K + lane * 8;
  float acc[RPS];
  for (int j = 0; j < RPS; j++) acc[j] = 0.0f;
  for (int k0 = 0; k0 < K; k0 += 256) {
    float xt[8];
    for (int i = 0; i < 8; i++) xt[i] = float(x[i]);
    for (int j = 0; j < RPS; j++)
      for (int i = 0; i < 8; i++) acc[j] = fma(xt[i], float(w[j * K + i]), acc[j]);
    w += 256; x += 256;
  }
  for (int j = 0; j < RPS; j++) {
    const float total = simd_sum(acc[j]);
    if (lane == 0) OUT[size_t(r) * N + row0 + j] = bfloat(total);
  }
"""

# one simdgroup a row: sigmoid scores, top K of score + bias (ties to the lower id), weights = scores / their sum
_ROUTE = r"""
  const uint lane = thread_index_in_simdgroup;
  const uint r = threadgroup_position_in_grid.x;
  float sc[NE / 32], sel[NE / 32];
  for (int j = 0; j < NE / 32; j++) {
    float g = float(G[int(r) * NE + int(lane) + 32 * j]);
    if (CAP > 0.0f) g = metal::precise::tanh(g / CAP) * CAP;
    sc[j] = 1.0f / (1.0f + metal::precise::exp(-g));
    sel[j] = sc[j] + float(BIAS[int(lane) + 32 * j]);
  }
  float picked[K];
  int ids[K];
  for (int k = 0; k < K; k++) {
    float best = -INFINITY;
    int best_e = 1 << 20;
    for (int j = 0; j < NE / 32; j++) {
      if (sel[j] > best) { best = sel[j]; best_e = int(lane) + 32 * j; }
    }
    const float top = simd_max(best);
    const int winner = simd_min(best == top ? best_e : (1 << 20));
    float mine = 0.0f;
    for (int j = 0; j < NE / 32; j++) {
      if (int(lane) + 32 * j == winner) { mine = sc[j]; sel[j] = -INFINITY; }
    }
    picked[k] = simd_sum(mine);
    ids[k] = winner;
  }
  if (lane == 0) {
    float total = 0.0f;
    for (int k = 0; k < K; k++) total += picked[k];
    for (int k = 0; k < K; k++) {
      IDX[int(r) * K + k] = uint(ids[k]);
      WT[int(r) * K + k] = picked[k] / total;
    }
  }
"""

_SILU_HEADER = gemma_moe._QDOT_HEADER + r"""
// mlx_lm's swiglu on bf16: silu(g) = g * sigmoid(g), then times up
inline float silu_up(float g, float u) {
  const float s = float(bfloat(1.0f / (1.0f + metal::precise::exp(-g))));
  return float(bfloat(g * s)) * u;
}
"""


def _replaced(source: str, old: str, new: str) -> str:
    if old not in source:
        raise RuntimeError("Gemma's expert kernel source changed: Laguna's variant needs updating")
    return source.replace(old, new)


_GATEUP = _replaced(gemma_moe._EXPERT_GATEUP, "bfloat(gelu_tanh(float(bfloat(gv))) * float(bfloat(uv)))",
                    "bfloat(silu_up(float(bfloat(gv)), float(bfloat(uv))))")
# mlx_lm: (y * weights).sum(-2) in fp32 (the weights are fp32), then bf16
_DOWN = _replaced(gemma_moe._EXPERT_DOWN, "routed += float(bfloat(ys[kk][lane] * float(WT[r * TOPK + kk])));",
                  "routed += ys[kk][lane] * float(WT[r * TOPK + kk]);")

_prep = Kernel("laguna_qkv_prep", _QKV_PREP, ["QKV", "QW", "KW", "INVF", "POS", "eps"], ["Q", "K", "V"])
_router = Kernel("laguna_router", _ROUTER, ["X", "W"], ["OUT"])
_route = Kernel("laguna_route", _ROUTE, ["G", "BIAS"], ["IDX", "WT"])
_gateup = Kernel("laguna_expert_gateup", _GATEUP, ["X", "IDX", "GW", "GSC", "GBI", "UW", "USC", "UBI"], ["ACT"],
                 header=_SILU_HEADER)
_down = Kernel("laguna_expert_down", _DOWN, ["ACT", "IDX", "WT", "DW", "DSC", "DBI"], ["OUT"],
               header=gemma_moe._QDOT_HEADER)


def qkv_prep(qkv: mx.array, q_w: mx.array, k_w: mx.array, inv_freq: mx.array, positions: mx.array, eps: mx.array, *,
             heads: int, kv_heads: int, head_dim: int, rotated: int, mscale: float = 1.0
             ) -> tuple[mx.array, mx.array, mx.array]:
    """Stacked q|k|v [R, W] -> q [R, H, Dh], k and v [Hk, R, Dh] (bf16): head norms, RoPE of the first ``rotated``."""

    rows, width = qkv.shape
    if head_dim % 64 or rotated % 64 or not 0 < rotated <= head_dim:
        raise ValueError("qkv_prep: head_dim and the rotated dims must be multiples of 64")
    if width != (heads + 2 * kv_heads) * head_dim:
        raise ValueError(f"qkv_prep: {width} columns for {heads} + 2 x {kv_heads} heads of {head_dim}")
    consts = (("DH", head_dim), ("RD", rotated), ("NQ", heads), ("NK", kv_heads), ("W", width),
              ("MS", float(mscale)))
    return _prep(consts, inputs=[qkv, q_w, k_w, inv_freq, positions, eps],
                 grid=(32 * (heads + 2 * kv_heads), rows, 1), threadgroup=(32, 1, 1),
                 output_shapes=[(rows, heads, head_dim), (kv_heads, rows, head_dim), (kv_heads, rows, head_dim)],
                 output_dtypes=[mx.bfloat16, mx.bfloat16, mx.bfloat16])


def router_logits(x: mx.array, weight: mx.array, *, simdgroups: int = 4, rows_per_simdgroup: int = 2) -> mx.array:
    """x [R, K] bf16 times the bf16 router weight [E, K] transposed -> [R, E] bf16."""

    rows, dims = x.shape
    experts = int(weight.shape[0])
    block = simdgroups * rows_per_simdgroup
    if weight.dtype != mx.bfloat16 or x.dtype != mx.bfloat16:
        raise ValueError("router_logits reads bf16 inputs and a bf16 weight")
    if dims % 256 or experts % block or int(weight.shape[1]) != dims:
        raise ValueError(f"router_logits: needs K a multiple of 256 and E a multiple of {block}")
    consts = (("K", dims), ("N", experts), ("SG", simdgroups), ("RPS", rows_per_simdgroup))
    return _router(consts, inputs=[x, weight], grid=((experts // block) * 32 * simdgroups, rows, 1),
                   threadgroup=(32 * simdgroups, 1, 1), output_shapes=[(rows, experts)],
                   output_dtypes=[mx.bfloat16])[0]


def route(logits: mx.array, bias: mx.array, top_k: int, softcap: float = 0.0) -> tuple[mx.array, mx.array]:
    """Logits [R, E] -> ids (best first) and fp32 weights, row r's at [r K, r K + K), padded to MIN_ELEMENTS."""

    rows, experts = logits.shape
    if experts % 32:
        raise ValueError("route: the expert count must be a multiple of 32")
    size = max(rows * top_k, MIN_ELEMENTS)
    return _route((("NE", experts), ("K", int(top_k)), ("CAP", float(softcap))), inputs=[logits, bias],
                  grid=(32 * rows, 1, 1), threadgroup=(32, 1, 1),
                  output_shapes=[(size,), (size,)], output_dtypes=[mx.uint32, mx.float32])


def expert_gateup(x: mx.array, ids: mx.array, top_k: int, gate: Any, up: Any, *, simdgroups: int = 2,
                  rows_per_simdgroup: int = 4) -> mx.array:
    """x [R, K] and ``route``'s ids -> bf16(silu(x W_gate^T) * (x W_up^T)) for every (row, slot): [R * TOPK, N]."""

    gemma_moe.check_q4(gate)
    gemma_moe.check_q4(up)
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
    """act [R * TOPK, NI], ``route``'s ids and fp32 weights -> bf16(sum_k w_k * bf16(act_k W_down^T)): [R, D]."""

    gemma_moe.check_q4(down)
    rows = int(act.shape[0]) // top_k
    inner = act.shape[-1]
    dims = down.weight.shape[1]
    if inner % 16 or not 32 <= inner // 16 <= 64 or dims % 8 or 32 * top_k > 1024:
        raise ValueError("expert_down: the expert width must be 512 to 1,024 in chunks of 16, at most 32 experts")
    consts = (("NI", inner), ("D", dims), ("TOPK", top_k), ("GS", down.group_size))
    return _down(consts, inputs=[act, ids, weights, down.weight, down.scales, down.biases],
                 grid=(32 * top_k, dims // 8, rows), threadgroup=(32 * top_k, 1, 1),
                 output_shapes=[(rows, dims)], output_dtypes=[mx.bfloat16])[0]


__all__ = ["expert_down", "expert_gateup", "qkv_prep", "route", "router_logits"]
