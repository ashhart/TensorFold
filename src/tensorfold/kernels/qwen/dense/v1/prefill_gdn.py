"""Prompt Gated DeltaNet scans with mlx_lm's bits: four state rows a simdgroup share each step's loads, 1.39x faster."""

from __future__ import annotations

from typing import Any

import mlx.core as mx

ROWS = 4        # state rows a simdgroup: four independent chains hide each step's two reductions (M5 Max: 1.39x)
GROUPS = 4      # simdgroups a threadgroup

# Kernel source comments belong to the decoder version hash and must change only with the arithmetic.
_SCAN = r"""
  // R state rows a simdgroup, lane l their columns dk = NPT l .. NPT l + NPT - 1 (mlx_lm's layout); each row's
  // arithmetic is mlx_lm's gated_delta_step, verbatim, and the next step's inputs load while this step runs
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const uint n = threadgroup_position_in_grid.z;
  const uint b_idx = n / Hv;
  const uint hv_idx = n % Hv;
  const uint hk_idx = hv_idx / (Hv / Hk);
  constexpr int NPT = Dk / 32;
  const int dv0 = (int(threadgroup_position_in_grid.y) * SG + int(sg)) * R;
  const int TT = T;
  const device InT* q_ = q + b_idx * TT * Hk * Dk + hk_idx * Dk + NPT * lane;
  const device InT* k_ = k + b_idx * TT * Hk * Dk + hk_idx * Dk + NPT * lane;
  const device InT* v_ = v + b_idx * TT * Hv * Dv + hv_idx * Dv + dv0;
  device InT* y_ = y + b_idx * TT * Hv * Dv + hv_idx * Dv + dv0;
  auto g_ = g + b_idx * TT * Hv + hv_idx;
  auto beta_ = beta + b_idx * TT * Hv + hv_idx;
  float state[R][NPT];
  for (int j = 0; j < R; j++)
    for (int i = 0; i < NPT; i++)
      state[j][i] = static_cast<float>(state_in[(n * Dv + dv0 + j) * Dk + NPT * lane + i]);
  InT qn[NPT], kn[NPT], vn[R];
  auto gn = g_[0];
  auto bn = beta_[0];
  for (int i = 0; i < NPT; i++) { qn[i] = q_[i]; kn[i] = k_[i]; }
  for (int j = 0; j < R; j++) vn[j] = v_[j];
  for (int t = 0; t < TT; ++t) {
    InT qq[NPT], kk[NPT], vv[R];
    for (int i = 0; i < NPT; i++) { qq[i] = qn[i]; kk[i] = kn[i]; }
    for (int j = 0; j < R; j++) vv[j] = vn[j];
    const auto gt = gn;
    const auto bt = bn;
    if (t + 1 < TT) {
      q_ += Hk * Dk; k_ += Hk * Dk; v_ += Hv * Dv; g_ += Hv; beta_ += Hv;
      for (int i = 0; i < NPT; i++) { qn[i] = q_[i]; kn[i] = k_[i]; }
      for (int j = 0; j < R; j++) vn[j] = v_[j];
      gn = g_[0];
      bn = beta_[0];
    }
    for (int j = 0; j < R; j++) {
      float kv_mem = 0.0f;
      for (int i = 0; i < NPT; ++i) {
        state[j][i] = state[j][i] * gt;
        kv_mem += state[j][i] * kk[i];
      }
      kv_mem = simd_sum(kv_mem);
      auto delta = (vv[j] - kv_mem) * bt;
      float out = 0.0f;
      for (int i = 0; i < NPT; ++i) {
        state[j][i] = state[j][i] + kk[i] * delta;
        out += state[j][i] * qq[i];
      }
      out = simd_sum(out);
      if (lane == uint(j)) y_[j] = static_cast<InT>(out);
    }
    y_ += Hv * Dv;
  }
  for (int j = 0; j < R; j++)
    for (int i = 0; i < NPT; i++)
      state_out[(n * Dv + dv0 + j) * Dk + NPT * lane + i] = static_cast<StT>(state[j][i]);
"""

_kernel: Any = None
_STOCK: Any = None


def takes(q: mx.array, k: mx.array, v: mx.array, g: mx.array, mask: Any) -> bool:
    """Scalar gates, no mask, heads of a multiple of 32 keys and ROWS * GROUPS values: other calls stay mlx_lm's."""

    if mask is not None or g.ndim != 3 or q.ndim != 4 or v.ndim != 4:
        return False
    hk, dk = int(k.shape[2]), int(k.shape[3])
    hv, dv = int(v.shape[2]), int(v.shape[3])
    return dk % 32 == 0 and dv % (ROWS * GROUPS) == 0 and hv % hk == 0 and int(k.shape[1]) > 0


def scan(q: mx.array, k: mx.array, v: mx.array, g: mx.array, beta: mx.array, state: mx.array,
         mask: Any = None) -> tuple[mx.array, mx.array]:
    """``mlx_lm.models.gated_delta.gated_delta_kernel``, the same bits: y [B, T, Hv, Dv] and the final state."""

    global _kernel
    if not takes(q, k, v, g, mask):
        return _STOCK(q, k, v, g, beta, state, mask)
    if _kernel is None:
        _kernel = mx.fast.metal_kernel(name="tf_prefill_gdn_scan", input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
                                       output_names=["y", "state_out"], source=_SCAN)
    B, T, Hk, Dk = (int(s) for s in k.shape)
    Hv, Dv = int(v.shape[2]), int(v.shape[3])
    return _kernel(
        inputs=[q, k, v, g, beta, state, T],
        template=[("InT", q.dtype), ("StT", state.dtype), ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv),
                  ("R", ROWS), ("SG", GROUPS)],
        grid=(32 * GROUPS, Dv // (ROWS * GROUPS), B * Hv), threadgroup=(32 * GROUPS, 1, 1),
        output_shapes=[(B, T, Hv, Dv), state.shape], output_dtypes=[q.dtype, state.dtype])


def install() -> None:
    """Route mlx_lm's prompt scans (``gated_delta_update`` on the GPU) through ``scan``. Idempotent."""

    global _STOCK
    from mlx_lm.models import gated_delta

    if _STOCK is None:
        _STOCK = gated_delta.gated_delta_kernel
    gated_delta.gated_delta_kernel = scan


def uninstall() -> None:
    if _STOCK is not None:
        from mlx_lm.models import gated_delta

        gated_delta.gated_delta_kernel = _STOCK


__all__ = ["GROUPS", "ROWS", "install", "scan", "takes", "uninstall"]
