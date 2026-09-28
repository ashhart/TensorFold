"""A hyper-connection boundary in three kernels, each in the row-by-row path's partitions and order."""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from tensorfold.kernels.glm.flash.v1.fused import SQ_FMA, _kernel

HC_MIX_U = 8             # iterations of loads the hyper-connection mix issues ahead

_HC_COMMON = r"""
  constexpr int S = 4;
  constexpr int F = S * D;                          // flattened streams
  constexpr int MIX = (2 + S) * S;                  // 24 mixes
"""

# write-back, stream RMS, mix, sinkhorn split and RMSNorm, each in the row-by-row path's MLX partition and order
_HC_EXPAND = _HC_COMMON + r"""
  // Threadgroup r (1024 threads): the pending write-back (EXPAND) and the streams' RMS scale (SPLIT).
  const int r = int(threadgroup_position_in_grid.x);
  const uint t = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
  threadgroup float red[32];
  device const bfloat* xo = XOLD + size_t(r) * F;
  device bfloat* xn = XNEW + size_t(r) * F;
  float ss = 0.0f;
  for (int k = 0; k < F / 4096; ++k) {
    for (int i = 0; i < 4; ++i) {
      const int f = int(t) * 4 + 4096 * k + i;
      float v;
      if (EXPAND) {
        const int s = f / D, d = f - s * D;
        const float y = POST[r * S + s] * float(BRANCH[size_t(r) * D + d]);
        const device float* c = COMB + r * S * S;
        float mm = c[0 * S + s] * float(xo[0 * D + d]);
        mm = fma(c[1 * S + s], float(xo[1 * D + d]), mm);
        mm = fma(c[2 * S + s], float(xo[2 * D + d]), mm);
        mm = fma(c[3 * S + s], float(xo[3 * D + d]), mm);
        const bfloat nb = bfloat(add_nc(y, mm));
        xn[f] = nb;
        v = float(nb);
      } else {
        v = float(xo[f]);
      }
      ss = sq_acc<SQ_FMA>(ss, v);
    }
  }
  if (!SPLIT) return;
  ss = simd_sum(ss);
  if (sg == 0) red[lane] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) red[sg] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    const float a = simd_sum(red[lane]);
    if (lane == 0) INV[r] = metal::precise::rsqrt(a / float(F) + EPS[0]);
  }
"""

_HC_MIX = _HC_COMMON + r"""
  // Threadgroup (og, r): MLX's gemv for mixes og 4 .. og 4 + 3 of row r on z = x inv (the rms_norm output).
  const int og = int(threadgroup_position_in_grid.x);
  const int r = int(threadgroup_position_in_grid.y);
  const uint lane = thread_index_in_simdgroup, sgn = simdgroup_index_in_threadgroup;
  threadgroup float part[8][4];
  const float inv = INV[r];
  device const bfloat* xs = X + size_t(r) * F;
  float res[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int bn = (32 * int(sgn) + int(lane)) * 4; bn < F; bn += 1024) {
    float vc[4];
    for (int tn = 0; tn < 4; tn++) vc[tn] = float(xs[bn + tn]) * inv;
    for (int tm = 0; tm < 4; tm++) {
      const device float* mrow = FN + size_t(og * 4 + tm) * F;
      float inter[4];
      for (int tn = 0; tn < 4; tn++) inter[tn] = mrow[bn + tn];
      for (int tn = 0; tn < 4; tn++) res[tm] += inter[tn] * vc[tn];
    }
  }
  for (int tm = 0; tm < 4; tm++)
    for (ushort sn = 16; sn >= 1; sn >>= 1) res[tm] += simd_shuffle_down(res[tm], sn);
  if (lane == 0) for (int tm = 0; tm < 4; tm++) part[sgn][tm] = res[tm];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sgn == 0 && lane < 4) {
    float a = part[0][lane];
    for (int k = 1; k < 8; k++) a += part[k][lane];
    MIXES[r * MIX + og * 4 + lane] = a;
  }
"""

_HC_MIX_PACKED = _HC_COMMON + r"""
  // _HC_MIX's arithmetic on the bf16 matrix repacked per thread, loads issued U iterations ahead
  const int og = int(threadgroup_position_in_grid.x);
  const int r = int(threadgroup_position_in_grid.y);
  const uint lane = thread_index_in_simdgroup, sgn = simdgroup_index_in_threadgroup;
  constexpr int ITERS = F / 1024;
  threadgroup float part[8][4];
  const float inv = INV[r];
  device const bfloat* xs = X + size_t(r) * F;
  const device uint4* w = (const device uint4*)(FNP + ((size_t(og) * 8 + sgn) * 32 + lane) * ITERS * 16);
  const int bn0 = (32 * int(sgn) + int(lane)) * 4;
  float res[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int i0 = 0; i0 < ITERS; i0 += U) {
    uint4 raw[U][2];
    float xv[U][4];
    for (int u = 0; u < U; u++) {
      raw[u][0] = w[(i0 + u) * 2]; raw[u][1] = w[(i0 + u) * 2 + 1];
      for (int tn = 0; tn < 4; tn++) xv[u][tn] = float(xs[bn0 + 1024 * (i0 + u) + tn]);
    }
    for (int u = 0; u < U; u++) {
      float vc[4];
      for (int tn = 0; tn < 4; tn++) vc[tn] = xv[u][tn] * inv;
      float inter[4][4];
      for (int h = 0; h < 2; h++) {
        const uint4 v = raw[u][h];
        const uint words[4] = {v.x, v.y, v.z, v.w};
        for (int j = 0; j < 4; j++) {
          const int e = h * 8 + j * 2;
          inter[e / 4][e % 4] = as_type<float>(words[j] << 16);
          inter[(e + 1) / 4][(e + 1) % 4] = as_type<float>(words[j] & 0xffff0000u);
        }
      }
      for (int tm = 0; tm < 4; tm++)
        for (int tn = 0; tn < 4; tn++) res[tm] += inter[tm][tn] * vc[tn];
    }
  }
  for (int tm = 0; tm < 4; tm++)
    for (ushort sn = 16; sn >= 1; sn >>= 1) res[tm] += simd_shuffle_down(res[tm], sn);
  if (lane == 0) for (int tm = 0; tm < 4; tm++) part[sgn][tm] = res[tm];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sgn == 0 && lane < 4) {
    float a = part[0][lane];
    for (int k = 1; k < 8; k++) a += part[k][lane];
    MIXES[r * MIX + og * 4 + lane] = a;
  }
"""

_HC_SPLIT_NORM = _HC_COMMON + r"""
  // Threadgroup r: sinkhorn, pre / post / comb (hc_sinkhorn_collapse), the collapse and the RMSNorm
  constexpr float HC_EPS = HC_EPS_INT * 1e-9;       // as the hc_split kernel spells its eps
  const int r = int(threadgroup_position_in_grid.x);
  const uint t = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
  threadgroup float red[32];
  threadgroup float pre_s[S];
  threadgroup float inv_s[1];
  device const float* mixes = MIXES + r * MIX;
  device const bfloat* xs = X + size_t(r) * F;
  if (sg == 0) {
    constexpr int BASE_OFF = 2 * S;
    const float pre_scale = SCALE[0], post_scale = SCALE[1], comb_scale = SCALE[2];
    const float active = (lane < (uint)S) ? 1.0f : 0.0f;
    const uint llane = metal::min(lane, (uint)(S - 1));
    const float pre_z = mixes[llane] * pre_scale + BASEV[llane];
    const float post_z = mixes[S + llane] * post_scale + BASEV[S + llane];
    const float pre_v = 1.0f / (1.0f + metal::fast::exp(-pre_z)) + HC_EPS;
    const float post_v = 2.0f / (1.0f + metal::fast::exp(-post_z));
    if (lane < (uint)S) { pre_s[lane] = pre_v; POST_OUT[r * S + lane] = post_v; }
    float4 v = (*(const device float4*)(mixes + BASE_OFF + llane * S) * comb_scale
                + *(const device float4*)(BASEV + BASE_OFF + llane * S)) * active;
    const float row_max = metal::max(metal::max(v.x, v.y), metal::max(v.z, v.w));
    const float4 e = metal::fast::exp(v - row_max) * active;
    float4 rr = e * (1.0f / (e.x + e.y + e.z + e.w + HC_EPS)) + HC_EPS * active;
    float4 col_inv = 1.0f / (float4(simd_sum(rr.x), simd_sum(rr.y), simd_sum(rr.z), simd_sum(rr.w)) + HC_EPS);
    rr *= col_inv;
    for (int iter = 1; iter < ITERS; ++iter) {
      rr *= (1.0f / (rr.x + rr.y + rr.z + rr.w + HC_EPS)) * active;
      col_inv = 1.0f / (float4(simd_sum(rr.x), simd_sum(rr.y), simd_sum(rr.z), simd_sum(rr.w)) + HC_EPS);
      rr *= col_inv;
    }
    if (lane < (uint)S) *(device float4*)(COMB_OUT + r * S * S + lane * S) = rr;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float p0 = pre_s[0], p1 = pre_s[1], p2 = pre_s[2], p3 = pre_s[3];
  float xc[4];
  float acc = 0.0f;
  for (int i = 0; i < 4; ++i) {
    const int d = int(t) * 4 + i;
    const float res = fma(p0, float(xs[d]), fma(p1, float(xs[D + d]),
                          fma(p2, float(xs[2 * D + d]), p3 * float(xs[3 * D + d]))));
    xc[i] = float(bfloat(res));
    acc = sq_acc<SQ_FMA>(acc, xc[i]);
  }
  acc = simd_sum(acc);
  if (sg == 0) red[lane] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) red[sg] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    const float a = simd_sum(red[lane]);
    if (lane == 0) inv_s[0] = metal::precise::rsqrt(a / float(D) + EPS[0]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = 0; i < 4; ++i) {
    const int d = int(t) * 4 + i;
    NORMED[size_t(r) * D + d] = NORMW[d] * bfloat(xc[i] * inv_s[0]);
  }
"""


def hc_fits(hc: Any, dims: int) -> bool:
    cfg = hc.cfg
    return (cfg.hc_mult == 4 and dims == 4096 and int(hc.fn.shape[0]) == 24 and int(hc.fn.shape[1]) == 4 * dims)


def hc_step(x: mx.array, pending: tuple[mx.array, mx.array, mx.array] | None, hc: Any | None, norm_w: mx.array | None,
            eps: float) -> tuple[mx.array, mx.array | None, mx.array | None, mx.array | None]:
    """A block boundary on streams [R, 4, D]: the pending write-back, then the next block's split and RMSNorm."""

    rows, streams, dims = x.shape
    expand, split = pending is not None, hc is not None
    if expand or split:
        branch, post, comb = pending if expand else (x[:, 0], mx.zeros((rows, 4), mx.float32),
                                                       mx.zeros((rows, 4, 4), mx.float32))
        eps_arr = mx.array([hc.cfg.rms_norm_eps if split else eps], dtype=mx.float32)
        k1 = _kernel("hc_expand", _HC_EXPAND, ["XOLD", "BRANCH", "POST", "COMB", "EPS"], ["XNEW", "INV"])
        xn, inv = k1(inputs=[x, branch, post, comb, eps_arr],
                     template=[("D", dims), ("EXPAND", int(expand)), ("SPLIT", int(split)), ("SQ_FMA", SQ_FMA)],
                     grid=(1024 * rows, 1, 1), threadgroup=(1024, 1, 1),
                     output_shapes=[x.shape if expand else (1,), (rows,)], output_dtypes=[mx.bfloat16, mx.float32])
        if expand:
            x = xn
    if not split:
        return x, None, None, None
    if hc.fn_packed is not None:
        k2 = _kernel("hc_mix_packed", _HC_MIX_PACKED, ["X", "INV", "FNP"], ["MIXES"])
        mixes = k2(inputs=[x, inv, hc.fn_packed], template=[("D", dims), ("U", HC_MIX_U)], grid=(6 * 256, rows, 1),
                   threadgroup=(256, 1, 1), output_shapes=[(rows, 24)], output_dtypes=[mx.float32])[0]
    else:
        k2 = _kernel("hc_mix", _HC_MIX, ["X", "INV", "FN"], ["MIXES"])
        mixes = k2(inputs=[x, inv, hc.fn], template=[("D", dims)], grid=(6 * 256, rows, 1), threadgroup=(256, 1, 1),
                   output_shapes=[(rows, 24)], output_dtypes=[mx.float32])[0]
    k3 = _kernel("hc_split_norm", _HC_SPLIT_NORM, ["X", "MIXES", "SCALE", "BASEV", "NORMW", "EPS"],
                 ["NORMED", "POST_OUT", "COMB_OUT"])
    normed, post_o, comb_o = k3(
        inputs=[x, mixes, hc.scale, hc.base, norm_w, eps_arr],
        template=[("D", dims), ("SQ_FMA", SQ_FMA), ("ITERS", hc.cfg.hc_sinkhorn_iters),
                  ("HC_EPS_INT", round(hc.cfg.hc_eps / 1e-9))],
        grid=(1024 * rows, 1, 1), threadgroup=(1024, 1, 1),
        output_shapes=[(rows, dims), (rows, 4), (rows, 4, 4)], output_dtypes=[mx.bfloat16, mx.float32, mx.float32])
    return x, normed, post_o, comb_o


def pack_hc_fn(fn_bf16: mx.array) -> mx.array:
    """The stored bf16 mix matrix [24, 16384] repacked per thread for _HC_MIX_PACKED."""

    m = fn_bf16.reshape(6, 4, 16, 8, 32, 4)                     # [og, tm, i, sgn, lane, tn]
    return mx.contiguous(m.transpose(0, 3, 4, 2, 1, 5))
