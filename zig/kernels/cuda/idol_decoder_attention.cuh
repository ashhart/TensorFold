// The decoders' tiled attention: full<D, WARPS>'s row arithmetic (P to nearest fp16 for P V), queries to other keys.
#pragma once

#include "idol_attention.cuh"
#include "idol_decoder.cuh"

namespace tf_idol_dec {

using tf_idol_attn::BN;
using tf_idol_attn::mma;
using tf_idol_attn::mma0;
using tf_idol_attn::pack;
using tf_idol_attn::rowsum;
using tf_idol_attn::tdiv;
using tf_idol_attn::texp;

// o [LQ, HO D] = softmax(q k^T scale) v, v at column (HO + h) D of k's rows. BREAK 1: keys past LK kept; 2: fp16 P V.
template <int D, int WARPS, int BREAK>
__device__ __forceinline__ void rows(const __half* __restrict__ qp, int qs, const __half* __restrict__ kvp, int ks,
                                     __half* __restrict__ out, int LQ, int LK, int HO, float scale) {
  static_assert(D % 16 == 0, "whole 16-dim steps");
  constexpr int C = D / 8, RS = C + 1, K16 = D / 16, NT = BN / 8, THREADS = 32 * WARPS;
  extern __shared__ uint4 smem[];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, qd = lane % 4, mt = lane / 8;
  const int head = blockIdx.y, wrow = (blockIdx.x * WARPS + warp) * 16, last = LK - 1, tiles = (LK + BN - 1) / BN;
  const __half* __restrict__ q = qp + head * D;
  const __half* __restrict__ kc = kvp + head * D;
  const __half* __restrict__ vc = kvp + (HO + head) * D;
  auto stage = [&](int t) {  // tile t's K rows then V rows into buffer t % 2
    uint4* dst = smem + (t & 1) * 2 * BN * RS;
    for (int i = threadIdx.x; i < 2 * BN * C; i += THREADS) {
      const int kv = i / (BN * C), r = (i / C) % BN, c = i % C, key = t * BN + r;
      const bool in = key < LK;
      qmm_frag::cp16z(dst + (kv * BN + r) * RS + c, (kv ? vc : kc) + (int64_t)(in ? key : 0) * ks + c * 8, in);
    }
    qmm_frag::commit();
  };
  uint32_t qa[K16][4];
  const int r0 = wrow + g, r1 = r0 + 8;
  {
    const uint32_t* q0 = reinterpret_cast<const uint32_t*>(q + (int64_t)min(r0, LQ - 1) * qs);
    const uint32_t* q1 = reinterpret_cast<const uint32_t*>(q + (int64_t)min(r1, LQ - 1) * qs);
#pragma unroll
    for (int k = 0; k < K16; ++k) {
      qa[k][0] = r0 < LQ ? q0[8 * k + qd] : 0u;
      qa[k][1] = r1 < LQ ? q1[8 * k + qd] : 0u;
      qa[k][2] = r0 < LQ ? q0[8 * k + qd + 4] : 0u;
      qa[k][3] = r1 < LQ ? q1[8 * k + qd + 4] : 0u;
    }
  }
  float o[C][4];
#pragma unroll
  for (int i = 0; i < C; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
  stage(0);
  for (int t = 0; t < tiles; ++t) {
    qmm_frag::wait<0>();
    __syncthreads();  // tile t landed, and every warp is done with tile t - 1
    if (t + 1 < tiles) stage(t + 1);
    const uint4* kt = smem + (t & 1) * 2 * BN * RS;
    const uint4* vt = kt + BN * RS;
    float sc[NT][4];
#pragma unroll
    for (int k = 0; k < K16; ++k) {
#pragma unroll
      for (int n = 0; n < NT; n += 2) {
        uint32_t b[4];
        qmm_frag::ldmatrix4(b, kt + (8 * (n + mt / 2) + lane % 8) * RS + 2 * k + mt % 2);
        if (k == 0) {
          mma0(sc[n], qa[k], b[0], b[1]);
          mma0(sc[n + 1], qa[k], b[2], b[3]);
        } else {
          mma(sc[n], qa[k], b[0], b[1]);
          mma(sc[n + 1], qa[k], b[2], b[3]);
        }
      }
    }
    // keys past LK: their K rows read as zeros, so unmasked (BREAK 1) they score 0 and join the softmax
    const int kb = t * BN + 2 * qd, end = BREAK == 1 ? tiles * BN - 1 : last;
    float tm0 = -INFINITY, tm1 = -INFINITY;
#pragma unroll
    for (int n = 0; n < NT; ++n) {
      const int c = kb + 8 * n;
      sc[n][0] = c <= end ? __fmul_rn(sc[n][0], scale) : -INFINITY;
      sc[n][1] = c + 1 <= end ? __fmul_rn(sc[n][1], scale) : -INFINITY;
      sc[n][2] = c <= end ? __fmul_rn(sc[n][2], scale) : -INFINITY;
      sc[n][3] = c + 1 <= end ? __fmul_rn(sc[n][3], scale) : -INFINITY;
      tm0 = fmaxf(tm0, fmaxf(sc[n][0], sc[n][1]));
      tm1 = fmaxf(tm1, fmaxf(sc[n][2], sc[n][3]));
    }
    for (int x = 1; x <= 2; x *= 2) {
      tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, x));
      tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, x));
    }
    const bool act0 = tm0 != -INFINITY, act1 = tm1 != -INFINITY;
    const float n0 = act0 ? fmaxf(m0, tm0) : m0, n1 = act1 ? fmaxf(m1, tm1) : m1;
    const float a0 = act0 ? (m0 == -INFINITY ? 0.0f : texp(__fsub_rn(m0, n0))) : 1.0f;
    const float a1 = act1 ? (m1 == -INFINITY ? 0.0f : texp(__fsub_rn(m1, n1))) : 1.0f;
    float c0[NT], c1[NT];
#pragma unroll
    for (int n = 0; n < NT; ++n) {
      const int c = kb + 8 * n;
      sc[n][0] = act0 && c <= end ? texp(__fsub_rn(sc[n][0], n0)) : 0.0f;
      sc[n][1] = act0 && c + 1 <= end ? texp(__fsub_rn(sc[n][1], n0)) : 0.0f;
      sc[n][2] = act1 && c <= end ? texp(__fsub_rn(sc[n][2], n1)) : 0.0f;
      sc[n][3] = act1 && c + 1 <= end ? texp(__fsub_rn(sc[n][3], n1)) : 0.0f;
      c0[n] = __fadd_rn(sc[n][0], sc[n][1]);
      c1[n] = __fadd_rn(sc[n][2], sc[n][3]);
    }
    l0 = __fmaf_rn(l0, a0, rowsum(c0));
    l1 = __fmaf_rn(l1, a1, rowsum(c1));
    m0 = n0;
    m1 = n1;
#pragma unroll
    for (int i = 0; i < C; ++i) {
      o[i][0] = __fmul_rn(o[i][0], a0);
      o[i][1] = __fmul_rn(o[i][1], a0);
      o[i][2] = __fmul_rn(o[i][2], a1);
      o[i][3] = __fmul_rn(o[i][3], a1);
    }
#pragma unroll
    for (int j = 0; j < NT / 2; ++j) {  // 16-key chunks, in key order for every output tile
      const uint32_t pa[4] = {pack(sc[2 * j][0], sc[2 * j][1]), pack(sc[2 * j][2], sc[2 * j][3]),
                              pack(sc[2 * j + 1][0], sc[2 * j + 1][1]), pack(sc[2 * j + 1][2], sc[2 * j + 1][3])};
#pragma unroll
      for (int i = 0; i < C; i += 2) {
        uint32_t b[4];
        qmm_frag::ldmatrix4t(b, vt + (16 * j + 8 * (mt % 2) + lane % 8) * RS + i + mt / 2);
        mma(o[i], pa, b[0], b[1]);
        mma(o[i + 1], pa, b[2], b[3]);
        if constexpr (BREAK == 2) {
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            o[i][e] = h(o[i][e]);
            o[i + 1][e] = h(o[i + 1][e]);
          }
        }
      }
    }
  }
#pragma unroll
  for (int i = 0; i < C; ++i) {
    const int d = 8 * i + 2 * qd;
    if (r0 < LQ)
      *reinterpret_cast<uint32_t*>(out + ((int64_t)r0 * HO + head) * D + d) = pack(tdiv(o[i][0], l0), tdiv(o[i][1], l0));
    if (r1 < LQ)
      *reinterpret_cast<uint32_t*>(out + ((int64_t)r1 * HO + head) * D + d) = pack(tdiv(o[i][2], l1), tdiv(o[i][3], l1));
  }
}

}  // namespace tf_idol_dec
