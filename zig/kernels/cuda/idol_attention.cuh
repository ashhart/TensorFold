// Idol's attention: vision_kernel's row arithmetic over fp16 q, k, v and P (fp32 scores, softmax, sums), its own copy.
#pragma once

#include <cuda_fp16.h>
#include <math.h>
#include <stdint.h>

#include "common/qmm_frag.cuh"

namespace tf_idol_attn {

using qmm_frag::commit;
using qmm_frag::cp16z;
using qmm_frag::ldmatrix4;
using qmm_frag::ldmatrix4t;

constexpr int BN = 64;  // keys a tile
constexpr float LOG2E = 1.4426950408889634f;

__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma0(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  d[0] = d[1] = d[2] = d[3] = 0.0f;
  mma(d, a, b0, b1);
}

__device__ __forceinline__ float texp(float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;\n" : "=f"(y) : "f"(__fmul_rn(x, LOG2E)));
  return y;
}

__device__ __forceinline__ float tdiv(float a, float b) {
  float y;
  asm("div.full.f32 %0, %1, %2;\n" : "=f"(y) : "f"(a), "f"(b));
  return y;
}

__device__ __forceinline__ uint32_t pack(float lo, float hi) {
  __half2 v = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}

__device__ __forceinline__ float rowsum(const float (&c)[8]) {
  float w[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const float x = __fadd_rn(c[j], __shfl_xor_sync(0xffffffffu, c[j], 2));
    w[j] = __fadd_rn(x, __shfl_xor_sync(0xffffffffu, x, 1));
  }
  return __fadd_rn(__fadd_rn(__fadd_rn(w[0], w[4]), __fadd_rn(w[2], w[6])),
                   __fadd_rn(__fadd_rn(w[1], w[5]), __fadd_rn(w[3], w[7])));
}

// Every row over every key, a block a head and 16 WARPS rows: qkv [W, 3 HO D] fp16 -> out [W, HO D] fp16.
template <int D, int WARPS>
__device__ __forceinline__ void full(const __half* __restrict__ qkv, __half* __restrict__ out, int W, int HO,
                                     float scale) {
  static_assert(D % 16 == 0, "whole 16-dim steps");
  constexpr int C = D / 8, RS = C + 1, K16 = D / 16, NT = BN / 8, THREADS = 32 * WARPS;
  extern __shared__ uint4 smem[];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, qd = lane % 4, mt = lane / 8;
  const int head = blockIdx.y, wrow = (blockIdx.x * WARPS + warp) * 16, last = W - 1, tiles = (W + BN - 1) / BN;
  const int64_t stride = 3LL * HO * D;
  const __half* __restrict__ q = qkv + head * D;
  const __half* __restrict__ kc = qkv + (HO + head) * D;
  const __half* __restrict__ vc = qkv + (2 * HO + head) * D;
  auto stage = [&](int t) {  // tile t's K rows then V rows into buffer t % 2
    uint4* dst = smem + (t & 1) * 2 * BN * RS;
    for (int i = threadIdx.x; i < 2 * BN * C; i += THREADS) {
      const int kv = i / (BN * C), r = (i / C) % BN, c = i % C, key = t * BN + r;
      const bool in = key < W;
      cp16z(dst + (kv * BN + r) * RS + c, (kv ? vc : kc) + (int64_t)(in ? key : 0) * stride + c * 8, in);
    }
    commit();
  };
  uint32_t qa[K16][4];
  const int r0 = wrow + g, r1 = r0 + 8;
  {
    const uint32_t* q0 = reinterpret_cast<const uint32_t*>(q + (int64_t)min(r0, last) * stride);
    const uint32_t* q1 = reinterpret_cast<const uint32_t*>(q + (int64_t)min(r1, last) * stride);
#pragma unroll
    for (int k = 0; k < K16; ++k) {
      qa[k][0] = r0 < W ? q0[8 * k + qd] : 0u;
      qa[k][1] = r1 < W ? q1[8 * k + qd] : 0u;
      qa[k][2] = r0 < W ? q0[8 * k + qd + 4] : 0u;
      qa[k][3] = r1 < W ? q1[8 * k + qd + 4] : 0u;
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
    const uint4* ks = smem + (t & 1) * 2 * BN * RS;
    const uint4* vs = ks + BN * RS;
    float sc[NT][4];
#pragma unroll
    for (int k = 0; k < K16; ++k) {
#pragma unroll
      for (int n = 0; n < NT; n += 2) {
        uint32_t b[4];
        ldmatrix4(b, ks + (8 * (n + mt / 2) + lane % 8) * RS + 2 * k + mt % 2);
        if (k == 0) {
          mma0(sc[n], qa[k], b[0], b[1]);
          mma0(sc[n + 1], qa[k], b[2], b[3]);
        } else {
          mma(sc[n], qa[k], b[0], b[1]);
          mma(sc[n + 1], qa[k], b[2], b[3]);
        }
      }
    }
    const int kb = t * BN + 2 * qd;
    float tm0 = -INFINITY, tm1 = -INFINITY;
#pragma unroll
    for (int n = 0; n < NT; ++n) {
      const int c = kb + 8 * n;
      sc[n][0] = c <= last ? __fmul_rn(sc[n][0], scale) : -INFINITY;
      sc[n][1] = c + 1 <= last ? __fmul_rn(sc[n][1], scale) : -INFINITY;
      sc[n][2] = c <= last ? __fmul_rn(sc[n][2], scale) : -INFINITY;
      sc[n][3] = c + 1 <= last ? __fmul_rn(sc[n][3], scale) : -INFINITY;
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
      sc[n][0] = act0 && c <= last ? texp(__fsub_rn(sc[n][0], n0)) : 0.0f;
      sc[n][1] = act0 && c + 1 <= last ? texp(__fsub_rn(sc[n][1], n0)) : 0.0f;
      sc[n][2] = act1 && c <= last ? texp(__fsub_rn(sc[n][2], n1)) : 0.0f;
      sc[n][3] = act1 && c + 1 <= last ? texp(__fsub_rn(sc[n][3], n1)) : 0.0f;
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
        ldmatrix4t(b, vs + (16 * j + 8 * (mt % 2) + lane % 8) * RS + i + mt / 2);
        mma(o[i], pa, b[0], b[1]);
        mma(o[i + 1], pa, b[2], b[3]);
      }
    }
  }
#pragma unroll
  for (int i = 0; i < C; ++i) {
    const int d = 8 * i + 2 * qd;
    if (r0 < W)
      *reinterpret_cast<uint32_t*>(out + ((int64_t)r0 * HO + head) * D + d) = pack(tdiv(o[i][0], l0), tdiv(o[i][1], l0));
    if (r1 < W)
      *reinterpret_cast<uint32_t*>(out + ((int64_t)r1 * HO + head) * D + d) = pack(tdiv(o[i][2], l1), tdiv(o[i][3], l1));
  }
}

}  // namespace tf_idol_attn
