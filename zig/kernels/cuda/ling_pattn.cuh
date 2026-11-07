// Ling's MLA prompt rows in the expanded form: q [rows][heads][192] (nope, roped), expanded keys a block at a time.
#pragma once
#include <cuda_bf16.h>
#include <stdint.h>

#include "common/qmm_frag.cuh"

namespace ling_pattn {

constexpr int QK = 192, DV = 128, BN = 64, KC = QK / 8, VC = DV / 8;  // dims, keys a tile, 16-byte chunks a row
constexpr int WARPS = 8, ROWS = 16 * WARPS, STAGE = BN * (KC + VC);   // a block's query rows; uint4 a stage
constexpr float LOG2E = 1.4426950408889634f;

__device__ __forceinline__ float texp(float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;\n" : "=f"(y) : "f"(__fmul_rn(x, LOG2E)));
  return y;
}

__device__ __forceinline__ uint32_t pack(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}

template <int C>
__device__ __forceinline__ int sw(int r, int c) { return r * C + (c ^ (r & 7)); }

// The 4 lanes of a row's quad summed in a fixed order (xor 2, then xor 1), as prefill_attention's rowsum.
__device__ __forceinline__ float quad(float x) {
  x = __fadd_rn(x, __shfl_xor_sync(0xffffffffu, x, 2));
  return __fadd_rn(x, __shfl_xor_sync(0xffffffffu, x, 1));
}

// fmaxf, but a NaN wins (a NaN score or carry makes the row NaN, as the float64 softmax does).
__device__ __forceinline__ float nmax(float a, float b) { return (a > b || a != a) ? a : b; }

}  // namespace ling_pattn

// Block (128 rows, head) over keys kb0 + j < kb0 + n, j <= p0 + r - kb0; fp32 o/m/l carried, last: o / l as bf16.
extern "C" __global__ void __launch_bounds__(256, 1)
    tf_ling_pattn(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kv,
                  const __nv_bfloat16* __restrict__ rk, float* __restrict__ co, float* __restrict__ cm,
                  float* __restrict__ cl, __nv_bfloat16* __restrict__ out, int rows, int heads, int n, int kb0, int p0,
                  float scale, int first, int last) {
  using namespace ling_pattn;
  using qmm_frag::cp16z;
  using qmm_frag::ldmatrix4;
  using qmm_frag::ldmatrix4t;
  using qmm_frag::mma;
  using qmm_frag::mma0;
  extern __shared__ uint4 smem[];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, qd = lane % 4, mt = lane / 8;
  const int h = blockIdx.y, row0 = blockIdx.x * ROWS, wrow = row0 + warp * 16;
  const int lastrow = min(row0 + ROWS, rows) - 1;
  const int span = min(n, p0 + lastrow - kb0 + 1), tiles = (span + BN - 1) / BN;  // keys any row of the block sees
  const int64_t kvrow = (int64_t)heads * 256;
  auto stage = [&](int t, int slot) {
    uint4* ks = smem + slot * STAGE;
    uint4* vs = ks + BN * KC;
    for (int i = threadIdx.x; i < BN * (KC + VC); i += 32 * WARPS) {
      const int key = i < BN * KC ? i / KC : (i - BN * KC) / VC, j = t * BN + key;
      const bool in = j < span;
      const int jj = in ? j : 0;
      if (i < BN * KC) {
        const int c = i % KC;
        const __nv_bfloat16* src = c < 16 ? kv + jj * kvrow + h * 256 + c * 8 : rk + (int64_t)jj * 576 + 512 + (c - 16) * 8;
        cp16z(ks + sw<KC>(key, c), src, in);
      } else {
        const int c = (i - BN * KC) % VC;
        cp16z(vs + sw<VC>(key, c), kv + jj * kvrow + h * 256 + 128 + c * 8, in);
      }
    }
    qmm_frag::commit();
  };
  uint32_t qa[QK / 16][4];
  {
    const int r0 = wrow + g, r1 = r0 + 8;
    const uint32_t* q0 = reinterpret_cast<const uint32_t*>(q + ((int64_t)min(r0, rows - 1) * heads + h) * QK);
    const uint32_t* q1 = reinterpret_cast<const uint32_t*>(q + ((int64_t)min(r1, rows - 1) * heads + h) * QK);
#pragma unroll
    for (int k = 0; k < QK / 16; ++k) {
      qa[k][0] = r0 < rows ? q0[8 * k + qd] : 0u;
      qa[k][1] = r1 < rows ? q1[8 * k + qd] : 0u;
      qa[k][2] = r0 < rows ? q0[8 * k + qd + 4] : 0u;
      qa[k][3] = r1 < rows ? q1[8 * k + qd + 4] : 0u;
    }
  }
  const int r0 = wrow + g, r1 = r0 + 8;
  float o[VC][4];
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
#pragma unroll
  for (int i = 0; i < VC; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
  if (!first) {  // the carry of the earlier key blocks
    if (r0 < rows) {
      m0 = cm[(int64_t)r0 * heads + h];
      l0 = cl[(int64_t)r0 * heads + h];
    }
    if (r1 < rows) {
      m1 = cm[(int64_t)r1 * heads + h];
      l1 = cl[(int64_t)r1 * heads + h];
    }
#pragma unroll
    for (int i = 0; i < VC; ++i) {
      const int d = 8 * i + 2 * qd;
      if (r0 < rows) {
        const float2 v = *reinterpret_cast<const float2*>(co + ((int64_t)r0 * heads + h) * DV + d);
        o[i][0] = v.x;
        o[i][1] = v.y;
      }
      if (r1 < rows) {
        const float2 v = *reinterpret_cast<const float2*>(co + ((int64_t)r1 * heads + h) * DV + d);
        o[i][2] = v.x;
        o[i][3] = v.y;
      }
    }
  }
  const int pos0 = p0 + r0 - kb0, pos1 = pos0 + 8;  // the last key (block-local) each of the thread's rows sees
  if (tiles > 0) stage(0, 0);
  for (int t = 0; t < tiles; ++t) {
    if (t + 1 < tiles) {
      stage(t + 1, (t + 1) & 1);
      qmm_frag::wait<1>();
    } else {
      qmm_frag::wait<0>();
    }
    __syncthreads();
    const uint4* ks = smem + (t & 1) * STAGE;
    const uint4* vs = ks + BN * KC;
    float sc[8][4];
#pragma unroll
    for (int k = 0; k < QK / 16; ++k) {
#pragma unroll
      for (int nt = 0; nt < 8; nt += 2) {
        uint32_t b[4];
        ldmatrix4(b, ks + sw<KC>(8 * (nt + mt / 2) + lane % 8, 2 * k + mt % 2));
        if (k == 0) {
          mma0(sc[nt], qa[k], b[0], b[1]);
          mma0(sc[nt + 1], qa[k], b[2], b[3]);
        } else {
          mma(sc[nt], qa[k], b[0], b[1]);
          mma(sc[nt + 1], qa[k], b[2], b[3]);
        }
      }
    }
    const int kb = t * BN + 2 * qd;
    float tm0 = -INFINITY, tm1 = -INFINITY;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      const int c = kb + 8 * nt;
      sc[nt][0] = c <= pos0 && c < span ? __fmul_rn(sc[nt][0], scale) : -INFINITY;
      sc[nt][1] = c + 1 <= pos0 && c + 1 < span ? __fmul_rn(sc[nt][1], scale) : -INFINITY;
      sc[nt][2] = c <= pos1 && c < span ? __fmul_rn(sc[nt][2], scale) : -INFINITY;
      sc[nt][3] = c + 1 <= pos1 && c + 1 < span ? __fmul_rn(sc[nt][3], scale) : -INFINITY;
      tm0 = nmax(tm0, nmax(sc[nt][0], sc[nt][1]));
      tm1 = nmax(tm1, nmax(sc[nt][2], sc[nt][3]));
    }
    for (int x = 1; x <= 2; x *= 2) {
      tm0 = nmax(tm0, __shfl_xor_sync(0xffffffffu, tm0, x));
      tm1 = nmax(tm1, __shfl_xor_sync(0xffffffffu, tm1, x));
    }
    const bool act0 = tm0 != -INFINITY, act1 = tm1 != -INFINITY;
    const float n0 = act0 ? nmax(m0, tm0) : m0, n1 = act1 ? nmax(m1, tm1) : m1;
    const float a0 = act0 ? (m0 == -INFINITY ? 0.0f : texp(__fsub_rn(m0, n0))) : 1.0f;
    const float a1 = act1 ? (m1 == -INFINITY ? 0.0f : texp(__fsub_rn(m1, n1))) : 1.0f;
    float s0 = 0.0f, s1 = 0.0f;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      sc[nt][0] = act0 && sc[nt][0] != -INFINITY ? texp(__fsub_rn(sc[nt][0], n0)) : 0.0f;
      sc[nt][1] = act0 && sc[nt][1] != -INFINITY ? texp(__fsub_rn(sc[nt][1], n0)) : 0.0f;
      sc[nt][2] = act1 && sc[nt][2] != -INFINITY ? texp(__fsub_rn(sc[nt][2], n1)) : 0.0f;
      sc[nt][3] = act1 && sc[nt][3] != -INFINITY ? texp(__fsub_rn(sc[nt][3], n1)) : 0.0f;
      s0 = __fadd_rn(s0, __fadd_rn(sc[nt][0], sc[nt][1]));
      s1 = __fadd_rn(s1, __fadd_rn(sc[nt][2], sc[nt][3]));
    }
    l0 = __fmaf_rn(l0, a0, quad(s0));
    l1 = __fmaf_rn(l1, a1, quad(s1));
    m0 = n0;
    m1 = n1;
#pragma unroll
    for (int i = 0; i < VC; ++i) {
      o[i][0] = __fmul_rn(o[i][0], a0);
      o[i][1] = __fmul_rn(o[i][1], a0);
      o[i][2] = __fmul_rn(o[i][2], a1);
      o[i][3] = __fmul_rn(o[i][3], a1);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      uint32_t pa[4];
      pa[0] = pack(sc[2 * j][0], sc[2 * j][1]);
      pa[1] = pack(sc[2 * j][2], sc[2 * j][3]);
      pa[2] = pack(sc[2 * j + 1][0], sc[2 * j + 1][1]);
      pa[3] = pack(sc[2 * j + 1][2], sc[2 * j + 1][3]);
#pragma unroll
      for (int i = 0; i < VC; i += 2) {
        uint32_t b[4];
        ldmatrix4t(b, vs + sw<VC>(16 * j + 8 * (mt % 2) + lane % 8, i + mt / 2));
        mma(o[i], pa, b[0], b[1]);
        mma(o[i + 1], pa, b[2], b[3]);
      }
    }
    __syncthreads();
  }
  if (last) {
    const float i0 = l0 == 0.0f ? 0.0f : 1.0f / l0, i1 = l1 == 0.0f ? 0.0f : 1.0f / l1;
#pragma unroll
    for (int i = 0; i < VC; ++i) {
      const int d = 8 * i + 2 * qd;
      if (r0 < rows)
        *reinterpret_cast<uint32_t*>(out + ((int64_t)r0 * heads + h) * DV + d) = pack(o[i][0] * i0, o[i][1] * i0);
      if (r1 < rows)
        *reinterpret_cast<uint32_t*>(out + ((int64_t)r1 * heads + h) * DV + d) = pack(o[i][2] * i1, o[i][3] * i1);
    }
    return;
  }
#pragma unroll
  for (int i = 0; i < VC; ++i) {
    const int d = 8 * i + 2 * qd;
    if (r0 < rows) *reinterpret_cast<float2*>(co + ((int64_t)r0 * heads + h) * DV + d) = make_float2(o[i][0], o[i][1]);
    if (r1 < rows) *reinterpret_cast<float2*>(co + ((int64_t)r1 * heads + h) * DV + d) = make_float2(o[i][2], o[i][3]);
  }
  if (qd == 0) {
    if (r0 < rows) {
      cm[(int64_t)r0 * heads + h] = m0;
      cl[(int64_t)r0 * heads + h] = l0;
    }
    if (r1 < rows) {
      cm[(int64_t)r1 * heads + h] = m1;
      cl[(int64_t)r1 * heads + h] = l1;
    }
  }
}
