// Ripple's CUDA glue: glue.metal's rc_embed, rc_add_norm, rc_gate and rc_silu_clamp at the same rounding points.
#pragma once

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

#include "common/act.cuh"
#include "common/reduce.cuh"

namespace tf_rc {

typedef __nv_bfloat16 bf16;

constexpr int NT = 256;   // threads a block in every glue launch
constexpr int MOST = 4;   // 8-value chunks a thread holds in add_norm: rows up to 8,192 wide

__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float ld(const bf16* p) { return __bfloat162float(*p); }

// A rounding point: bf16 when K keeps P; a control's K drops it.
template <int K, int P>
__device__ __forceinline__ float point(float x) { return (K & P) ? bf(x) : x; }

// add_norm's rounding points, a bit each; a control clears one.
constexpr int Y_ROW = 1;   // bf16(y * inv): Y's norm before w_post
constexpr int Y_OUT = 2;   // bf16(w_post * that): the normed branch
constexpr int RESID = 4;   // bf16(r + normed): the residual the next norm reads
constexpr int X_ROW = 8;   // bf16(r * inv): R's norm before w_next
constexpr int X_LERP = 16; // the lerp reads X as stored (bf16), not the fp32 product
constexpr int NORM_ALL = Y_ROW | Y_OUT | RESID | X_ROW | X_LERP;

constexpr int SIG = 1;  // gate: bf16(sigmoid(g)), the release's fp32 sigmoid then bf16
constexpr int SILU = 1; // silu_clamp: bf16(silu(g))
constexpr int LOW = 2;  // silu_clamp: up's lower clamp (not a rounding point; the Metal family's control drops it)

// R and V0 row r = bf16(E[ids[r]] * scale), the fp32 scale; BF_SCALE (a control) rounds the scale to bf16 first.
template <bool BF_SCALE>
__device__ __forceinline__ void embed(const unsigned* __restrict__ ids, const bf16* __restrict__ E, bf16* __restrict__ R,
                                      bf16* __restrict__ V0, int rows, int width, float scale) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
  if (x >= width || row >= rows) return;
  const size_t at = (size_t)row * width + x;
  const bf16 v = __float2bfloat16_rn(__fmul_rn(ld(E + (size_t)ids[row] * width + x), BF_SCALE ? bf(scale) : scale));
  R[at] = v;
  V0[at] = v;
}

// torch.lerp on bf16 (its vectorized form): fma(w, e - s, s) below one half, else fma(w - 1, e - s, e); unrounded.
__device__ __forceinline__ float lerp(float s, float e, float w) {
  const float d = __fsub_rn(e, s);
  return fabsf(w) < 0.5f ? __fmaf_rn(w, d, s) : __fmaf_rn(__fsub_rn(w, 1.0f), d, e);
}

// A thread's chunks t + j NT of a row's 8-value chunks, zeros past the row.
__device__ __forceinline__ void load_row(const bf16* __restrict__ p, int chunks, float (&v)[MOST][8]) {
  const uint4* q = reinterpret_cast<const uint4*>(p);
#pragma unroll
  for (int j = 0; j < MOST; ++j) {
    const int c = threadIdx.x + j * NT;
    if (c < chunks) {
      tf_common::unpack8(q[c], v[j]);
    } else {
#pragma unroll
      for (int e = 0; e < 8; ++e) v[j][e] = 0.0f;
    }
  }
}

// A row's fp32 sum of squares (each thread's fma chain, then the block's tree) into 1 / sqrt(mean + eps).
__device__ __forceinline__ float inv_rms(const float (&v)[MOST][8], float* part, int width, float eps) {
  float s = 0.0f;
#pragma unroll
  for (int j = 0; j < MOST; ++j)
#pragma unroll
    for (int e = 0; e < 8; ++e) s = __fmaf_rn(v[j][e], v[j][e], s);
  return tf_common::rms_scale(tf_common::block_sum_tree(s, part, NT / 32), width, eps);
}

// A row a block, R in place: R = bf16(R + bf16(w_post bf16(Y inv))); X = bf16(w_next bf16(R inv)); VX = lerp(V0, X).
template <int K>
__device__ __forceinline__ void add_norm(bf16* __restrict__ R, const bf16* __restrict__ Y, const bf16* __restrict__ w_post,
                                         const bf16* __restrict__ w_next, const bf16* __restrict__ V0, bf16* __restrict__ X,
                                         bf16* __restrict__ VX, int rows, int width, float eps, float blend) {
  __shared__ float part[NT / 32];
  const int row = blockIdx.x, chunks = width / 8;
  if (row >= rows) return;
  const size_t base = (size_t)row * width;
  float r[MOST][8];
  load_row(R + base, chunks, r);
  if (Y != nullptr) {
    float y[MOST][8];
    load_row(Y + base, chunks, y);
    const float inv = inv_rms(y, part, width, eps);
    const uint4* w4 = reinterpret_cast<const uint4*>(w_post);
    uint4* out = reinterpret_cast<uint4*>(R + base);
#pragma unroll
    for (int j = 0; j < MOST; ++j) {
      const int c = threadIdx.x + j * NT;
      if (c >= chunks) continue;
      float w[8];
      tf_common::unpack8(w4[c], w);
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const float normed = point<K, Y_OUT>(__fmul_rn(w[e], point<K, Y_ROW>(__fmul_rn(y[j][e], inv))));
        r[j][e] = point<K, RESID>(__fadd_rn(r[j][e], normed));
      }
      out[c] = tf_common::pack8(r[j]);
    }
  }
  const float inv = inv_rms(r, part, width, eps);
  const uint4* w4 = reinterpret_cast<const uint4*>(w_next);
#pragma unroll
  for (int j = 0; j < MOST; ++j) {
    const int c = threadIdx.x + j * NT;
    if (c >= chunks) continue;
    float w[8], x[8];
    tf_common::unpack8(w4[c], w);
#pragma unroll
    for (int e = 0; e < 8; ++e) x[e] = __fmul_rn(w[e], point<K, X_ROW>(__fmul_rn(r[j][e], inv)));
    const uint4 xs = tf_common::pack8(x);
    reinterpret_cast<uint4*>(X + base)[c] = xs;
    if (V0 == nullptr) continue;
    float s[8], e8[8];
    tf_common::unpack8(reinterpret_cast<const uint4*>(V0 + base)[c], s);
    if (K & X_LERP) {
      tf_common::unpack8(xs, e8);
    } else {
#pragma unroll
      for (int e = 0; e < 8; ++e) e8[e] = x[e];
    }
#pragma unroll
    for (int e = 0; e < 8; ++e) s[e] = lerp(s[e], e8[e], blend);
    reinterpret_cast<uint4*>(VX + base)[c] = tf_common::pack8(s);
  }
}

// The attention's output gate in place: A = bf16(A bf16(sigmoid(G))), G the bf16 g_proj rows.
template <int K>
__device__ __forceinline__ void gate(bf16* __restrict__ A, const bf16* __restrict__ G, int rows, int width) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
  if (x >= width || row >= rows) return;
  const size_t i = (size_t)row * width + x;
  const float s = point<K, SIG>(tf_common::sigmoid_rn(ld(G + i)));
  A[i] = __float2bfloat16_rn(__fmul_rn(ld(A + i), s));
}

// The FFN on a stacked gate | up row: gate at most limit, up within +-limit, then bf16(bf16(silu(g)) u).
template <int K>
__device__ __forceinline__ void silu_clamp(const bf16* __restrict__ GU, bf16* __restrict__ A, int rows, int width, float limit) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
  if (x >= width || row >= rows) return;
  const size_t at = (size_t)row * 2 * width + x;
  const float g0 = ld(GU + at), u0 = ld(GU + at + width);
  // a comparison is false for a NaN, so it passes both clamps as torch.clamp keeps it
  const float g = g0 > limit ? limit : g0;
  const float hi = u0 > limit ? limit : u0;
  const float u = (K & LOW) && u0 < -limit ? -limit : hi;
  const float silu = point<K, SILU>(tf_common::silu_rn(g));
  A[(size_t)row * width + x] = __float2bfloat16_rn(__fmul_rn(silu, u));
}

}  // namespace tf_rc
