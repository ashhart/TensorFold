// Idol's decoders' steps (idol_decoder.metal's op for op) as templates: BREAK 0 ships, `if constexpr` drops the rest.
#pragma once

#include <cuda_fp16.h>
#include <math.h>
#include <stdint.h>

#include "idol_attention.cuh"

namespace tf_idol_dec {

typedef __half f16;

__device__ __forceinline__ float h(float x) { return __half2float(__float2half_rn(x)); }
__device__ __forceinline__ float ld(const f16* p) { return __half2float(*p); }

// F.normalize's clamp_min(1e-12): a NaN norm stays NaN (fmaxf would drop it).
__device__ __forceinline__ float clamp_min(float n) { return n < 1e-12f ? 1e-12f : n; }

// h = in_proj(p) + pe(p) fp32: sin then cos of fp32(fp32(x f) pi) a axis, 4 zeros. BREAK 1: fp16 pe; 2: fp16 sums.
template <int BREAK>
__device__ __forceinline__ void embed(const float* __restrict__ P, const float* __restrict__ W,
                                      const float* __restrict__ B, const float* __restrict__ F, float* __restrict__ Hs,
                                      int rows) {
  const int c = blockIdx.y * 256 + threadIdx.x, r = blockIdx.x;
  if (c >= 1024 || r >= rows) return;
  const float x0 = P[r * 3], x1 = P[r * 3 + 1], x2 = P[r * 3 + 2];
  float s;
  if constexpr (BREAK == 2) {
    s = h(__fmaf_rn(h(x2), h(W[c * 3 + 2]), h(__fmaf_rn(h(x1), h(W[c * 3 + 1]), h(__fmul_rn(h(x0), h(W[c * 3])))))));
  } else {
    s = __fmaf_rn(x2, W[c * 3 + 2], __fmaf_rn(x1, W[c * 3 + 1], __fmul_rn(x0, W[c * 3])));
  }
  s = __fadd_rn(s, B[c]);
  float pe = 0.0f;
  if (c < 1020) {
    const int a = c / 340, j = c % 340;
    const float x = a == 0 ? x0 : (a == 1 ? x1 : x2);
    const float ang = __fmul_rn(__fmul_rn(x, F[j % 170]), 3.14159265358979323846f);
    pe = j < 170 ? sinf(ang) : cosf(ang);
    if constexpr (BREAK == 1) pe = h(pe);
  }
  Hs[(size_t)r * 1024 + c] = __fadd_rn(s, pe);
}

// o = fp16(y + b) by column, an fp32 Linear's bias then the blocks' cast. BREAK 1: fp16 y first; 2: writes its input.
template <int BREAK>
__device__ __forceinline__ void to_half(const float* Y, const float* __restrict__ B, f16* __restrict__ O, int width,
                                        int rows) {
  const int c = blockIdx.y * 256 + threadIdx.x, r = blockIdx.x;
  if (c >= width || r >= rows) return;
  const size_t i = (size_t)r * width + c;
  const float y = BREAK == 1 ? h(Y[i]) : Y[i];
  O[i] = __float2half_rn(__fadd_rn(y, B[c]));
  if constexpr (BREAK == 2) const_cast<float*>(Y)[i] = 0.0f;
}

// kv [rows, 2048] = fp16(ctx w^T + b), fp32 FMA sums of the fp16 latent. BREAK 1: fp16 ctx w first; 2: fp16 sums.
template <int BREAK>
__device__ __forceinline__ void kv(const f16* __restrict__ X, const f16* __restrict__ W, const f16* __restrict__ B,
                                   f16* __restrict__ KV, int rows) {
  const int c = blockIdx.y * 256 + threadIdx.x, r = blockIdx.x;
  if (c >= 2048 || r >= rows) return;
  float s = 0.0f;
#pragma unroll
  for (int k = 0; k < 16; ++k) {
    s = __fmaf_rn(ld(X + r * 16 + k), ld(W + c * 16 + k), s);
    if constexpr (BREAK == 2) s = h(s);
  }
  if constexpr (BREAK == 1) s = h(s);
  KV[(size_t)r * 2048 + c] = __float2half_rn(__fadd_rn(s, ld(B + c)));
}

// MultiHeadRMSNorm in fp32 of 16 heads at `offset`, a warp a head, Y may be X. BREAK 1: fmaxf; 2: fp16 x / n; 3: sums.
template <int BREAK>
__device__ __forceinline__ void heads(const f16* X, const float* __restrict__ G, f16* Y, int stride, int offset,
                                      int rows) {
  const int r = blockIdx.x, hd = blockIdx.y * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (hd >= 16 || r >= rows) return;
  const size_t at = (size_t)r * stride + offset + hd * 64 + lane * 2;
  const float a0 = ld(X + at), a1 = ld(X + at + 1);
  float sq = __fadd_rn(__fmul_rn(a0, a0), __fmul_rn(a1, a1));
  if constexpr (BREAK == 3) sq = h(sq);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    sq = __fadd_rn(sq, __shfl_xor_sync(0xffffffffu, sq, o));
    if constexpr (BREAK == 3) sq = h(sq);
  }
  const float len = __fsqrt_rn(sq);
  const float n = BREAK == 1 ? fmaxf(len, 1e-12f) : clamp_min(len);
  float q0 = __fdiv_rn(a0, n), q1 = __fdiv_rn(a1, n);
  if constexpr (BREAK == 2) {
    q0 = h(q0);
    q1 = h(q1);
  }
  const int g = hd * 64 + lane * 2;
  Y[at] = __float2half_rn(__fmul_rn(__fmul_rn(q0, G[g]), 8.0f));
  Y[at + 1] = __float2half_rn(__fmul_rn(__fmul_rn(q1, G[g + 1]), 8.0f));
}

// lq <= 8 queries, lk <= 8192 keys in fp32 as the cases' MPS kernel; a block a row and head. BREAK 1: fp16 P; 2: sums.
template <int BREAK>
__device__ __forceinline__ void few(const f16* __restrict__ Q, int qs, const f16* __restrict__ KV, int ks,
                                    f16* __restrict__ O, int lq, int lk, int heads, float scale) {
  __shared__ float q[64];
  __shared__ float part[8][64];
  __shared__ float red[9];
  const int r = blockIdx.x, hd = blockIdx.y, t = threadIdx.x, lane = t & 31, w = t >> 5;
  if (r >= lq) return;
  if (t < 64) q[t] = __fmul_rn(ld(Q + (size_t)r * qs + hd * 64 + t), scale);
  __syncthreads();
  const f16* kh = KV + hd * 64;
  const f16* vh = KV + (heads + hd) * 64;
  float s[32];
  float top = -INFINITY;
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    const int j = t + i * 256;
    s[i] = -INFINITY;
    if (j >= lk) continue;
    const __half2* kr = reinterpret_cast<const __half2*>(kh + (size_t)j * ks);
    float d = 0.0f;
#pragma unroll
    for (int c = 0; c < 32; ++c) {
      const float2 kv2 = __half22float2(kr[c]);
      d = __fmaf_rn(q[2 * c], kv2.x, __fmaf_rn(q[2 * c + 1], kv2.y, d));
    }
    s[i] = d;
    top = fmaxf(top, d);
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, o));
  if (lane == 0) red[w] = top;
  __syncthreads();
  if (t == 0) {
    float m = red[0];
    for (int i = 1; i < 8; ++i) m = fmaxf(m, red[i]);
    red[8] = m;
  }
  __syncthreads();
  top = red[8];
  float o[64];
#pragma unroll
  for (int c = 0; c < 64; ++c) o[c] = 0.0f;
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    const int j = t + i * 256;
    if (j >= lk) continue;
    float e = expf(__fsub_rn(s[i], top));
    if constexpr (BREAK == 1) e = h(e);
    sum = __fadd_rn(sum, e);
    const __half2* vr = reinterpret_cast<const __half2*>(vh + (size_t)j * ks);
#pragma unroll
    for (int c = 0; c < 32; ++c) {
      const float2 v2 = __half22float2(vr[c]);
      o[2 * c] = __fmaf_rn(e, v2.x, o[2 * c]);
      o[2 * c + 1] = __fmaf_rn(e, v2.y, o[2 * c + 1]);
      if constexpr (BREAK == 2) {
        o[2 * c] = h(o[2 * c]);
        o[2 * c + 1] = h(o[2 * c + 1]);
      }
    }
  }
  __syncthreads();
#pragma unroll
  for (int x = 16; x > 0; x >>= 1) sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, x));
  if (lane == 0) red[w] = sum;
#pragma unroll
  for (int c = 0; c < 64; ++c) {
    float v = o[c];
#pragma unroll
    for (int x = 16; x > 0; x >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, x));
    if (lane == 0) part[w][c] = v;
  }
  __syncthreads();
  if (t < 64) {
    float total = 0.0f, acc = 0.0f;
    for (int i = 0; i < 8; ++i) {
      total = __fadd_rn(total, red[i]);
      acc = __fadd_rn(acc, part[i][t]);
    }
    O[(size_t)r * heads * 64 + hd * 64 + t] = __float2half_rn(__fdiv_rn(acc, total));
  }
}

}  // namespace tf_idol_dec
