// Unlimited-OCR-2's CUDA helpers: bf16 rounding, warp and block reductions, 16-byte loads, erf.
#pragma once
#include <cuda_bf16.h>
#include <stdint.h>
#include "common/reduce.cuh"

namespace tf_uocr {

typedef __nv_bfloat16 bf16;

#define UD_NEG -1.0e9f
#define UD_ENTMAX_NEG -1.0e4f
#define UD_CLAMP 1.0e4f

__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float ld(const bf16* p) { return __bfloat162float(*p); }
__device__ __forceinline__ bf16 to(float x) { return __float2bfloat16_rn(x); }
__device__ __forceinline__ float clampf(float x, float lo, float hi) { return fminf(fmaxf(x, lo), hi); }

using tf_common::warp_sum;

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

__device__ __forceinline__ int warp_min_int(int v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = min(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

// A block's sum of every thread's `v` (blockDim.x a multiple of 32, at most 1,024), returned to all of them.
__device__ __forceinline__ float block_sum(float v, float* part) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, n = (blockDim.x + 31) >> 5;
  v = warp_sum(v);
  if (lane == 0) part[w] = v;
  __syncthreads();
  if (w == 0) {
    float s = lane < n ? part[lane] : 0.0f;
    s = warp_sum(s);
    if (lane == 0) part[32] = s;
  }
  __syncthreads();
  const float out = part[32];
  __syncthreads();
  return out;
}

// 8 bf16 at p (16-byte aligned) widened.
__device__ __forceinline__ void load8(const bf16* p, float* out) {
  const uint4 u = *reinterpret_cast<const uint4*>(p);
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 f = __bfloat1622float2(h[i]);
    out[2 * i] = f.x;
    out[2 * i + 1] = f.y;
  }
}

// erf to under one ulp (Juffa's faithfully rounded erff, as uv_erf).
__device__ __forceinline__ float erf_(float x) {
  const float t = fabsf(x), s = x * x;
  float r;
  if (t > 0.927734375f) {
    r = fmaf(-1.72853470e-5f, t, 3.83197126e-4f);
    const float u = fmaf(-3.88396438e-3f, t, 2.42546219e-2f);
    r = fmaf(r, s, u);
    r = fmaf(r, t, -1.06777877e-1f);
    r = fmaf(r, t, -6.34846687e-1f);
    r = fmaf(r, t, -1.28717512e-1f);
    r = fmaf(r, t, -t);
    r = copysignf(1.0f - expf(r), x);
  } else {
    r = -5.96761703e-4f;
    r = fmaf(r, s, 4.99119423e-3f);
    r = fmaf(r, s, -2.67681349e-2f);
    r = fmaf(r, s, 1.12819925e-1f);
    r = fmaf(r, s, -3.76125336e-1f);
    r = fmaf(r, s, 1.28379166e-1f);
    r = fmaf(r, x, x);
  }
  return r;
}

}  // namespace tf_uocr

using namespace tf_uocr;
