// music3 kernels: the types, roundings and reductions every part uses, and the Philox counter draws.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

#define FULL 0xffffffffu
typedef __nv_bfloat16 bf16;

__device__ __forceinline__ float bfu(unsigned short h) { return __uint_as_float(((unsigned)h) << 16); }
__device__ __forceinline__ float b2f(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ bf16 f2b(float v) { return __float2bfloat16_rn(v); }
__device__ __forceinline__ float rb(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
__device__ __forceinline__ float h2f(__half v) { return __half2float(v); }
__device__ __forceinline__ __half f2h(float v) { return __float2half_rn(v); }
__device__ __forceinline__ float rh(float v) { return __half2float(__float2half_rn(v)); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL, v, o);
  return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(FULL, v, o));
  return v;
}
// Block sum (blockDim a multiple of 32, <= 1024): warp trees, then the first warp's tree. All threads get the sum.
__device__ __forceinline__ float block_sum(float v, float* sh) {
  v = warp_sum(v);
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31, nw = blockDim.x >> 5;
  __syncthreads();
  if (l == 0) sh[w] = v;
  __syncthreads();
  float t = l < nw ? sh[l] : 0.f;
  t = warp_sum(t);
  return t;
}
__device__ __forceinline__ float block_max(float v, float* sh) {
  v = warp_max(v);
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31, nw = blockDim.x >> 5;
  __syncthreads();
  if (l == 0) sh[w] = v;
  __syncthreads();
  float t = l < nw ? sh[l] : -INFINITY;
  return warp_max(t);
}


struct Philox {
  __device__ static void round(uint32_t& c0, uint32_t& c1, uint32_t& c2, uint32_t& c3, uint32_t k0, uint32_t k1) {
    const uint64_t p0 = (uint64_t)0xD2511F53u * c0, p1 = (uint64_t)0xCD9E8D57u * c2;
    const uint32_t h0 = (uint32_t)(p0 >> 32), l0 = (uint32_t)p0, h1 = (uint32_t)(p1 >> 32), l1 = (uint32_t)p1;
    const uint32_t n0 = h1 ^ c1 ^ k0, n2 = h0 ^ c3 ^ k1;
    c0 = n0;
    c1 = l1;
    c2 = n2;
    c3 = l0;
  }
  // Philox4x32-10 of counter (a, b) under key seed: the first word as a uniform in [0, 1) (24 bits).
  __device__ static float uniform(unsigned long long seed, unsigned long long a, unsigned long long b) {
    uint32_t c0 = (uint32_t)a, c1 = (uint32_t)(a >> 32), c2 = (uint32_t)b, c3 = (uint32_t)(b >> 32);
    uint32_t k0 = (uint32_t)seed, k1 = (uint32_t)(seed >> 32);
#pragma unroll
    for (int i = 0; i < 10; ++i) {
      round(c0, c1, c2, c3, k0, k1);
      k0 += 0x9E3779B9u;
      k1 += 0xBB67AE85u;
    }
    return (float)(c0 >> 8) * (1.0f / 16777216.0f);
  }
  // Two normals (Box-Muller) of counter (a, b).
  __device__ static float2 normal2(unsigned long long seed, unsigned long long a, unsigned long long b) {
    uint32_t c0 = (uint32_t)a, c1 = (uint32_t)(a >> 32), c2 = (uint32_t)b, c3 = (uint32_t)(b >> 32);
    uint32_t k0 = (uint32_t)seed, k1 = (uint32_t)(seed >> 32);
#pragma unroll
    for (int i = 0; i < 10; ++i) {
      round(c0, c1, c2, c3, k0, k1);
      k0 += 0x9E3779B9u;
      k1 += 0xBB67AE85u;
    }
    const float u1 = ((float)(c0 >> 8) + 1.0f) * (1.0f / 16777216.0f);
    const float u2 = (float)(c1 >> 8) * (1.0f / 16777216.0f);
    const float rr = sqrtf(-2.f * logf(u1));
    return make_float2(rr * cosf(6.2831853071795864f * u2), rr * sinf(6.2831853071795864f * u2));
  }
};
