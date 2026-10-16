// drama3 device helpers: bf16 conversions, warp and block reductions, streaming loads and 8-wide bf16 dots.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <math.h>

typedef __nv_bfloat16 bf16;

namespace {

constexpr unsigned FULL = 0xFFFFFFFFu;

__device__ __forceinline__ float b2f(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16 f2b(float x) { return __float2bfloat16(x); }
__device__ __forceinline__ float rb(float x) { return __bfloat162float(__float2bfloat16(x)); }  // round to bf16

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

// Sum over the block (blockDim.x a multiple of 32, at most 1024); every thread gets the total.
__device__ float block_sum(float v) {
  __shared__ float part[32];
  __shared__ float total;
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
  v = warp_sum(v);
  __syncthreads();
  if (lane == 0) part[w] = v;
  __syncthreads();
  if (w == 0) {
    float s = lane < nw ? part[lane] : 0.f;
    s = warp_sum(s);
    if (lane == 0) total = s;
  }
  __syncthreads();
  return total;
}

__device__ float block_max(float v) {
  __shared__ float part[32];
  __shared__ float total;
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
  v = warp_max(v);
  __syncthreads();
  if (lane == 0) part[w] = v;
  __syncthreads();
  if (w == 0) {
    float s = lane < nw ? part[lane] : -INFINITY;
    s = warp_max(s);
    if (lane == 0) total = s;
  }
  __syncthreads();
  return total;
}

__device__ __forceinline__ uint4 ld_stream(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p));
  return r;
}

__device__ __forceinline__ float lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float hi(uint32_t w) { return __uint_as_float(w & 0xFFFF0000u); }

__device__ __forceinline__ float dot8(uint4 a, uint4 b) {
  float s = lo(a.x) * lo(b.x);
  s = fmaf(hi(a.x), hi(b.x), s);
  s = fmaf(lo(a.y), lo(b.y), s);
  s = fmaf(hi(a.y), hi(b.y), s);
  s = fmaf(lo(a.z), lo(b.z), s);
  s = fmaf(hi(a.z), hi(b.z), s);
  s = fmaf(lo(a.w), lo(b.w), s);
  s = fmaf(hi(a.w), hi(b.w), s);
  return s;
}

__device__ __forceinline__ float sigmoidf_(float x) { return 1.f / (1.f + expf(-x)); }
__device__ __forceinline__ float siluf_(float x) { return x / (1.f + expf(-x)); }
__device__ __forceinline__ float softplusf_(float x) { return x > 20.f ? x : log1pf(expf(x)); }

__device__ __forceinline__ bool is_semantic(int tok, int sem_begin, int sem_end) {
  return tok >= sem_begin && tok <= sem_end;
}

}  // namespace

// The samplers' launch settings, passed by value: temperature, top-k, top-p and the request seed.
struct D3Sampling {
  float temperature;
  int top_k;
  float top_p;
  int pad;
  unsigned long long seed;
};
