// idol_dino.cu's glue with one bf16 rounding dropped or added (gate 3) or its sums in fp16 (gate 4), same arguments.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math.h>

namespace tf_ctl_idol_dino {
typedef __nv_bfloat16 bf16;
__device__ __forceinline__ float b(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float ld(const bf16* p) { return __bfloat162float(*p); }
__device__ __forceinline__ float h(float x) { return __half2float(__float2half_rn(x)); }

// A block of 256's sum, every partial an fp16 value (`half`) or fp32: lanes, then the 8 warps in order.
__device__ __forceinline__ float block_sum(float v, float* part, bool half) {
  for (int o = 16; o > 0; o >>= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, o);
    if (half) v = h(v);
  }
  __syncthreads();
  if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = v;
  __syncthreads();
  float total = 0.0f;
  for (int w = 0; w < 8; ++w) total = half ? h(total + part[w]) : total + part[w];
  return total;
}

// A bf16 row's mean and 1 / sqrt(var + eps), its sums kept in fp16 when `half`.
__device__ __forceinline__ void stats(const float* x, int width, float eps, float* part, bool half, float& mean,
                                      float& inv) {
  float a = 0.0f;
  for (int i = threadIdx.x; i < width; i += 256) a = half ? h(a + x[i]) : a + x[i];
  mean = block_sum(a, part, half) / (float)width;
  float q = 0.0f;
  for (int i = threadIdx.x; i < width; i += 256) q = half ? h(q + (x[i] - mean) * (x[i] - mean)) : q + (x[i] - mean) * (x[i] - mean);
  inv = 1.0f / sqrtf(block_sum(q, part, half) / (float)width + eps);
}
}  // namespace tf_ctl_idol_dino

// tf_idol_dino_bias rounding the sums before the bias: bf16(bf16(acc) + b), where torch rounds once after it.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_idol_dino_bias_twice(const float* ACC, const __nv_bfloat16* B,
                                                                              __nv_bfloat16* Y, int width, int rows) {
  using namespace tf_ctl_idol_dino;
  const int c = blockIdx.y * 256 + threadIdx.x, r = blockIdx.x;
  if (c >= width || r >= rows) return;
  const size_t i = (size_t)r * width + c;
  Y[i] = __float2bfloat16_rn(b(ACC[i]) + ld(B + c));
}

// tf_idol_dino_norm with its statistics summed in fp16.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_idol_dino_norm_half(const __nv_bfloat16* X, const __nv_bfloat16* W,
                                                                             const __nv_bfloat16* B, __nv_bfloat16* Y,
                                                                             int width, float eps) {
  using namespace tf_ctl_idol_dino;
  __shared__ float part[8];
  __shared__ float v[1280];
  if (width > 1280) return;
  const size_t row = blockIdx.x;
  for (int i = threadIdx.x; i < width; i += 256) v[i] = ld(X + row * width + i);
  __syncthreads();
  float mean, inv;
  stats(v, width, eps, part, true, mean, inv);
  for (int i = threadIdx.x; i < width; i += 256) Y[row * width + i] = __float2bfloat16_rn((v[i] - mean) * inv * ld(W + i) + ld(B + i));
}

// tf_idol_dino_final with F.layer_norm's sums (the second stage) in fp16.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_idol_dino_final_half(const __nv_bfloat16* X, const __nv_bfloat16* W,
                                                                              const __nv_bfloat16* B, __nv_bfloat16* H,
                                                                              float* Y, int width, float eps) {
  using namespace tf_ctl_idol_dino;
  __shared__ float part[8];
  __shared__ float v[1280];
  if (width > 1280) return;
  const size_t row = blockIdx.x;
  for (int i = threadIdx.x; i < width; i += 256) v[i] = ld(X + row * width + i);
  __syncthreads();
  float mean, inv;
  stats(v, width, eps, part, false, mean, inv);
  __syncthreads();
  for (int i = threadIdx.x; i < width; i += 256) {
    const __nv_bfloat16 o = __float2bfloat16_rn((v[i] - mean) * inv * ld(W + i) + ld(B + i));
    H[row * width + i] = o;
    v[i] = __bfloat162float(o);
  }
  __syncthreads();
  float mean2, inv2;
  stats(v, width, eps, part, true, mean2, inv2);
  for (int i = threadIdx.x; i < width; i += 256) Y[row * width + i] = (v[i] - mean2) * inv2;
}

// tf_idol_dino_heads turning q and k with one rounding, bf16(x cos + rot(x) sin): transformers rounds each product.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_idol_dino_heads_once(const float* QKV, const __nv_bfloat16* CS,
                                                                              const __nv_bfloat16* SN, __nv_bfloat16* OUT,
                                                                              int rows, int prefix) {
  using namespace tf_ctl_idol_dino;
  const int r = blockIdx.x, at = blockIdx.y * 256 + threadIdx.x;
  if (r >= rows || at >= 20 * 32) return;
  const int hd = at / 32, j = at % 32;
  const size_t from = (size_t)r * 3840 + hd * 64 + j;
  for (int p = 0; p < 3; ++p) {
    const float x0 = b(QKV[from + p * 1280]), x1 = b(QKV[from + p * 1280 + 32]);
    if (p == 2 || r < prefix) {
      OUT[from + p * 1280] = __float2bfloat16_rn(x0);
      OUT[from + p * 1280 + 32] = __float2bfloat16_rn(x1);
      continue;
    }
    const size_t t = (size_t)(r - prefix) * 64 + j;
    OUT[from + p * 1280] = __float2bfloat16_rn(x0 * ld(CS + t) - x1 * ld(SN + t));
    OUT[from + p * 1280 + 32] = __float2bfloat16_rn(x1 * ld(CS + t + 32) + x0 * ld(SN + t + 32));
  }
}

// tf_idol_dino_scale_add leaving g y in fp32: bf16(x + g y), where transformers rounds LayerScale's product first.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_idol_dino_scale_add_once(__nv_bfloat16* X, const float* ACC,
                                                                                  const __nv_bfloat16* B,
                                                                                  const __nv_bfloat16* G, int width,
                                                                                  int rows) {
  using namespace tf_ctl_idol_dino;
  const int c = blockIdx.y * 256 + threadIdx.x, r = blockIdx.x;
  if (c >= width || r >= rows) return;
  const size_t i = (size_t)r * width + c;
  X[i] = __float2bfloat16_rn(ld(X + i) + ld(G + c) * b(ACC[i] + ld(B + c)));
}

// tf_idol_dino_swiglu with silu(g) left in fp32: bf16(silu(g) u), where torch rounds silu's output to bf16 first.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_idol_dino_swiglu_late(const float* ACC, const __nv_bfloat16* B,
                                                                               __nv_bfloat16* Hd, int width, int rows) {
  using namespace tf_ctl_idol_dino;
  const int c = blockIdx.y * 256 + threadIdx.x, r = blockIdx.x;
  if (c >= width || r >= rows) return;
  const size_t i = (size_t)r * 2 * width + c;
  const float g = b(ACC[i] + ld(B + c));
  const float u = b(ACC[i + width] + ld(B + width + c));
  Hd[(size_t)r * width + c] = __float2bfloat16_rn(g / (1.0f + expf(-g)) * u);
}
