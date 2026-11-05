// Gate 3 controls for uocr.cu's rows: each kernel with one of the publisher's bf16 points dropped.
#pragma once
#include <cuda_bf16.h>
#include "common/reduce.cuh"

namespace tf_ctl_uocr_rows {

typedef __nv_bfloat16 bf16;

__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float ld(const bf16* p) { return __bfloat162float(*p); }
__device__ __forceinline__ bf16 to(float x) { return __float2bfloat16_rn(x); }
__device__ __forceinline__ float clampf(float x, float lo, float hi) { return fminf(fmaxf(x, lo), hi); }

// uocr_common.cuh's block_sum: the warp butterfly, then warp 0's over the partials.
__device__ __forceinline__ float block_sum(float v, float* part) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, n = (blockDim.x + 31) >> 5;
  v = tf_common::warp_sum(v);
  if (lane == 0) part[w] = v;
  __syncthreads();
  if (w == 0) {
    float s = lane < n ? part[lane] : 0.0f;
    s = tf_common::warp_sum(s);
    if (lane == 0) part[32] = s;
  }
  __syncthreads();
  const float out = part[32];
  __syncthreads();
  return out;
}

__device__ __forceinline__ float angle(const float* inv_tok, unsigned tok, unsigned page, int i) {
  const int j = i % 64;
  return j < 63 ? (float)tok * inv_tok[j] : (float)(page % 2);
}

}  // namespace tf_ctl_uocr_rows

// tf_uocr_layer_norm2d without bf16((x - u) / den).
extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_layer_norm2d_quotient(const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w, const __nv_bfloat16* __restrict__ b, __nv_bfloat16* __restrict__ y, int width, float eps) {
  using namespace tf_ctl_uocr_rows;
  __shared__ float part[33];
  const bf16* xr = x + (size_t)blockIdx.x * width;
  bf16* yr = y + (size_t)blockIdx.x * width;
  float s = 0.0f;
  for (int i = threadIdx.x; i < width; i += blockDim.x) s += ld(xr + i);
  const float u = bf(block_sum(s, part) / (float)width);
  float v = 0.0f;
  for (int i = threadIdx.x; i < width; i += blockDim.x) {
    const float d = bf(ld(xr + i) - u);
    v += bf(d * d);
  }
  const float var = bf(block_sum(v, part) / (float)width);
  const float den = bf(sqrtf(bf(var + eps)));
  for (int i = threadIdx.x; i < width; i += blockDim.x) {
    const float d = bf(ld(xr + i) - u) / den;
    yr[i] = to(bf(ld(w + i) * d) + ld(b + i));
  }
}

// tf_uocr_quick_gelu without bf16(sigmoid).
extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_quick_gelu_sigmoid(__nv_bfloat16* __restrict__ x, long long n) {
  using namespace tf_ctl_uocr_rows;
  const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
  if (i >= n) return;
  const float v = ld(x + i);
  const float z = bf(1.702f * v);
  const float s = 1.0f / (1.0f + expf(-z));
  x[i] = to(v * s);
}

// tf_uocr_rms_norm without bf16(x rsqrt).
extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_rms_norm_scaled(const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ y, int width, float eps) {
  using namespace tf_ctl_uocr_rows;
  __shared__ float part[33];
  const bf16* xr = x + (size_t)blockIdx.x * width;
  float s = 0.0f;
  for (int i = threadIdx.x; i < width; i += blockDim.x) {
    const float v = ld(xr + i);
    s += v * v;
  }
  s = block_sum(s, part);
  const float r = 1.0f / sqrtf(s / (float)width + eps);
  bf16* yr = y + (size_t)blockIdx.x * width;
  for (int i = threadIdx.x; i < width; i += blockDim.x) yr[i] = to(ld(w + i) * (ld(xr + i) * r));
}

// tf_uocr_rope without its two bf16 products.
extern "C" __global__ void __launch_bounds__(128) tf_ctl_uocr_rope_products(__nv_bfloat16* __restrict__ x, const unsigned* __restrict__ tok,
    const unsigned* __restrict__ page, const float* __restrict__ inv_tok, int stride) {
  using namespace tf_ctl_uocr_rows;
  const int d = threadIdx.x, row = blockIdx.x, head = blockIdx.y;
  bf16* h = x + (size_t)row * stride + head * 128;
  const float ang = angle(inv_tok, tok[row], page[row], d);
  const float c = bf(cosf(ang)), s = bf(sinf(ang));
  const float rh = d < 64 ? -ld(h + d + 64) : ld(h + d - 64);
  const bf16 out = to(ld(h + d) * c + rh * s);
  __syncthreads();
  h[d] = out;
}

// tf_uocr_router_norm with its normed row left in fp32.
extern "C" __global__ void __launch_bounds__(128) tf_ctl_uocr_router_norm_normed(const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w, const unsigned* __restrict__ tok, const unsigned* __restrict__ page,
    const float* __restrict__ inv_tok, float* __restrict__ y) {
  using namespace tf_ctl_uocr_rows;
  __shared__ float part[33];
  __shared__ float normed[128];
  const int row = blockIdx.x, d = threadIdx.x;
  float v = ld(x + (size_t)row * 128 + d);
  if (!isfinite(v)) v = 0.0f;
  const float s = block_sum(v * v, part);
  float n = clampf(v * (1.0f / sqrtf(s / 128.0f + 1e-6f)), -1.0e4f, 1.0e4f);
  n = clampf(n * (1.0f + ld(w + d)), -1.0e4f, 1.0e4f);
  normed[d] = n;
  __syncthreads();
  const float ang = angle(inv_tok, tok[row], page[row], d);
  const float c = bf(cosf(ang)), sn = bf(sinf(ang));
  const float rh = d < 64 ? -normed[d + 64] : normed[d - 64];
  float out = bf(bf(normed[d] * c) + bf(rh * sn));
  if (!isfinite(out)) out = 0.0f;
  y[(size_t)row * 128 + d] = out;
}

// tf_uocr_silu_mul without bf16(silu).
extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_silu_mul_silu(const __nv_bfloat16* __restrict__ g,
    const __nv_bfloat16* __restrict__ u, __nv_bfloat16* __restrict__ y, long long n) {
  using namespace tf_ctl_uocr_rows;
  const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
  if (i >= n) return;
  const float v = ld(g + i);
  y[i] = to(v / (1.0f + expf(-v)) * ld(u + i));
}
