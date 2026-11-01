// Idol's DINOv3 glue on CUDA: idol_dino.metal's steps, bf16 rows rounded where transformers' torch code rounds.
#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

#include "idol_attention.cuh"

namespace tf_idol_dino {

typedef __nv_bfloat16 bf16;

constexpr int WIDTH = 1280;  // the model's width: dn_final keeps one row of it in shared memory
constexpr int HEADS = 20;
constexpr int D = 64;

__device__ __forceinline__ float b(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float ld(const bf16* p) { return __bfloat162float(*p); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// A block of 256's sum of every thread's `v`, returned to all of them: lanes, then the 8 warps in order.
__device__ __forceinline__ float block_sum(float v, float* part) {
  v = warp_sum(v);
  __syncthreads();
  if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = v;
  __syncthreads();
  float total = 0.0f;
#pragma unroll
  for (int w = 0; w < 8; ++w) total += part[w];
  return total;
}

// nn.LayerNorm's statistics of a bf16 row: the mean, then 1 / sqrt(var + eps), each sum in fp32.
__device__ __forceinline__ void stats(const bf16* x, int width, float eps, float* part, float& mean, float& inv) {
  float a = 0.0f;
  for (int i = threadIdx.x; i < width; i += 256) a += ld(x + i);
  mean = block_sum(a, part) / (float)width;
  float q = 0.0f;
  for (int i = threadIdx.x; i < width; i += 256) {
    const float d = ld(x + i) - mean;
    q += d * d;
  }
  inv = 1.0f / sqrtf(block_sum(q, part) / (float)width + eps);
}

}  // namespace tf_idol_dino

using namespace tf_idol_dino;

// y = bf16(acc + b): a linear's fp32 sums (cuBLASLt) and its bf16 bias, one rounding, as torch's bf16 addmm.
extern "C" __global__ void __launch_bounds__(256) tf_idol_dino_bias(const float* __restrict__ ACC, const bf16* __restrict__ B,
                                                                    bf16* __restrict__ Y, int width, int rows) {
  const int c = blockIdx.y * 256 + threadIdx.x, r = blockIdx.x;
  if (c >= width || r >= rows) return;
  const size_t i = (size_t)r * width + c;
  Y[i] = __float2bfloat16_rn(ACC[i] + ld(B + c));
}

// nn.LayerNorm of a bf16 row: fp32 statistics, then bf16(n w + b) with bf16 w and b. A block a row.
extern "C" __global__ void __launch_bounds__(256) tf_idol_dino_norm(const bf16* __restrict__ X, const bf16* __restrict__ W,
                                                                    const bf16* __restrict__ B, bf16* __restrict__ Y,
                                                                    int width, float eps) {
  __shared__ float part[8];
  const bf16* x = X + (size_t)blockIdx.x * width;
  bf16* y = Y + (size_t)blockIdx.x * width;
  float mean, inv;
  stats(x, width, eps, part, mean, inv);
  for (int i = threadIdx.x; i < width; i += 256) y[i] = __float2bfloat16_rn((ld(x + i) - mean) * inv * ld(W + i) + ld(B + i));
}

// The model's last norm (as tf_idol_dino_norm) into H, then ImageEncoder's F.layer_norm of it in fp32 into Y.
extern "C" __global__ void __launch_bounds__(256) tf_idol_dino_final(const bf16* __restrict__ X, const bf16* __restrict__ W,
                                                                     const bf16* __restrict__ B, bf16* __restrict__ H,
                                                                     float* __restrict__ Y, int width, float eps) {
  __shared__ float part[8];
  __shared__ float v[WIDTH];
  if (width > WIDTH) return;
  const size_t row = blockIdx.x;
  const bf16* x = X + row * width;
  float mean, inv;
  stats(x, width, eps, part, mean, inv);
  float a = 0.0f;
  for (int i = threadIdx.x; i < width; i += 256) {
    const bf16 o = __float2bfloat16_rn((ld(x + i) - mean) * inv * ld(W + i) + ld(B + i));
    H[row * width + i] = o;
    v[i] = __bfloat162float(o);
    a += v[i];
  }
  const float mean2 = block_sum(a, part) / (float)width;
  float q = 0.0f;
  for (int i = threadIdx.x; i < width; i += 256) {
    const float d = v[i] - mean2;
    q += d * d;
  }
  const float inv2 = 1.0f / sqrtf(block_sum(q, part) / (float)width + eps);
  for (int i = threadIdx.x; i < width; i += 256) Y[row * width + i] = (v[i] - mean2) * inv2;
}

// Packed bf16 [rows, q | k | v] from qkv's fp32 sums; q, k past the prefix bf16(bf16(x cos) + bf16(rot(x) sin)).
extern "C" __global__ void __launch_bounds__(256) tf_idol_dino_heads(const float* __restrict__ QKV, const bf16* __restrict__ CS,
                                                                     const bf16* __restrict__ SN, bf16* __restrict__ OUT,
                                                                     int rows, int prefix) {
  const int r = blockIdx.x, at = blockIdx.y * 256 + threadIdx.x;
  if (r >= rows || at >= HEADS * 32) return;
  const int h = at / 32, j = at % 32;
  const size_t from = (size_t)r * 3 * WIDTH + h * D + j;
  OUT[from + 2 * WIDTH] = __float2bfloat16_rn(QKV[from + 2 * WIDTH]);
  OUT[from + 2 * WIDTH + 32] = __float2bfloat16_rn(QKV[from + 2 * WIDTH + 32]);
  const float q0 = b(QKV[from]), q1 = b(QKV[from + 32]);
  const float k0 = b(QKV[from + WIDTH]), k1 = b(QKV[from + WIDTH + 32]);
  if (r < prefix) {
    OUT[from] = __float2bfloat16_rn(q0);
    OUT[from + 32] = __float2bfloat16_rn(q1);
    OUT[from + WIDTH] = __float2bfloat16_rn(k0);
    OUT[from + WIDTH + 32] = __float2bfloat16_rn(k1);
    return;
  }
  const size_t t = (size_t)(r - prefix) * D + j;
  const float c0 = ld(CS + t), c1 = ld(CS + t + 32), s0 = ld(SN + t), s1 = ld(SN + t + 32);
  OUT[from] = __float2bfloat16_rn(b(q0 * c0) + b(-q1 * s0));
  OUT[from + 32] = __float2bfloat16_rn(b(q1 * c1) + b(q0 * s1));
  OUT[from + WIDTH] = __float2bfloat16_rn(b(k0 * c0) + b(-k1 * s0));
  OUT[from + WIDTH + 32] = __float2bfloat16_rn(b(k1 * c1) + b(k0 * s1));
}

// x = bf16(x + bf16(g bf16(acc + b))) by column, in place: a linear's bias (one rounding), LayerScale, the residual.
extern "C" __global__ void __launch_bounds__(256) tf_idol_dino_scale_add(bf16* __restrict__ X, const float* __restrict__ ACC,
                                                                         const bf16* __restrict__ B, const bf16* __restrict__ G,
                                                                         int width, int rows) {
  const int c = blockIdx.y * 256 + threadIdx.x, r = blockIdx.x;
  if (c >= width || r >= rows) return;
  const size_t i = (size_t)r * width + c;
  const float y = b(ACC[i] + ld(B + c));
  X[i] = __float2bfloat16_rn(ld(X + i) + b(ld(G + c) * y));
}

// h [rows, width] = bf16(bf16(silu(g)) u), g = bf16(gate + bg) and u = bf16(up + bu) from fp32 sums [gate | up].
extern "C" __global__ void __launch_bounds__(256) tf_idol_dino_swiglu(const float* __restrict__ ACC, const bf16* __restrict__ B,
                                                                      bf16* __restrict__ Hd, int width, int rows) {
  const int c = blockIdx.y * 256 + threadIdx.x, r = blockIdx.x;
  if (c >= width || r >= rows) return;
  const size_t i = (size_t)r * 2 * width + c;
  const float g = b(ACC[i] + ld(B + c));
  const float u = b(ACC[i + width] + ld(B + width + c));
  Hd[(size_t)r * width + c] = __float2bfloat16_rn(b(g / (1.0f + expf(-g))) * u);
}

// Full attention at head width 64 over bf16 packed rows: idol_attention.cuh's body, its bf16 instance.
extern "C" __global__ void __launch_bounds__(256) tf_idol_dino_attention(const bf16* __restrict__ qkv, bf16* __restrict__ out,
                                                                         int W, int HO, float scale) {
  tf_idol_attn::full<64, 8, bf16>(qkv, out, W, HO, scale);
}
