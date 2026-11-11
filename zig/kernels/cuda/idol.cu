// Idol's denoiser glue on CUDA: idol.metal's steps op for op, fp16 rows rounded where idol-sdk's torch code rounds.
#include <cuda_fp16.h>
#include <math.h>
#include <stdint.h>

#include "idol_attention.cuh"

namespace tf_idol {

typedef __half f16;

__device__ __forceinline__ float h(float x) { return __half2float(__float2half_rn(x)); }
__device__ __forceinline__ float ld(const f16* p) { return __half2float(*p); }

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

// The row's mean and 1 / sqrt(var + eps), each sum in fp32: LayerNorm32's statistics.
__device__ __forceinline__ void stats(const f16* x, int width, float eps, float* part, float& mean, float& inv) {
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

// F.normalize's clamp_min(1e-12) as torch does it: a NaN norm stays NaN (fmaxf would drop it).
__device__ __forceinline__ float clamp_min(float n) { return n < 1e-12f ? 1e-12f : n; }

// torch's tanh GELU of an fp16 value, in fp32.
__device__ __forceinline__ float gelu(float x) {
  const float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
  return 0.5f * x * (1.0f + tanhf(inner));
}

}  // namespace tf_idol

using namespace tf_idol;

// LayerNorm32 of an fp16 row to fp16: affine (mode 1) in fp32 before the rounding, modulated (mode 2) in fp16 after.
extern "C" __global__ void __launch_bounds__(256) tf_idol_norm(const f16* __restrict__ X, const float* __restrict__ W,
                                                               const float* __restrict__ B, const f16* __restrict__ S1,
                                                               const f16* __restrict__ SH, f16* __restrict__ Y, int width,
                                                               float eps, int mode) {
  __shared__ float part[8];
  const f16* x = X + (size_t)blockIdx.x * width;
  f16* y = Y + (size_t)blockIdx.x * width;
  float mean, inv;
  stats(x, width, eps, part, mean, inv);
  for (int i = threadIdx.x; i < width; i += 256) {
    float v = (ld(x + i) - mean) * inv;
    if (mode == 1) v = v * W[i] + B[i];
    v = h(v);
    if (mode == 2) v = h(h(v * ld(S1 + i)) + ld(SH + i));
    y[i] = __float2half_rn(v);
  }
}

// LayerNorm of an fp16 row kept in fp32: affine (mode 1) y w + b, or modulated (mode 2) y (1 + w) + b.
extern "C" __global__ void __launch_bounds__(256) tf_idol_norm32(const f16* __restrict__ X, const float* __restrict__ W,
                                                                 const float* __restrict__ B, float* __restrict__ Y,
                                                                 int width, float eps, int mode) {
  __shared__ float part[8];
  const f16* x = X + (size_t)blockIdx.x * width;
  float* y = Y + (size_t)blockIdx.x * width;
  float mean, inv;
  stats(x, width, eps, part, mean, inv);
  for (int i = threadIdx.x; i < width; i += 256) {
    const float v = (ld(x + i) - mean) * inv;
    y[i] = mode == 2 ? v * (1.0f + W[i]) + B[i] : v * W[i] + B[i];
  }
}

// RePo: d = final_map(silu(gate) content), 3 a head; angles clamp_mul(d, freqs) pi as cos, sin [rows, 16, 32].
extern "C" __global__ void __launch_bounds__(64) tf_idol_repo(const float* __restrict__ GC, const float* __restrict__ WF,
                                                              const float* __restrict__ F, float* __restrict__ CS,
                                                              float* __restrict__ SN, int rows) {
  __shared__ float u[128];
  __shared__ float d[48];
  const int row = blockIdx.x, t = threadIdx.x;
  if (row >= rows) return;
  const float* g = GC + (size_t)row * 256;
  for (int i = t; i < 128; i += 64) {
    const float a = g[i];
    u[i] = (a / (1.0f + expf(-a))) * g[128 + i];
  }
  __syncthreads();
  if (t < 48) {
    float s = 0.0f;
    for (int k = 0; k < 128; ++k) s = __fmaf_rn(u[k], WF[t * 128 + k], s);
    d[t] = s;
  }
  __syncthreads();
  for (int i = t; i < 512; i += 64) {
    const int hd = i / 32, j = i % 32;
    const int axis = j < 10 ? 0 : (j < 20 ? 1 : 2);
    const float x = d[hd * 3 + axis], fr = F[j];
    const float ft = tanhf(fr);
    const float ang = (x * ft + x * (fr - ft)) * 3.14159265358979323846f;
    CS[(size_t)row * 512 + i] = cosf(ang);
    SN[(size_t)row * 512 + i] = sinf(ang);
  }
}

// q, k of qkv rows turned by cos, sin (to fp16), F.normalize times gamma times `scale`; v copied. A warp a head.
extern "C" __global__ void __launch_bounds__(256) tf_idol_qk(const f16* __restrict__ QKV, const float* __restrict__ GQ,
                                                             const float* __restrict__ GK, const float* __restrict__ CS,
                                                             const float* __restrict__ SN, f16* __restrict__ OUT, int rows,
                                                             float scale) {
  const int r = blockIdx.x, hd = blockIdx.y * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (hd >= 16 || r >= rows) return;
  const size_t from = (size_t)r * 3072 + hd * 64 + lane * 2;
  const size_t at = (size_t)r * 512 + hd * 32 + lane;
  const float c = CS[at], s = SN[at];
  const float q0 = ld(QKV + from), q1 = ld(QKV + from + 1);
  const float k0 = ld(QKV + from + 1024), k1 = ld(QKV + from + 1025);
  const float rq0 = h(q0 * c - q1 * s), rq1 = h(q0 * s + q1 * c);
  const float rk0 = h(k0 * c - k1 * s), rk1 = h(k0 * s + k1 * c);
  const float qn = clamp_min(sqrtf(warp_sum(rq0 * rq0 + rq1 * rq1)));
  const float kn = clamp_min(sqrtf(warp_sum(rk0 * rk0 + rk1 * rk1)));
  const int gi = hd * 64 + lane * 2;
  f16* o = OUT + from;
  o[0] = __float2half_rn(rq0 / qn * GQ[gi] * scale);
  o[1] = __float2half_rn(rq1 / qn * GQ[gi + 1] * scale);
  o[1024] = __float2half_rn(rk0 / kn * GK[gi] * scale);
  o[1025] = __float2half_rn(rk1 / kn * GK[gi + 1] * scale);
  o[2048] = QKV[from + 2048];
  o[2049] = QKV[from + 2049];
}

// x = fp16(x + fp16(g y)) by column; g is a modulation gate, or ones for an ungated residual. In place.
extern "C" __global__ void __launch_bounds__(256) tf_idol_gate_add(f16* __restrict__ X, const f16* __restrict__ Y,
                                                                   const f16* __restrict__ G, int width, int rows) {
  const int c = blockIdx.y * 256 + threadIdx.x, r = blockIdx.x;
  if (c >= width || r >= rows) return;
  const size_t i = (size_t)r * width + c;
  X[i] = __float2half_rn(ld(X + i) + h(ld(G + c) * ld(Y + i)));
}

// y = fp16(acc + b): a block linear's fp32 sums and its fp16 bias, one rounding; then fp16(GELU(y)) when `act`.
extern "C" __global__ void __launch_bounds__(256) tf_idol_bias(const float* __restrict__ ACC, const f16* __restrict__ B,
                                                               f16* __restrict__ Y, int width, int rows, int act) {
  const int c = blockIdx.y * 256 + threadIdx.x, r = blockIdx.x;
  if (c >= width || r >= rows) return;
  const size_t i = (size_t)r * width + c;
  float v = h(ACC[i] + ld(B + c));
  if (act) v = gelu(v);
  Y[i] = __float2half_rn(v);
}

// x = fp16(fp16(z w^T + b) + pe): the latent's input projection (16 inputs, fp32) and its fp16 position embedding.
extern "C" __global__ void __launch_bounds__(256) tf_idol_embed(const float* __restrict__ Z, const float* __restrict__ W,
                                                                const float* __restrict__ B, const f16* __restrict__ PE,
                                                                f16* __restrict__ X, int rows) {
  const int c = blockIdx.y * 256 + threadIdx.x, r = blockIdx.x;
  if (c >= 1024 || r >= rows) return;
  float s = 0.0f;
#pragma unroll
  for (int k = 0; k < 16; ++k) s = __fmaf_rn(Z[(size_t)r * 16 + k], W[c * 16 + k], s);
  const size_t i = (size_t)r * 1024 + c;
  X[i] = __float2half_rn(h(s + B[c]) + ld(PE + i));
}

// y = fp16((a + ba) + (c + bc)) by column: the two condition projections' fp32 sums and biases.
extern "C" __global__ void __launch_bounds__(256) tf_idol_cond(const float* __restrict__ A, const float* __restrict__ BA,
                                                               const float* __restrict__ C, const float* __restrict__ BC,
                                                               f16* __restrict__ Y, int width, int rows) {
  const int c = blockIdx.y * 256 + threadIdx.x, r = blockIdx.x;
  if (c >= width || r >= rows) return;
  const size_t i = (size_t)r * width + c;
  Y[i] = __float2half_rn((A[i] + BA[c]) + (C[i] + BC[c]));
}

// y += b by column in fp32 (an fp32 Linear's bias), in place.
extern "C" __global__ void __launch_bounds__(256) tf_idol_bias32(float* __restrict__ Y, const float* __restrict__ B,
                                                                 int width, int rows) {
  const int c = blockIdx.y * 256 + threadIdx.x, r = blockIdx.x;
  if (c >= width || r >= rows) return;
  Y[(size_t)r * width + c] += B[c];
}

// Full attention at head width 64 over fp16 packed rows; idol_attention.cuh has the body.
extern "C" __global__ void __launch_bounds__(256) tf_idol_attention(const f16* __restrict__ qkv, f16* __restrict__ out,
                                                                    int W, int HO, float scale) {
  tf_idol_attn::full<64, 8>(qkv, out, W, HO, scale);
}
