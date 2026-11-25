// MOSS-Image's prompt encoder (Qwen3.5's text model) on CUDA: moss_image_encoder.metal's kernels op for op.
#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

#include "common/reduce.cuh"

namespace tf_me {

typedef __nv_bfloat16 bf16;

// torch's cast to bf16: round to nearest even, a NaN stays a NaN
__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float ld(const bf16* p) { return __bfloat162float(*p); }
// IEEE division and square root: the glue flags keep both exact enough for the host's references
__device__ __forceinline__ float silu(float x) { return __fdiv_rn(x, 1.0f + expf(-x)); }
__device__ __forceinline__ float sigmoid(float x) { return __fdiv_rn(1.0f, 1.0f + expf(-x)); }

}  // namespace tf_me

using namespace tf_me;

// y = bf16(x rsqrt(mean(x^2) + eps) (1 + w)) in fp32 (Qwen3.5's zero-centered RMSNorm); a block of 256 a row.
extern "C" __global__ void __launch_bounds__(256) tf_me_norm(const bf16* __restrict__ X, const bf16* __restrict__ W,
                                                             bf16* __restrict__ Y, int width, int in_stride,
                                                             int out_stride, float eps) {
  __shared__ float part[8];
  const bf16* x = X + (size_t)blockIdx.x * in_stride;
  bf16* y = Y + (size_t)blockIdx.x * out_stride;
  float a = 0.0f;
  for (int i = threadIdx.x; i < width; i += 256) a += ld(x + i) * ld(x + i);
  a = tf_common::block_sum_tree(a, part, 8);
  const float inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(a, (float)width), eps)));
  for (int i = threadIdx.x; i < width; i += 256) y[i] = __float2bfloat16_rn((ld(x + i) * inv) * (1.0f + ld(W + i)));
}

// 256-wide heads to [heads, rows, 256], normed and the first 64 channels turned by CS, SN (else copied); a warp a head.
extern "C" __global__ void __launch_bounds__(256) tf_me_head(const bf16* __restrict__ X, const bf16* __restrict__ W,
                                                             const float* __restrict__ CS, const float* __restrict__ SN,
                                                             bf16* __restrict__ Y, int rows, int heads, int in_stride,
                                                             int head_stride, float eps, int normed) {
  const int r = blockIdx.x, h = blockIdx.y * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (h >= heads) return;
  const size_t from = (size_t)r * in_stride + (size_t)h * head_stride + lane * 8;
  const size_t to = ((size_t)h * rows + r) * 256 + lane * 8;
  if (normed == 0) {
#pragma unroll
    for (int i = 0; i < 8; ++i) Y[to + i] = X[from + i];
    return;
  }
  float v[8];
  float ss = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    v[i] = ld(X + from + i);
    ss += v[i] * v[i];
  }
  const float inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(tf_common::warp_sum(ss), 256.0f), eps)));
  float t[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) t[i] = bf(v[i] * inv * (1.0f + ld(W + lane * 8 + i)));
  // lanes 0-7 hold the turned 64 channels; a channel's partner is 32 channels, four lanes, away
  const float sign = (lane & 4) != 0 ? 1.0f : -1.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float partner = sign * __shfl_xor_sync(0xffffffffu, t[i], 4);
    float o = t[i];
    if (lane < 8) {
      const int j = (lane & 3) * 8 + i;
      o = bf(t[i] * CS[(size_t)r * 32 + j]) + bf(partner * SN[(size_t)r * 32 + j]);
    }
    Y[to + i] = __float2bfloat16_rn(o);
  }
}

// o [rows, heads * 256] = causal softmax(q k^T scale) v in fp32, keys in order; a block a query row and head.
extern "C" __global__ void __launch_bounds__(256) tf_me_attention(const bf16* __restrict__ Q, const bf16* __restrict__ K,
                                                                  const bf16* __restrict__ V, bf16* __restrict__ O,
                                                                  int rows, int heads, int kv_heads, float scale) {
  __shared__ float part[8];
  const int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
  const int group = heads / kv_heads, kvh = h / group;
  const float q = ld(Q + ((size_t)h * rows + t) * 256 + d);
  float m = -INFINITY, l = 0.0f, acc = 0.0f;
  for (int j = 0; j <= t; ++j) {
    const size_t at = ((size_t)kvh * rows + j) * 256 + d;
    const float s = tf_common::block_sum_tree(q * ld(K + at), part, 8) * scale;
    const float top = fmaxf(m, s);
    const float old = expf(m - top), e = expf(s - top);
    l = l * old + e;
    acc = acc * old + e * ld(V + at);
    m = top;
  }
  O[((size_t)t * heads + h) * 256 + d] = __float2bfloat16_rn(__fdiv_rn(acc, l));
}

// o = bf16(a bf16(sigmoid(g))): the attention rows gated by the second half of each 512-wide q_proj head.
extern "C" __global__ void __launch_bounds__(256) tf_me_gate(const bf16* __restrict__ A, const bf16* __restrict__ QG,
                                                             bf16* __restrict__ O, int width) {
  const int row = blockIdx.y, col = blockIdx.x * 256 + threadIdx.x;
  if (col >= width) return;
  const size_t i = (size_t)row * width + col;
  const float g = ld(QG + (size_t)row * 2 * width + (col / 256) * 512 + 256 + col % 256);
  O[i] = __float2bfloat16_rn(ld(A + i) * bf(sigmoid(g)));
}

// o = bf16(bf16(silu(g)) u): the MLP's product.
extern "C" __global__ void __launch_bounds__(256) tf_me_silu_mul(const bf16* __restrict__ G, const bf16* __restrict__ U,
                                                                 bf16* __restrict__ O, int width) {
  const int row = blockIdx.y, col = blockIdx.x * 256 + threadIdx.x;
  if (col >= width) return;
  const size_t i = (size_t)row * width + col;
  const float g = ld(G + i);
  O[i] = __float2bfloat16_rn(bf(silu(g)) * ld(U + i));
}

// y = bf16(silu(bf16(the 4-tap causal depthwise conv of x))) by channel, the taps in order, rows before 0 zero.
extern "C" __global__ void __launch_bounds__(256) tf_me_conv(const bf16* __restrict__ X, const bf16* __restrict__ W,
                                                             bf16* __restrict__ Y, int width) {
  const int row = blockIdx.y, col = blockIdx.x * 256 + threadIdx.x;
  if (col >= width) return;
  float s = 0.0f;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    if (row + k >= 3) s += ld(W + col * 4 + k) * ld(X + (size_t)(row + k - 3) * width + col);
  }
  const float c = bf(s);
  Y[(size_t)row * width + col] = __float2bfloat16_rn(silu(c));
}

// The DeltaNet's fp32 inputs: q, k L2-normed (q scaled), beta = bf16(sigmoid(b)), g = -exp(A_log) softplus(a + dt).
extern "C" __global__ void __launch_bounds__(256) tf_me_gdn_prep(const bf16* __restrict__ MIX, const bf16* __restrict__ A,
                                                                 const bf16* __restrict__ B, const bf16* __restrict__ A_LOG,
                                                                 const bf16* __restrict__ DT_BIAS, float* __restrict__ QN,
                                                                 float* __restrict__ KN, float* __restrict__ G,
                                                                 float* __restrict__ BETA, int rows, int kheads, int vheads,
                                                                 int mix_stride, float scale) {
  const int r = blockIdx.x, h = blockIdx.y * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (h >= kheads) return;
  const size_t row = (size_t)r * mix_stride;
  const int kdim = kheads * 128;
  float q[4], k[4];
  float qq = 0.0f, kk = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    q[i] = ld(MIX + row + h * 128 + lane * 4 + i);
    k[i] = ld(MIX + row + kdim + h * 128 + lane * 4 + i);
    qq += q[i] * q[i];
    kk += k[i] * k[i];
  }
  const float qi = __fdiv_rn(1.0f, __fsqrt_rn(tf_common::warp_sum(qq) + 1e-6f));
  const float ki = __fdiv_rn(1.0f, __fsqrt_rn(tf_common::warp_sum(kk) + 1e-6f));
  const size_t to = ((size_t)r * kheads + h) * 128 + lane * 4;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    QN[to + i] = q[i] * qi * scale;
    KN[to + i] = k[i] * ki;
  }
  const int per = vheads / kheads;
  if (lane < per) {
    const int v = h * per + lane;
    const size_t at = (size_t)r * vheads + v;
    BETA[at] = bf(sigmoid(ld(B + at)));
    const float x = ld(A + at) + ld(DT_BIAS + v);
    const float soft = x > 20.0f ? x : logf(1.0f + expf(x));
    G[at] = -expf(ld(A_LOG + v)) * soft;
  }
}

// S = S exp(g); d = (v - S^T k) beta; S += k d^T; o = S^T q over the rows in order, a thread a value column in fp32.
extern "C" __global__ void __launch_bounds__(128) tf_me_gdn(const float* __restrict__ QN, const float* __restrict__ KN,
                                                            const bf16* __restrict__ MIX, const float* __restrict__ G,
                                                            const float* __restrict__ BETA, bf16* __restrict__ OUT,
                                                            int rows, int kheads, int vheads, int mix_stride) {
  __shared__ float kq[256];
  const int h = blockIdx.x, j = threadIdx.x;
  const int per = vheads / kheads, kh = h / per;
  const size_t voff = (size_t)2 * kheads * 128 + (size_t)h * 128 + j;
  float s[128];
#pragma unroll
  for (int i = 0; i < 128; ++i) s[i] = 0.0f;
  for (int t = 0; t < rows; ++t) {
    const size_t at = ((size_t)t * kheads + kh) * 128 + j;
    kq[j] = KN[at];
    kq[128 + j] = QN[at];
    __syncthreads();
    const float decay = expf(G[(size_t)t * vheads + h]);
    const float v = ld(MIX + (size_t)t * mix_stride + voff);
    float kv = 0.0f;
#pragma unroll
    for (int i = 0; i < 128; ++i) {
      s[i] *= decay;
      kv += s[i] * kq[i];
    }
    const float delta = (v - kv) * BETA[(size_t)t * vheads + h];
    float o = 0.0f;
#pragma unroll
    for (int i = 0; i < 128; ++i) {
      s[i] += kq[i] * delta;
      o += s[i] * kq[128 + i];
    }
    OUT[((size_t)t * vheads + h) * 128 + j] = __float2bfloat16_rn(o);
    __syncthreads();
  }
}

// y = bf16(bf16(w bf16(x rsqrt(mean x^2 + eps))) silu(z)) over each 128-wide head; a warp a row and head.
extern "C" __global__ void __launch_bounds__(256) tf_me_gated_norm(const bf16* __restrict__ X, const bf16* __restrict__ Z,
                                                                   const bf16* __restrict__ W, bf16* __restrict__ Y,
                                                                   int width, float eps) {
  const int r = blockIdx.x, h = blockIdx.y * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (h * 128 >= width) return;
  const size_t at = (size_t)r * width + h * 128 + lane * 4;
  float v[4];
  float ss = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    v[i] = ld(X + at + i);
    ss += v[i] * v[i];
  }
  const float inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(tf_common::warp_sum(ss), 128.0f), eps)));
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float n = bf(ld(W + lane * 4 + i) * bf(v[i] * inv));
    Y[at + i] = __float2bfloat16_rn(n * silu(ld(Z + at + i)));
  }
}
