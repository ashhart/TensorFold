// ours: ZONOS3's SSL teacher (Wav2Vec2-BERT) on CUDA in fp32: relative-key attention, the conformer's convolution.
#include <math.h>
#include <stdint.h>
#include "common/reduce.cuh"
#include "common/act.cuh"

namespace {

using tf_common::block_max_tree;

}  // namespace

// Swish in place: x * sigmoid(x).
extern "C" __global__ void zt_swish(float* __restrict__ x, long long n) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = tf_common::silu(x[i]);
}

// GLU: out[r, c] = in[r, c] * sigmoid(in[r, C + c]).
extern "C" __global__ void zt_glu(const float* __restrict__ in, float* __restrict__ out, int C) {
  const size_t r = blockIdx.y;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < C) out[r * C + c] = in[r * 2 * C + c] * tf_common::sigmoidf_(in[r * 2 * C + C + c]);
}

// Depthwise conv of K taps, the newest input last, zeros before row 0: out[t, c] = sum_j w[c][j] x[t - K + 1 + j, c].
extern "C" __global__ void zt_dwconv(const float* __restrict__ x, const float* __restrict__ w, int K, int C,
                                     float* __restrict__ out) {
  const long long t = blockIdx.y;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  float a = 0.f;
  for (int j = 0; j < K; ++j) {
    const long long p = t - K + 1 + j;
    if (p >= 0) a = fmaf(w[c * K + j], x[p * C + c], a);
  }
  out[t * C + c] = a;
}

// Attention of (row l, head h) over all T rows: score(r) = (q . k_r + q . E[clamp(r - l, -64, 8) + 64]) / 8.
extern "C" __global__ void __launch_bounds__(128) zt_attention(const float* __restrict__ qkv, int D, int T,
                                                               const float* __restrict__ E,
                                                               float* __restrict__ out) {
  __shared__ float part[32];
  __shared__ float pe[73];
  __shared__ float acc[128][65];
  const int l = blockIdx.x, h = blockIdx.y;
  float qv[64];
  const float* qr = qkv + (size_t)l * 3 * D + h * 64;
#pragma unroll
  for (int i = 0; i < 64; ++i) qv[i] = qr[i];
  if (threadIdx.x < 73) {
    const float* e = E + threadIdx.x * 64;
    float d = 0.f;
#pragma unroll
    for (int i = 0; i < 64; ++i) d = fmaf(qv[i], e[i], d);
    pe[threadIdx.x] = d;
  }
  __syncthreads();
  float m = -INFINITY;
  for (int r = threadIdx.x; r < T; r += blockDim.x) {
    const float* k = qkv + (size_t)r * 3 * D + D + h * 64;
    float d = 0.f;
#pragma unroll
    for (int i = 0; i < 64; ++i) d = fmaf(qv[i], k[i], d);
    m = fmaxf(m, (d + pe[min(max(r - l, -64), 8) + 64]) * 0.125f);
  }
  m = block_max_tree(m, part, 4);
  float a[65];
#pragma unroll
  for (int i = 0; i < 65; ++i) a[i] = 0.f;
  for (int r = threadIdx.x; r < T; r += blockDim.x) {
    const float* k = qkv + (size_t)r * 3 * D + D + h * 64;
    const float* v = k + D;
    float d = 0.f;
#pragma unroll
    for (int i = 0; i < 64; ++i) d = fmaf(qv[i], k[i], d);
    const float e = expf((d + pe[min(max(r - l, -64), 8) + 64]) * 0.125f - m);
    a[64] += e;
#pragma unroll
    for (int i = 0; i < 64; ++i) a[i] = fmaf(e, v[i], a[i]);
  }
#pragma unroll
  for (int i = 0; i < 65; ++i) acc[threadIdx.x][i] = a[i];
  __syncthreads();
  if (threadIdx.x < 65) {
    float s = 0.f;
    for (int w = 0; w < 128; ++w) s += acc[w][threadIdx.x];
    acc[0][threadIdx.x] = s;
  }
  __syncthreads();
  if (threadIdx.x < 64) out[(size_t)l * D + h * 64 + threadIdx.x] = acc[0][threadIdx.x] / acc[0][64];
}
