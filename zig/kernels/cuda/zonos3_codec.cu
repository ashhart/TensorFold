// ours: ZONOS3's codec decoder on CUDA in fp32; a row or sample depends on its position, never on a stream's chunks.
#include <math.h>
#include <stdint.h>
#include "common/reduce.cuh"

namespace {

using tf_common::block_max_tree;
using tf_common::block_sum_tree;
using tf_common::warp_sum;

}  // namespace

// y[r, n] = sum_k W[n, k] x[r, k] (+ bias[n]): fp32 W [N, K], a warp an output, x staged in K * 4 bytes of shared.
extern "C" __global__ void __launch_bounds__(256) zc_linear(const float* __restrict__ W, const float* __restrict__ x,
                                                            int x_stride, int K, int N, float* __restrict__ y,
                                                            int y_stride, const float* __restrict__ bias) {
  extern __shared__ float xs[];
  const int r = blockIdx.y;
  const float* xr = x + (size_t)r * x_stride;
  for (int i = threadIdx.x; i < K; i += blockDim.x) xs[i] = xr[i];
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const int n = blockIdx.x * 8 + (threadIdx.x >> 5);
  const float* wr = W + (size_t)min(n, N - 1) * K;
  float acc = 0.f;
  for (int k = lane; k < K; k += 32) acc = fmaf(wr[k], xs[k], acc);
  acc = warp_sum(acc);
  if (lane == 0 && n < N) y[(size_t)r * y_stride + n] = bias ? acc + bias[n] : acc;
}

// LayerNorm over D: y = (x - mean) / sqrt(var + eps) * w + b; a block a row.
extern "C" __global__ void __launch_bounds__(256) zc_layernorm(const float* __restrict__ x, const float* __restrict__ w,
                                                               const float* __restrict__ b, float* __restrict__ y,
                                                               int D, float eps) {
  __shared__ float part[32];
  const float* xr = x + (size_t)blockIdx.x * D;
  float* yr = y + (size_t)blockIdx.x * D;
  float s = 0.f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) s += xr[i];
  const float mean = block_sum_tree(s, part, blockDim.x >> 5) / D;
  float v = 0.f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    const float d = xr[i] - mean;
    v = fmaf(d, d, v);
  }
  const float inv = 1.f / sqrtf(block_sum_tree(v, part, blockDim.x >> 5) / D + eps);
  for (int i = threadIdx.x; i < D; i += blockDim.x) yr[i] = (xr[i] - mean) * inv * w[i] + b[i];
}

// Rows [q | k | v] (each D = heads * 64) at pos0 + row: q and k rotate in adjacent pairs; k and v go to the cache row.
extern "C" __global__ void __launch_bounds__(32) zc_attn_prep(float* __restrict__ qkv, int D, int pos0,
                                                              float* __restrict__ kc, float* __restrict__ vc) {
  const int t = blockIdx.x, h = blockIdx.y, m = threadIdx.x;
  const float freq = expf((float)m * (-9.210340371976184f * 2.f / 64.f));
  const float phase = (float)(pos0 + t) * freq;
  const float c = cosf(phase), s = sinf(phase);
  float* q = qkv + (size_t)t * 3 * D + h * 64 + 2 * m;
  const float qr = q[0], qi = q[1];
  q[0] = qr * c - qi * s;
  q[1] = qr * s + qi * c;
  const float kr = q[D], ki = q[D + 1];
  const size_t at = (size_t)(pos0 + t) * D + h * 64 + 2 * m;
  kc[at] = kr * c - ki * s;
  kc[at + 1] = kr * s + ki * c;
  vc[at] = q[2 * D];
  vc[at + 1] = q[2 * D + 1];
}

// Causal local attention of (row, head) at p = pos0 + row over cache rows max(0, p - ctx + 1) .. p, in a fixed order.
extern "C" __global__ void __launch_bounds__(128) zc_attention(const float* __restrict__ qkv, int D,
                                                               const float* __restrict__ kc,
                                                               const float* __restrict__ vc, int pos0, int ctx,
                                                               float* __restrict__ out) {
  __shared__ float part[32];
  __shared__ float acc[128][65];
  const int t = blockIdx.x, h = blockIdx.y;
  const int p = pos0 + t;
  const int lo = max(0, p - ctx + 1);
  float qv[64];
  const float* qr = qkv + (size_t)t * 3 * D + h * 64;
#pragma unroll
  for (int i = 0; i < 64; ++i) qv[i] = qr[i];
  float m = -INFINITY;
  for (int j = lo + threadIdx.x; j <= p; j += blockDim.x) {
    const float* k = kc + (size_t)j * D + h * 64;
    float d = 0.f;
#pragma unroll
    for (int i = 0; i < 64; ++i) d = fmaf(qv[i], k[i], d);
    m = fmaxf(m, d * 0.125f);
  }
  m = block_max_tree(m, part, 4);
  float a[65];
#pragma unroll
  for (int i = 0; i < 65; ++i) a[i] = 0.f;
  for (int j = lo + threadIdx.x; j <= p; j += blockDim.x) {
    const float* k = kc + (size_t)j * D + h * 64;
    const float* v = vc + (size_t)j * D + h * 64;
    float d = 0.f;
#pragma unroll
    for (int i = 0; i < 64; ++i) d = fmaf(qv[i], k[i], d);
    const float e = expf(d * 0.125f - m);
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
  if (threadIdx.x < 64) out[(size_t)t * D + h * 64 + threadIdx.x] = acc[0][threadIdx.x] / acc[0][64];
}

// x[r, i] += scale[i] * y[r, i] (scale null: a plain residual add).
extern "C" __global__ void zc_add_scaled(float* __restrict__ x, const float* __restrict__ y,
                                         const float* __restrict__ scale, int D) {
  const size_t r = blockIdx.y;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < D) x[r * D + i] += scale ? scale[i] * y[r * D + i] : y[r * D + i];
}

// GELU (erf) in place.
extern "C" __global__ void zc_gelu(float* __restrict__ x, long long n) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = 0.5f * x[i] * (1.f + erff(x[i] * 0.70710678118654752f));
}

// Patch up by 2: in [T][C] to out [2T][C / 2], out[2t + j][c] = in[t][2c + j].
extern "C" __global__ void zc_patch_up(const float* __restrict__ in, float* __restrict__ out, int C) {
  const size_t r = blockIdx.y;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < C / 2) out[r * (C / 2) + c] = in[(r >> 1) * C + 2 * c + (r & 1)];
}

// Causal depthwise conv (kernel 7) + bias over T rows of C channels; hist [C][6] holds the 6 inputs before them.
extern "C" __global__ void zc_dwconv(const float* __restrict__ x, int T, const float* __restrict__ w,
                                     const float* __restrict__ b, float* __restrict__ hist, float* __restrict__ out,
                                     int C) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  float s[7];
#pragma unroll
  for (int j = 0; j < 6; ++j) s[j] = hist[c * 6 + j];
  for (int t = 0; t < T; ++t) {
    s[6] = x[(size_t)t * C + c];
    float acc = b[c];
#pragma unroll
    for (int j = 0; j < 7; ++j) acc = fmaf(w[c * 7 + j], s[j], acc);
    out[(size_t)t * C + c] = acc;
#pragma unroll
    for (int j = 0; j < 6; ++j) s[j] = s[j + 1];
  }
#pragma unroll
  for (int j = 0; j < 6; ++j) hist[c * 6 + j] = s[j];
}

// out[t] = table[codes[t][col]]: codes int32 [T][10], table [4096][D].
extern "C" __global__ void zc_gather(const int* __restrict__ codes, int col, const float* __restrict__ table,
                                     float* __restrict__ out, int D) {
  const size_t t = blockIdx.y;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < D) out[t * D + i] = table[(size_t)codes[t * 10 + col] * D + i];
}

// z[t] += tables[i][codes[t][1 + i]] for the nine residual codebooks in order: tables [9][4096][D].
extern "C" __global__ void zc_latent(const int* __restrict__ codes, const float* __restrict__ tables,
                                     float* __restrict__ z, int D) {
  const size_t t = blockIdx.y;
  const int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d >= D) return;
  float a = z[t * D + d];
#pragma unroll
  for (int i = 0; i < 9; ++i) a += tables[((size_t)i * 4096 + codes[t * 10 + 1 + i]) * D + d];
  z[t * D + d] = a;
}

// The head's [log-magnitude | phase] rows (each half bins wide) to [real | imaginary] in place, magnitude at most 100.
extern "C" __global__ void zc_spec(float* __restrict__ y, int half) {
  float* r = y + (size_t)blockIdx.y * 2 * half;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= half) return;
  const float lm = r[i];
  const float mag = expf(lm > 4.605170185988092f ? 4.605170185988092f : lm);  // a comparison: a NaN stays NaN
  const float ph = r[half + i];
  r[i] = mag * cosf(ph);
  r[half + i] = mag * sinf(ph);
}

// Centered overlap-add of windowed frames [F][N] at hop: out[i] is sample n0 + i over the same frames' window envelope.
extern "C" __global__ void zc_ola(const float* __restrict__ frames, const float* __restrict__ win2, int F, int N,
                                  int hop, long long n0, long long count, float* __restrict__ out) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const long long p = n0 + i + N / 2;
  long long f0 = (p - N) / hop + 1;
  if (p - N < 0) f0 = 0;
  long long f1 = p / hop;
  if (f1 > F - 1) f1 = F - 1;
  float s = 0.f, e = 0.f;
  for (long long f = f0; f <= f1; ++f) {
    const int o = (int)(p - f * hop);
    s += frames[(size_t)f * N + o];
    e += win2[o];
  }
  out[i] = s / e;
}

// The mixer: out[i] = sum_b sum_j w[b][j] s_b[n0 + i + j - 63], zero outside [0, total); s rows are `stride` apart.
extern "C" __global__ void zc_mix(const float* __restrict__ s, long long stride, long long total,
                                  const float* __restrict__ w, long long n0, long long count,
                                  float* __restrict__ out) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  float a = 0.f;
  for (int b = 0; b < 2; ++b)
    for (int j = 0; j < 127; ++j) {
      const long long p = n0 + i + j - 63;
      if (p >= 0 && p < total) a = fmaf(w[b * 127 + j], s[(size_t)b * stride + p], a);
    }
  out[i] = a;
}

// 44160 to 44100 Hz: output m0 + i = 735 q + c is sum_j kernel[c][j] x[736 q + j - 17], zero outside [0, total).
extern "C" __global__ void zc_resample(const float* __restrict__ x, long long total, const float* __restrict__ kernel,
                                       long long m0, long long count, float* __restrict__ out) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const long long m = m0 + i;
  const long long q = m / 735;
  const int c = (int)(m % 735);
  float a = 0.f;
  for (int j = 0; j < 770; ++j) {
    const long long p = 736 * q + j - 17;
    if (p >= 0 && p < total) a = fmaf(kernel[c * 770 + j], x[p], a);
  }
  out[i] = a;
}
