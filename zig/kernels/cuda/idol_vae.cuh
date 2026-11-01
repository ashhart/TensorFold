// Idol's image encoder (FLUX.2's VAE encoder) in fp32 on CUDA, each picture tensor [rows, columns, channels].
#pragma once
#include <cuda_fp16.h>
#include <math.h>
#include <stdint.h>

namespace tf_iv {

// Each body takes a flaw F: idol_vae.cu builds F = none alone; contract_controls.cu builds the controls' flaws.
enum Flaw {
  none = 0,
  conv_in_half = 1,   // conv_in stages 2 x - 1 in half
  conv_half_in = 2,   // conv stages its input tile in half
  conv_sym_pad = 3,   // conv's stride-2 3x3 pads above and left as padding=1 would
  conv_half_sum = 4,  // conv sums in half
  norm_half = 5,      // norm's SiLU in half
  part_half = 6,      // group_part sums in half
  finish_half = 7,    // group_finish sums in half
  softmax_half = 8,   // softmax's denominator summed in half
  bias_swap = 9,      // bias by row reads the column's bias
  tokens_axes = 10,   // tokens swaps a patch's axes
  tokens_clamp = 11,  // tokens clamps a NaN logvar (fminf, fmaxf)
};

__device__ __forceinline__ float h(float x) { return __half2float(__float2half_rn(x)); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

// y [pixels, 128] = conv3x3(2 x - 1) + b of the picture x [pixels, 3] in [0, 1], zero padded; w [27, 128] by tap.
template <int F>
__device__ __forceinline__ void conv_in(const float* __restrict__ x, const float* __restrict__ w,
                                        const float* __restrict__ b, float* __restrict__ y, int rows, int cols) {
  const int o = threadIdx.x;
  const long long p = (long long)blockIdx.x * 2 + threadIdx.y;
  if (p >= (long long)rows * cols) return;
  const int py = (int)(p / cols), px = (int)(p % cols);
  float s = 0.0f;
#pragma unroll
  for (int ky = 0; ky < 3; ++ky) {
#pragma unroll
    for (int kx = 0; kx < 3; ++kx) {
      const int iy = py + ky - 1, ix = px + kx - 1;
      if (iy < 0 || ix < 0 || iy >= rows || ix >= cols) continue;
      const float* v = x + ((size_t)iy * cols + ix) * 3;
#pragma unroll
      for (int c = 0; c < 3; ++c) {
        float a = v[c] * 2.0f - 1.0f;
        if (F == conv_in_half) a = h(a);
        s = __fmaf_rn(a, w[((ky * 3 + kx) * 3 + c) * 128 + o], s);
      }
    }
  }
  y[(size_t)p * 128 + o] = s + b[o];
}

// The implicit GEMM's tile: 128 pixels by 128 output channels a block of 256, 8 of K a step, 8 x 8 outputs a thread.
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 8;
constexpr int LDA = BM + 4;

// y = conv(x) + b (+ r when add), w [taps * cin, ldw]: one fp32 FMA chain over the taps' input channels, then b, r.
template <int F>
__device__ __forceinline__ void conv(const float* __restrict__ X, const float* __restrict__ W,
                                     const float* __restrict__ B, const float* __restrict__ R, float* __restrict__ Y,
                                     int rows, int cols, int in_rows, int in_cols, int cin, int cout, int ldw, int taps,
                                     int stride, int pad, int add) {
  __shared__ __align__(16) float As[2][BK][LDA];
  __shared__ __align__(16) float Bs[2][BK][BN];
  const int t = threadIdx.x;
  const long long pixels = (long long)rows * cols;
  const long long m0 = (long long)blockIdx.x * BM;
  const int n0 = blockIdx.y * BN;
  // pad: rows and columns above and left (a same-size 3x3: 1; the stride-2 downsample, padded right and below: 0)
  if (F == conv_sym_pad && stride == 2) pad = 1;
  // this thread's loads: four input channels of one pixel, four weight columns of one K row
  const int am = t >> 1, aq = (t & 1) * 4;
  const long long apix = m0 + am;
  const bool alive = apix < pixels;
  const int oy = alive ? (int)(apix / cols) : 0;
  const int ox = alive ? (int)(apix % cols) : 0;
  const int bk = t >> 5, bc = (t & 31) * 4;
  const int K = taps * cin;
  const int tx = t & 15, ty = t >> 4;
  float acc[8][8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[i][j] = 0.0f;
  }
  float4 a, bq;
  // K row k0's tap and channel, the input pixel it reads (zero outside the picture), and the weights
  auto load = [&](int k0) {
    const int tap = k0 / cin;
    const int c0 = k0 - tap * cin;
    const int ky = taps == 9 ? tap / 3 : 0;
    const int kx = taps == 9 ? tap % 3 : 0;
    const int iy = oy * stride + ky - pad, ix = ox * stride + kx - pad;
    a = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (alive && iy >= 0 && ix >= 0 && iy < in_rows && ix < in_cols)
      a = *reinterpret_cast<const float4*>(X + ((size_t)iy * in_cols + ix) * cin + c0 + aq);
    bq = *reinterpret_cast<const float4*>(W + (size_t)(k0 + bk) * ldw + n0 + bc);
  };
  auto store = [&](int s) {
    if (F == conv_half_in) a = make_float4(h(a.x), h(a.y), h(a.z), h(a.w));
    As[s][aq + 0][am] = a.x;
    As[s][aq + 1][am] = a.y;
    As[s][aq + 2][am] = a.z;
    As[s][aq + 3][am] = a.w;
    *reinterpret_cast<float4*>(&Bs[s][bk][bc]) = bq;
  };
  load(0);
  store(0);
  __syncthreads();
  int s = 0;
  for (int k0 = 0; k0 < K; k0 += BK) {
    const bool more = k0 + BK < K;
    if (more) load(k0 + BK);
#pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
      const float4 a0 = *reinterpret_cast<const float4*>(&As[s][kk][ty * 4]);
      const float4 a1 = *reinterpret_cast<const float4*>(&As[s][kk][64 + ty * 4]);
      const float4 b0 = *reinterpret_cast<const float4*>(&Bs[s][kk][tx * 4]);
      const float4 b1 = *reinterpret_cast<const float4*>(&Bs[s][kk][64 + tx * 4]);
      const float av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
      const float bv[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
      for (int i = 0; i < 8; ++i) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          acc[i][j] = __fmaf_rn(av[i], bv[j], acc[i][j]);
          if (F == conv_half_sum) acc[i][j] = h(acc[i][j]);
        }
      }
    }
    // the other buffer was last read before the previous step's barrier
    if (more) store(s ^ 1);
    __syncthreads();
    s ^= 1;
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const long long row = m0 + (i < 4 ? ty * 4 + i : 64 + ty * 4 + i - 4);
    if (row >= pixels) continue;
#pragma unroll
    for (int jh = 0; jh < 2; ++jh) {
      const int col = n0 + jh * 64 + tx * 4;
      if (col >= cout) continue;
      const size_t at = (size_t)row * cout + col;
      float4 v = make_float4(acc[i][jh * 4] + B[col], acc[i][jh * 4 + 1] + B[col + 1], acc[i][jh * 4 + 2] + B[col + 2],
                             acc[i][jh * 4 + 3] + B[col + 3]);
      if (add) {
        const float4 r = *reinterpret_cast<const float4*>(R + at);
        v = make_float4(v.x + r.x, v.y + r.y, v.z + r.z, v.w + r.w);
      }
      *reinterpret_cast<float4*>(Y + at) = v;
    }
  }
}

// One chunk of pixels' sums of x a group, or of (x - mean)^2 when centered, into part [32, chunks]: a thread a channel.
template <int F>
__device__ __forceinline__ void group_part(const float* __restrict__ x, const float* __restrict__ stats,
                                           float* __restrict__ part, int pixels, int channels, int chunk, int chunks,
                                           int centered) {
  __shared__ float sums[512];
  const int t = threadIdx.x;
  const int size = channels / 32;
  const long long first = (long long)blockIdx.x * chunk;
  const long long last = min(first + chunk, (long long)pixels);
  const float mean = centered ? stats[t / size] : 0.0f;
  float s = 0.0f;
  for (long long px = first; px < last; ++px) {
    const float v = x[(size_t)px * channels + t];
    const float d = v - mean;
    const float term = centered ? d * d : v;
    s = F == part_half ? h(h(s) + h(term)) : s + term;
  }
  sums[t] = s;
  __syncthreads();
  if (t < 32) {
    float all = 0.0f;
    for (int c = 0; c < size; ++c) all = F == part_half ? h(h(all) + h(sums[t * size + c])) : all + sums[t * size + c];
    part[t * chunks + blockIdx.x] = all;
  }
}

// stats from part's chunk sums, a warp a group: mean[32] when not centered, else rstd[32] = 1 / sqrt(var + eps).
template <int F>
__device__ __forceinline__ void group_finish(const float* __restrict__ part, float* __restrict__ stats, int pixels,
                                             int channels, int chunks, int centered, float eps) {
  const int lane = threadIdx.x & 31, g = threadIdx.x >> 5;
  float s = 0.0f;
  for (int i = lane; i < chunks; i += 32) s = F == finish_half ? h(h(s) + h(part[g * chunks + i])) : s + part[g * chunks + i];
  s = warp_sum(s);
  if (lane != 0) return;
  const float n = (float)pixels * (float)(channels / 32);
  if (centered)
    stats[32 + g] = 1.0f / sqrtf(s / n + eps);
  else
    stats[g] = s / n;
}

// y = GroupNorm(x) by stats (32 means, then 32 rstd) with g and beta, then SiLU when silu; x, y [pixels, channels].
template <int F>
__device__ __forceinline__ void norm(const float* __restrict__ x, const float* __restrict__ stats,
                                     const float* __restrict__ g, const float* __restrict__ beta, float* __restrict__ y,
                                     int pixels, int channels, int silu) {
  const size_t i = (size_t)blockIdx.x * 256 + threadIdx.x;
  if (i >= (size_t)pixels * channels) return;
  const int c = (int)(i % channels);
  const int grp = c / (channels / 32);
  float v = (x[i] - stats[grp]) * stats[32 + grp];
  v = v * g[c] + beta[c];
  if (silu && F == norm_half) {
    v = h(h(v) / h(1.0f + h(expf(-h(v)))));
  } else if (silu) {
    v = v / (1.0f + expf(-v));
  }
  y[i] = v;
}

// y = acc + b (+ r when add), b by column, or by row when by_row; y may be acc.
template <int F>
__device__ __forceinline__ void bias(const float* acc, const float* __restrict__ b, const float* r, float* y, int rows,
                                     int cols, int by_row, int add) {
  const size_t i = (size_t)blockIdx.x * 256 + threadIdx.x;
  if (i >= (size_t)rows * cols) return;
  const int col = (int)(i % cols);
  const int row = (int)(i / cols);
  float v = acc[i] + b[by_row && F != bias_swap ? row : col];
  if (add) v = v + r[i];
  y[i] = v;
}

// The softmax of each row of s [rows, n] times scale, in place, n at most 256 * SOFTMAX_PER: a block of 256 a row.
constexpr int SOFTMAX_PER = 64;

template <int F>
__device__ __forceinline__ void softmax(float* s, int n, float scale) {
  __shared__ float part[8];
  float* row = s + (size_t)blockIdx.x * n;
  const int t = threadIdx.x;
  float v[SOFTMAX_PER];
  float m = -INFINITY;
#pragma unroll
  for (int j = 0; j < SOFTMAX_PER; ++j) {
    const int i = t + j * 256;
    v[j] = i < n ? row[i] * scale : -INFINITY;
    m = fmaxf(m, v[j]);
  }
  m = warp_max(m);
  if ((t & 31) == 0) part[t >> 5] = m;
  __syncthreads();
#pragma unroll
  for (int w = 0; w < 8; ++w) m = fmaxf(m, part[w]);
  __syncthreads();
  float sum = 0.0f;
#pragma unroll
  for (int j = 0; j < SOFTMAX_PER; ++j) {
    const int i = t + j * 256;
    if (i < n) {
      v[j] = expf(v[j] - m);
      sum = F == softmax_half ? h(h(sum) + h(v[j])) : sum + v[j];
    }
  }
  sum = warp_sum(sum);
  if ((t & 31) == 0) part[t >> 5] = sum;
  __syncthreads();
  float total = 0.0f;
#pragma unroll
  for (int w = 0; w < 8; ++w) total += part[w];
#pragma unroll
  for (int j = 0; j < SOFTMAX_PER; ++j) {
    const int i = t + j * 256;
    if (i < n) row[i] = v[j] / total;
  }
}

// Rows [pad + patches, 128]: pad zeros, then each 2 x 2 patch of z = mean + std * noise as (z - m) / s.
template <int F>
__device__ __forceinline__ void tokens(const float* __restrict__ moments, const float* __restrict__ noise,
                                       const float* __restrict__ m, const float* __restrict__ s, float* __restrict__ out,
                                       int rows, int cols, int pad) {
  const int k = threadIdx.x;
  const int r = blockIdx.x * 2 + threadIdx.y;
  const int w2 = cols / 2;
  if (r >= pad + (rows / 2) * w2) return;
  const size_t o = (size_t)r * 128 + k;
  if (r < pad) {
    out[o] = 0.0f;
    return;
  }
  // moments [rows, cols, 64] as mean | logvar, noise [rows, cols, 32]; a patch's channel c * 4 + dy * 2 + dx
  const int patch = r - pad;
  const int c = k / 4;
  const int dy = F == tokens_axes ? k % 2 : (k % 4) / 2;
  const int dx = F == tokens_axes ? (k % 4) / 2 : k % 2;
  const size_t pixel = (size_t)((patch / w2) * 2 + dy) * cols + (patch % w2) * 2 + dx;
  // torch.clamp keeps a NaN, where fminf and fmaxf would drop it
  const float raw = moments[pixel * 64 + 32 + c];
  const float clamped = fminf(fmaxf(raw, -30.0f), 20.0f);
  const float logvar = F == tokens_clamp ? clamped : (isnan(raw) ? raw : clamped);
  const float z = moments[pixel * 64 + c] + expf(0.5f * logvar) * noise[pixel * 32 + c];
  out[o] = (z - m[k]) / s[k];
}

}  // namespace tf_iv
