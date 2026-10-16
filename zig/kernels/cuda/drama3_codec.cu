// ours: drama3's codec (Fish S2 modded DAC) in fp32: the glue between cuBLAS SGEMMs, time-major [rows, channels].
#include <stdint.h>
#include <math.h>
#include <cuda_bf16.h>

namespace {
constexpr unsigned FULL = 0xFFFFFFFFu;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL, v, o);
  return v;
}

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
}  // namespace

// out[t] = table[code 0] + sum of residual codebook rows of codes 1..9 (frames [t, fs] columns 1..10, clamped).
extern "C" __global__ void c_vq_gather(const int* __restrict__ frames, int fs, const float* __restrict__ table,
                                       float* __restrict__ out, int D) {
  const int t = blockIdx.x;
  const int* f = frames + (size_t)t * fs;
  int idx[10];
  idx[0] = min(max(f[1], 0), 4095);
#pragma unroll
  for (int c = 1; c < 10; ++c) idx[c] = 4096 + (c - 1) * 1024 + min(max(f[1 + c], 0), 1023);
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    float r = table[(size_t)idx[1] * D + i];
#pragma unroll
    for (int c = 2; c < 10; ++c) r += table[(size_t)idx[c] * D + i];
    out[(size_t)t * D + i] = table[(size_t)idx[0] * D + i] + r;
  }
}

// Encoder RVQ, a block a row: 10 stages of project, l2-normalize, nearest normalized code, subtract; codes [t, 10].
extern "C" __global__ void c_vq_encode(float* __restrict__ z, int D, const float* __restrict__ in_w,
                                       const float* __restrict__ in_b, const float* __restrict__ cbn,
                                       const float* __restrict__ table, int* __restrict__ codes) {
  const int t = blockIdx.x;
  float* r = z + (size_t)t * D;
  __shared__ float e[8];
  __shared__ float bv[32];
  __shared__ int bi[32];
  __shared__ int chosen;
  for (int st = 0; st < 10; ++st) {
    const int N = st == 0 ? 4096 : 1024;
    const int base = st == 0 ? 0 : 4096 + (st - 1) * 1024;
    for (int k = 0; k < 8; ++k) {
      float acc = 0.f;
      for (int i = threadIdx.x; i < D; i += blockDim.x) acc += in_w[((size_t)st * 8 + k) * D + i] * r[i];
      acc = block_sum(acc);
      if (threadIdx.x == 0) e[k] = acc + in_b[st * 8 + k];
      __syncthreads();
    }
    float nrm = 0.f;
#pragma unroll
    for (int k = 0; k < 8; ++k) nrm += e[k] * e[k];
    const float inv = 1.f / fmaxf(sqrtf(nrm), 1e-12f);
    float en[8], esq = 0.f;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      en[k] = e[k] * inv;
      esq += en[k] * en[k];
    }
    float best = -INFINITY;
    int besti = 0x7FFFFFFF;
    for (int n = threadIdx.x; n < N; n += blockDim.x) {
      const float* c = cbn + (size_t)(base + n) * 8;
      float dot = 0.f, csq = 0.f;
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        dot += en[k] * c[k];
        csq += c[k] * c[k];
      }
      const float s = -(esq - 2.f * dot + csq);
      if (s > best || (s == best && n < besti)) {
        best = s;
        besti = n;
      }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor_sync(FULL, best, o);
      const int oi = __shfl_xor_sync(FULL, besti, o);
      if (ov > best || (ov == best && oi < besti)) {
        best = ov;
        besti = oi;
      }
    }
    if ((threadIdx.x & 31) == 0) {
      bv[threadIdx.x >> 5] = best;
      bi[threadIdx.x >> 5] = besti;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      for (int w = 1; w < (int)(blockDim.x >> 5); ++w)
        if (bv[w] > best || (bv[w] == best && bi[w] < besti)) {
          best = bv[w];
          besti = bi[w];
        }
      int b0 = bi[0];
      float v0 = bv[0];
      for (int w = 1; w < (int)(blockDim.x >> 5); ++w)
        if (bv[w] > v0 || (bv[w] == v0 && bi[w] < b0)) {
          v0 = bv[w];
          b0 = bi[w];
        }
      chosen = b0;
      codes[t * 10 + st] = b0;
    }
    __syncthreads();
    const float* q = table + (size_t)(base + chosen) * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) r[i] -= q[i];
    __syncthreads();
  }
}

// RMSNorm (the codec transformers'): y = x * rsqrt(mean(x^2) + eps) * w.
extern "C" __global__ void c_rmsnorm(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ y,
                                     int D, float eps) {
  const float* xr = x + (size_t)blockIdx.x * D;
  float ss = 0.f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) ss += xr[i] * xr[i];
  const float inv = rsqrtf(block_sum(ss) / D + eps);
  for (int i = threadIdx.x; i < D; i += blockDim.x) y[(size_t)blockIdx.x * D + i] = xr[i] * inv * w[i];
}

// LayerNorm (ConvNeXt, eps 1e-6): y = (x - mean) / sqrt(var + eps) * w + b.
extern "C" __global__ void c_layernorm(const float* __restrict__ x, const float* __restrict__ w,
                                       const float* __restrict__ b, float* __restrict__ y, int D, float eps) {
  const float* xr = x + (size_t)blockIdx.x * D;
  float s = 0.f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) s += xr[i];
  const float mean = block_sum(s) / D;
  float v = 0.f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    const float d = xr[i] - mean;
    v += d * d;
  }
  const float inv = rsqrtf(block_sum(v) / D + eps);
  for (int i = threadIdx.x; i < D; i += blockDim.x) y[(size_t)blockIdx.x * D + i] = (xr[i] - mean) * inv * w[i] + b[i];
}

// Codec rotary on q and k of qkv rows (16 heads of 64, pairs 2i, 2i + 1, bf16 table); k / v to ring rows; q in place.
extern "C" __global__ void c_rope_kv(float* __restrict__ qkv, int pos0, float* __restrict__ kc,
                                     float* __restrict__ vc, int ring) {
  const int t = blockIdx.x, i = threadIdx.x;  // pair i of 512 (head i / 32, pair i % 32)
  const int pos = pos0 + t;
  const int pr = i & 31;
  const float freq = 1.f / powf(10000.f, (float)(2 * pr) / 64.f);
  const float ang = (float)pos * freq;
  const float c = __bfloat162float(__float2bfloat16(cosf(ang))), s = __bfloat162float(__float2bfloat16(sinf(ang)));
  float* row = qkv + (size_t)t * 3072;
  const size_t cr = (size_t)(pos % ring) * 1024;
  const float q0 = row[2 * i], q1 = row[2 * i + 1];
  row[2 * i] = q0 * c - q1 * s;
  row[2 * i + 1] = q1 * c + q0 * s;
  const float k0 = row[1024 + 2 * i], k1 = row[1024 + 2 * i + 1];
  kc[cr + 2 * i] = k0 * c - k1 * s;
  kc[cr + 2 * i + 1] = k1 * c + k0 * s;
  vc[cr + 2 * i] = row[2048 + 2 * i];
  vc[cr + 2 * i + 1] = row[2048 + 2 * i + 1];
}

// Window-limited causal attention over the ring caches: a warp a (row, head), 2 dims a lane; out [t, 1024].
extern "C" __global__ void c_attn(const float* __restrict__ qkv, const float* __restrict__ kc,
                                  const float* __restrict__ vc, float* __restrict__ out, int pos0, int window, int ring) {
  const int t = blockIdx.x, h = blockIdx.y * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  const int pos = pos0 + t;
  const float* q = qkv + (size_t)t * 3072 + h * 64;
  const float q0 = q[2 * lane] * 0.125f, q1 = q[2 * lane + 1] * 0.125f;
  float m = -INFINITY, l = 0.f, a0 = 0.f, a1 = 0.f;
  for (int p = max(0, pos - window + 1); p <= pos; ++p) {
    const size_t r = (size_t)(p % ring) * 1024 + h * 64 + 2 * lane;
    const float sc = warp_sum(q0 * kc[r] + q1 * kc[r + 1]);
    const float nm = fmaxf(m, sc);
    const float corr = expf(m - nm), e = expf(sc - nm);
    l = l * corr + e;
    a0 = a0 * corr + e * vc[r];
    a1 = a1 * corr + e * vc[r + 1];
    m = nm;
  }
  out[(size_t)t * 1024 + h * 64 + 2 * lane] = a0 / l;
  out[(size_t)t * 1024 + h * 64 + 2 * lane + 1] = a1 / l;
}

// x [rows, D] += gamma * y (LayerScale and the residual add; ConvNeXt's input + gamma * x); contiguous rows.
extern "C" __global__ void c_add_gamma(float* __restrict__ x, const float* __restrict__ y,
                                       const float* __restrict__ gamma, size_t total, int D) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x)
    x[i] += (gamma ? gamma[i % D] : 1.f) * y[i];
}

// SwiGLU of [w1 | w3] rows: y = silu(a) * b.
extern "C" __global__ void c_silu_mul(const float* __restrict__ x, float* __restrict__ y, int F) {
  const int t = blockIdx.y;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < F; i += gridDim.x * blockDim.x) {
    const float a = x[(size_t)t * 2 * F + i], b = x[(size_t)t * 2 * F + F + i];
    y[(size_t)t * F + i] = a / (1.f + expf(-a)) * b;
  }
}

// GELU (exact, erf) in place over n values.
extern "C" __global__ void c_gelu(float* __restrict__ x, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    x[i] = 0.5f * x[i] * (1.f + erff(x[i] * 0.70710678118654752f));
}

// y [rows, n] += b[col] (bias of a GEMM's columns), optionally tanh.
extern "C" __global__ void c_bias(float* __restrict__ y, const float* __restrict__ b, size_t rows, int n, int act) {
  const size_t total = rows * (size_t)n;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
    const float v = y[i] + b[i % n];
    y[i] = act ? tanhf(v) : v;
  }
}

// Snake: y = x + 1 / (alpha + 1e-9) * sin(alpha x)^2 per channel (alpha [C]); or a copy (alpha null).
extern "C" __global__ void c_snake(const float* __restrict__ x, float* __restrict__ y, size_t rows, int C,
                                   const float* __restrict__ alpha) {
  const size_t total = rows * (size_t)C;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
    const float v = x[i];
    if (alpha) {
      const float a = alpha[i % C];
      const float s = sinf(a * v);
      y[i] = v + (1.f / (a + 1e-9f)) * (s * s);
    } else {
      y[i] = v;
    }
  }
}

// ConvNeXt's causal depthwise conv, kernel 7: y[t, c] = b[c] + sum_j w[c, j] x[t - 6 + j, c], x with 6 margin rows.
extern "C" __global__ void c_dwconv7(const float* __restrict__ xm, const float* __restrict__ w,
                                     const float* __restrict__ b, float* __restrict__ y, int C) {
  const int t = blockIdx.y;
  for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < C; c += gridDim.x * blockDim.x) {
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < 7; ++j) acc += w[c * 7 + j] * xm[(size_t)(t + j) * C + c];
    y[(size_t)t * C + c] = acc + b[c];
  }
}

// Move n floats (the carried rows of a staging buffer, through a scratch copy when they overlap).
extern "C" __global__ void c_copy(const float* __restrict__ src, float* __restrict__ dst, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    dst[i] = src[i];
}
