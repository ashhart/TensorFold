// ours: ZONOS3 Mini's backbone on CUDA: fp32 rows over bf16 weights, one fixed order a row for prompt and decode.
#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>
#include "common/reduce.cuh"
#include "common/act.cuh"

typedef __nv_bfloat16 bf16;

namespace {

__device__ __forceinline__ float b2f(bf16 x) { return __bfloat162float(x); }

using tf_common::block_max_tree;
using tf_common::block_sum_tree;
using tf_common::rms_scale;
using tf_common::warp_sum;

// Eight bf16 weights against eight fp32 inputs, in index order.
__device__ __forceinline__ float dot8(const uint4& w, const float* x, float acc) {
  float f[8];
  tf_common::unpack8(w, f);
#pragma unroll
  for (int i = 0; i < 8; ++i) acc = fmaf(f[i], x[i], acc);
  return acc;
}

}  // namespace

// Frame rows [T, 11] to the sum of their eleven embeddings, column 0 first: tables bf16 [rows, D], ten audio then text.
extern "C" __global__ void z3_embed(const int* __restrict__ rows, const bf16* __restrict__ tables, int audio_rows,
                                    float* __restrict__ out, int D) {
  const int* r = rows + (size_t)blockIdx.x * 11;
  float* o = out + (size_t)blockIdx.x * D;
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    float s = 0.f;
#pragma unroll
    for (int c = 0; c < 11; ++c) s += b2f(tables[((size_t)c * audio_rows + r[c]) * D + i]);
    o[i] = s;
  }
}

// y = x * rsqrt(mean(x^2) + eps) * w (w bf16, or null for a weightless norm); a block a row, D a multiple of 32.
extern "C" __global__ void __launch_bounds__(256) z3_rmsnorm(const float* __restrict__ x, int x_stride,
                                                             const bf16* __restrict__ w, float* __restrict__ y,
                                                             int y_stride, int D, float eps) {
  __shared__ float part[32];
  const float* xr = x + (size_t)blockIdx.x * x_stride;
  float* yr = y + (size_t)blockIdx.x * y_stride;
  float ss = 0.f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) ss = fmaf(xr[i], xr[i], ss);
  const float inv = rms_scale(block_sum_tree(ss, part, blockDim.x >> 5), D, eps);
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    const float v = xr[i] * inv;
    yr[i] = w ? v * b2f(w[i]) : v;
  }
}

// Full AttnRes: a row's S sources (bank [S][cap][D]) mixed by softmax over s of sum(rms_norm(source) * query).
extern "C" __global__ void __launch_bounds__(256) z3_attnres(const float* __restrict__ bank, int S, int cap,
                                                             const bf16* __restrict__ query, float* __restrict__ out,
                                                             int D, float eps) {
  __shared__ float part[32];
  __shared__ float wgt[64];
  const size_t row = blockIdx.x;
  for (int s = 0; s < S; ++s) {
    const float* x = bank + ((size_t)s * cap + row) * D;
    float ss = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) ss = fmaf(x[i], x[i], ss);
    const float inv = rms_scale(block_sum_tree(ss, part, blockDim.x >> 5), D, eps);
    float d = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) d = fmaf(x[i] * inv, b2f(query[i]), d);
    d = block_sum_tree(d, part, blockDim.x >> 5);
    if (threadIdx.x == 0) wgt[s] = d;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float m = wgt[0];
    for (int s = 1; s < S; ++s) m = fmaxf(m, wgt[s]);
    float total = 0.f;
    for (int s = 0; s < S; ++s) {
      wgt[s] = expf(wgt[s] - m);
      total += wgt[s];
    }
    for (int s = 0; s < S; ++s) wgt[s] /= total;
  }
  __syncthreads();
  float* o = out + row * D;
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    float a = 0.f;
    for (int s = 0; s < S; ++s) a = fmaf(bank[((size_t)s * cap + row) * D + i], wgt[s], a);
    o[i] = a;
  }
}

// y[r, n] = sum_k W[n, k] x[r / xrep, k] (+ bias[n]): W bf16 [N, K], K a multiple of 8. A warp an output, x staged
// in K * 4 bytes of dynamic shared. With picks, row r reads W + picks[r] * pick_stride.
extern "C" __global__ void __launch_bounds__(256) z3_linear(const bf16* __restrict__ W, const float* __restrict__ x,
                                                            int x_stride, int xrep, int K, int N,
                                                            float* __restrict__ y, int y_stride,
                                                            const bf16* __restrict__ bias,
                                                            const int* __restrict__ picks, long long pick_stride) {
  extern __shared__ float xs[];
  const int r = blockIdx.y;
  const float* xr = x + (size_t)(r / xrep) * x_stride;
  for (int i = threadIdx.x; i < K; i += blockDim.x) xs[i] = xr[i];
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const int n = blockIdx.x * 8 + (threadIdx.x >> 5);
  const bf16* base = picks ? W + (size_t)picks[r] * (size_t)pick_stride : W;
  const uint4* wr = reinterpret_cast<const uint4*>(base + (size_t)min(n, N - 1) * K);
  const int chunks = K >> 3;
  float acc = 0.f;
  for (int c = lane; c < chunks; c += 32) acc = dot8(wr[c], xs + c * 8, acc);
  acc = warp_sum(acc);
  if (lane == 0 && n < N) y[(size_t)r * y_stride + n] = bias ? acc + b2f(bias[n]) : acc;
}

// GDN depthwise causal conv (kernel 4, the newest input last) + SiLU over C channels; state keeps the last 3 inputs.
extern "C" __global__ void z3_conv_silu(const float* __restrict__ proj, int ps, int T, const bf16* __restrict__ w,
                                        float* __restrict__ state, float* __restrict__ out, int C) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  const float w0 = b2f(w[c * 4 + 0]), w1 = b2f(w[c * 4 + 1]), w2 = b2f(w[c * 4 + 2]), w3 = b2f(w[c * 4 + 3]);
  float s0 = state[c * 3 + 0], s1 = state[c * 3 + 1], s2 = state[c * 3 + 2];
  for (int t = 0; t < T; ++t) {
    const float x = proj[(size_t)t * ps + c];
    float acc = s0 * w0;
    acc = fmaf(s1, w1, acc);
    acc = fmaf(s2, w2, acc);
    acc = fmaf(x, w3, acc);
    out[(size_t)t * C + c] = tf_common::silu(acc);
    s0 = s1;
    s1 = s2;
    s2 = x;
  }
  state[c * 3 + 0] = s0;
  state[c * 3 + 1] = s1;
  state[c * 3 + 2] = s2;
}

// A row's GDN inputs for head blockIdx.y: l2-normed q (scaled by 128^-1/2) and k in place, decay and beta.
// conv is [T][3 * 512] (q | k | v); proj rows hold a at a_off + h and b at a_off + 4 + h.
extern "C" __global__ void __launch_bounds__(128) z3_gdn_prep(float* __restrict__ conv, const float* __restrict__ proj,
                                                              int ps, int a_off, const bf16* __restrict__ A_log,
                                                              const bf16* __restrict__ dt_bias,
                                                              float* __restrict__ gates) {
  __shared__ float part[32];
  const int t = blockIdx.x, h = blockIdx.y;
  float* q = conv + (size_t)t * 1536 + h * 128;
  float* k = q + 512;
  const float qv = q[threadIdx.x], kv = k[threadIdx.x];
  const float qs = block_sum_tree(qv * qv, part, 4);
  const float ks = block_sum_tree(kv * kv, part, 4);
  q[threadIdx.x] = qv * rsqrtf(qs + 1e-6f) * 0.088388347648318447f;
  k[threadIdx.x] = kv * rsqrtf(ks + 1e-6f);
  if (threadIdx.x == 0) {
    const float* p = proj + (size_t)t * ps + a_off;
    gates[(t * 4 + h) * 2 + 0] = expf(-expf(b2f(A_log[h])) * tf_common::softplusf_(p[h] + b2f(dt_bias[h])));
    gates[(t * 4 + h) * 2 + 1] = tf_common::sigmoidf_(p[4 + h]);
  }
}

// GDN recurrence, token by token in fp32: a thread a value row of head blockIdx.x's state S [4][128][128].
extern "C" __global__ void __launch_bounds__(128) z3_gdn(const float* __restrict__ conv,
                                                         const float* __restrict__ gates, int T,
                                                         float* __restrict__ S, float* __restrict__ o) {
  const int h = blockIdx.x, v = threadIdx.x;
  float* srow = S + ((size_t)h * 128 + v) * 128;
  float row[128];
  for (int j = 0; j < 128; ++j) row[j] = srow[j];
  for (int t = 0; t < T; ++t) {
    const float* q = conv + (size_t)t * 1536 + h * 128;
    const float* k = q + 512;
    const float val = q[1024 + v];
    const float decay = gates[(t * 4 + h) * 2 + 0], beta = gates[(t * 4 + h) * 2 + 1];
    float pred = 0.f;
    for (int j = 0; j < 128; ++j) {
      row[j] *= decay;
      pred = fmaf(row[j], k[j], pred);
    }
    const float delta = beta * (val - pred);
    float out = 0.f;
    for (int j = 0; j < 128; ++j) {
      row[j] = fmaf(delta, k[j], row[j]);
      out = fmaf(row[j], q[j], out);
    }
    o[(size_t)t * 512 + h * 128 + v] = out;
  }
  for (int j = 0; j < 128; ++j) srow[j] = row[j];
}

// y = rms_norm(o, w, 1e-5) * silu(z) per head of 128: o [T][512], z in proj rows at z_off.
extern "C" __global__ void __launch_bounds__(128) z3_gated_norm(const float* __restrict__ o,
                                                                const float* __restrict__ proj, int ps, int z_off,
                                                                const bf16* __restrict__ w, float* __restrict__ y) {
  __shared__ float part[32];
  const int t = blockIdx.x, h = blockIdx.y, i = threadIdx.x;
  const float x = o[(size_t)t * 512 + h * 128 + i];
  const float inv = rms_scale(block_sum_tree(x * x, part, 4), 128, 1e-5f);
  y[(size_t)t * 512 + h * 128 + i] = x * inv * b2f(w[i]) * tf_common::silu(proj[(size_t)t * ps + z_off + h * 128 + i]);
}

// Attention inputs of row blockIdx.x at position pos0 + row: q [16][32] rms-normed, scaled by |temp| and rotated in
// place; k [4][32] rms-normed and rotated into the cache row, v copied. A warp a head; pairs rotate (2m, 2m + 1).
extern "C" __global__ void __launch_bounds__(512) z3_attn_prep(float* __restrict__ q, const float* __restrict__ kv,
                                                               const bf16* __restrict__ temp, int pos0,
                                                               float* __restrict__ kc, float* __restrict__ vc,
                                                               float theta) {
  const int t = blockIdx.x, lane = threadIdx.x & 31, h = threadIdx.x >> 5;
  const float pos = (float)(pos0 + t);
  const float freq = powf(theta, -(float)(lane & ~1) / 32.f);
  const float ang = pos * freq;
  const float c = cosf(ang), s = sinf(ang);
  float* qr = q + (size_t)t * 512 + threadIdx.x;
  float x = *qr;
  x = x * rms_scale(warp_sum(x * x), 32, 1e-6f) * fabsf(b2f(temp[h]));
  float other = __shfl_xor_sync(0xffffffffu, x, 1);
  *qr = (lane & 1) ? other * s + x * c : x * c - other * s;
  if (h < 4) {
    const float* kr = kv + (size_t)t * 256;
    float kx = kr[threadIdx.x];
    kx = kx * rms_scale(warp_sum(kx * kx), 32, 1e-6f);
    other = __shfl_xor_sync(0xffffffffu, kx, 1);
    const size_t at = (size_t)(pos0 + t) * 128 + threadIdx.x;
    kc[at] = (lane & 1) ? other * s + kx * c : kx * c - other * s;
    vc[at] = kr[128 + threadIdx.x];
  }
}

// Causal attention of (row blockIdx.x, head blockIdx.y) over cache rows 0 .. pos0 + row, then the head's sigmoid gate.
// Keys go to threads by index mod 256 and the block folds in a fixed order, so a row's bits depend on its position.
extern "C" __global__ void __launch_bounds__(256) z3_attention(const float* __restrict__ q,
                                                               const float* __restrict__ kc,
                                                               const float* __restrict__ vc,
                                                               const float* __restrict__ gate, int pos0,
                                                               float* __restrict__ out) {
  __shared__ float part[32];
  __shared__ float acc[256][33];
  const int t = blockIdx.x, h = blockIdx.y, kh = h >> 2;
  const int L = pos0 + t + 1;
  float qv[32];
  const float* qr = q + (size_t)t * 512 + h * 32;
#pragma unroll
  for (int i = 0; i < 32; ++i) qv[i] = qr[i];
  const float scale = 0.17677669529663688f;
  float m = -INFINITY;
  for (int j = threadIdx.x; j < L; j += blockDim.x) {
    const float* k = kc + (size_t)j * 128 + kh * 32;
    float d = 0.f;
#pragma unroll
    for (int i = 0; i < 32; ++i) d = fmaf(qv[i], k[i], d);
    m = fmaxf(m, d * scale);
  }
  m = block_max_tree(m, part, 8);
  float a[33];
#pragma unroll
  for (int i = 0; i < 33; ++i) a[i] = 0.f;
  for (int j = threadIdx.x; j < L; j += blockDim.x) {
    const float* k = kc + (size_t)j * 128 + kh * 32;
    const float* v = vc + (size_t)j * 128 + kh * 32;
    float d = 0.f;
#pragma unroll
    for (int i = 0; i < 32; ++i) d = fmaf(qv[i], k[i], d);
    const float e = expf(d * scale - m);
    a[32] += e;
#pragma unroll
    for (int i = 0; i < 32; ++i) a[i] = fmaf(e, v[i], a[i]);
  }
#pragma unroll
  for (int i = 0; i < 33; ++i) acc[threadIdx.x][i] = a[i];
  __syncthreads();
  if (threadIdx.x < 33) {
    float s = 0.f;
    for (int w = 0; w < 256; ++w) s += acc[w][threadIdx.x];
    acc[0][threadIdx.x] = s;
  }
  __syncthreads();
  if (threadIdx.x < 32) {
    const float g = tf_common::sigmoidf_(gate[(size_t)t * 16 + h]);
    out[(size_t)t * 512 + h * 32 + threadIdx.x] = acc[0][threadIdx.x] / acc[0][32] * g;
  }
}

// out[r, i] = in[r, a_off + i] * silu(in[r, b_off + i]) for i < n.
extern "C" __global__ void z3_mul_silu(const float* __restrict__ in, int stride, int a_off, int b_off, int n,
                                       float* __restrict__ out) {
  const float* r = in + (size_t)blockIdx.y * stride;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[(size_t)blockIdx.y * n + i] = r[a_off + i] * tf_common::silu(r[b_off + i]);
}

// A row's two experts: the top two of logits - bias (ties to the lower index), weights softmax of their raw logits.
extern "C" __global__ void z3_route(const float* __restrict__ logits, const float* __restrict__ bias, int E,
                                    int* __restrict__ picks, float* __restrict__ w) {
  if (threadIdx.x != 0) return;
  const float* l = logits + (size_t)blockIdx.x * E;
  int b0 = -1, b1 = -1;
  float v0 = -INFINITY, v1 = -INFINITY;
  for (int e = 0; e < E; ++e) {
    const float v = l[e] - bias[e];
    if (b0 < 0 || v > v0) {
      b1 = b0;
      v1 = v0;
      b0 = e;
      v0 = v;
    } else if (b1 < 0 || v > v1) {
      b1 = e;
      v1 = v;
    }
  }
  const float m = fmaxf(l[b0], l[b1]);
  const float e0 = expf(l[b0] - m), e1 = expf(l[b1] - m);
  picks[blockIdx.x * 2] = b0;
  picks[blockIdx.x * 2 + 1] = b1;
  w[blockIdx.x * 2] = e0 / (e0 + e1);
  w[blockIdx.x * 2 + 1] = e1 / (e0 + e1);
}

// out[r] = down[2r] * w[2r] + down[2r + 1] * w[2r + 1], the first pick first.
extern "C" __global__ void z3_combine(const float* __restrict__ down, const float* __restrict__ w, int n,
                                      float* __restrict__ out) {
  const size_t r = blockIdx.y;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float a = down[(2 * r) * n + i] * w[2 * r];
  a += down[(2 * r + 1) * n + i] * w[2 * r + 1];
  out[r * n + i] = a;
}

// Logits in place: cap * tanh(x / cap).
extern "C" __global__ void z3_softcap(float* __restrict__ x, long long n, float cap) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = cap * tanhf(x[i] / cap);
}
