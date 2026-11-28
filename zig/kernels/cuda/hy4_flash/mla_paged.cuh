// Hy4-flash's absorbed gated MLA (docs/recipes/hy4-flash.md): device bodies templated on Fault (kNone as served).
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <stdint.h>

#include "../common/reduce.cuh"

namespace tf_hy4 {

// Planted faults a gate must catch: NaN read as 0, mutating an input, dropping the last term, or fp16 running sums.
enum Fault { kNone = 0, kBadNan = 1, kBadRepeat = 2, kBadTail = 3, kBadSums = 4 };

constexpr int kLat = 512;   // the normed latent
constexpr int kRope = 64;   // the roped columns a q head and the shared key carry
constexpr int kRow = 576;   // a cache row: the latent, then the roped key
constexpr int kNope = 192;  // nope columns a q head
constexpr int kQ = 256;     // a q_b head: nope, then rope
constexpr int kV = 256;     // value columns a head
constexpr int kKv = 448;    // a kv_b head: its 192 key rows, then its 256 value rows

constexpr int kHB = 8;          // heads an attention block takes, one warp a head in the scores
constexpr int kKC = 32;         // keys an attention chunk holds, one lane a key in the scores
constexpr int kKs = kRow + 2;   // a key's shared row, padded so lanes reading different keys use different banks

// The attention's dynamic shared bytes: q (fp32), the chunk's keys, weights, key positions, then m, l and the rescale.
constexpr int kAttnSmem = kHB * kRow * 4 + kKC * kKs * 2 + kHB * kKC * 4 + kKC * 4 + 3 * kHB * 4;

__device__ __forceinline__ float bf(__nv_bfloat16 v) { return __bfloat162float(v); }

// One fma step of a sum, rounded once in fp32 (kBadSums: the running sum rounded to fp16 at every step).
template <int F>
__device__ __forceinline__ float step(float a, float b, float acc) {
  const float s = __fmaf_rn(a, b, acc);
  return F == kBadSums ? __half2float(__float2half_rn(s)) : s;
}

template <int F>
__device__ __forceinline__ float add(float a, float b) {
  const float s = __fadd_rn(a, b);
  return F == kBadSums ? __half2float(__float2half_rn(s)) : s;
}

template <int F>
__device__ __forceinline__ float load(float v) {
  return (F == kBadNan && v != v) ? 0.0f : v;
}

// kBadRepeat: one thread bumps the first element of an input by one unit in its last place on every run.
template <int F, typename T>
__device__ __forceinline__ void bump(const T* p) {
  if (F != kBadRepeat || blockIdx.x != 0 || blockIdx.y != 0 || blockIdx.z != 0 || threadIdx.x != 0) return;
  if (sizeof(T) == 2) {
    unsigned short* w = (unsigned short*)p;
    *w = (unsigned short)(*w + 1);
  } else {
    unsigned* w = (unsigned*)p;
    *w = *w + 1;
  }
}

// A max that keeps a NaN, as float64 arithmetic on the formula does (kBadNan: fmaxf, which drops it).
template <int F>
__device__ __forceinline__ float pick(float a, float b) {
  if (F == kBadNan) return fmaxf(a, b);
  return (a != a || b != b) ? CUDART_NAN_F : fmaxf(a, b);
}

// ql [rows][heads][512] fp32 = q_nope . wk (kv_b[h][0..192][c], fp32 fma chain); block (head, 128 cols, 16 rows).
template <int F>
__device__ __forceinline__ void absorb(const __nv_bfloat16* __restrict__ q, int64_t qs,
                                       const __nv_bfloat16* __restrict__ kvb, float* __restrict__ ql, int rows,
                                       int heads) {
  constexpr int RB = 16;
  __shared__ float qn[RB][kNope];
  const int h = blockIdx.x, c = blockIdx.y * 128 + threadIdx.x, r0 = blockIdx.z * RB;
  const int nr = min(RB, rows - r0);
  bump<F>(q);
  for (int i = threadIdx.x; i < RB * kNope; i += 128) {
    const int r = i / kNope, d = i % kNope;
    qn[r][d] = r < nr ? load<F>(bf(q[(int64_t)(r0 + r) * qs + (int64_t)h * kQ + d])) : 0.0f;
  }
  __syncthreads();
  const __nv_bfloat16* w = kvb + (int64_t)h * kKv * kLat + c;
  float acc[RB];
#pragma unroll
  for (int r = 0; r < RB; ++r) acc[r] = 0.0f;
  const int nd = F == kBadTail ? kNope - 1 : kNope;
  for (int d = 0; d < nd; ++d) {
    const float wv = bf(w[(int64_t)d * kLat]);
#pragma unroll
    for (int r = 0; r < RB; ++r) acc[r] = step<F>(qn[r][d], wv, acc[r]);
  }
#pragma unroll
  for (int r = 0; r < RB; ++r)
    if (r < nr) ql[((int64_t)(r0 + r) * heads + h) * kLat + c] = acc[r];
}

// Key j of row t's walk: its window or causal span from lo, or its list (an entry < 0 or past t is masked); -1 masked.
__device__ __forceinline__ int key_at(int j, int n, int t, int lo, const int* list) {
  if (j >= n) return -1;
  const int p = list ? list[j] : lo + j;
  return (p >= 0 && p <= t) ? p : -1;
}

// ol [rows][heads][512] fp32: sinked online softmax over 32-key chunks (window, causal span or nidx list; p % ring).
template <int F>
__device__ __forceinline__ void attn(const float* __restrict__ ql, const __nv_bfloat16* __restrict__ qr,
                                     const __nv_bfloat16* __restrict__ cache, const int* __restrict__ pos,
                                     const int* __restrict__ idx, const float* __restrict__ sinks,
                                     float* __restrict__ ol, int heads, int window, int nidx, int ring, float scale) {
  extern __shared__ __align__(16) unsigned char smem[];
  float* qs = (float*)smem;
  __nv_bfloat16* ks = (__nv_bfloat16*)(qs + kHB * kRow);
  float* ps = (float*)(ks + kKC * kKs);
  int* kp = (int*)(ps + kHB * kKC);
  float* alpha = (float*)(kp + kKC);
  float* ls = alpha + kHB;
  float* ms = ls + kHB;
  const int r = blockIdx.x, h0 = blockIdx.y * kHB;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  bump<F>(ql);
  const int t = pos[r];
  const int lo = window > 0 ? max(0, t - window + 1) : 0;
  const int n = nidx > 0 ? nidx : t - lo + 1;
  const int* list = nidx > 0 ? idx + (int64_t)r * nidx : nullptr;
  for (int i = tid; i < kHB * kRow; i += 256) {
    const int hh = i / kRow, c = i % kRow;
    const int64_t at = (int64_t)r * heads + h0 + hh;
    qs[i] = load<F>(c < kLat ? ql[at * kLat + c] : bf(qr[at * kRope + c - kLat]));
  }
  if (tid < kHB) {
    ms[tid] = sinks[h0 + tid];
    ls[tid] = 1.0f;
  }
  int last = -1;  // kBadTail: the walk's last visible key is left out
  if (F == kBadTail)
    for (int j = n - 1; j >= 0 && last < 0; --j)
      if (key_at(j, n, t, lo, list) >= 0) last = j;
  float acc[kHB][2];
#pragma unroll
  for (int hh = 0; hh < kHB; ++hh) acc[hh][0] = acc[hh][1] = 0.0f;
  __syncthreads();
  for (int base = 0; base < n; base += kKC) {
    if (tid < kKC) kp[tid] = (F == kBadTail && base + tid == last) ? -1 : key_at(base + tid, n, t, lo, list);
    __syncthreads();
    for (int i = tid; i < kKC * (kRow / 2); i += 256) {
      const int j = i / (kRow / 2), w2 = i % (kRow / 2), p = kp[j];
      __nv_bfloat162 v = __float2bfloat162_rn(0.0f);
      if (p >= 0) v = ((const __nv_bfloat162*)(cache + (int64_t)(ring > 0 ? p % ring : p) * kRow))[w2];
      ((__nv_bfloat162*)(ks + j * kKs))[w2] = v;
    }
    __syncthreads();
    {
      const float* q = qs + warp * kRow;
      const __nv_bfloat162* k = (const __nv_bfloat162*)(ks + lane * kKs);
      float s = 0.0f;
      for (int c2 = 0; c2 < kRow / 2; ++c2) {
        const float2 kv = __bfloat1622float2(k[c2]);
        s = step<F>(q[2 * c2], kv.x, s);
        s = step<F>(q[2 * c2 + 1], kv.y, s);
      }
      s = __fmul_rn(s, scale);
      const bool ok = kp[lane] >= 0 && (F != kBadNan || s == s);
      float cm = ok ? s : -CUDART_INF_F;
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) cm = pick<F>(cm, __shfl_xor_sync(0xffffffffu, cm, off));
      const float m = ms[warp];
      const float mn = pick<F>(m, cm);
      const float a = expf(__fsub_rn(m, mn));
      const float p = ok ? expf(__fsub_rn(s, mn)) : 0.0f;
      float sum = p;  // a fixed tree into lane 0
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) sum = __fadd_rn(sum, __shfl_down_sync(0xffffffffu, sum, off));
      ps[warp * kKC + lane] = p;
      if (lane == 0) {
        ms[warp] = mn;
        ls[warp] = __fadd_rn(__fmul_rn(ls[warp], a), sum);
        alpha[warp] = a;
      }
    }
    __syncthreads();
#pragma unroll
    for (int hh = 0; hh < kHB; ++hh) {  // thread tid holds latent columns 2 tid and 2 tid + 1 of every head
      const float a = alpha[hh];
      float x0 = __fmul_rn(acc[hh][0], a), x1 = __fmul_rn(acc[hh][1], a);
      for (int j = 0; j < kKC; ++j) {
        const float p = ps[hh * kKC + j];
        const float2 kv = __bfloat1622float2(((const __nv_bfloat162*)(ks + j * kKs))[tid]);
        x0 = step<F>(p, kv.x, x0);
        x1 = step<F>(p, kv.y, x1);
      }
      acc[hh][0] = x0;
      acc[hh][1] = x1;
    }
    __syncthreads();
  }
#pragma unroll
  for (int hh = 0; hh < kHB; ++hh) {
    const float l = ls[hh];
    float2 o;
    o.x = __fdiv_rn(acc[hh][0], l);
    o.y = __fdiv_rn(acc[hh][1], l);
    ((float2*)(ol + ((int64_t)r * heads + h0 + hh) * kLat))[tid] = o;
  }
}

// out [rows][heads * 256] bf16 = (ol . wv) * sigmoid(gate), fp32 fma over latent; block (head, 32 values, 16 rows).
template <int F>
__device__ __forceinline__ void value(const float* __restrict__ ol, const __nv_bfloat16* __restrict__ kvb,
                                      const __nv_bfloat16* __restrict__ gate, int64_t gs, __nv_bfloat16* __restrict__ out,
                                      int64_t os, int rows, int heads) {
  constexpr int RB = 16, VB = 32, CB = 64;
  __shared__ float lat[RB][CB];
  __shared__ __nv_bfloat16 ws[VB][CB + 2];
  const int h = blockIdx.x, v0 = blockIdx.y * VB, r0 = blockIdx.z * RB;
  const int tid = threadIdx.x, v = tid & 31, rg = tid >> 5;
  bump<F>(ol);
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  const int nc = F == kBadTail ? kLat - 1 : kLat;
  for (int c0 = 0; c0 < kLat; c0 += CB) {
    for (int i = tid; i < RB * CB; i += 128) {
      const int rr = i / CB, c = i % CB;
      lat[rr][c] = r0 + rr < rows ? load<F>(ol[((int64_t)(r0 + rr) * heads + h) * kLat + c0 + c]) : 0.0f;
    }
    for (int i = tid; i < VB * CB; i += 128) {
      const int vv = i / CB, c = i % CB;
      ws[vv][c] = kvb[((int64_t)h * kKv + kNope + v0 + vv) * kLat + c0 + c];
    }
    __syncthreads();
    for (int c = 0; c < CB && c0 + c < nc; ++c) {
      const float w = bf(ws[v][c]);
#pragma unroll
      for (int k = 0; k < 4; ++k) acc[k] = step<F>(lat[rg * 4 + k][c], w, acc[k]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int r = r0 + rg * 4 + k;
    if (r >= rows) continue;
    const int64_t col = (int64_t)h * kV + v0 + v;
    const float g = bf(gate[r * gs + col]);
    const float sg = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-g)));
    out[r * os + col] = __float2bfloat16_rn(__fmul_rn(acc[k], sg));
  }
}

// out [rows][width] bf16: x = bf16(rank-ordered sum of world planes), then post_proj_norm(x, w); 256 threads/row.
template <int F>
__device__ __forceinline__ void post(const float* __restrict__ parts, int64_t plane, int world,
                                     const __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ out, int width,
                                     float eps) {
  constexpr int T = 256, MOST = 16;
  __shared__ float red[T / 32];
  const int r = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  bump<F>(parts);
  const int n = F == kBadTail ? width - 1 : width;
  float x[MOST];
  float ss = 0.0f;
#pragma unroll
  for (int k = 0; k < MOST; ++k) {
    const int i = tid + k * T;
    x[k] = 0.0f;
    if (i >= n) continue;
    const float* p = parts + (int64_t)r * width + i;
    float s = load<F>(p[0]);
    for (int q = 1; q < world; ++q) s = __fadd_rn(s, load<F>(p[q * plane]));
    x[k] = bf(__float2bfloat16_rn(s));
    ss = step<F>(x[k], x[k], ss);
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) ss = add<F>(ss, __shfl_down_sync(0xffffffffu, ss, off));
  if (lane == 0) red[warp] = ss;
  __syncthreads();
  if (warp == 0) {
    float v = lane < T / 32 ? red[lane] : 0.0f;
#pragma unroll
    for (int off = 4; off > 0; off >>= 1) v = add<F>(v, __shfl_down_sync(0xffffffffu, v, off));
    if (lane == 0) red[0] = v;
  }
  __syncthreads();
  const float inv = tf_common::rms_scale(red[0], width, eps);
#pragma unroll
  for (int k = 0; k < MOST; ++k) {
    const int i = tid + k * T;
    if (i < width) out[(int64_t)r * width + i] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(x[k], inv), bf(w[i])));
  }
}

// attn over a batched round's sequences: row r reads its own latent cache at caches[r] (a device address).
template <int F>
__device__ __forceinline__ void attn_rows(const float* __restrict__ ql, const __nv_bfloat16* __restrict__ qr,
                                          const unsigned long long* __restrict__ caches, const int* __restrict__ pos,
                                          const int* __restrict__ idx, const float* __restrict__ sinks,
                                          float* __restrict__ ol, int heads, int window, int nidx, int ring,
                                          float scale) {
  const __nv_bfloat16* cache = reinterpret_cast<const __nv_bfloat16*>(caches[blockIdx.x]);
  attn<F>(ql, qr, cache, pos, idx, sinks, ol, heads, window, nidx, ring, scale);
}

}  // namespace tf_hy4
