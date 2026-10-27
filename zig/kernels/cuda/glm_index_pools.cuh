// GLM-5.3-Flash DSA indexer, a prompt chunk's top-K pools in one pass over its scores: BadAd84's patch 0101 in CUDA.
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

namespace glm_index {

constexpr int T = 256, WARPS = T / 32, PER = 8, CHUNK = T * PER;  // threads, warps, scores a thread a step, a step
constexpr int CAP_MAX = 4096, NS = 512;                           // candidates in shared memory, sample runs of 8

struct Smem {
  uint32_t key[CAP_MAX];  // candidates' order keys, in pool order
  int idx[CAP_MAX];       // their pools
  int hist[256];
  int warp[2][WARPS];  // block_scan's warp totals, two buffers taken in turn
  int digit, need;     // a radix step's results
};

// sparse._order_key: a score's bits as a uint32 whose unsigned order is the scores' (-0 counted as +0).
__device__ __forceinline__ uint32_t order_key(float s) {
  const int bits = __float_as_int(__fadd_rn(s, 0.0f));
  return (uint32_t)(bits ^ ((bits >> 31) | (int)0x80000000u));
}

// Exclusive sum of v over the block's threads in thread order; `total` the block's sum. Every thread calls it.
__device__ __forceinline__ int block_scan(Smem& sm, int v, int& phase, int& total) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, buf = phase & 1;
  phase++;
  int x = v;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, x, d);
    if (lane >= d) x += y;
  }
  if (lane == 31) sm.warp[buf][w] = x;
  __syncthreads();
  int before = 0;
  total = 0;
#pragma unroll
  for (int i = 0; i < WARPS; ++i) {
    const int s = sm.warp[buf][i];
    if (i < w) before += s;
    total += s;
  }
  return before + x - v;
}

// The row's scores: element i's key and pool, and the keys of elements i0 .. i0 + PER - 1 (0 past n).
struct Row {
  const float* s;
  int n;
  __device__ __forceinline__ uint32_t key(int i) const { return order_key(s[i]); }
  __device__ __forceinline__ int pool(int i) const { return i; }
  __device__ __forceinline__ void keys(int i0, uint32_t (&u)[PER]) const {
    if (i0 + PER <= n && ((uintptr_t)(s + i0) & 15) == 0) {
      const float4 a = *reinterpret_cast<const float4*>(s + i0), b = *reinterpret_cast<const float4*>(s + i0 + 4);
      const float v[PER] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
      for (int j = 0; j < PER; ++j) u[j] = order_key(v[j]);
    } else {
#pragma unroll
      for (int j = 0; j < PER; ++j) u[j] = i0 + j < n ? order_key(s[i0 + j]) : 0u;
    }
  }
};

// The candidates in shared memory, in the same shape.
struct Cand {
  const Smem* sm;
  int n;
  __device__ __forceinline__ uint32_t key(int i) const { return sm->key[i]; }
  __device__ __forceinline__ int pool(int i) const { return sm->idx[i]; }
  __device__ __forceinline__ void keys(int i0, uint32_t (&u)[PER]) const {
#pragma unroll
    for (int j = 0; j < PER; ++j) u[j] = i0 + j < n ? sm->key[i0 + j] : 0u;
  }
};

// sparse._radix_step from sm.hist: the pass's digit of the need-th best key under prefix, and need less those above.
__device__ __forceinline__ void radix_step(Smem& sm, int& need, uint32_t& prefix, uint32_t& fixed, int p, int& phase) {
  const int b = T - 1 - threadIdx.x;  // thread t holds bin 255 - t
  const int h = sm.hist[b];
  int total;
  const int above = block_scan(sm, h, phase, total);  // pools in bins above b
  if (above + h >= need) atomicMax(&sm.digit, b);
  __syncthreads();
  const int hit = sm.digit;  // -1: no bin reaches need, as tf_glm_select_rows takes digit 0 and need less every pool
  if (hit >= 0 && b == hit) sm.need = need - above;
  __syncthreads();
  const uint32_t digit = hit >= 0 ? (uint32_t)hit : 0u;
  need = hit >= 0 ? sm.need : need - total;
  prefix |= digit << (24 - 8 * p);
  fixed |= 0xFFu << (24 - 8 * p);
}

// A pass's histogram cleared and the digit reset; the caller's previous reads of them are done first.
__device__ __forceinline__ void hist_clear(Smem& sm) {
  __syncthreads();
  sm.hist[threadIdx.x] = 0;
  if (threadIdx.x == 0) sm.digit = -1;
  __syncthreads();
}

// sparse._select_rows over src's n elements: the K best's pools ascending into out, ties to the lower pool.
template <class Src>
__device__ void select(Smem& sm, const Src& src, int K, int64_t* __restrict__ out, int& phase) {
  uint32_t prefix = 0, fixed = 0;
  int need = K;
  for (int p = 0; p < 4; ++p) {
    hist_clear(sm);
    const int sh = 24 - 8 * p;
    for (int i = threadIdx.x; i < src.n; i += T) {
      const uint32_t u = src.key(i);
      if ((u & fixed) == prefix) atomicAdd(&sm.hist[(u >> sh) & 0xFFu], 1);
    }
    __syncthreads();
    radix_step(sm, need, prefix, fixed, p, phase);
  }
  // in pool order: every key above the K-th best's, and the first `need` equal to it, each at its rank among those
  int gt_run = 0, eq_run = 0;
  for (int c0 = 0; c0 < src.n; c0 += CHUNK) {
    const int i0 = c0 + (int)threadIdx.x * PER;
    uint32_t u[PER];
    src.keys(i0, u);
    int g = 0, e = 0;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
      const bool ok = i0 + j < src.n;
      g += ok && u[j] > prefix;
      e += ok && u[j] == prefix;
    }
    int total;
    const int ex = block_scan(sm, (g << 16) | e, phase, total);
    int gb = gt_run + (ex >> 16), eb = eq_run + (ex & 0xFFFF);
#pragma unroll
    for (int j = 0; j < PER; ++j) {
      if (i0 + j >= src.n) break;
      if (u[j] > prefix) {
        out[gb + min(eb, need)] = (int64_t)src.pool(i0 + j);
        gb++;
      } else if (u[j] == prefix) {
        if (eb < need) out[gb + eb] = (int64_t)src.pool(i0 + j);
        eb++;
      }
    }
    gt_run += total >> 16;
    eq_run += total & 0xFFFF;
  }
}

// The 16-bit key prefix of the rank of NS runs of 8 scores at even spacing that scales to TARGET pools. Speed only.
__device__ uint32_t sample_floor(Smem& sm, const float* row, int NP, int TARGET, int& phase) {
  uint32_t prefix = 0, fixed = 0;
  int need = max((int)((int64_t)TARGET * (NS * 8) / NP), 1);
  for (int p = 0; p < 2; ++p) {
    hist_clear(sm);
    const int sh = 24 - 8 * p;
    for (int s = threadIdx.x; s < NS * 8; s += T) {
      const int64_t j = (int64_t)(s / 8) * NP / NS / 8 * 8 + s % 8;
      const uint32_t u = order_key(j < NP ? row[j] : 0.0f);
      if ((u & fixed) == prefix) atomicAdd(&sm.hist[(u >> sh) & 0xFFu], 1);
    }
    __syncthreads();
    radix_step(sm, need, prefix, fixed, p, phase);
  }
  return prefix;
}

}  // namespace glm_index

// tf_glm_select_rows' pools for scores [R][np]; CAP (<= 4096), TARGET speed only; PATH null or [R] int32; grid (R).
extern "C" __global__ void __launch_bounds__(glm_index::T)
    tf_glm_index_pools(const float* __restrict__ S, int64_t* __restrict__ OUT, int64_t np, const int* __restrict__ pos,
                       int K, int CAP, int TARGET, int* __restrict__ PATH) {
  using namespace glm_index;
  __shared__ Smem sm;
  CAP = min(CAP, CAP_MAX);
  const int64_t r = blockIdx.x, vis = ((int64_t)pos[0] + r + 1) / 4;
  const float* row = S + r * np;
  int64_t* out = OUT + r * K;
  const int NP = (int)min(np, max(vis, (int64_t)K));
  int phase = 0;
  if (NP <= CAP) {
    select(sm, Row{row, NP}, K, out, phase);
    if (PATH != nullptr && threadIdx.x == 0) PATH[r] = 0;
    return;
  }
  const uint32_t lo = sample_floor(sm, row, NP, TARGET, phase);
  const Row src{row, NP};
  int n = 0;
  for (int c0 = 0; c0 < NP; c0 += CHUNK) {
    const int i0 = c0 + (int)threadIdx.x * PER;
    uint32_t u[PER];
    src.keys(i0, u);
    int m = 0;
#pragma unroll
    for (int j = 0; j < PER; ++j) m += i0 + j < NP && u[j] >= lo;
    int total;
    int at = n + block_scan(sm, m, phase, total);
#pragma unroll
    for (int j = 0; j < PER; ++j)
      if (i0 + j < NP && u[j] >= lo) {
        if (at < CAP) {
          sm.key[at] = u[j];
          sm.idx[at] = i0 + j;
        }
        at++;
      }
    n += total;
    if (n > CAP) break;  // the whole row's then: no need to count on
  }
  __syncthreads();  // the candidates other threads stored
  const bool fits = n >= K && n <= CAP;
  if (fits) select(sm, Cand{&sm, n}, K, out, phase);
  else select(sm, src, K, out, phase);
  if (PATH != nullptr && threadIdx.x == 0) PATH[r] = fits ? 1 : 2;
}
