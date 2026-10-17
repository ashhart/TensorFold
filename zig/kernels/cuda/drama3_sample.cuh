// drama3 samplers: top-k / top-p with an exponential race on Triton's Philox stream, and RAS for the main token.
#pragma once
#include "drama3_common.cuh"

namespace {

__device__ __forceinline__ uint32_t umulhi32(uint32_t a, uint32_t b) { return __umulhi(a, b); }

__device__ uint32_t philox_first(uint64_t seed, uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3) {
  uint32_t k0 = (uint32_t)(seed & 0xFFFFFFFFull), k1 = (uint32_t)(seed >> 32);
#pragma unroll
  for (int r = 0; r < 10; ++r) {
    const uint32_t _c0 = c0, _c2 = c2;
    c0 = umulhi32(0xCD9E8D57u, _c2) ^ c1 ^ k0;
    c2 = umulhi32(0xD2511F53u, _c0) ^ c3 ^ k1;
    c1 = 0xCD9E8D57u * _c2;
    c3 = 0xD2511F53u * _c0;
    k0 += 0x9E3779B9u;
    k1 += 0xBB67AE85u;
  }
  return c0;
}

__device__ __forceinline__ float exp_noise(uint64_t seed, uint64_t step, int row, int stream, int index) {
  const uint64_t key = seed ^ ((step >> 32) << 32);
  const uint32_t bits = philox_first(key, (uint32_t)index, (uint32_t)row, (uint32_t)stream, (uint32_t)(step & 0xFFFFFFFFull));
  const float u = ((float)(bits >> 9) + 0.5f) * (1.0f / 8388608.0f);
  return -logf(u);
}

// Order-preserving key of a float score with the index (lower index wins ties), 0 for no candidate.
__device__ __forceinline__ unsigned long long score_key(float s, int idx) {
  uint32_t b = __float_as_uint(s == 0.f ? 0.f : s);
  b = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
  return ((unsigned long long)b << 32) | (uint32_t)(0xFFFFFFFFu - (uint32_t)idx);
}
__device__ __forceinline__ float key_score(unsigned long long k) {
  uint32_t b = (uint32_t)(k >> 32);
  b = (b & 0x80000000u) ? (b & 0x7FFFFFFFu) : ~b;
  return __uint_as_float(b);
}
__device__ __forceinline__ int key_index(unsigned long long k) { return (int)(0xFFFFFFFFu - (uint32_t)k); }

// Bitonic sort of keys[0..N) descending, N a power of two, blockDim.x threads.
template <int N>
__device__ void sort_desc(unsigned long long* keys) {
  for (int k = 2; k <= N; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      for (int i = threadIdx.x; i < N; i += blockDim.x) {
        const int ixj = i ^ j;
        if (ixj > i) {
          const unsigned long long a = keys[i], b = keys[ixj];
          const bool desc = (i & k) == 0;
          if (desc ? (a < b) : (a > b)) {
            keys[i] = b;
            keys[ixj] = a;
          }
        }
      }
      __syncthreads();
    }
  }
}

// One draw from keys sorted descending by raw logit (count valid). Returns the candidate index (block-uniform).
template <int N>
__device__ int draw_sorted(const unsigned long long* keys, int count, float temperature, int top_k, float top_p,
                           uint64_t seed, uint64_t step, int row, int stream, float* scan) {
  const float t = fmaxf(temperature, 1e-5f);
  const float mx = key_score(keys[0]) / t;
  // weights over the valid candidates, their sum, and the inclusive prefix sum in sorted order (per-thread chunks)
  constexpr int PER = N / 1024 > 0 ? N / 1024 : 1;
  const int base = threadIdx.x * PER;
  float local[PER];
  float run = 0.f;
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    const int i = base + j;
    const float wv = (i < count && i < N) ? expf(key_score(keys[i]) / t - mx) : 0.f;
    run += wv;
    local[j] = run;
  }
  // exclusive scan of the per-thread totals
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  float incl = run;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const float n = __shfl_up_sync(FULL, incl, o);
    if (lane >= o) incl += n;
  }
  __shared__ float wsum[32];
  __shared__ float total;
  __syncthreads();
  if (lane == 31) wsum[w] = incl;
  __syncthreads();
  if (w == 0) {
    float x = lane < (int)(blockDim.x >> 5) ? wsum[lane] : 0.f;
    float xi = x;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const float n = __shfl_up_sync(FULL, xi, o);
      if (lane >= o) xi += n;
    }
    wsum[lane] = xi - x;  // exclusive
    if (lane == 31) total = xi;
  }
  __syncthreads();
  const float before = wsum[w] + (incl - run);
  const float sum = total;
  // race over the kept candidates
  float best = -1.f;
  int best_rank = 0x7FFFFFFF;
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    const int i = base + j;
    if (i >= count || i >= N) break;
    const float cum = (before + local[j]) / sum;
    const bool kept = i == 0 || (i < top_k && cum <= top_p);
    if (!kept) continue;
    const float wv = expf(key_score(keys[i]) / t - mx);
    const float race = wv / exp_noise(seed, step, row, stream, key_index(keys[i]));
    if (race > best) {
      best = race;
      best_rank = i;
    }
  }
  // block argmax (ties to the lower rank)
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float ob = __shfl_xor_sync(FULL, best, o);
    const int orank = __shfl_xor_sync(FULL, best_rank, o);
    if (ob > best || (ob == best && orank < best_rank)) {
      best = ob;
      best_rank = orank;
    }
  }
  __shared__ float bb[32];
  __shared__ int br[32];
  __shared__ int chosen;
  if (lane == 0) {
    bb[w] = best;
    br[w] = best_rank;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float b = bb[0];
    int r = br[0];
    for (int k = 1; k < (int)(blockDim.x >> 5); ++k)
      if (bb[k] > b || (bb[k] == b && br[k] < r)) {
        b = bb[k];
        r = br[k];
      }
    chosen = key_index(keys[r]);
  }
  __syncthreads();
  (void)scan;
  return chosen;
}

}  // namespace

// Main token and code 0: request draw (stream 0), RAS draw (stream 1) taken when the first repeats in the window.
extern "C" __global__ void __launch_bounds__(1024) d3_sample_main(const float* __restrict__ logits, int count,
                                                                  const int* __restrict__ cand_ids,
                                                                  const int* __restrict__ window, D3Sampling sp,
                                                                  const unsigned long long* __restrict__ step_ptr,
                                                                  int* __restrict__ frame, int sem_begin, int cb) {
  constexpr int N = 8192;
  extern __shared__ unsigned long long keys[];
  for (int i = threadIdx.x; i < N; i += blockDim.x) keys[i] = i < count ? score_key(logits[i], i) : 0ull;
  __syncthreads();
  sort_desc<N>(keys);
  const uint64_t step = *step_ptr;
  const int first = draw_sorted<N>(keys, count, sp.temperature, sp.top_k, sp.top_p, sp.seed, step, 0, 0, nullptr);
  const int ras = draw_sorted<N>(keys, count, 1.0f, 30, 0.95f, sp.seed, step, 0, 1, nullptr);
  if (threadIdx.x == 0) {
    const int ft = cand_ids[first], rt = cand_ids[ras];
    bool repeat = false;
    for (int i = 0; i < 16; ++i) repeat |= window[i] == ft;
    const bool sem = ft >= sem_begin && ft < sem_begin + cb;
    const int main = (repeat && sem) ? rt : ft;
    frame[0] = main;
    frame[1] = min(max(main - sem_begin, 0), cb - 1);
  }
}

// A codebook head's code: logits [4096] fp32, the request's settings, stream 2 + head; *out = code.
extern "C" __global__ void __launch_bounds__(1024) d3_sample_code(const float* __restrict__ logits, int count,
                                                                  D3Sampling sp,
                                                                  const unsigned long long* __restrict__ step_ptr,
                                                                  int stream, int* __restrict__ out) {
  constexpr int N = 4096;
  extern __shared__ unsigned long long keys[];
  for (int i = threadIdx.x; i < N; i += blockDim.x) keys[i] = i < count ? score_key(logits[i], i) : 0ull;
  __syncthreads();
  sort_desc<N>(keys);
  const int c = draw_sorted<N>(keys, count, sp.temperature, sp.top_k, sp.top_p, sp.seed, *step_ptr, 0, stream, nullptr);
  if (threadIdx.x == 0) *out = c;
}

// Fast samplers for top_k <= 64: radix-select the candidates, then the same keys and draw rule as the full sort.
namespace {

constexpr int CAP = 256;

__device__ __forceinline__ uint32_t ord32(float s) {
  uint32_t b = __float_as_uint(s == 0.f ? 0.f : s);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// The `need` largest of count logits plus boundary ties (at most CAP) into cand, sorted descending; returns the count.
__device__ int select_top(const float* __restrict__ logits, int count, int need, unsigned long long* cand) {
  __shared__ int hist[256];
  __shared__ uint32_t s_prefix, s_mask;
  __shared__ int s_above, s_rem, s_done, s_n;
  if (threadIdx.x == 0) {
    s_prefix = 0;
    s_mask = 0;
    s_above = 0;
    s_rem = need;
    s_done = 0;
    s_n = 0;
  }
  __syncthreads();
  for (int shift = 24; shift >= 0; shift -= 8) {
    for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0;
    __syncthreads();
    const uint32_t prefix = s_prefix, mask = s_mask;
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
      const uint32_t u = ord32(logits[i]);
      if ((u & mask) == prefix) atomicAdd(&hist[(u >> shift) & 255], 1);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      int acc = 0, b = 255;
      for (; b > 0; --b) {
        if (acc + hist[b] >= s_rem) break;
        acc += hist[b];
      }
      if (s_above + acc + hist[b] <= CAP || shift == 0) {
        s_prefix = prefix | ((uint32_t)b << shift);  // keep every key >= this (lower bits zero)
        s_mask = 0;
        s_done = 1;
      } else {
        s_prefix = prefix | ((uint32_t)b << shift);
        s_mask = mask | (255u << shift);
        s_above += acc;
        s_rem -= acc;
      }
    }
    __syncthreads();
    if (s_done) break;
  }
  const uint32_t thr = s_prefix;
  for (int i = threadIdx.x; i < CAP; i += blockDim.x) cand[i] = 0ull;
  __syncthreads();
  for (int i = threadIdx.x; i < count; i += blockDim.x) {
    const float v = logits[i];
    if (ord32(v) >= thr) {
      const int at = atomicAdd(&s_n, 1);
      if (at < CAP) cand[at] = score_key(v, i);
    }
  }
  __syncthreads();
  sort_desc<CAP>(cand);
  return min(s_n, CAP);
}

// sum over all count logits of exp(l / t - mx) (mx = the largest l / t), the full softmax's denominator.
__device__ float softmax_sum(const float* __restrict__ logits, int count, float t, float mx) {
  float s = 0.f;
  for (int i = threadIdx.x; i < count; i += blockDim.x) s += expf(logits[i] / t - mx);
  return block_sum(s);
}

// One draw among sorted candidates: kept = rank 0 or rank < top_k with cumulative p <= top_p, race w / Exp(1).
__device__ int draw_top(const unsigned long long* cand, int n, float temperature, int top_k, float top_p, float sum,
                        uint64_t seed, uint64_t step, int stream) {
  const float t = fmaxf(temperature, 1e-5f);
  const float mx = key_score(cand[0]) / t;
  const int i = threadIdx.x, lane = i & 31, w = i >> 5;
  const float wv = (i < n) ? expf(key_score(cand[i]) / t - mx) : 0.f;
  float incl = wv;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const float x = __shfl_up_sync(FULL, incl, o);
    if (lane >= o) incl += x;
  }
  __shared__ float wsum[32];
  __shared__ float bbest[32];
  __shared__ int brank[32];
  __shared__ int chosen;
  if (lane == 31) wsum[w] = incl;
  __syncthreads();
  float before = 0.f;
  for (int k = 0; k < w && k < CAP / 32; ++k) before += wsum[k];
  const float cum = (before + incl) / sum;
  float best = -1.f;
  int rank = 0x7FFFFFFF;
  if (i < n && (i == 0 || (i < top_k && cum <= top_p))) {
    best = wv / exp_noise(seed, step, 0, stream, key_index(cand[i]));
    rank = i;
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float ob = __shfl_xor_sync(FULL, best, o);
    const int orr = __shfl_xor_sync(FULL, rank, o);
    if (ob > best || (ob == best && orr < rank)) {
      best = ob;
      rank = orr;
    }
  }
  if (lane == 0) {
    bbest[w] = best;
    brank[w] = rank;
  }
  __syncthreads();
  if (i == 0) {
    float b = bbest[0];
    int r = brank[0];
    for (int k = 1; k < (int)(blockDim.x >> 5); ++k)
      if (bbest[k] > b || (bbest[k] == b && brank[k] < r)) {
        b = bbest[k];
        r = brank[k];
      }
    chosen = key_index(cand[r]);
  }
  __syncthreads();
  return chosen;
}

}  // namespace

// d3_sample_main's draws over the radix-selected candidates (identical picks).
extern "C" __global__ void __launch_bounds__(1024) d3_sample_main_fast(const float* __restrict__ logits, int count,
                                                                       const int* __restrict__ cand_ids,
                                                                       const int* __restrict__ window, D3Sampling sp,
                                                                       const unsigned long long* __restrict__ step_ptr,
                                                                       int* __restrict__ frame, int sem_begin, int cb) {
  __shared__ unsigned long long cand[CAP];
  const int n = select_top(logits, count, max(sp.top_k, 30), cand);
  const float t = fmaxf(sp.temperature, 1e-5f);
  const float top = key_score(cand[0]);
  const float zt = softmax_sum(logits, count, t, top / t);
  const float z1 = softmax_sum(logits, count, 1.0f, top);
  const uint64_t step = *step_ptr;
  const int first = draw_top(cand, n, sp.temperature, sp.top_k, sp.top_p, zt, sp.seed, step, 0);
  const int ras = draw_top(cand, n, 1.0f, 30, 0.95f, z1, sp.seed, step, 1);
  if (threadIdx.x == 0) {
    const int ft = cand_ids[first], rt = cand_ids[ras];
    bool repeat = false;
    for (int i = 0; i < 16; ++i) repeat |= window[i] == ft;
    const bool sem = ft >= sem_begin && ft < sem_begin + cb;
    const int main = (repeat && sem) ? rt : ft;
    frame[0] = main;
    frame[1] = min(max(main - sem_begin, 0), cb - 1);
  }
}

// d3_sample_code's draw over the radix-selected candidates (identical picks).
extern "C" __global__ void __launch_bounds__(1024) d3_sample_code_fast(const float* __restrict__ logits, int count,
                                                                       D3Sampling sp,
                                                                       const unsigned long long* __restrict__ step_ptr,
                                                                       int stream, int* __restrict__ out) {
  __shared__ unsigned long long cand[CAP];
  const int n = select_top(logits, count, sp.top_k, cand);
  const float t = fmaxf(sp.temperature, 1e-5f);
  const float z = softmax_sum(logits, count, t, key_score(cand[0]) / t);
  const int c = draw_top(cand, n, sp.temperature, sp.top_k, sp.top_p, z, sp.seed, *step_ptr, stream);
  if (threadIdx.x == 0) *out = c;
}
