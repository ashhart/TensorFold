// music3 AR sampling: the CFG top-k draw and the frame bookkeeping inside the frame graph.
#pragma once
#include "music3_common.cuh"

// ------------------------------------------------------------------------------------------------ AR: sampling


__device__ __forceinline__ unsigned ord(float v) {  // float -> unsigned with the same order
  const unsigned u = __float_as_uint(v);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

#define SAMPLE_THREADS 1024
#define SAMPLE_PER 17  // values a thread: 1024 x 17 >= 16385
#define SAMPLE_CAP 1024

// A CFG top-k draw of logits [2, V] (cond, uncond) by inverse CDF in index order on a Philox uniform; code 0 also.
extern "C" __global__ void __launch_bounds__(SAMPLE_THREADS) m3_sample(const float* __restrict__ logits, int V, float cfg, int top_k, int rank_cond, unsigned long long seed, const unsigned long long* __restrict__ step, int slot, int* __restrict__ pick, int* __restrict__ code, int* __restrict__ tokens, const int* __restrict__ count, const int* __restrict__ force) {
  __shared__ unsigned hist[256];
  __shared__ unsigned sel[2];
  __shared__ int cnt_sh[SAMPLE_THREADS / 32];
  __shared__ int cidx[SAMPLE_CAP];
  __shared__ float cval[SAMPLE_CAP];
  __shared__ int ncand;
  const int t = threadIdx.x;
  const int lo = t * SAMPLE_PER;
  unsigned key[SAMPLE_PER];
  float gv[SAMPLE_PER];
#pragma unroll
  for (int j = 0; j < SAMPLE_PER; ++j) {
    const int i = lo + j;
    if (i < V) {
      const float c = logits[i], u = logits[V + i];
      gv[j] = u + (c - u) * cfg;
      key[j] = ord(rank_cond ? c : gv[j]);
    } else {
      gv[j] = -INFINITY;
      key[j] = 0;
    }
  }
  // radix select of the top_k-th largest key, 8 bits at a time from the top
  unsigned prefix = 0, mask = 0;
  int want = max(1, min(top_k, V));
  for (int shift = 24; shift >= 0; shift -= 8) {
    if (t < 256) hist[t] = 0;
    __syncthreads();
#pragma unroll
    for (int j = 0; j < SAMPLE_PER; ++j)
      if (lo + j < V && (key[j] & mask) == prefix) atomicAdd(&hist[(key[j] >> shift) & 255], 1u);
    __syncthreads();
    if (t == 0) {
      int acc = 0, dgt = 0;
      for (int dd = 255; dd >= 0; --dd) {
        if (acc + (int)hist[dd] >= want) {
          dgt = dd;
          break;
        }
        acc += hist[dd];
      }
      sel[0] = dgt;
      sel[1] = (unsigned)(want - acc);
    }
    __syncthreads();
    prefix |= sel[0] << shift;
    mask |= 255u << shift;
    want = (int)sel[1];
    __syncthreads();
  }
  const unsigned thr = prefix;  // keys >= thr are the candidates
  // compact candidates in index order
  int mine = 0;
#pragma unroll
  for (int j = 0; j < SAMPLE_PER; ++j) mine += (lo + j < V && key[j] >= thr);
  int incl = mine;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int v = __shfl_up_sync(FULL, incl, o);
    if ((t & 31) >= o) incl += v;
  }
  if ((t & 31) == 31) cnt_sh[t >> 5] = incl;
  __syncthreads();
  if (t == 0) {
    int a = 0;
    for (int w = 0; w < SAMPLE_THREADS / 32; ++w) {
      const int v = cnt_sh[w];
      cnt_sh[w] = a;
      a += v;
    }
    ncand = min(a, SAMPLE_CAP);
  }
  __syncthreads();
  int at = cnt_sh[t >> 5] + incl - mine;
#pragma unroll
  for (int j = 0; j < SAMPLE_PER; ++j)
    if (lo + j < V && key[j] >= thr) {
      if (at < SAMPLE_CAP) {
        cidx[at] = lo + j;
        cval[at] = gv[j];
      }
      ++at;
    }
  __syncthreads();
  if (t != 0) return;
  int n = ncand;
  int chosen;
  if (force) {
    chosen = *force;
  } else {
    if (rank_cond && n > top_k) {
      // ComfyUI's second sample_topk: keep values with fewer than top_k strictly greater ones (the top_k with ties)
      float kth = INFINITY;
      for (int i = 0; i < n; ++i) {
        int greater = 0;
        for (int j = 0; j < n; ++j) greater += cval[j] > cval[i];
        if (greater < top_k) kth = fminf(kth, cval[i]);
      }
      int w = 0;
      for (int i = 0; i < n; ++i)
        if (cval[i] >= kth) {
          cidx[w] = cidx[i];
          cval[w] = cval[i];
          ++w;
        }
      n = w;
    }
    float m = -INFINITY;
    for (int i = 0; i < n; ++i) m = fmaxf(m, cval[i]);
    if (top_k <= 1) {
      chosen = cidx[0];
      for (int i = 0; i < n; ++i)
        if (cval[i] == m) {
          chosen = cidx[i];
          break;
        }
    } else {
      float sum = 0.f;
      for (int i = 0; i < n; ++i) sum += __expf(cval[i] - m);
      const float u = Philox::uniform(seed, *step, (unsigned long long)slot) * sum;
      float run = 0.f;
      chosen = cidx[n - 1];
      for (int i = 0; i < n; ++i) {
        run += __expf(cval[i] - m);
        if (u < run) {
          chosen = cidx[i];
          break;
        }
      }
    }
  }
  *pick = chosen;
  if (code) *code = max(chosen - 1, 0);
  if (tokens) tokens[*count] = chosen;
}

// After a frame: the next position and step, one frame more.
extern "C" __global__ void m3_frame_done(int* __restrict__ pos, unsigned long long* __restrict__ step, int* __restrict__ count) {
  if (threadIdx.x == 0) {
    *pos += 1;
    *step += 1;
    *count += 1;
  }
}

// A frame state's part: states[*count, part] = src (bf16 4096).
extern "C" __global__ void m3_state_part(const bf16* __restrict__ src, bf16* __restrict__ states, const int* __restrict__ count, int part) {
  const size_t base = ((size_t)*count * 8 + part) * 4096;
  for (int i = threadIdx.x; i < 512; i += blockDim.x) reinterpret_cast<uint4*>(states + base)[i] = reinterpret_cast<const uint4*>(src)[i];
}

// A forced frame's codes (teacher forcing): codes[0..8) = forced[*count, 0..8).
extern "C" __global__ void m3_force_codes(const int* __restrict__ forced, const int* __restrict__ count, int* __restrict__ codes, int* __restrict__ token) {
  if (threadIdx.x < 8) codes[threadIdx.x] = forced[*count * 8 + threadIdx.x];
  if (threadIdx.x == 0) *token = forced[*count * 8] + 1;
}
