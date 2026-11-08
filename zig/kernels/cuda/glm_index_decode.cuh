// GLM-5.3-Flash DSA indexer, a decode window's top-512 pools of long rows on many CTAs: tf_glm_select_rows' bytes.
// tf_glm_dsel_scan: every CTA of a row takes the same floor (the need-th best of 4,096 sampled keys) and marks its
// span's pools above it (gt) and equal to it (eq). tf_glm_dsel_merge, a CTA a row, then selects exactly:
//   - fewer than K above the floor and at least K at or above it: the K-th best key is the floor itself, so the picks
//     are every gt pool and the first K - |gt| eq pools in pool order (no select at all);
//   - at most CAP candidates (gt and eq, or gt alone when it holds K): tf_glm_select_rows' radix select over them;
//   - otherwise (the floor overshot, or far too many above it): tf_glm_select_rows_wide's full-row select.
// The picks are the K best keys (sparse._order_key of the fp32 scores), ties to the lower pool, in ascending pool order,
// over each row's visible pools [0, max(vis, K)) clipped to np: tf_glm_select_rows' result whichever path a row takes.
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

namespace glm_dsel {

constexpr int T1 = 256;     // scan threads
constexpr int SPAN = 4096;  // pools a scan CTA marks: 16 words of 32 a warp
constexpr int NS8 = 4096;   // sampled keys: 512 runs of 8 at even spacing (glm_index_pools' sample)
constexpr int T2 = 1024;    // merge threads
constexpr int CAP = 4096;   // candidates the merge holds in shared memory
constexpr int PER = 8;      // keys a merge thread takes a select step

// sparse._order_key: a score's bits as a uint32 whose unsigned order is the scores' (-0 counted as +0).
__device__ __forceinline__ uint32_t order_key(float s) {
  const int b = __float_as_int(__fadd_rn(s, 0.0f));
  return (uint32_t)(b ^ ((b >> 31) | (int)0x80000000u));
}

// Row r's selection range [0, NP): its visible pools, at least K, at most np (tf_glm_select_rows' limit).
__device__ __forceinline__ int row_limit(int64_t np, const int* pos, int64_t r, int K) {
  const int64_t vis = ((int64_t)pos[0] + r + 1) / 4;
  return (int)min(np, max(vis, (int64_t)K));
}

// hist[b] += 1 for each lane's bin b (256: none), each bin's lanes adding once. The whole warp calls it.
__device__ __forceinline__ void hist_add(int* hist, int b) {
  if (__any_sync(0xffffffffu, b < 256)) {
    const unsigned same = __match_any_sync(0xffffffffu, b);
    if (b < 256 && (same & ((1u << (threadIdx.x & 31)) - 1)) == 0) atomicAdd(&hist[b], __popc(same));
  }
}

// Warp 0: the digit of the need-th largest key under prefix from hist, and need less the keys in the bins above it
// (no bin reaching need: digit 0 and need less every key, as tf_glm_select_rows takes it).
__device__ __forceinline__ void digit_of(const int* hist, int need, uint32_t prefix, int shift, uint32_t* sh_prefix,
                                         int* sh_need) {
  const int lane = threadIdx.x & 31;
  int h[8], own = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) own += h[j] = hist[255 - 8 * lane - j];
  int upto = own;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int v = __shfl_up_sync(0xffffffffu, upto, o);
    if (lane >= o) upto += v;
  }
  const unsigned hit = __ballot_sync(0xffffffffu, upto >= need);
  if (lane == (hit ? __ffs(hit) - 1 : 31)) {
    int above = upto - own, digit = 0;
    for (int j = 0; j < 8; ++j) {
      if (above + h[j] >= need) {
        digit = 255 - 8 * lane - j;
        break;
      }
      above += h[j];
    }
    *sh_need = need - above;
    *sh_prefix = prefix | ((uint32_t)digit << shift);
  }
}

// T threads' exclusive prefix of v in thread order and its total, via `slot` of wsum; its barrier orders slot reuse
// (callers alternate slots).
template <int T>
__device__ __forceinline__ int block_scan(int v, int (*wsum)[T / 32], int slot, int* total) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int inc = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int u = __shfl_up_sync(0xffffffffu, inc, o);
    if (lane >= o) inc += u;
  }
  if (lane == 31) wsum[slot][warp] = inc;
  __syncthreads();
  int before = inc - v, all = 0;
#pragma unroll
  for (int w = 0; w < T / 32; ++w) {
    const int s = wsum[slot][w];
    if (w < warp) before += s;
    all += s;
  }
  *total = all;
  return before;
}

template <int T>
struct Sel {
  int hist[256];
  int wsum[2][T / 32];
  uint32_t prefix;
  int need;
};

// A row's scores [0, n) from global memory: the keys of elements i0 .. i0 + 7 (0 past n), and element i's pool.
struct RowSrc {
  const float* s;
  int n;
  bool vec;
  __device__ __forceinline__ void keys8(int i0, uint32_t (&u)[PER]) const {
    if (vec && i0 + PER <= n) {
      const float4 a = *reinterpret_cast<const float4*>(s + i0), b = *reinterpret_cast<const float4*>(s + i0 + 4);
      const float v[PER] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
      for (int j = 0; j < PER; ++j) u[j] = order_key(v[j]);
    } else {
#pragma unroll
      for (int j = 0; j < PER; ++j) u[j] = i0 + j < n ? order_key(s[i0 + j]) : 0u;
    }
  }
  __device__ __forceinline__ int64_t pool(int i) const { return i; }
};

// Candidates in shared memory, in pool order: their keys and pools.
struct CandSrc {
  const uint32_t* key;
  const int* idx;
  int n;
  __device__ __forceinline__ void keys8(int i0, uint32_t (&u)[PER]) const {
#pragma unroll
    for (int j = 0; j < PER; ++j) u[j] = i0 + j < n ? key[i0 + j] : 0u;
  }
  __device__ __forceinline__ int64_t pool(int i) const { return idx[i]; }
};

// tf_glm_select_rows over src's n elements, T threads: the K best keys' pools ascending into out, ties to the lower
// pool. Four 8-bit radix passes find the K-th best key; an index-order pass emits every key above it and the first
// `need` equal to it. Every thread calls it; it starts and ends at a barrier of its own.
template <int T, class Src>
__device__ void select_core(Sel<T>& sh, const Src& src, int n, int K, int64_t* __restrict__ out) {
  const int t = threadIdx.x;
  uint32_t prefix = 0, fixed = 0;
  int need = K;
  for (int shift = 24; shift >= 0; shift -= 8) {
    for (int b = t; b < 256; b += T) sh.hist[b] = 0;
    __syncthreads();
    for (int c = 0; c < n; c += PER * T) {
      uint32_t u[PER];
      src.keys8(c + PER * t, u);
#pragma unroll
      for (int j = 0; j < PER; ++j) {
        const int i = c + PER * t + j;
        hist_add(sh.hist, i < n && (u[j] & fixed) == prefix ? (int)((u[j] >> shift) & 0xFF) : 256);
      }
    }
    __syncthreads();
    if (t < 32) digit_of(sh.hist, need, prefix, shift, &sh.prefix, &sh.need);
    __syncthreads();
    need = sh.need;
    prefix = sh.prefix;
    fixed |= 0xFFu << shift;
  }
  int gt_run = 0, eq_run = 0, slot = 0;
  for (int c = 0; c < n; c += PER * T) {
    const int i0 = c + PER * t;
    uint32_t u[PER];
    src.keys8(i0, u);
    int g = 0, e = 0;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
      const bool ok = i0 + j < n;
      g += ok && u[j] > prefix;
      e += ok && u[j] == prefix;
    }
    int total;
    const int ex = block_scan<T>((g << 16) | e, sh.wsum, slot, &total);  // a step's counts stay under 2^16
    slot ^= 1;
    int gb = gt_run + (ex >> 16), eb = eq_run + (ex & 0xFFFF);
#pragma unroll
    for (int j = 0; j < PER; ++j) {
      if (i0 + j >= n) break;
      if (u[j] > prefix) {
        out[gb + min(eb, need)] = src.pool(i0 + j);
        gb++;
      } else if (u[j] == prefix) {
        if (eb < need) out[gb + eb] = src.pool(i0 + j);
        eb++;
      }
    }
    gt_run += total >> 16;
    eq_run += total & 0xFFFF;
  }
  __syncthreads();  // shared state free for the caller
}

struct Merge {
  uint32_t key[CAP];
  int idx[CAP];
  Sel<T2> sel;
};

}  // namespace glm_dsel

// Scores [R][np] fp32; grid (cdiv(the last row's NP, SPAN), R). Row r's gt and eq bit words at bits[r][0][..] and
// bits[r][1][..] (words a part; bit b of word w is pool 32 w + b). Rows of NP <= CAP mark nothing (the merge takes
// them whole); each CTA writes only its span's words below its row's NP.
extern "C" __global__ void __launch_bounds__(glm_dsel::T1)
    tf_glm_dsel_scan(const float* __restrict__ S, int64_t np, const int* __restrict__ pos, int K, int target,
                     uint32_t* __restrict__ bits, int64_t words) {
  using namespace glm_dsel;
  __shared__ int hist[256];
  __shared__ uint32_t sh_prefix;
  __shared__ int sh_need;
  const int64_t r = blockIdx.y;
  const int NP = row_limit(np, pos, r, K);
  const int64_t p0 = (int64_t)blockIdx.x * SPAN;
  if (NP <= CAP || p0 >= NP) return;
  const float* row = S + r * np;
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  // The floor: the need-th largest of the sampled keys, need the target's share of them. Speed only.
  constexpr int PT = NS8 / T1;
  uint32_t u[PT];
#pragma unroll
  for (int j = 0; j < PT; ++j) {
    const int s = t + j * T1;
    const int64_t i = (int64_t)(s / 8) * NP / (NS8 / 8) / 8 * 8 + s % 8;
    u[j] = order_key(i < NP ? row[i] : 0.0f);
  }
  uint32_t prefix = 0, fixed = 0;
  int need = max((int)((int64_t)target * NS8 / NP), 1);
  for (int shift = 24; shift >= 0; shift -= 8) {
    hist[t] = 0;  // T1 == 256 bins
    __syncthreads();
#pragma unroll
    for (int j = 0; j < PT; ++j) hist_add(hist, (u[j] & fixed) == prefix ? (int)((u[j] >> shift) & 0xFF) : 256);
    __syncthreads();
    if (warp == 0) digit_of(hist, need, prefix, shift, &sh_prefix, &sh_need);
    __syncthreads();
    prefix = sh_prefix;
    need = sh_need;
    fixed |= 0xFFu << shift;
  }
  const uint32_t lo = prefix;
  // The span's words, a warp's every 8th: its pools above and at the floor.
  constexpr int WARPS = T1 / 32, WPW = SPAN / 32 / WARPS;
  const int64_t w0 = p0 / 32;
  const int nwc = (int)min((int64_t)(SPAN / 32), ((int64_t)NP + 31) / 32 - w0);
  float v[WPW];
#pragma unroll
  for (int m = 0; m < WPW; ++m) {
    const int wi = warp + m * WARPS;
    const int64_t i = (w0 + wi) * 32 + lane;
    v[m] = wi < nwc && i < NP ? row[i] : 0.0f;
  }
  uint32_t* gt = bits + r * 2 * words;
#pragma unroll
  for (int m = 0; m < WPW; ++m) {
    const int wi = warp + m * WARPS;
    const int64_t i = (w0 + wi) * 32 + lane;
    const bool in = wi < nwc && i < NP;
    const uint32_t k = order_key(v[m]);
    const unsigned g = __ballot_sync(0xffffffffu, in && k > lo), e = __ballot_sync(0xffffffffu, in && k == lo);
    if (lane == 0 && wi < nwc) {
      gt[w0 + wi] = g;
      gt[words + w0 + wi] = e;
    }
  }
}

// The picks of tf_glm_dsel_scan's rows: scores [R][np], out [R][K] int64 ascending; grid (R). PATH null or [R] int32:
// 0 the whole row in one select (NP <= CAP), 1 the candidates, 2 the whole row after the candidates missed, 3 ties
// at the floor.
extern "C" __global__ void __launch_bounds__(glm_dsel::T2)
    tf_glm_dsel_merge(const float* __restrict__ S, int64_t* __restrict__ OUT, int64_t np, const int* __restrict__ pos,
                      int K, const uint32_t* __restrict__ bits, int64_t words, int* __restrict__ PATH) {
  using namespace glm_dsel;
  __shared__ Merge sm;
  const int64_t r = blockIdx.x;
  const int NP = row_limit(np, pos, r, K);
  const float* row = S + r * np;
  int64_t* out = OUT + r * K;
  const int t = threadIdx.x;
  const RowSrc whole{row, NP, (reinterpret_cast<uintptr_t>(row) & 15) == 0};
  if (NP <= CAP) {
    select_core<T2>(sm.sel, whole, NP, K, out);
    if (PATH != nullptr && t == 0) PATH[r] = 0;
    return;
  }
  const uint32_t* gt = bits + r * 2 * words;
  const uint32_t* eq = gt + words;
  const int nw = (NP + 31) / 32, per = (nw + T2 - 1) / T2;
  const int wa = min(nw, t * per), wb = min(nw, wa + per);  // this thread's words, in pool order
  int cg = 0, ce = 0;
  for (int w = wa; w < wb; ++w) {
    cg += __popc(gt[w]);
    ce += __popc(eq[w]);
  }
  int n_gt, n_eq;
  const int gt_before = block_scan<T2>(cg, sm.sel.wsum, 0, &n_gt);
  const int eq_before = block_scan<T2>(ce, sm.sel.wsum, 1, &n_eq);
  const int n_ge = n_gt + n_eq;
  int path;
  if (n_gt < K && K <= n_ge) {
    // The K-th best key is the floor: every gt pool, and the eq pools of eq rank under K - n_gt, in pool order.
    const int need = K - n_gt;
    int seen = eq_before, cs = 0;
    for (int w = wa; w < wb; ++w) {
      const uint32_t e = eq[w];
      uint32_t take = 0, m = e;
      for (int q = min(__popc(e), max(0, need - seen)); q > 0; --q) {
        take |= m & (0u - m);
        m &= m - 1;
      }
      cs += __popc(gt[w] | take);
      seen += __popc(e);
    }
    int all;
    int at = block_scan<T2>(cs, sm.sel.wsum, 0, &all);
    seen = eq_before;
    for (int w = wa; w < wb; ++w) {
      const uint32_t e = eq[w];
      uint32_t take = 0, m = e;
      for (int q = min(__popc(e), max(0, need - seen)); q > 0; --q) {
        take |= m & (0u - m);
        m &= m - 1;
      }
      seen += __popc(e);
      for (uint32_t s = gt[w] | take; s != 0; s &= s - 1) out[at++] = (int64_t)w * 32 + (__ffs(s) - 1);
    }
    path = 3;
  } else if ((K <= n_ge && n_ge <= CAP) || (K <= n_gt && n_gt <= CAP)) {
    // Every pool at or above the floor (or above it, when those hold K) in pool order: the K best are among them.
    const bool with_eq = n_ge <= CAP;
    const int n = with_eq ? n_ge : n_gt;
    int at = with_eq ? gt_before + eq_before : gt_before;
    for (int w = wa; w < wb; ++w)
      for (uint32_t s = gt[w] | (with_eq ? eq[w] : 0u); s != 0; s &= s - 1) sm.idx[at++] = w * 32 + (__ffs(s) - 1);
    __syncthreads();  // then their keys, spread over the threads (candidates may crowd a few threads' words)
    for (int c = t; c < n; c += T2) sm.key[c] = order_key(row[sm.idx[c]]);
    __syncthreads();
    select_core<T2>(sm.sel, CandSrc{sm.key, sm.idx, n}, n, K, out);
    path = 1;
  } else {
    select_core<T2>(sm.sel, whole, NP, K, out);
    path = 2;
  }
  if (PATH != nullptr && t == 0) PATH[r] = path;
}
