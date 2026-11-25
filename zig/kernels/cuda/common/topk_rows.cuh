// An exact row top-k for every family: an 8-bit radix select of fp32 scores, ties to the lower index, picks in order.
#pragma once

#include <stdint.h>

namespace tf_topk {

// A score's bits as a uint32 whose unsigned order is the scores' (-0 counted as +0, any NaN above +inf).
__device__ __forceinline__ uint32_t order_key(float s) {
  const int32_t b = __float_as_int(s + 0.0f);
  return (uint32_t)(b ^ ((b >> 31) | (int32_t)0x80000000));
}

// W warps' exclusive prefix of v in thread order and its total, via slot of `wsum`; its barrier orders slot reuse.
template <int W>
__device__ __forceinline__ int block_scan(int v, int (*wsum)[W], int slot, int* total) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int inc = v;
  for (int o = 1; o < 32; o <<= 1) {
    const int u = __shfl_up_sync(0xffffffffu, inc, o);
    if (lane >= o) inc += u;
  }
  if (lane == 31) wsum[slot][warp] = inc;
  __syncthreads();
  int before = inc - v;
  for (int w = 0; w < warp; ++w) before += wsum[slot][w];
  int all = 0;
  for (int w = 0; w < W; ++w) all += wsum[slot][w];
  *total = all;
  return before;
}

// The k best of row[0, limit) by an 8-bit radix select, ties to the lower index, written to out in index order.
template <int T, typename O>
__device__ __forceinline__ void select_row(const float* row, int64_t limit, int k, O* out) {
  __shared__ int hist[256];
  __shared__ uint32_t sh_prefix;
  __shared__ int sh_need, wsum[2][T / 32];
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  const unsigned below = (1u << lane) - 1;
  const bool vec = (reinterpret_cast<uintptr_t>(row) & 15) == 0;
  // The order keys of pools c + 8 t .. + 7 (0 past limit; callers test the index).
  auto keys = [&](int64_t c, uint32_t u[8]) {
    const int64_t i0 = c + 8 * t;
    if (vec && i0 + 8 <= limit) {
      const float4 a = *reinterpret_cast<const float4*>(row + i0), b = *reinterpret_cast<const float4*>(row + i0 + 4);
      const float v[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
      for (int j = 0; j < 8; ++j) u[j] = order_key(v[j]);
    } else {
      for (int j = 0; j < 8; ++j) u[j] = i0 + j < limit ? order_key(row[i0 + j]) : 0u;
    }
  };
  // T only sets the pools a pass reads (8 a thread): digit counts and index-order scans give every T the same picks
  uint32_t prefix = 0, fixed = 0;
  int need = k;
  for (int shift = 24; shift >= 0; shift -= 8) {
    for (int b = t; b < 256; b += blockDim.x) hist[b] = 0;
    __syncthreads();
    for (int64_t c = 0; c < limit; c += 8 * T) {
      uint32_t u[8];
      keys(c, u);
      for (int j = 0; j < 8; ++j) {
        const int b = c + 8 * t + j < limit && (u[j] & fixed) == prefix ? (int)((u[j] >> shift) & 0xFF) : 256;
        if (__any_sync(0xffffffffu, b < 256)) {
          const unsigned same = __match_any_sync(0xffffffffu, b);
          if (b < 256 && (same & below) == 0) atomicAdd(&hist[b], __popc(same));
        }
      }
    }
    __syncthreads();
    if (warp == 0) {
      // Lane l holds bins 255 - 8 l .. 248 - 8 l; the digit is in the first lane whose count reaches need, else 0.
      int h[8], own = 0;
      for (int j = 0; j < 8; ++j) own += h[j] = hist[255 - 8 * lane - j];
      int upto = own;
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
        sh_need = need - above;
        sh_prefix = prefix | ((uint32_t)digit << shift);
      }
    }
    __syncthreads();
    need = sh_need;
    prefix = sh_prefix;
    fixed |= 0xFFu << shift;
    __syncthreads();
  }
  int run_take = 0, run_eq = 0;
  for (int64_t c = 0; c < limit; c += 8 * T) {
    uint32_t u[8];
    keys(c, u);
    bool eq[8], gt[8];
    int eqs = 0;
    for (int j = 0; j < 8; ++j) {
      const bool in = c + 8 * t + j < limit;
      eq[j] = in && u[j] == prefix;
      gt[j] = in && u[j] > prefix;
      eqs += eq[j];
    }
    int all_eq, all_take;
    int seen = run_eq + block_scan<T / 32>(eqs, wsum, 0, &all_eq);
    bool take[8];
    int takes = 0;
    for (int j = 0; j < 8; ++j) {
      take[j] = gt[j] || (eq[j] && seen < need);
      seen += eq[j];
      takes += take[j];
    }
    int at = run_take + block_scan<T / 32>(takes, wsum, 1, &all_take);
    for (int j = 0; j < 8; ++j)
      if (take[j]) out[at++] = (O)(c + 8 * t + j);
    run_eq += all_eq;
    run_take += all_take;
  }
}

}  // namespace tf_topk
