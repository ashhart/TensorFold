// sparse.py's DSA indexer glue by hand: the index key write, the 4-token pools and the per-row top-512 pools.
#pragma once

#include "glm_triton.cuh"

using namespace glm_triton;

// sparse._index_write, a warp a row (D 128): LayerNorm of k_raw (div.full, sqrt.approx) into ik, gate row into ig.
extern "C" __global__ void __launch_bounds__(32)
    tf_glm_index_write(const __nv_bfloat16* kr, int64_t k_stride, const float* gr, const __nv_bfloat16* lnw,
                       const __nv_bfloat16* lnb, __nv_bfloat16* ik, __nv_bfloat16* ig, const int* pos, float eps) {
  constexpr int D = 128;
  const int64_t r = blockIdx.x, at = (int64_t)pos[0] + r;
  const int d0 = threadIdx.x * 4;
  float x[4];
  for (int j = 0; j < 4; ++j) x[j] = bf(kr[r * k_stride + d0 + j]);
  float s = x[0] + x[1];
  s = x[2] + s;
  s = x[3] + s;
  for (int o = 16; o > 0; o >>= 1) s = s + __shfl_xor_sync(0xffffffffu, s, o);
  const float mean = div_full(s, (float)D);
  float xc[4];
  for (int j = 0; j < 4; ++j) xc[j] = __fsub_rn(x[j], mean);  // sub.rn in Triton: never contracted
  float v = fmul(xc[1], xc[1]);
  v = __fmaf_rn(xc[0], xc[0], v);
  v = __fmaf_rn(xc[2], xc[2], v);
  v = __fmaf_rn(xc[3], xc[3], v);
  for (int o = 16; o > 0; o >>= 1) v = v + __shfl_xor_sync(0xffffffffu, v, o);
  const float sd = sqrt_approx(fadd(eps, div_full(v, (float)D)));
  for (int j = 0; j < 4; ++j) {
    const int d = d0 + j;
    ik[at * D + d] = __float2bfloat16_rn(__fmaf_rn(div_full(xc[j], sd), bf(lnw[d]), bf(lnb[d])));
    ig[at * D + d] = __float2bfloat16_rn(gr[r * D + d]);
  }
}

// sparse._pool_keys, a warp a completed pool: softmax of its 4 gate rows (+ ape) times bf16 keys, summed in order.
extern "C" __global__ void __launch_bounds__(32)
    tf_glm_pool_keys(const __nv_bfloat16* ik, const __nv_bfloat16* ig, const __nv_bfloat16* ape, __nv_bfloat16* pk,
                     const int* pos, int64_t rows, const int* pt, int64_t ps) {
  constexpr int D = 128;
  const int64_t P = pos[0], p = P / 4 + blockIdx.x;
  if (4 * p + 3 > P + rows - 1) return;
  __nv_bfloat16* out = pk + glm_pages::at<128>(pt, ps / 2, p, D);  // a page holds 512 rows' 128 pools
  for (int j = 0; j < 4; ++j) {
    const int d = threadIdx.x * 4 + j;
    float l[4], e[4];
    for (int q = 0; q < 4; ++q) l[q] = bf(ig[(4 * p + q) * D + d]) + bf(ape[q * D + d]);
    const float m = fmaxf(fmaxf(l[0], l[1]), fmaxf(l[2], l[3]));
    for (int q = 0; q < 4; ++q) e[q] = exp_approx(l[q] - m);
    const float s = ((e[0] + e[1]) + e[2]) + e[3];
    float t = 0.0f;
    for (int q = 0; q < 4; ++q) {
      const float tq = rbf(__fmul_rn(rbf(div_full(e[q], s)), bf(ik[(4 * p + q) * D + d])));
      t = q == 0 ? tq : t + tq;
    }
    out[d] = __float2bfloat16_rn(t);
  }
}

namespace glm_index {

// sparse._order_key: a score's bits as a uint32 whose unsigned order is the scores' (-0 counted as +0).
__device__ __forceinline__ uint32_t order_key(float s) {
  const int32_t b = __float_as_int(s + 0.0f);
  return (uint32_t)(b ^ ((b >> 31) | (int32_t)0x80000000));
}

}  // namespace glm_index

namespace glm_index {

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

// sparse._select_rows (VISIBLE), a row per T-thread block: K best visible pools by 8-bit radix select, ties to lower.
template <int T>
__device__ __forceinline__ void select_row(const float* scores, int64_t* out, int64_t np, const int* pos, int k) {
  __shared__ int hist[256];
  __shared__ uint32_t sh_prefix;
  __shared__ int sh_need, wsum[2][T / 32];
  const int64_t r = blockIdx.x;
  const float* row = scores + r * np;
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  const unsigned below = (1u << lane) - 1;
  const int64_t vis = ((int64_t)pos[0] + r + 1) / 4;
  const int64_t limit = np < (vis > k ? vis : k) ? np : (vis > k ? vis : k);
  const bool vec = (reinterpret_cast<uintptr_t>(row) & 15) == 0;
  // The order keys of pools c + 8 t .. + 7 (0 past limit; callers test the index).
  auto keys = [&](int64_t c, uint32_t u[8]) {
    const int64_t i0 = c + 8 * t;
    if (vec && i0 + 8 <= limit) {
      const float4 a = *reinterpret_cast<const float4*>(row + i0), b = *reinterpret_cast<const float4*>(row + i0 + 4);
      const float v[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
      for (int j = 0; j < 8; ++j) u[j] = glm_index::order_key(v[j]);
    } else {
      for (int j = 0; j < 8; ++j) u[j] = i0 + j < limit ? glm_index::order_key(row[i0 + j]) : 0u;
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
      if (take[j]) out[r * k + at++] = c + 8 * t + j;
    run_eq += all_eq;
    run_take += all_take;
  }
}

}  // namespace glm_index

// sparse._select_rows (VISIBLE), a row per 4-warp block: K best visible pools by 8-bit radix select, ties to lower.
extern "C" __global__ void __launch_bounds__(128)
    tf_glm_select_rows(const float* scores, int64_t* out, int64_t np, const int* pos, int k) {
  glm_index::select_row<128>(scores, out, np, pos, k);
}

// tf_glm_select_rows a row per 1,024-thread block, for a few rows over many pools (a decode step's): its bytes.
extern "C" __global__ void __launch_bounds__(1024)
    tf_glm_select_rows_wide(const float* scores, int64_t* out, int64_t np, const int* pos, int k) {
  glm_index::select_row<1024>(scores, out, np, pos, k);
}
