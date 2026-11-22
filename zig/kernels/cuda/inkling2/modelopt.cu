// Mini routed experts: checkpoint NVFP4/MXFP8 codes decode to BF16 mma fragments; no activation quantization.
#include <cuda_bf16.h>
#include <stdint.h>
#include "experts.cuh"

namespace ink2o {

struct Weight {
  const uint8_t* codes;
  const uint8_t* scales;
  const float* amax;
  const float* den;
  long long E, N, K;
  int G, format, row0, row_step;
};

__device__ __forceinline__ float e2m1(uint32_t n) {
  const uint32_t s = (n & 8) ? 0x80000000u : 0;
  const uint32_t e = (n >> 1) & 3, m = n & 1;
  if (!e) return __uint_as_float(s | (m ? 0x3f000000u : 0));
  return __uint_as_float(s | ((e + 126) << 23) | (m << 22));
}

__device__ __forceinline__ float e4m3(uint32_t n) {
  const uint32_t s = (n & 128) ? 0x80000000u : 0;
  const uint32_t e = (n >> 3) & 15, m = n & 7;
  if (e == 15 && m == 7) return __uint_as_float(s | 0x7fc00000u);
  if (!e) return (s ? -1.f : 1.f) * (float)m * 0.001953125f;
  return __uint_as_float(s | ((e + 120) << 23) | (m << 20));
}

__device__ __forceinline__ float e8m0(uint32_t n) {
  if (n == 0) return 0x1p-127f;
  return __uint_as_float(n << 23);
}

__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ __nv_bfloat16 decoded(const Weight& w, long long e, long long row, long long col) {
  const long long z = e * w.N * w.K + row * w.K + col;
  const uint32_t nib = w.format ? w.codes[z] : ((w.codes[z / 2] >> ((col & 1) * 4)) & 15);
  float v;
  if (w.format) {
    v = __fmul_rn(e4m3(nib), e8m0(w.scales[z / w.G]));
  } else {
    const float outer = __fdiv_rn(bf(__ldg(w.amax + e)), bf(__ldg(w.den)));
    v = __fmul_rn(__fmul_rn(e2m1(nib), e4m3(w.scales[z / w.G])), outer);
  }
  return __float2bfloat16_rn(v);
}

// One uint4 is eight BF16 weights in the fragment layout of experts_bf16.cu.
__device__ __forceinline__ uint4 ld_w(const Weight& w, long long e, int cb, int g, int lane, int j) {
  const int row = w.row0 + w.row_step * (cb * 32 + 8 * j + (lane >> 2)), col = g * 32 + 4 * (lane & 3);
  uint16_t h[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int c = col + (i & 3) + 16 * (i >> 2);
    h[i] = __bfloat16_as_ushort(decoded(w, e, row, c));
  }
  uint4 o;
  o.x = h[0] | (uint32_t)h[1] << 16;
  o.y = h[2] | (uint32_t)h[3] << 16;
  o.z = h[4] | (uint32_t)h[5] << 16;
  o.w = h[6] | (uint32_t)h[7] << 16;
  return o;
}

template <int RT, int WARPS>
__device__ __forceinline__ void body(const __nv_bfloat16* __restrict__ X, int x_stride, int slots, Weight w,
                                     int KG, int NB, const int* __restrict__ items, const int* __restrict__ item_count,
                                     const int* __restrict__ members, void* __restrict__ out, int N, int skip) {
  const int lane = threadIdx.x & 31, gq = lane >> 2, t = lane & 3;
  const int units = __ldg(item_count) * NB;
  for (int unit = blockIdx.x * WARPS + (threadIdx.x >> 5); unit < units; unit += gridDim.x * WARPS) {
    const int it = unit / NB, cb = unit - it * NB;
    const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
    if (e == skip) continue;
    for (int r0 = 0; r0 < cnt; r0 += 16 * RT) {
      bool v[RT][2];
      int pr[RT][2];
      const __nv_bfloat16* x[RT][2];
#pragma unroll
      for (int r = 0; r < RT; ++r)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int i = r0 + 16 * r + gq + 8 * h;
          v[r][h] = i < cnt;
          pr[r][h] = v[r][h] ? __ldg(members + first + i) : 0;
          x[r][h] = X + (size_t)(slots ? pr[r][h] / slots : pr[r][h]) * x_stride + 4 * t;
        }
      float acc[1][RT][NTW][4];
#pragma unroll
      for (int r = 0; r < RT; ++r)
#pragma unroll
        for (int j = 0; j < NTW; ++j) acc[0][r][j][0] = acc[0][r][j][1] = acc[0][r][j][2] = acc[0][r][j][3] = 0.f;
      const uint2 zero = make_uint2(0u, 0u);
      for (int g = 0; g < KG; ++g) {
        uint4 ww[NTW];
#pragma unroll
        for (int j = 0; j < NTW; ++j) ww[j] = ld_w(w, e, cb, g, lane, j);
#pragma unroll
        for (int r = 0; r < RT; ++r) {
          uint2 xa[2], xb[2];
#pragma unroll
          for (int q = 0; q < 2; ++q) {
            xa[q] = v[r][0] ? __ldg(reinterpret_cast<const uint2*>(x[r][0] + 32 * g + 16 * q)) : zero;
            xb[q] = v[r][1] ? __ldg(reinterpret_cast<const uint2*>(x[r][1] + 32 * g + 16 * q)) : zero;
          }
#pragma unroll
          for (int q = 0; q < 2; ++q)
#pragma unroll
            for (int j = 0; j < NTW; ++j)
              mma(acc[0][r][j], xa[q].x, xb[q].x, xa[q].y, xb[q].y, q ? ww[j].z : ww[j].x, q ? ww[j].w : ww[j].y);
        }
      }
#pragma unroll
      for (int r = 0; r < RT; ++r)
        epilogue<3, 1, RT>(acc, r, out, N, cb * COLS + 2 * t, pr[r][0], pr[r][1], v[r][0], v[r][1], 0.f);
    }
  }
}

__device__ __forceinline__ void act(const __nv_bfloat16* __restrict__ gate, const __nv_bfloat16* __restrict__ up,
                                    __nv_bfloat16* __restrict__ out, int n, float gate_cap, float up_cap) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float raw = __bfloat162float(gate[i]);
  const float g = gate_cap * tanhf(raw / gate_cap);
  const float u = up_cap * tanhf(__bfloat162float(up[i]) / up_cap);
  // σ(raw) is its own FP32 rounding before the two left-to-right products (§6.4).
  const float sigmoid = 1.f / (1.f + expf(-raw));
  const float a = g * sigmoid * u;
  out[i] = __float2bfloat16_rn(a);
}

__device__ __forceinline__ void combine(const int32_t* __restrict__ pick, const float* __restrict__ wts,
                                        const __nv_bfloat16* __restrict__ ey, __nv_bfloat16* __restrict__ out,
                                        int rows, int hidden, int slots, int experts) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * hidden) return;
  const int r = i / hidden, h = i - r * hidden;
  float total = 0.f;
  uint32_t seen = 0;
  for (int n = 0; n < slots; ++n) {
    int best = -1;
    int32_t id = 0;
    for (int s = 0; s < slots; ++s) {
      if (seen & (1u << s)) continue;
      const int32_t expert = pick[r * slots + s];
      if (expert < 0 || expert >= experts) continue;
      if (best < 0 || expert < id) {
        best = s;
        id = expert;
      }
    }
    if (best < 0) break;
    seen |= 1u << best;
    const int pair = r * slots + best;
    const float term = __fmul_rn(__ldg(wts + pair), __bfloat162float(ey[pair * hidden + h]));
    total = __fadd_rn(total, term);
  }
  out[i] = __float2bfloat16_rn(total);
}

// combine, 8 adjacent columns a thread (hidden % 8 == 0, 16-byte aligned rows): one 16-byte load a pair, the
// pairs taken in the same ascending-expert order, each column's terms in the same __fmul_rn/__fadd_rn order.
__device__ __forceinline__ void combine8(const int32_t* __restrict__ pick, const float* __restrict__ wts,
                                         const __nv_bfloat16* __restrict__ ey, __nv_bfloat16* __restrict__ out,
                                         int rows, int hidden, int slots, int experts) {
  const int groups = hidden / 8;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * groups) return;
  const int r = i / groups, h = (i - r * groups) * 8;
  float total[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  uint32_t seen = 0;
  for (int n = 0; n < slots; ++n) {
    int best = -1;
    int32_t id = 0;
    for (int s = 0; s < slots; ++s) {
      if (seen & (1u << s)) continue;
      const int32_t expert = pick[r * slots + s];
      if (expert < 0 || expert >= experts) continue;
      if (best < 0 || expert < id) {
        best = s;
        id = expert;
      }
    }
    if (best < 0) break;
    seen |= 1u << best;
    const int pair = r * slots + best;
    const float w = __ldg(wts + pair);
    const uint4 v = __ldg(reinterpret_cast<const uint4*>(ey + static_cast<size_t>(pair) * hidden + h));
    const uint32_t u[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      total[2 * k] = __fadd_rn(total[2 * k], __fmul_rn(w, __uint_as_float(u[k] << 16)));
      total[2 * k + 1] = __fadd_rn(total[2 * k + 1], __fmul_rn(w, __uint_as_float(u[k] & 0xffff0000u)));
    }
  }
  uint32_t o[4];
#pragma unroll
  for (int k = 0; k < 4; ++k)
    o[k] = static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16_rn(total[2 * k]))) |
           (static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16_rn(total[2 * k + 1]))) << 16);
  *reinterpret_cast<uint4*>(out + static_cast<size_t>(r) * hidden + h) = make_uint4(o[0], o[1], o[2], o[3]);
}

constexpr int kPlanChunk = 1024;  // the multi-block plan: the most pairs a chunk (its picks staged in shared memory)

} // namespace ink2o

// Pair indices are row * slots + slot. Items hold a physical expert ID, a global member offset and a count.
// Single-block 256-thread parallel counting sort: shared-memory counts, exclusive prefix sums, then one
// thread a expert writes its members by scanning pairs in ascending order, so `items`, `members` and
// `item_count` are 100% bit-identical to the serial scan (experts <= 512, arbitrary pairs).
extern "C" __global__ void __launch_bounds__(256) tf_ink2moe_plan(const int32_t* __restrict__ pick,
    int* __restrict__ items, int* __restrict__ members, int* __restrict__ item_count,
    int rows, int slots, int experts, int tile) {
  if (blockIdx.x) return;
  const int tid = threadIdx.x;
  const int pairs = rows * slots;
  __shared__ int cnt[512];
  __shared__ int off[512];
  __shared__ int item_off[512];
  __shared__ int spick[256];
  for (int e = tid; e < experts; e += 256) cnt[e] = 0;
  __syncthreads();
  for (int p = tid; p < pairs; p += 256) {
    const int e = pick[p];
    if (e >= 0 && e < experts) atomicAdd(&cnt[e], 1);
  }
  __syncthreads();
  if (tid == 0) {
    int run = 0, total_items = 0;
    for (int e = 0; e < experts; ++e) {
      off[e] = run;
      run += cnt[e];
      item_off[e] = total_items;
      total_items += (cnt[e] + tile - 1) / tile;
    }
    *item_count = total_items;
  }
  __syncthreads();
  if (pairs <= 256) {
    if (tid < pairs) spick[tid] = pick[tid];
    __syncthreads();
    for (int e = tid; e < experts; e += 256) {
      int k = 0;
      for (int pair = 0; pair < pairs; ++pair)
        if (spick[pair] == e) members[off[e] + k++] = pair;
      const int nt = (cnt[e] + tile - 1) / tile;
      for (int t = 0; t < nt; ++t) {
        items[3 * (item_off[e] + t)] = e;
        items[3 * (item_off[e] + t) + 1] = off[e] + t * tile;
        items[3 * (item_off[e] + t) + 2] = (cnt[e] - t * tile < tile) ? cnt[e] - t * tile : tile;
      }
    }
  } else {
    for (int e = tid; e < experts; e += 256) {
      int k = 0;
      for (int pair = 0; pair < pairs; ++pair)
        if (pick[pair] == e) members[off[e] + k++] = pair;
      const int nt = (cnt[e] + tile - 1) / tile;
      for (int t = 0; t < nt; ++t) {
        items[3 * (item_off[e] + t)] = e;
        items[3 * (item_off[e] + t) + 1] = off[e] + t * tile;
        items[3 * (item_off[e] + t) + 2] = (cnt[e] - t * tile < tile) ? cnt[e] - t * tile : tile;
      }
    }
  }
}

// The multi-block plan (wide calls): the same items, members and item_count as tf_ink2moe_plan, bit for bit.
// Pass 1, a block a chunk of `chunk` pairs: the chunk's per-expert counts into hist[chunk][expert].
extern "C" __global__ void __launch_bounds__(256) tf_ink2moe_plan_count(const int32_t* __restrict__ pick,
    int* __restrict__ hist, int pairs, int experts, int chunk) {
  __shared__ int cnt[512];
  const int tid = threadIdx.x, b = blockIdx.x;
  const int p0 = b * chunk, p1 = min(pairs, p0 + chunk);
  for (int e = tid; e < experts; e += 256) cnt[e] = 0;
  __syncthreads();
  for (int p = p0 + tid; p < p1; p += 256) {
    const int e = pick[p];
    if (e >= 0 && e < experts) atomicAdd(&cnt[e], 1);
  }
  __syncthreads();
  for (int e = tid; e < experts; e += 256) hist[b * experts + e] = cnt[e];
}

// Pass 2, one block: each expert's total, the serial plan's offsets and items; hist[b][e] becomes the first member
// slot of chunk b's pairs of expert e (chunks ascending, so pair order holds across chunks).
extern "C" __global__ void __launch_bounds__(512) tf_ink2moe_plan_scan(int* __restrict__ hist, int* __restrict__ items,
    int* __restrict__ item_count, int blocks, int experts, int tile) {
  __shared__ int cnt[512];
  __shared__ int off[512];
  __shared__ int item_off[512];
  const int e = threadIdx.x;
  if (e < experts) {
    int c = 0;
    for (int b = 0; b < blocks; ++b) c += hist[b * experts + e];
    cnt[e] = c;
  }
  __syncthreads();
  if (e == 0) {
    int run = 0, total_items = 0;
    for (int x = 0; x < experts; ++x) {
      off[x] = run;
      run += cnt[x];
      item_off[x] = total_items;
      total_items += (cnt[x] + tile - 1) / tile;
    }
    *item_count = total_items;
  }
  __syncthreads();
  if (e >= experts) return;
  int run = off[e];
  for (int b = 0; b < blocks; ++b) {
    const int h = hist[b * experts + e];
    hist[b * experts + e] = run;
    run += h;
  }
  const int nt = (cnt[e] + tile - 1) / tile;
  for (int t = 0; t < nt; ++t) {
    items[3 * (item_off[e] + t)] = e;
    items[3 * (item_off[e] + t) + 1] = off[e] + t * tile;
    items[3 * (item_off[e] + t) + 2] = (cnt[e] - t * tile < tile) ? cnt[e] - t * tile : tile;
  }
}

// Pass 3, a block a chunk: each pair at its chunk's slot for its expert plus its rank among the chunk's earlier
// pairs of that expert, which is the serial scan's ascending pair order.
extern "C" __global__ void __launch_bounds__(256) tf_ink2moe_plan_scatter(const int32_t* __restrict__ pick,
    const int* __restrict__ hist, int* __restrict__ members, int pairs, int experts, int chunk) {
  __shared__ int sp[ink2o::kPlanChunk];
  const int tid = threadIdx.x, b = blockIdx.x;
  const int p0 = b * chunk, n = min(pairs - p0, chunk);
  for (int i = tid; i < n; i += 256) sp[i] = pick[p0 + i];
  __syncthreads();
  for (int i = tid; i < n; i += 256) {
    const int e = sp[i];
    if (e < 0 || e >= experts) continue;
    int k = 0;
    for (int j = 0; j < i; ++j) k += sp[j] == e;
    members[hist[b * experts + e] + k] = p0 + i;
  }
}

#define TF_INK2_MOE(NAME, RT)                                                                                         \
  extern "C" __global__ void __launch_bounds__(128) NAME(const __nv_bfloat16* __restrict__ X, int x_stride, int slots, \
      ink2o::Weight w, int KG, int NB, const int* __restrict__ items, const int* __restrict__ item_count,            \
      const int* __restrict__ members, void* __restrict__ out, int N, int skip) {                                    \
    ink2o::body<RT, 4>(X, x_stride, slots, w, KG, NB, items, item_count, members, out, N, skip);                       \
  }
TF_INK2_MOE(tf_ink2moe_up, 1)
TF_INK2_MOE(tf_ink2moe_prompt_up, 4)
extern "C" __global__ void __launch_bounds__(256) tf_ink2moe_act(const __nv_bfloat16* g, const __nv_bfloat16* u, __nv_bfloat16* o, int n, float gc, float uc) {
  ink2o::act(g, u, o, n, gc, uc);
}
extern "C" __global__ void __launch_bounds__(256) tf_ink2moe_combine(const int32_t* p, const float* w, const __nv_bfloat16* e,
    __nv_bfloat16* o, int rows, int hidden, int slots, int experts) {
  ink2o::combine(p, w, e, o, rows, hidden, slots, experts);
}
extern "C" __global__ void __launch_bounds__(256) tf_ink2moe_combine8(const int32_t* p, const float* w, const __nv_bfloat16* e,
    __nv_bfloat16* o, int rows, int hidden, int slots, int experts) {
  ink2o::combine8(p, w, e, o, rows, hidden, slots, experts);
}
