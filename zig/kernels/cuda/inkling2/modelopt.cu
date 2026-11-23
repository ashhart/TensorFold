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
__device__ __forceinline__ uint32_t pair(float lo, float hi) {
  return __bfloat16_as_ushort(__float2bfloat16_rn(lo)) | (uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(hi)) << 16;
}

// An NVFP4 expert's outer factor: the same quotient every element of expert e used to recompute.
__device__ __forceinline__ float outer_of(const Weight& w, long long e) {
  return w.format ? 0.f : __fdiv_rn(bf(__ldg(w.amax + e)), bf(__ldg(w.den)));
}

// A half-fragment's stored bytes: columns col..col+3 and col+16..col+19 of one row, their two scale bytes.
struct Raw {
  uint32_t lo, hi, sc;
};

constexpr int kRowBytes = 144;  // a staged row: 128 code bytes + 16, so the 8 rows a load touches use distinct banks

// A warp's tile in registers: up to 128 code bytes of its 32 rows (8 lanes a row), the lane's row's scales; K % 32 == 0
struct Tile {
  uint4 codes[8];
  uint32_t scales[4];
};

// Up to four scale bytes as one little-endian word, none read past `n`: a tail or an unaligned row's scales.
__device__ __forceinline__ uint32_t scale_word(const uint8_t* p, int n) {
  uint32_t v = 0;
#pragma unroll
  for (int b = 0; b < 4; ++b)
    if (b < n) v |= (uint32_t)__ldg(p + b) << (8 * b);
  return v;
}

__device__ __forceinline__ void fetch(const Weight& w, long long e, int cb, int g0, int steps, int lane, Tile& t) {
  const int bytes = steps * (w.format ? 32 : 16);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int q = lane + 32 * i, lr = q >> 3, chunk = q & 7;
    if (chunk * 16 >= bytes) continue;
    const long long row = w.row0 + w.row_step * (cb * 32 + lr);
    const long long z = (e * w.N + row) * w.K + 32LL * g0;  // first element of the tile
    t.codes[i] = __ldg(reinterpret_cast<const uint4*>(w.codes + (w.format ? z : z / 2) + chunk * 16));
  }
  const long long row = w.row0 + w.row_step * (cb * 32 + lane);
  const long long z = (e * w.N + row) * w.K + 32LL * g0;
  const int n = w.format ? steps : 2 * steps;  // the row's scale bytes in this tile; unpadded, the last ends the data
  const uint8_t* src = w.scales + z / w.G;
  const bool words = (reinterpret_cast<uintptr_t>(src) & 3) == 0;
#pragma unroll
  for (int i = 0; i < 4; ++i)
    if (4 * i < n)
      t.scales[i] = words && 4 * i + 4 <= n ? __ldg(reinterpret_cast<const uint32_t*>(src) + i)
                                            : scale_word(src + 4 * i, n - 4 * i);
}

__device__ __forceinline__ void put(const Weight& w, int steps, int lane, const Tile& t, uint8_t* __restrict__ codes,
                                    uint8_t* __restrict__ scales) {
  const int bytes = steps * (w.format ? 32 : 16);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int q = lane + 32 * i, lr = q >> 3, chunk = q & 7;
    if (chunk * 16 < bytes) *reinterpret_cast<uint4*>(codes + lr * kRowBytes + chunk * 16) = t.codes[i];
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) reinterpret_cast<uint32_t*>(scales + lane * 16)[i] = t.scales[i];
}

// Step gg of a staged tile, fragment rows 8j + lane / 4: the bytes ld_w used to read from global memory.
__device__ __forceinline__ Raw staged(const Weight& w, const uint8_t* codes, const uint8_t* scales, int gg, int lane,
                                      int j) {
  const int lr = 8 * j + (lane >> 2), t = lane & 3;
  const uint8_t* c = codes + lr * kRowBytes;
  Raw r;
  if (w.format) {
    r.lo = *reinterpret_cast<const uint32_t*>(c + gg * 32 + 4 * t);
    r.hi = *reinterpret_cast<const uint32_t*>(c + gg * 32 + 4 * t + 16);
    r.sc = scales[lr * 16 + gg] * 0x101u;  // one MXFP8 scale covers both halves
  } else {
    r.lo = *reinterpret_cast<const uint16_t*>(c + gg * 16 + 2 * t);
    r.hi = *reinterpret_cast<const uint16_t*>(c + gg * 16 + 2 * t + 8);
    r.sc = scales[lr * 16 + 2 * gg] | (uint32_t)scales[lr * 16 + 2 * gg + 1] << 8;
  }
  return r;
}

// Eight BF16 weights in experts_bf16.cu's fragment layout: each the checkpoint's per-element product, rounded alike.
__device__ __forceinline__ uint4 decode(const Weight& w, const Raw& r, float outer) {
  float v[8];
  if (w.format) {
    const float s0 = e8m0(r.sc & 255), s1 = e8m0(r.sc >> 8);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      v[i] = __fmul_rn(e4m3((r.lo >> (8 * i)) & 255), s0);
      v[4 + i] = __fmul_rn(e4m3((r.hi >> (8 * i)) & 255), s1);
    }
  } else {
    const float s0 = e4m3(r.sc & 255), s1 = e4m3(r.sc >> 8);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      v[i] = __fmul_rn(__fmul_rn(e2m1((r.lo >> (4 * i)) & 15), s0), outer);
      v[4 + i] = __fmul_rn(__fmul_rn(e2m1((r.hi >> (4 * i)) & 15), s1), outer);
    }
  }
  uint4 o;
  o.x = pair(v[0], v[1]);
  o.y = pair(v[2], v[3]);
  o.z = pair(v[4], v[5]);
  o.w = pair(v[6], v[7]);
  return o;
}

template <int RT, int WARPS>
__device__ __forceinline__ void body(const __nv_bfloat16* __restrict__ X, int x_stride, int slots, Weight w,
                                     int KG, int NB, const int* __restrict__ items, const int* __restrict__ item_count,
                                     const int* __restrict__ members, void* __restrict__ out, int N, int skip) {
  const int lane = threadIdx.x & 31, gq = lane >> 2, t = lane & 3;
  const int units = __ldg(item_count) * NB;
  __shared__ __align__(16) uint8_t tiles[WARPS][32 * kRowBytes];
  __shared__ __align__(16) uint8_t tile_scales[WARPS][32 * 16];
  uint8_t* codes = tiles[threadIdx.x >> 5];
  uint8_t* scales = tile_scales[threadIdx.x >> 5];
  for (int unit = blockIdx.x * WARPS + (threadIdx.x >> 5); unit < units; unit += gridDim.x * WARPS) {
    const int it = unit / NB, cb = unit - it * NB;
    const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
    if (e == skip) continue;
    const float outer = outer_of(w, e);
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
      const int ts = w.format ? 4 : 8;  // steps a staged tile
      uint2 xn[2][2] = {{zero, zero}, {zero, zero}};  // decode: the next step's inputs
      if constexpr (RT == 1)
#pragma unroll
        for (int q = 0; q < 2; ++q) {
          xn[0][q] = v[0][0] ? __ldg(reinterpret_cast<const uint2*>(x[0][0] + 16 * q)) : zero;
          xn[1][q] = v[0][1] ? __ldg(reinterpret_cast<const uint2*>(x[0][1] + 16 * q)) : zero;
        }
      Tile next;  // decode (RT 1) fetches the next tile while this one sums; prompt tiles load in place
      if (RT == 1) fetch(w, e, cb, 0, KG < ts ? KG : ts, lane, next);
      for (int g0 = 0; g0 < KG; g0 += ts) {
        const int steps = KG - g0 < ts ? KG - g0 : ts;
        __syncwarp();
        if (RT != 1) fetch(w, e, cb, g0, steps, lane, next);
        put(w, steps, lane, next, codes, scales);
        __syncwarp();
        if (RT == 1 && g0 + ts < KG) fetch(w, e, cb, g0 + ts, KG - g0 - ts < ts ? KG - g0 - ts : ts, lane, next);
        for (int gg = 0; gg < steps; ++gg) {
          const int g = g0 + gg;
          uint2 cur[2][2];  // decode: this step's inputs, fetched a step early
          if constexpr (RT == 1) {
#pragma unroll
            for (int q = 0; q < 2; ++q) {
              cur[0][q] = xn[0][q];
              cur[1][q] = xn[1][q];
              const uint2* pa = reinterpret_cast<const uint2*>(x[0][0] + 32 * (g + 1) + 16 * q);
              const uint2* pb = reinterpret_cast<const uint2*>(x[0][1] + 32 * (g + 1) + 16 * q);
              xn[0][q] = g + 1 < KG && v[0][0] ? __ldg(pa) : zero;
              xn[1][q] = g + 1 < KG && v[0][1] ? __ldg(pb) : zero;
            }
          }
          uint4 ww[NTW];
#pragma unroll
          for (int j = 0; j < NTW; ++j) ww[j] = decode(w, staged(w, codes, scales, gg, lane, j), outer);
#pragma unroll
          for (int r = 0; r < RT; ++r) {
            uint2 xa[2], xb[2];
#pragma unroll
            for (int q = 0; q < 2; ++q) {
              if constexpr (RT == 1) {
                xa[q] = cur[0][q];
                xb[q] = cur[1][q];
              } else {
                xa[q] = v[r][0] ? __ldg(reinterpret_cast<const uint2*>(x[r][0] + 32 * g + 16 * q)) : zero;
                xb[q] = v[r][1] ? __ldg(reinterpret_cast<const uint2*>(x[r][1] + 32 * g + 16 * q)) : zero;
              }
            }
#pragma unroll
            for (int q = 0; q < 2; ++q)
#pragma unroll
              for (int j = 0; j < NTW; ++j)
                mma(acc[0][r][j], xa[q].x, xb[q].x, xa[q].y, xb[q].y, q ? ww[j].z : ww[j].x, q ? ww[j].w : ww[j].y);
          }
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

} // namespace ink2o

// Pair indices are row * slots + slot. Items hold a physical expert ID, a global member offset and a count.
extern "C" __global__ void __launch_bounds__(32) tf_ink2moe_plan(const int32_t* __restrict__ pick,
    int* __restrict__ items, int* __restrict__ members, int* __restrict__ item_count,
    int rows, int slots, int experts, int tile) {
  if (blockIdx.x || threadIdx.x) return;
  const int pairs = rows * slots;
  int used = 0, at = 0;
  for (int e = 0; e < experts; ++e) {
    const int first = used;
    for (int pair = 0; pair < pairs; ++pair) {
      if (pick[pair] == e) members[used++] = pair;
    }
    const int count = used - first;
    for (int off = 0; off < count; off += tile) {
      items[3 * at] = e;
      items[3 * at + 1] = first + off;
      items[3 * at + 2] = (count - off < tile) ? count - off : tile;
      ++at;
    }
  }
  *item_count = at;
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
