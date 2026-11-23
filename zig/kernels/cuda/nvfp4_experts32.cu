// GLM's routed experts at block 32 with csf0 scales: nvfp4_experts.cu's two forms, its sums in the same order.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#include "experts.cuh"
#include "common/cp_async.cuh"
#include "common/fp_pairs.cuh"

namespace tf_nvfp4_experts32 {

constexpr int PAIR4 = 66;  // a matrix's two k32 groups: 32 uint4 of codes each, then each group's offsets, 1 uint4
constexpr int BASE4 = 2;   // a column block opens with each matrix's 32 row bases: quad t's 8 bytes at [t][j][c]

using tf_common::ld_nc;
using tf_common::ld_nc2;
using tf_common::ld_nc32;
using tf_common::fp4pair;
using tf_common::fp4pair_nz5;

using tf_common::e4m3f;

// Two codes as a bf16 pair: E2M1's values, or NVFP4_NZ5's (0b1000 is +5) when NZ.
template <bool NZ>
__device__ __forceinline__ uint32_t e2m1pair(uint32_t w, int s) {
  if constexpr (NZ) return fp4pair_nz5(w, s);
  else return fp4pair(w, s);
}

template <int M>
struct Stage {
  uint4 w[M];
  uint32_t s[M];  // the group's offsets of quad t: nibble 2j + c, each added to its column's base
  uint2 xa[2], xb[2];
};

template <int M>
__device__ __forceinline__ void load_stage(Stage<M>& st, const uint4* blk, int g, int odd, int lane, int t,
                                           const __nv_bfloat16* x0, const __nv_bfloat16* x1, bool v0, bool v1) {
  const uint4* b = blk + (size_t)(g >> 1) * (M * PAIR4);
#pragma unroll
  for (int m = 0; m < M; ++m) {
    st.w[m] = ld_nc(b + m * PAIR4 + odd * 32 + lane);
    st.s[m] = ld_nc32(reinterpret_cast<const uint32_t*>(b + m * PAIR4 + 64 + odd) + t);
  }
  const uint2 zero = make_uint2(0u, 0u);
  const int k0 = g * 32 + 4 * t;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    st.xa[h] = v0 ? __ldg(reinterpret_cast<const uint2*>(x0 + k0 + 16 * h)) : zero;
    st.xb[h] = v1 ? __ldg(reinterpret_cast<const uint2*>(x1 + k0 + 16 * h)) : zero;
  }
}

template <int M, bool NZ>
__device__ __forceinline__ void compute_stage(float (&acc)[M][1][NTW][4], const Stage<M>& st, const uint2* bq, bool hi) {
#pragma unroll
  for (int m = 0; m < M; ++m) {
    float sf[NTW][2];
#pragma unroll
    for (int j = 0; j < NTW; ++j) {
      const uint32_t sw = j < 2 ? bq[m].x : bq[m].y, o = st.s[m] >> (j * 8);
      const int sh = (j & 1) * 16;
      sf[j][0] = e4m3f(((sw >> sh) & 0xFFu) + (o & 15u));
      sf[j][1] = e4m3f(((sw >> (sh + 8)) & 0xFFu) + ((o >> 4) & 15u));
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const uint32_t a0 = st.xa[h].x, a2 = st.xa[h].y, a1 = st.xb[h].x, a3 = st.xb[h].y;
#pragma unroll
      for (int j = 0; j < NTW; ++j) {
        const uint32_t word = comp(st.w[m], j);
        float p[4] = {0.f, 0.f, 0.f, 0.f};
        mma(p, a0, a1, a2, a3, e2m1pair<NZ>(word, 8 * h), e2m1pair<NZ>(word, 8 * h + 4));
        float(&a)[4] = acc[m][0][j];
        a[0] = fmaf(p[0], sf[j][0], a[0]);
        a[1] = fmaf(p[1], sf[j][1], a[1]);
        if (hi) {
          a[2] = fmaf(p[2], sf[j][0], a[2]);
          a[3] = fmaf(p[3], sf[j][1], a[3]);
        }
      }
    }
  }
}

template <int M, bool NZ>
__device__ __forceinline__ void k_loop(float (&acc)[M][1][NTW][4], const uint4* blk, const uint2* bq, int KG, int lane,
                                       int t, const __nv_bfloat16* x0, const __nv_bfloat16* x1, bool v0, bool v1) {
  constexpr int D = 2;
  Stage<M> st[D];
#pragma unroll
  for (int d = 0; d < D; ++d)
    if (d < KG) load_stage<M>(st[d], blk, d, d, lane, t, x0, x1, v0, v1);
  for (int g0 = 0; g0 < KG; g0 += D) {
#pragma unroll
    for (int d = 0; d < D; ++d) {
      const int g = g0 + d;
      if (g < KG) {
        compute_stage<M, NZ>(acc, st[d], bq, v1);
        if (g + D < KG) load_stage<M>(st[d], blk, g + D, d, lane, t, x0, x1, v0, v1);
      }
    }
  }
}

// One unit of the decode form: a warp's 32 columns of item `unit / NB` over all K, then the epilogue.
template <int M, int EPI, bool NZ>
__device__ __forceinline__ void unit_full(int unit, int lane, int gq, int t, const __nv_bfloat16* __restrict__ X,
                                          int x_stride, int slots, const uint4* __restrict__ W,
                                          const float* __restrict__ scale, int KG, int NB, const int* __restrict__ items,
                                          const int* __restrict__ members, void* __restrict__ out, int N, float limit,
                                          int skip) {
  const int it = unit / NB, cb = unit - it * NB;
  const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
  if (e == skip) return;
  const uint4* cbp = W + ((size_t)e * NB + cb) * (size_t)(M * BASE4 + (KG / 2) * M * PAIR4);
  const uint4* blk = cbp + M * BASE4;
  uint2 bq[M];
#pragma unroll
  for (int m = 0; m < M; ++m) bq[m] = ld_nc2(reinterpret_cast<const uint2*>(cbp + m * BASE4) + t);
  for (int r0 = 0; r0 < cnt; r0 += 16) {
    const bool v0 = r0 + gq < cnt, v1 = r0 + gq + 8 < cnt;
    const int pr0 = v0 ? __ldg(members + first + r0 + gq) : 0;
    const int pr1 = v1 ? __ldg(members + first + r0 + gq + 8) : 0;
    const int x0r = slots ? pr0 / slots : pr0, x1r = slots ? pr1 / slots : pr1;
    const __nv_bfloat16* x0 = X + (size_t)x0r * x_stride;
    const __nv_bfloat16* x1 = X + (size_t)x1r * x_stride;
    float acc[M][1][NTW][4];
#pragma unroll
    for (int m = 0; m < M; ++m)
#pragma unroll
      for (int j = 0; j < NTW; ++j) acc[m][0][j][0] = acc[m][0][j][1] = acc[m][0][j][2] = acc[m][0][j][3] = 0.f;
    k_loop<M, NZ>(acc, blk, bq, KG, lane, t, x0, x1, v0, v1);
#pragma unroll
    for (int m = 0; m < M; ++m) {
      const float g = __ldg(scale + e * M + m);
#pragma unroll
      for (int j = 0; j < NTW; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[m][0][j][q] *= g;
    }
    epilogue<EPI, M, 1>(acc, 0, out, N, cb * COLS + 2 * t, pr0, pr1, v0, v1, limit);
  }
}

template <int M, int EPI, int WARPS, bool NZ>
__device__ __forceinline__ void decode_body(const __nv_bfloat16* __restrict__ X, int x_stride, int slots,
                                            const uint4* __restrict__ W, const float* __restrict__ scale, int KG, int NB,
                                            const int* __restrict__ items, const int* __restrict__ counts,
                                            const int* __restrict__ members, void* __restrict__ out, int N, float limit,
                                            int skip) {
  const int lane = threadIdx.x & 31, gq = lane >> 2, t = lane & 3;
  const int units = __ldg(counts) * NB;
  for (int unit = blockIdx.x * WARPS + (threadIdx.x >> 5); unit < units; unit += gridDim.x * WARPS)
    unit_full<M, EPI, NZ>(unit, lane, gq, t, X, x_stride, slots, W, scale, KG, NB, items, members, out, N, limit, skip);
}

// The prompt form's tile: 16 RT rows a warp, WM x WN warps (BM pairs of one item by WN 32-column blocks a CTA).
template <int M, int RT, int WM, int WN>
struct Prompt4 {
  static constexpr int THREADS = WM * WN * 32, BM = 16 * RT * WM, XC = 8, SG = 2, WB = M * PAIR4;
  static constexpr int XU = BM * XC, WU = WN * WB, SU = XU + WU;  // uint4 a stage: 64 inputs of X, then a pair's weights
  static constexpr int XPT = (XU + THREADS - 1) / THREADS;
  static constexpr int STAGES = SU * 16 * 4 <= 49152 ? 4 : SU * 16 * 3 <= 49152 ? 3 : 2;
  // row r's 16-byte chunk c: rows r .. r + 3 put a 16-input block's lanes on distinct banks
  static __device__ __forceinline__ int chunk(int r, int c) { return r * XC + (c ^ ((r & 3) << 1)); }
};

// An e4m3 scale in both bf16 halves (bytes 0 and 2 of x, the others zero), exact.
__device__ __forceinline__ uint32_t e4m3x2(uint32_t x) {
  const uint32_t t = ((x & 0x007F007Fu) << 4) | ((x & 0x00800080u) << 8);
  uint32_t r;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(t), "r"(0x7B807B80u), "r"(0x80008000u));
  return r;
}

// a * b in bf16 pairs; exact for e2m1 codes and NZ5's 5 times e4m3 scales (seven significant bits at most).
__device__ __forceinline__ uint32_t mul2(uint32_t a, uint32_t b) {
  uint32_t r;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(a), "r"(b), "r"(0x80008000u));
  return r;
}

// Prompt rows, 64 inputs a stage, NT row tiles a warp.
template <int M, int EPI, int RT, int WM, int WN, int NT, bool NZ>
__device__ __forceinline__ void prompt_tiles(const __nv_bfloat16* __restrict__ X, int x_stride, int slots,
                                             const uint4* __restrict__ W, const float* __restrict__ scale, int KG,
                                             int NB, const int* __restrict__ members, void* __restrict__ out, int N,
                                             float limit, int cbt, int e, int first, int cnt) {
  using P = Prompt4<M, RT, WM, WN>;
  constexpr int STAGES = P::STAGES;
  extern __shared__ uint4 sm[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wm = warp / WN, wn = warp - wm * WN;
  const int t = lane & 3, gq = lane >> 2, cb = cbt * WN + wn, cbs = min(WN, NB - cbt * WN);
  const __nv_bfloat16* xsrc[P::XPT];
  int xdst[P::XPT];
#pragma unroll
  for (int i = 0; i < P::XPT; ++i) {
    const int q = tid + i * P::THREADS, r = q / P::XC, c = q - r * P::XC;
    xdst[i] = P::chunk(r, c);
    xsrc[i] = nullptr;
    if (q < P::XU && r < cnt) {
      const int p = __ldg(members + first + r);
      xsrc[i] = X + (size_t)(slots ? p / slots : p) * x_stride + 8 * c;
    }
  }
  const size_t CB = M * BASE4 + (size_t)(KG / 2) * P::WB;  // uint4 of a column block
  const uint4* wsrc = W + ((size_t)e * NB + cbt * WN) * CB;
  auto stage = [&](int s, int g0) {  // k32 groups g0 .. g0 + ng - 1 (ng < SG only at an odd end)
    const int ng = min(P::SG, KG - g0);
    uint4* xs = sm + s * P::SU;
#pragma unroll
    for (int i = 0; i < P::XPT; ++i)
      if (xsrc[i] && ((tid + i * P::THREADS) % P::XC) < ng * 4) cp16(xs + xdst[i], xsrc[i] + (size_t)g0 * 32);
    uint4* ws = xs + P::XU;
    for (int q = tid; q < cbs * P::WB; q += P::THREADS) {
      const int j = q / P::WB, o = q - j * P::WB;
      cp16(ws + q, wsrc + (size_t)j * CB + M * BASE4 + (size_t)(g0 / 2) * P::WB + o);
    }
  };
  constexpr int TA = NT > 0 ? NT : 1;  // array extent
  const int base = 16 * RT * wm;
  const bool active = NT > 0 && cb < NB;
  uint2 bq[M];  // this warp's column block's row bases of quad gq >> 1
#pragma unroll
  for (int m = 0; m < M; ++m)
    bq[m] = active ? ld_nc2(reinterpret_cast<const uint2*>(wsrc + (size_t)wn * CB + m * BASE4) + (gq >> 1))
                   : make_uint2(0u, 0u);
  float acc[M][TA][NTW][4];
#pragma unroll
  for (int m = 0; m < M; ++m)
#pragma unroll
    for (int r = 0; r < NT; ++r)
#pragma unroll
      for (int j = 0; j < NTW; ++j) acc[m][r][j][0] = acc[m][r][j][1] = acc[m][r][j][2] = acc[m][r][j][3] = 0.f;
  const int steps = (KG + P::SG - 1) / P::SG;
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < steps) stage(s, s * P::SG);
    cp_commit();
  }
  for (int k = 0; k < steps; ++k) {
    cp_wait<STAGES - 2>();
    __syncthreads();
    if (k + STAGES - 1 < steps) stage((k + STAGES - 1) % STAGES, (k + STAGES - 1) * P::SG);
    cp_commit();
    if (!active) continue;
    const uint4* xs = sm + (k % STAGES) * P::SU;
#pragma unroll
    for (int gi = 0; gi < P::SG; ++gi) {
      if (k * P::SG + gi >= KG) break;
      const uint4* ws = xs + P::XU + wn * P::WB;
      uint4 wv[M];
      uint32_t sv[M][NTW];  // column j * 8 + gq's scale in both bf16 halves
#pragma unroll
      for (int m = 0; m < M; ++m) {
        wv[m] = ws[m * PAIR4 + gi * 32 + lane];
        // column j * 8 + gq's scale: base byte 2j + (gq & 1), offset nibble 2j + (gq & 1) of a 4-byte run
        const uint32_t o = reinterpret_cast<const uint32_t*>(ws + m * PAIR4 + 64 + gi)[gq >> 1];
#pragma unroll
        for (int j = 0; j < NTW; ++j) {
          const int c = gq & 1;
          const uint32_t v = (((j < 2 ? bq[m].x : bq[m].y) >> (8 * (2 * (j & 1) + c))) & 0xFFu) + ((o >> (8 * j + 4 * c)) & 15u);
          sv[m][j] = e4m3x2(v | (v << 16));
        }
      }
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        uint2 xa[TA], xb[TA];
        const int u = gi * 8 + h * 4 + t;  // the 8-byte unit holding lane t's inputs 16h + 4t .. + 3 of group gi
#pragma unroll
        for (int r = 0; r < NT; ++r) {
          const int r0 = base + 16 * r + gq;
          xa[r] = reinterpret_cast<const uint2*>(xs + P::chunk(r0, u >> 1))[u & 1];
          xb[r] = reinterpret_cast<const uint2*>(xs + P::chunk(r0 + 8, u >> 1))[u & 1];
        }
#pragma unroll
        for (int m = 0; m < M; ++m)
#pragma unroll
          for (int j = 0; j < NTW; ++j) {
            const uint32_t word = comp(wv[m], j);
            const uint32_t b0 = mul2(e2m1pair<NZ>(word, 8 * h), sv[m][j]), b1 = mul2(e2m1pair<NZ>(word, 8 * h + 4), sv[m][j]);
#pragma unroll
            for (int r = 0; r < NT; ++r) mma(acc[m][r][j], xa[r].x, xb[r].x, xa[r].y, xb[r].y, b0, b1);
          }
      }
    }
  }
  if (!active) return;
  if constexpr (NT > 0) {
#pragma unroll
    for (int m = 0; m < M; ++m) {
      const float g = __ldg(scale + e * M + m);
#pragma unroll
      for (int r = 0; r < NT; ++r)
#pragma unroll
        for (int j = 0; j < NTW; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) acc[m][r][j][q] *= g;
    }
#pragma unroll
    for (int r = 0; r < NT; ++r) {
      const int m0 = base + 16 * r + gq, m1 = m0 + 8;
      const bool v0 = m0 < cnt, v1 = m1 < cnt;
      const int p0 = v0 ? __ldg(members + first + m0) : 0, p1 = v1 ? __ldg(members + first + m1) : 0;
      epilogue<EPI, M, TA>(acc, r, out, N, cb * COLS + 2 * t, p0, p1, v0, v1, limit);
    }
  }
}

template <int M, int EPI, int RT, int WM, int WN, bool NZ>
__device__ __forceinline__ void prompt_body(const __nv_bfloat16* __restrict__ X, int x_stride, int slots,
                                            const uint4* __restrict__ W, const float* __restrict__ scale, int KG,
                                            int NB, const int* __restrict__ items, const int* __restrict__ counts,
                                            const int* __restrict__ members, void* __restrict__ out, int N,
                                            float limit, int skip) {
  static_assert(RT == 4, "the dispatch below spells out 0 .. 4 row tiles");
  const int nbt = (NB + WN - 1) / WN, it = blockIdx.x / nbt, cbt = blockIdx.x - it * nbt;
  if (it >= __ldg(counts)) return;
  const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
  if (e == skip) return;  // an expert the family computes itself, left alone as the decode form does
  const int nt = max(0, min(RT, (cnt - 16 * RT * ((threadIdx.x >> 5) / WN) + 15) >> 4));
#define TF_TILES(NT) \
  prompt_tiles<M, EPI, RT, WM, WN, NT, NZ>(X, x_stride, slots, W, scale, KG, NB, members, out, N, limit, cbt, e, first, cnt)
  switch (nt) {
    case 4: TF_TILES(4); break;
    case 3: TF_TILES(3); break;
    case 2: TF_TILES(2); break;
    case 1: TF_TILES(1); break;
    default: TF_TILES(0);
  }
#undef TF_TILES
}
}

// The decode form's entries, 4 warps a CTA: SwiGLU gate-up, fp32 and bf16 sums down; NZ reads NVFP4_NZ5's codes.
#define TF_NVFP4_DECODE32(NAME, M, EPI, NZ)                                                                            \
  extern "C" __global__ void __launch_bounds__(4 * 32) NAME(                                                           \
      const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W,                       \
      const float* __restrict__ scale, int KG, int NB, const int* __restrict__ items, const int* __restrict__ counts,  \
      const int* __restrict__ members, void* __restrict__ out, int N, float limit, int skip) {                         \
    tf_nvfp4_experts32::decode_body<M, EPI, 4, NZ>(X, x_stride, slots, W, scale, KG, NB, items, counts, members, out,  \
                                                   N, limit, skip);                                                    \
  }
TF_NVFP4_DECODE32(tf_nvfp4_decode32_swiglu, 2, 2, false)
TF_NVFP4_DECODE32(tf_nvfp4_decode32_down_f32, 1, 0, false)
TF_NVFP4_DECODE32(tf_nvfp4_decode32_down_bf16, 1, 3, false)
TF_NVFP4_DECODE32(tf_nvfp4_nz5_decode32_swiglu, 2, 2, true)
TF_NVFP4_DECODE32(tf_nvfp4_nz5_decode32_down_f32, 1, 0, true)
TF_NVFP4_DECODE32(tf_nvfp4_nz5_decode32_down_bf16, 1, 3, true)

// The prompt form's entries, 4 row tiles a warp in one row of WN warps: SwiGLU gate-up, bf16 sums down.
#define TF_NVFP4_PROMPT32(NAME, M, EPI, WN, NZ)                                                                        \
  extern "C" __global__ void __launch_bounds__(WN * 32) NAME(                                                          \
      const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W,                       \
      const float* __restrict__ scale, int KG, int NB, const int* __restrict__ items, const int* __restrict__ counts,  \
      const int* __restrict__ members, void* __restrict__ out, int N, float limit, int skip) {                         \
    tf_nvfp4_experts32::prompt_body<M, EPI, 4, 1, WN, NZ>(X, x_stride, slots, W, scale, KG, NB, items, counts,         \
                                                          members, out, N, limit, skip);                               \
  }
TF_NVFP4_PROMPT32(tf_nvfp4_prompt32_swiglu, 2, 2, 4, false)
TF_NVFP4_PROMPT32(tf_nvfp4_prompt32_down, 1, 3, 8, false)
TF_NVFP4_PROMPT32(tf_nvfp4_nz5_prompt32_swiglu, 2, 2, 4, true)
TF_NVFP4_PROMPT32(tf_nvfp4_nz5_prompt32_down, 1, 3, 8, true)
