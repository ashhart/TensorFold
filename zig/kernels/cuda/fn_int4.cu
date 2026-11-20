// GPTQ int4 W4A16 matmuls for AutoGPTQ group-128 checkpoints: the routed experts (gate/up SwiGLU, down) and the head.

// From MiaAI-Lab's dual-Spark recipe at 04f2b53826, patches/0002-kernels.patch, Apache-2.0, Copyright 2026 MiaAI-Lab.

// TensorFold changed three things: the head's items of 16 MT rows and the gate/up prompt form, each marked TF:, and the

// mma, the non-allocating load and the cp.async copies, which now come from common/ with the same instruction text.

// Weights are exact in bf16 MMAs: code q (0..15) becomes the pair (128 + q) - 136 = q - 8 and a group's products run

// in one fp32 MMA chain (k32 blocks in order, each block's two k16 steps in order), then fmaf(group sum, scale, acc).

// So a row's bits depend on its own inputs and weights alone, never on the other rows, the items or the column tiles.

// Packed words uint32 [N/8][K/GS][32 lanes][GS/32]: lane l holds column 8 tile + l/4 and, a k32 block, inputs

// 8 (l%4) .. 8 (l%4) + 7, nibble i = input 2i and nibble i + 4 = input 2i + 1, so the pair at 4p is inputs 2p, 2p + 1.

// The k16 step s takes pairs 2s and 2s + 1 against a uint4 of the live rows (4s, 4s + 1, 4s + 2, 4s + 3) and fp16

// scales [N/8][K/GS][8]; cuda/int4.zig packs all of it from the checkpoint's qweight [K/8, N] int32.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

#include "common/cp_async.cuh"
#include "common/qmm_frag.cuh"

namespace tf_fn_int4 {

constexpr uint32_t LOW = 0x000F000Fu;
constexpr uint32_t K128 = 0x43004300u;   // the bf16 pair (128, 128)
constexpr uint32_t K136 = 0x43084308u;   // (136, 136): 128 + the zero point 8

__device__ __forceinline__ uint32_t pair(uint32_t w, int p) {
  uint32_t v = ((w >> (4 * p)) & LOW) | K128;
  const uint32_t k = K136;
  __nv_bfloat162 r = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&v), *reinterpret_cast<const __nv_bfloat162*>(&k));
  return *reinterpret_cast<uint32_t*>(&r);
}

__device__ __forceinline__ void mma(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                    uint32_t b1) {
  const uint32_t a[4] = {a0, a1, a2, a3};
  qmm_frag::mma(d, a, b0, b1);
}

using tf_common::ld_nc;

__device__ __forceinline__ uint32_t comp(const uint4& v, int c) {
  return c == 0 ? v.x : c == 1 ? v.y : c == 2 ? v.z : v.w;
}

__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// experts.cuh's SwiGLU (the NVFP4 experts' epilogue 2): gate and up rounded to bf16, silu rounded, times up.
__device__ __forceinline__ float swiglu(float g, float u, float limit) {
  float gv = bf(g), uv = bf(u);
  if (limit > 0.f) {
    gv = fminf(gv, limit);
    uv = fminf(fmaxf(uv, -limit), limit);
  }
  return bf(gv / (1.f + expf(-gv))) * uv;
}

// A lane's words of one n8 tile and group: GS/32 uint32, four for groups of 128 inputs.
template <int GS>
struct Words {
  static_assert(GS == 128, "groups of 128 inputs");
  uint32_t v[GS / 32];
};

template <int GS>
__device__ __forceinline__ Words<GS> load_words(const uint32_t* p) {
  const uint4 u = ld_nc(reinterpret_cast<const uint4*>(p));
  Words<GS> w;
  w.v[0] = u.x;
  w.v[1] = u.y;
  w.v[2] = u.z;
  w.v[3] = u.w;
  return w;
}

// One stage: a group's weights and scales of every (matrix, n8 tile) and rows gq, gq + 8 of each 16-row tile.
template <int GS, int NT, int MATS, int MT>
struct Stage {
  Words<GS> w[MATS][NT];
  uint32_t s[MATS][NT];               // half2: the scales of the lane's two columns
  uint4 xa[MT][GS / 32], xb[MT][GS / 32];  // rows gq, gq + 8: inputs 8 t .. 8 t + 7 of each k32 block
};

template <int GS, int NT, int MATS, int MT>
__device__ __forceinline__ void load_stage(Stage<GS, NT, MATS, MT>& st, const uint32_t* const (&w)[MATS],
                                           const __half* const (&s)[MATS], int KG, int g, int lane, int t,
                                           const __nv_bfloat16* const (&x0)[MT], const __nv_bfloat16* const (&x1)[MT],
                                           const bool (&v0)[MT], const bool (&v1)[MT]) {
#pragma unroll
  for (int m = 0; m < MATS; ++m)
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      const size_t tg = (size_t)j * KG + g;   // tile j (relative) and group g
      st.w[m][j] = load_words<GS>(w[m] + (tg * 32 + lane) * (GS / 32));
      st.s[m][j] = __ldg(reinterpret_cast<const unsigned int*>(s[m] + tg * 8 + 2 * t));
    }
  const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
  for (int r = 0; r < MT; ++r)
#pragma unroll
    for (int b = 0; b < GS / 32; ++b) {
      const int k0 = g * GS + b * 32 + 8 * t;
      st.xa[r][b] = v0[r] ? __ldg(reinterpret_cast<const uint4*>(x0[r] + k0)) : zero;
      st.xb[r][b] = v1[r] ? __ldg(reinterpret_cast<const uint4*>(x1[r] + k0)) : zero;
    }
}

template <int GS, int NT, int MATS, int MT>
__device__ __forceinline__ void compute_stage(float (&acc)[MT][MATS][NT][4], const Stage<GS, NT, MATS, MT>& st,
                                              const bool (&hi)[MT]) {
#pragma unroll
  for (int m = 0; m < MATS; ++m)
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      const __half2 h = *reinterpret_cast<const __half2*>(&st.s[m][j]);
      const float s0 = __low2float(h), s1 = __high2float(h);
#pragma unroll
      for (int r = 0; r < MT; ++r) {
        float d[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int b = 0; b < GS / 32; ++b) {
          const uint32_t word = st.w[m][j].v[b];
#pragma unroll
          for (int s = 0; s < 2; ++s)
            mma(d, comp(st.xa[r][b], 2 * s), comp(st.xb[r][b], 2 * s), comp(st.xa[r][b], 2 * s + 1),
                comp(st.xb[r][b], 2 * s + 1), pair(word, 2 * s), pair(word, 2 * s + 1));
        }
        float(&a)[4] = acc[r][m][j];
        a[0] = fmaf(d[0], s0, a[0]);
        a[1] = fmaf(d[1], s1, a[1]);
        if (hi[r]) {
          a[2] = fmaf(d[2], s0, a[2]);
          a[3] = fmaf(d[3], s1, a[3]);
        }
      }
    }
}

// EPI 2: SwiGLU of matrices 0 (gate) and 1 (up), bf16; 0: fp32; 3: bf16.
template <int NT, int MATS, int EPI>
__device__ __forceinline__ void store(const float (&acc)[MATS][NT][4], void* out, int out_stride, int col0, int pr0,
                                      int pr1, bool v0, bool v1, float limit) {
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    if (!(h ? v1 : v0)) continue;
    const size_t row = (size_t)(h ? pr1 : pr0) * out_stride;
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      const int col = col0 + 8 * j;
      const float a0 = acc[0][j][2 * h], a1 = acc[0][j][2 * h + 1];
      if constexpr (EPI == 0) {
        *reinterpret_cast<float2*>(reinterpret_cast<float*>(out) + row + col) = make_float2(a0, a1);
      } else {
        float o0 = a0, o1 = a1;
        if constexpr (EPI == 2) {
          o0 = swiglu(a0, acc[MATS - 1][j][2 * h], limit);
          o1 = swiglu(a1, acc[MATS - 1][j][2 * h + 1], limit);
        }
        *reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<__nv_bfloat16*>(out) + row + col) =
            __floats2bfloat162_rn(o0, o1);
      }
    }
  }
}

// A warp a unit: (item, NT n8 tiles of every matrix); items (expert, first member, count) from the plan, or `items`

// null for the dense rows [16 MT i, 16 MT (i + 1)) of `rows` (the head). `slots`: a member p reads input row p /

// slots (gate/up), 0: row p (down). A pass takes MT tiles of 16 rows and loads a group's weights once for them.
template <int GS, int NT, int MT, int MATS, int EPI, int WARPS>
__global__ void __launch_bounds__(WARPS * 32)
    int4_kernel(const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint32_t* __restrict__ W,
                const __half* __restrict__ S, int K, int N, const int* __restrict__ items,
                const int* __restrict__ counts, const int* __restrict__ members, int rows, void* __restrict__ out,
                int out_stride, float limit, int skip) {
  const int lane = threadIdx.x & 31, gq = lane >> 2, t = lane & 3;
  const int KG = K / GS, NB = N / (8 * NT);
  // TF: dense items of 16 MT rows (the head of rows > 16 at MT 2: its weights read once for 32 rows)
  const int n_items = items ? __ldg(counts) : (rows + 16 * MT - 1) / (16 * MT);
  const int units = n_items * NB;
  const size_t mat_words = (size_t)N * K / 8, mat_scales = (size_t)N * KG;
  for (int unit = blockIdx.x * WARPS + (threadIdx.x >> 5); unit < units; unit += gridDim.x * WARPS) {
    const int it = unit / NB, cb = unit - it * NB;
    int e = 0, first = 16 * MT * it, cnt = min(16 * MT, rows - 16 * MT * it);
    if (items) {
      e = __ldg(items + 3 * it);
      first = __ldg(items + 3 * it + 1);
      cnt = __ldg(items + 3 * it + 2);
      if (e == skip) continue;
    }
    const uint32_t* w[MATS];
    const __half* s[MATS];
#pragma unroll
    for (int m = 0; m < MATS; ++m) {
      const size_t mi = (size_t)e * MATS + m;
      w[m] = W + mi * mat_words + (size_t)cb * NT * KG * 32 * (GS / 32);
      s[m] = S + mi * mat_scales + (size_t)cb * NT * KG * 8;
    }
    for (int r0 = 0; r0 < cnt; r0 += 16 * MT) {
      bool v0[MT], v1[MT];
      int pr0[MT], pr1[MT];
      const __nv_bfloat16* x0[MT];
      const __nv_bfloat16* x1[MT];
#pragma unroll
      for (int r = 0; r < MT; ++r) {
        const int a = r0 + 16 * r + gq;
        v0[r] = a < cnt;
        v1[r] = a + 8 < cnt;
        if (items) {
          pr0[r] = v0[r] ? __ldg(members + first + a) : 0;
          pr1[r] = v1[r] ? __ldg(members + first + a + 8) : 0;
        } else {
          pr0[r] = v0[r] ? first + a : 0;
          pr1[r] = v1[r] ? first + a + 8 : 0;
        }
        const int x0r = slots ? pr0[r] / slots : pr0[r], x1r = slots ? pr1[r] / slots : pr1[r];
        x0[r] = X + (size_t)x0r * x_stride;
        x1[r] = X + (size_t)x1r * x_stride;
      }
      float acc[MT][MATS][NT][4];
#pragma unroll
      for (int r = 0; r < MT; ++r)
#pragma unroll
        for (int m = 0; m < MATS; ++m)
#pragma unroll
          for (int j = 0; j < NT; ++j) acc[r][m][j][0] = acc[r][m][j][1] = acc[r][m][j][2] = acc[r][m][j][3] = 0.f;
      constexpr int D = 2;   // stages in flight
      Stage<GS, NT, MATS, MT> st[D];
#pragma unroll
      for (int d = 0; d < D; ++d)
        if (d < KG) load_stage<GS, NT, MATS, MT>(st[d], w, s, KG, d, lane, t, x0, x1, v0, v1);
      for (int g0 = 0; g0 < KG; g0 += D) {
#pragma unroll
        for (int d = 0; d < D; ++d) {
          const int g = g0 + d;
          if (g < KG) {
            compute_stage<GS, NT, MATS, MT>(acc, st[d], v1);
            if (g + D < KG) load_stage<GS, NT, MATS, MT>(st[d], w, s, KG, g + D, lane, t, x0, x1, v0, v1);
          }
        }
      }
#pragma unroll
      for (int r = 0; r < MT; ++r)
        store<NT, MATS, EPI>(acc[r], out, out_stride, cb * NT * 8 + 2 * t, pr0[r], pr1[r], v0[r], v1[r], limit);
    }
  }
}


// ---- prompt calls: a block an (item of up to 64 pairs, column block), weights and rows staged in shared memory --

// The same per-output arithmetic as int4_kernel, so the same bits: a group's fresh fp32 MMA chain over the k32 blocks

// in order and each block's two k16 steps, the same k permutation (a lane holds inputs 8 t .. 8 t + 7 of a block;

// nibble pair p of its word is inputs 2p, 2p + 1), then fmaf(group sum, scale, acc) in group order.

// What changes is where the operands come from: a block of WR x WC warps copies one group of the item's rows (by the

// plan's members, zero past the item's count) and of its column block's words and scales into shared memory with

// cp.async, STAGES groups in flight; each warp applies every k16 weight step to RT 16-row tiles and every row

// fragment to WN n8 tiles of each matrix. A row's bits never depend on the other rows (an MMA's rows are independent).

using tf_common::cp_commit;
using tf_common::cp_wait;

__device__ __forceinline__ uint4 lds128(const unsigned char* p) {
  return *reinterpret_cast<const uint4*>(p);
}

// The 16-byte chunk c of shared row r: odd rows' chunks move by four, so a quarter warp's rows gq, gq + 1 hit 32 banks.
__device__ __forceinline__ int swz(int r, int c) { return c ^ ((r & 1) << 2); }

template <int GS, int MATS, int WR, int WC, int RT, int WN>
struct PromptShape {
  static constexpr int THREADS = WR * WC * 32;
  static constexpr int BR = WR * RT * 16;              // rows a block (an item's pairs)
  static constexpr int BT = WC * WN;                   // n8 tiles a block of each matrix
  static constexpr int CH = GS * 2 / 16;               // 16-byte chunks of a row's group
  static constexpr int WPL = GS / 32;                  // words a lane of a tile's group
  static constexpr int A_BYTES = BR * GS * 2;
  static constexpr int W_BYTES = MATS * BT * 32 * WPL * 4;
  static constexpr int S_BYTES = MATS * BT * 16;
  static constexpr int STAGE = A_BYTES + W_BYTES + S_BYTES;
};

template <int GS, int MATS, int EPI, int WR, int WC, int RT, int WN, int STAGES>
__global__ void __launch_bounds__(WR * WC * 32)
    int4_prompt_kernel(const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint32_t* __restrict__ W,
                       const __half* __restrict__ S, int K, int N, const int* __restrict__ items,
                       const int* __restrict__ counts, const int* __restrict__ members, void* __restrict__ out,
                       int out_stride, float limit, int skip) {
  using P = PromptShape<GS, MATS, WR, WC, RT, WN>;
  extern __shared__ __align__(128) unsigned char smem[];
  __shared__ int src_row[P::BR];
  __shared__ int out_row[P::BR];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gq = lane >> 2, t = lane & 3;
  const int wr = warp / WC, wc = warp - wr * WC;
  const int KG = K / GS, NB = N / (8 * P::BT);
  const int units = __ldg(counts) * NB;
  const size_t mat_words = (size_t)N * K / 8, mat_scales = (size_t)N * KG;
  for (int unit = blockIdx.x; unit < units; unit += gridDim.x) {
    const int it = unit / NB, cb = unit - it * NB;
    const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
    if (e == skip) continue;
    __syncthreads();   // the previous unit's rows and stages are no longer read
    for (int i = threadIdx.x; i < P::BR; i += P::THREADS) {
      const bool v = i < cnt;
      const int p = v ? __ldg(members + first + i) : 0;
      out_row[i] = p;
      src_row[i] = v ? (slots ? p / slots : p) : -1;
    }
    __syncthreads();
    const uint32_t* wbase = W + (size_t)e * MATS * mat_words;
    const __half* sbase = S + (size_t)e * MATS * mat_scales;
    auto load = [&](int stage, int g) {
      unsigned char* st = smem + stage * P::STAGE;
      for (int c = threadIdx.x; c < P::BR * P::CH; c += P::THREADS) {
        const int r = c / P::CH, ch = c - r * P::CH;
        const int src = src_row[r];
        const __nv_bfloat16* gp = src >= 0 ? X + (size_t)src * x_stride + g * GS + ch * 8 : X;
        tf_common::cp16z(st + r * GS * 2 + swz(r, ch) * 16, gp, src >= 0);
      }
      constexpr int TCH = 32 * P::WPL * 4 / 16;   // 16-byte chunks of a tile's group
      for (int c = threadIdx.x; c < MATS * P::BT * TCH; c += P::THREADS) {
        const int q = c / TCH, w = c - q * TCH;
        const int m = q / P::BT, j = q - m * P::BT;
        const uint32_t* gp = wbase + m * mat_words + ((size_t)(cb * P::BT + j) * KG + g) * 32 * P::WPL + w * 4;
        tf_common::cp16z(st + P::A_BYTES + q * TCH * 16 + w * 16, gp, true);
      }
      for (int q = threadIdx.x; q < MATS * P::BT; q += P::THREADS) {
        const int m = q / P::BT, j = q - m * P::BT;
        const __half* gp = sbase + m * mat_scales + ((size_t)(cb * P::BT + j) * KG + g) * 8;
        tf_common::cp16z(st + P::A_BYTES + P::W_BYTES + q * 16, gp, true);
      }
    };
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
      if (s < KG) load(s, s);
      cp_commit();
    }
    float acc[RT][MATS][WN][4];
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
      for (int m = 0; m < MATS; ++m)
#pragma unroll
        for (int j = 0; j < WN; ++j) acc[r][m][j][0] = acc[r][m][j][1] = acc[r][m][j][2] = acc[r][m][j][3] = 0.f;
    // row tiles of this warp holding pairs (warp-uniform)
    const int row_base = wr * RT * 16;
    int live = (cnt - row_base + 15) / 16;
    live = live < 0 ? 0 : (live > RT ? RT : live);
    for (int g = 0; g < KG; ++g) {
      cp_wait<STAGES - 2>();
      __syncthreads();
      if (g + STAGES - 1 < KG) load((g + STAGES - 1) % STAGES, g + STAGES - 1);
      cp_commit();
      if (live == 0) continue;
      const unsigned char* st = smem + (g % STAGES) * P::STAGE;
      const unsigned char* sw = st + P::A_BYTES;
      const unsigned char* ss = sw + P::W_BYTES;
      Words<GS> wd[MATS][WN];
      float s0[MATS][WN], s1[MATS][WN];
#pragma unroll
      for (int m = 0; m < MATS; ++m)
#pragma unroll
        for (int j = 0; j < WN; ++j) {
          const int q = m * P::BT + wc * WN + j;
          const unsigned char* wp = sw + (q * 32 + lane) * P::WPL * 4;
          const uint4 u = lds128(wp);
          wd[m][j].v[0] = u.x;
          wd[m][j].v[1] = u.y;
          wd[m][j].v[2] = u.z;
          wd[m][j].v[3] = u.w;
          const __half2 h = *reinterpret_cast<const __half2*>(ss + q * 16 + 4 * t);
          s0[m][j] = __low2float(h);
          s1[m][j] = __high2float(h);
        }
      float d[RT][MATS][WN][4];
#pragma unroll
      for (int r = 0; r < RT; ++r)
#pragma unroll
        for (int m = 0; m < MATS; ++m)
#pragma unroll
          for (int j = 0; j < WN; ++j) d[r][m][j][0] = d[r][m][j][1] = d[r][m][j][2] = d[r][m][j][3] = 0.f;
#pragma unroll
      for (int b = 0; b < P::WPL; ++b) {
        uint4 xa[RT], xb[RT];
#pragma unroll
        for (int r = 0; r < RT; ++r) {
          if (r < live) {
            const int r0 = row_base + 16 * r + gq, r1 = r0 + 8;
            xa[r] = lds128(st + r0 * GS * 2 + swz(r0, 4 * b + t) * 16);
            xb[r] = lds128(st + r1 * GS * 2 + swz(r1, 4 * b + t) * 16);
          }
        }
#pragma unroll
        for (int s = 0; s < 2; ++s)
#pragma unroll
          for (int m = 0; m < MATS; ++m)
#pragma unroll
            for (int j = 0; j < WN; ++j) {
              const uint32_t b0 = pair(wd[m][j].v[b], 2 * s), b1 = pair(wd[m][j].v[b], 2 * s + 1);
#pragma unroll
              for (int r = 0; r < RT; ++r)
                if (r < live)
                  mma(d[r][m][j], comp(xa[r], 2 * s), comp(xb[r], 2 * s), comp(xa[r], 2 * s + 1),
                      comp(xb[r], 2 * s + 1), b0, b1);
            }
      }
#pragma unroll
      for (int r = 0; r < RT; ++r)
#pragma unroll
        for (int m = 0; m < MATS; ++m)
#pragma unroll
          for (int j = 0; j < WN; ++j) {
            float(&a)[4] = acc[r][m][j];
            a[0] = fmaf(d[r][m][j][0], s0[m][j], a[0]);
            a[1] = fmaf(d[r][m][j][1], s1[m][j], a[1]);
            a[2] = fmaf(d[r][m][j][2], s0[m][j], a[2]);
            a[3] = fmaf(d[r][m][j][3], s1[m][j], a[3]);
          }
    }
    cp_wait<0>();
#pragma unroll
    for (int r = 0; r < RT; ++r) {
      if (r >= live) continue;
      const int a0 = row_base + 16 * r + gq, a1 = a0 + 8;
      store<WN, MATS, EPI>(acc[r], out, out_stride, (cb * P::BT + wc * WN) * 8 + 2 * t, out_row[a0], out_row[a1],
                           a0 < cnt, a1 < cnt, limit);
    }
  }
}

}  // namespace tf_fn_int4

// gate/up SwiGLU (2 matrices), down fp32 / bf16 and the head (bf16), groups of 128, 1, 2 or 4 n8 tiles a warp.

// A pass takes 1 or 2 16-row tiles, and the bits are the same either way.
#define TF_INT4(GS, NT, MT, MATS, EPI) template __global__ void tf_fn_int4::int4_kernel<GS, NT, MT, MATS, EPI, 4>( \
    const __nv_bfloat16*, int, int, const uint32_t*, const __half*, int, int, const int*, const int*, const int*, int, \
    void*, int, float, int);
#define TF_INT4_NT(GS, MATS, EPI) TF_INT4(GS, 1, 1, MATS, EPI) TF_INT4(GS, 2, 1, MATS, EPI) TF_INT4(GS, 4, 1, MATS, EPI) \
    TF_INT4(GS, 1, 2, MATS, EPI) TF_INT4(GS, 2, 2, MATS, EPI) TF_INT4(GS, 4, 2, MATS, EPI)
TF_INT4_NT(128, 2, 2)
TF_INT4_NT(128, 1, 0)
TF_INT4_NT(128, 1, 3)

// Prompt calls of the routed down (int4_prompt_kernel, 8 warps: 2 row x 4 column, 2 row tiles, 4 n8 tiles a warp).

// Items of up to 64 pairs, fp32 / bf16 out, groups of 128, 2 stages.
#define TF_INT4_PROMPT(GS, EPI, ST) template __global__ void tf_fn_int4::int4_prompt_kernel<GS, 1, EPI, 2, 4, 2, 4, ST>( \
    const __nv_bfloat16*, int, int, const uint32_t*, const __half*, int, int, const int*, const int*, const int*, void*, \
    int, float, int);
TF_INT4_PROMPT(128, 0, 2)
TF_INT4_PROMPT(128, 3, 2)
// TF: prompt calls of the routed gate/up on the same kernel (2 matrices, SwiGLU, groups of 128), items of up to 64

// pairs: an expert's gate/up words read once an item, not once a 16-pair item (int4_kernel). 2 x 4 warps and 2 x 4

// tiles a warp (64 rows x 128 columns a block), 3 stages (cuda/int4.zig prompt_gu), the same per-output arithmetic.
#define TF_INT4_PROMPT_GU(WR, WC, RT, WN, ST) template __global__ void tf_fn_int4::int4_prompt_kernel<128, 2, 2, WR, WC, \
    RT, WN, ST>(const __nv_bfloat16*, int, int, const uint32_t*, const __half*, int, int, const int*, const int*, \
    const int*, void*, int, float, int);
TF_INT4_PROMPT_GU(2, 4, 2, 4, 3)
