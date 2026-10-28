// ours: fp8_experts.cu for drama3 with SGLang's per-token-group FP8 activation scales (xsc null: bf16 rows).

#include <cuda_bf16.h>
#include <stdint.h>

#include "experts.cuh"
#include "common/qmm_frag.cuh"
#include "common/fp_pairs.cuh"

namespace tf_d3_fp8 {

constexpr int LANE4 = 2;
constexpr int BLOCK4 = 32 * LANE4;
constexpr int PER_SCALE = 4;

using qmm_frag::ld_nc;

using tf_common::fp8pair;

template <int EPI, int M, int RT>
__device__ __forceinline__ void d3_epilogue(const float (&acc)[M][RT][NTW][4], int r, void* out, int N, int col0,
                                            int pr0, int pr1, bool v0, bool v1, float limit) {
  if constexpr (EPI != 4) {
    epilogue<EPI, M, RT>(acc, r, out, N, col0, pr0, pr1, v0, v1, limit);
  } else {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      if (!(h ? v1 : v0)) continue;
      const size_t row = (size_t)(h ? pr1 : pr0) * N;
#pragma unroll
      for (int j = 0; j < NTW; ++j) {
        const int col = col0 + 8 * j;
        float o[2];
#pragma unroll
        for (int c = 0; c < 2; ++c) {
          const float g = bf(acc[0][r][j][2 * h + c]), u = bf(acc[M - 1][r][j][2 * h + c]);
          o[c] = g / (1.f + expf(-g)) * u;
        }
        *reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<__nv_bfloat16*>(out) + row + col) =
            __floats2bfloat162_rn(o[0], o[1]);
      }
    }
  }
}

template <int M>
struct Stage {
  uint4 w[M][LANE4];
  uint2 xa[2], xb[2];
};

template <int M>
__device__ __forceinline__ void load_stage(Stage<M>& st, const uint4* blk, int g, int lane, int t,
                                           const __nv_bfloat16* x0, const __nv_bfloat16* x1, bool v0, bool v1) {
  const uint4* b = blk + (size_t)g * (M * BLOCK4) + lane * LANE4;
#pragma unroll
  for (int m = 0; m < M; ++m)
#pragma unroll
    for (int h = 0; h < LANE4; ++h) st.w[m][h] = ld_nc(b + m * BLOCK4 + h);
  const uint2 zero = make_uint2(0u, 0u);
  const int k0 = g * 32 + 4 * t;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    st.xa[h] = v0 ? __ldg(reinterpret_cast<const uint2*>(x0 + k0 + 16 * h)) : zero;
    st.xb[h] = v1 ? __ldg(reinterpret_cast<const uint2*>(x1 + k0 + 16 * h)) : zero;
  }
}

template <int M>
__device__ __forceinline__ void compute_stage(float (&part)[M][NTW][4], const Stage<M>& st) {
#pragma unroll
  for (int m = 0; m < M; ++m)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const uint32_t a0 = st.xa[h].x, a2 = st.xa[h].y, a1 = st.xb[h].x, a3 = st.xb[h].y;
#pragma unroll
      for (int j = 0; j < NTW; ++j) {
        const uint32_t word = comp(st.w[m][h], j);
        mma(part[m][j], a0, a1, a2, a3, fp8pair(word), fp8pair(word >> 16));
      }
    }
}

template <int M>
__device__ __forceinline__ void k_loop(float (&acc)[M][1][NTW][4], const uint4* blk, const float* sc, int KG,
                                       int lane, int t, const __nv_bfloat16* x0, const __nv_bfloat16* x1, bool v0,
                                       bool v1, int sstride, const float* xs0, const float* xs1) {
  constexpr int D = 2;
  Stage<M> st[D];
  float part[M][NTW][4];
#pragma unroll
  for (int d = 0; d < D; ++d)
    if (d < KG) load_stage<M>(st[d], blk, d, lane, t, x0, x1, v0, v1);
  for (int g0 = 0; g0 < KG; g0 += D) {
#pragma unroll
    for (int d = 0; d < D; ++d) {
      const int g = g0 + d;
      if (g < KG) {
        if (g % PER_SCALE == 0) {
#pragma unroll
          for (int m = 0; m < M; ++m)
#pragma unroll
            for (int j = 0; j < NTW; ++j) part[m][j][0] = part[m][j][1] = part[m][j][2] = part[m][j][3] = 0.f;
        }
        compute_stage<M>(part, st[d]);
        if (g + D < KG) load_stage<M>(st[d], blk, g + D, lane, t, x0, x1, v0, v1);
        if (g % PER_SCALE == PER_SCALE - 1) {
          const int sg = g / PER_SCALE;
#pragma unroll
          for (int m = 0; m < M; ++m) {
            const float s = __ldg(sc + m * sstride + sg);
            const float sa = xs0 ? s * __ldg(xs0 + sg) : s, sb = xs1 ? s * __ldg(xs1 + sg) : s;
#pragma unroll
            for (int j = 0; j < NTW; ++j) {
              acc[m][0][j][0] = fmaf(part[m][j][0], sa, acc[m][0][j][0]);
              acc[m][0][j][1] = fmaf(part[m][j][1], sa, acc[m][0][j][1]);
              acc[m][0][j][2] = fmaf(part[m][j][2], sb, acc[m][0][j][2]);
              acc[m][0][j][3] = fmaf(part[m][j][3], sb, acc[m][0][j][3]);
            }
          }
        }
      }
    }
  }
}

// Decode experts: FP8 block-scaled weights times rows (FP8 values with group scales, or bf16), grouped plan items.
template <int M, int EPI, int WARPS>
__global__ void __launch_bounds__(WARPS * 32)
    fp8_expert_kernel(const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W,
                      const float* __restrict__ scale, int KG, int NB, const int* __restrict__ items,
                      const int* __restrict__ counts, const int* __restrict__ members, void* __restrict__ out, int N,
                      float limit, int skip, const float* __restrict__ xsc) {
  const int lane = threadIdx.x & 31, gq = lane >> 2, t = lane & 3;
  const int units = __ldg(counts) * NB;
  const int SG = KG / PER_SCALE, NR = (NB * COLS) / 128;
  for (int unit = blockIdx.x * WARPS + (threadIdx.x >> 5); unit < units; unit += gridDim.x * WARPS) {
    const int it = unit / NB, cb = unit - it * NB;
    const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
    if (e == skip) continue;
    const uint4* blk = W + ((size_t)e * NB + cb) * (size_t)KG * (M * BLOCK4);
    const float* sc = scale + ((size_t)e * M * NR + (cb * COLS) / 128) * SG;
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
      const float* xs0 = xsc ? xsc + (size_t)x0r * SG : nullptr;
      const float* xs1 = xsc ? xsc + (size_t)x1r * SG : nullptr;
      k_loop<M>(acc, blk, sc, KG, lane, t, x0, x1, v0, v1, NR * SG, xs0, xs1);
      d3_epilogue<EPI, M, 1>(acc, 0, out, N, cb * COLS + 2 * t, pr0, pr1, v0, v1, limit);
    }
  }
}

constexpr int PWARPS = 4, PSTAGES = 4, PMIN = 1;

template <int M, int RT, int CW>
struct PromptCfg {
  static constexpr int RW = PWARPS / CW;
  static_assert(RW * RT * 16 == 64, "a prompt item holds 64 pairs");
};

// Prompt experts: the same arithmetic over 64-pair tiles of a routed plan.
template <int M, int EPI, int RT, int CW>
__global__ void __launch_bounds__(PWARPS * 32, PMIN)
    fp8_expert_prompt_kernel(const __nv_bfloat16* __restrict__ X, int x_stride, int slots,
                             const uint4* __restrict__ W, const float* __restrict__ scale, int KG, int NB,
                             const int* __restrict__ items, const int* __restrict__ counts,
                             const int* __restrict__ members, void* __restrict__ out, int N, float limit, int skip,
                             const float* __restrict__ xsc) {
  constexpr int RW = PromptCfg<M, RT, CW>::RW;
  __shared__ uint4 sw[PSTAGES][CW][M * BLOCK4];
  __shared__ uint4 sx[PSTAGES][64][4];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gq = lane >> 2, t = lane & 3;
  const int cw = warp % CW, rw = warp / CW;
  const int groups = NB / CW;
  const int units = __ldg(counts) * groups;
  const int SG = KG / PER_SCALE, NR = (NB * COLS) / 128;
  for (int unit = blockIdx.x; unit < units; unit += gridDim.x) {
    const int it = unit / groups, cg = unit - it * groups;
    const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
    if (e == skip) continue;
    const int cb = cg * CW + cw;
    const float* sc = scale + ((size_t)e * M * NR + (cb * COLS) / 128) * SG;
    int pr0[RT], pr1[RT];
    bool v0[RT], v1[RT];
#pragma unroll
    for (int r = 0; r < RT; ++r) {
      const int r0 = (rw * RT + r) * 16;
      v0[r] = r0 + gq < cnt;
      v1[r] = r0 + gq + 8 < cnt;
      pr0[r] = v0[r] ? __ldg(members + first + r0 + gq) : 0;
      pr1[r] = v1[r] ? __ldg(members + first + r0 + gq + 8) : 0;
    }
    const __nv_bfloat16* xrow[2];
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      const int p = min((threadIdx.x + c * PWARPS * 32) / 4, cnt - 1);
      const int pr = __ldg(members + first + p);
      xrow[c] = X + (size_t)(slots ? pr / slots : pr) * x_stride + ((threadIdx.x + c * PWARPS * 32) % 4) * 8;
    }
    const uint4* base = W + ((size_t)e * NB + cg * CW) * (size_t)KG * (M * BLOCK4);
    auto stage = [&](int g) {
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        const int i = threadIdx.x + c * PWARPS * 32;
        cp16(&sx[g % PSTAGES][i / 4][i % 4], xrow[c] + g * 32);
      }
      for (int i = threadIdx.x; i < CW * M * BLOCK4; i += PWARPS * 32) {
        const int c = i / (M * BLOCK4), j = i % (M * BLOCK4);
        cp16(&sw[g % PSTAGES][c][j], base + ((size_t)c * KG + g) * (M * BLOCK4) + j);
      }
    };
    __syncthreads();
#pragma unroll
    for (int g = 0; g < PSTAGES - 1; ++g) {
      if (g < KG) stage(g);
      cp_commit();
    }
    float acc[M][RT][NTW][4];
    float part[M][RT][NTW][4];
#pragma unroll
    for (int m = 0; m < M; ++m)
#pragma unroll
      for (int r = 0; r < RT; ++r)
#pragma unroll
        for (int j = 0; j < NTW; ++j) acc[m][r][j][0] = acc[m][r][j][1] = acc[m][r][j][2] = acc[m][r][j][3] = 0.f;
    const bool live = rw * RT * 16 < cnt;
    for (int g = 0; g < KG; ++g) {
      cp_wait<PSTAGES - 2>();
      __syncthreads();
      if (g + PSTAGES - 1 < KG) stage(g + PSTAGES - 1);
      cp_commit();
      if (!live) continue;
      if (g % PER_SCALE == 0) {
#pragma unroll
        for (int m = 0; m < M; ++m)
#pragma unroll
          for (int r = 0; r < RT; ++r)
#pragma unroll
            for (int j = 0; j < NTW; ++j)
              part[m][r][j][0] = part[m][r][j][1] = part[m][r][j][2] = part[m][r][j][3] = 0.f;
      }
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        uint2 xa[RT], xb[RT];
#pragma unroll
        for (int r = 0; r < RT; ++r) {
          const int r0 = (rw * RT + r) * 16 + gq;
          xa[r] = reinterpret_cast<const uint2*>(&sx[g % PSTAGES][r0][0])[4 * h + t];
          xb[r] = reinterpret_cast<const uint2*>(&sx[g % PSTAGES][r0 + 8][0])[4 * h + t];
        }
#pragma unroll
        for (int m = 0; m < M; ++m) {
          const uint4 wv = sw[g % PSTAGES][cw][m * BLOCK4 + lane * LANE4 + h];
#pragma unroll
          for (int j = 0; j < NTW; ++j) {
            const uint32_t word = comp(wv, j);
            const uint32_t b0 = fp8pair(word), b1 = fp8pair(word >> 16);
#pragma unroll
            for (int r = 0; r < RT; ++r) mma(part[m][r][j], xa[r].x, xb[r].x, xa[r].y, xb[r].y, b0, b1);
          }
        }
      }
      if (g % PER_SCALE == PER_SCALE - 1) {
#pragma unroll
        for (int m = 0; m < M; ++m) {
          const float s = __ldg(sc + m * NR * SG + g / PER_SCALE);
#pragma unroll
          for (int r = 0; r < RT; ++r) {
            float sa = s, sb = s;
            if (xsc) {
              const int a0 = slots ? pr0[r] / slots : pr0[r], a1 = slots ? pr1[r] / slots : pr1[r];
              if (v0[r]) sa = s * __ldg(xsc + (size_t)a0 * SG + g / PER_SCALE);
              if (v1[r]) sb = s * __ldg(xsc + (size_t)a1 * SG + g / PER_SCALE);
            }
#pragma unroll
            for (int j = 0; j < NTW; ++j) {
              acc[m][r][j][0] = fmaf(part[m][r][j][0], sa, acc[m][r][j][0]);
              acc[m][r][j][1] = fmaf(part[m][r][j][1], sa, acc[m][r][j][1]);
              acc[m][r][j][2] = fmaf(part[m][r][j][2], sb, acc[m][r][j][2]);
              acc[m][r][j][3] = fmaf(part[m][r][j][3], sb, acc[m][r][j][3]);
            }
          }
        }
      }
    }
    cp_wait<0>();
    if (live) {
#pragma unroll
      for (int r = 0; r < RT; ++r)
        d3_epilogue<EPI, M, RT>(acc, r, out, N, cb * COLS + 2 * t, pr0[r], pr1[r], v0[r], v1[r], limit);
    }
  }
}

// The instances: SwiGLU gate-up (reference epilogue 4), fp32 down; decode and prompt forms.
#define D3_ARGS const __nv_bfloat16* __restrict__, int, int, const uint4* __restrict__, const float* __restrict__, int, \
    int, const int* __restrict__, const int* __restrict__, const int* __restrict__, void* __restrict__, int, float, int, \
    const float* __restrict__
template __global__ void fp8_expert_kernel<2, 4, 4>(D3_ARGS);
template __global__ void fp8_expert_kernel<1, 0, 4>(D3_ARGS);
template __global__ void fp8_expert_prompt_kernel<2, 4, 2, 2>(D3_ARGS);
template __global__ void fp8_expert_prompt_kernel<1, 0, 4, 4>(D3_ARGS);
}
