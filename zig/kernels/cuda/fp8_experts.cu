// Device code of fp8/experts.cu (lines 12-326, host launchers, comments and ATen dropped) and its instances.

#include <cuda_bf16.h>
#include <stdint.h>

#include "experts.cuh"

namespace tf_fp8_experts {

constexpr int LANE4 = 2;
constexpr int BLOCK4 = 32 * LANE4;
constexpr int PER_SCALE = 4;

__device__ __forceinline__ uint4 ld_nc(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p));
  return r;
}

__device__ __forceinline__ uint32_t fp8pair(uint32_t v) {
  const uint32_t t = ((v & 0x7Fu) << 4) | ((v & 0x80u) << 8) | ((v & 0x7F00u) << 12) | ((v & 0x8000u) << 16);
  uint32_t r;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(t), "r"(0x7B807B80u), "r"(0x80008000u));
  return r;
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
                                       bool v1, int sstride) {
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
#pragma unroll
            for (int j = 0; j < NTW; ++j)
#pragma unroll
              for (int q = 0; q < 4; ++q) acc[m][0][j][q] = fmaf(part[m][j][q], s, acc[m][0][j][q]);
          }
        }
      }
    }
  }
}

template <int M, int EPI, int WARPS>
__global__ void __launch_bounds__(WARPS * 32)
    fp8_expert_kernel(const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W,
                      const float* __restrict__ scale, int KG, int NB, const int* __restrict__ items,
                      const int* __restrict__ counts, const int* __restrict__ members, void* __restrict__ out, int N,
                      float limit, int skip) {
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
      k_loop<M>(acc, blk, sc, KG, lane, t, x0, x1, v0, v1, NR * SG);
      epilogue<EPI, M, 1>(acc, 0, out, N, cb * COLS + 2 * t, pr0, pr1, v0, v1, limit);
    }
  }
}

constexpr int PWARPS = 4, PSTAGES = 4, PMIN = 1;

template <int M, int RT, int CW>
struct PromptCfg {
  static constexpr int RW = PWARPS / CW;
  static_assert(RW * RT * 16 == 64, "a prompt item holds 64 pairs");
};

template <int M, int EPI, int RT, int CW>
__global__ void __launch_bounds__(PWARPS * 32, PMIN)
    fp8_expert_prompt_kernel(const __nv_bfloat16* __restrict__ X, int x_stride, int slots,
                             const uint4* __restrict__ W, const float* __restrict__ scale, int KG, int NB,
                             const int* __restrict__ items, const int* __restrict__ counts,
                             const int* __restrict__ members, void* __restrict__ out, int N, float limit, int skip) {
  constexpr int RW = PromptCfg<M, RT, CW>::RW;
  __shared__ uint4 sw[PSTAGES][CW][M][LANE4][32];
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
        const int m = j / BLOCK4, l = (j % BLOCK4) / LANE4, h = j % LANE4;
        cp16(&sw[g % PSTAGES][c][m][h][l], base + ((size_t)c * KG + g) * (M * BLOCK4) + j);
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
          const uint4 wv = sw[g % PSTAGES][cw][m][h][lane];
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
          for (int r = 0; r < RT; ++r)
#pragma unroll
            for (int j = 0; j < NTW; ++j)
#pragma unroll
              for (int q = 0; q < 4; ++q) acc[m][r][j][q] = fmaf(part[m][r][j][q], s, acc[m][r][j][q]);
        }
      }
    }
    cp_wait<0>();
    if (live) {
#pragma unroll
      for (int r = 0; r < RT; ++r)
        epilogue<EPI, M, RT>(acc, r, out, N, cb * COLS + 2 * t, pr0[r], pr1[r], v0[r], v1[r], limit);
    }
  }
}

// The instances fp8_experts_cuda and fp8_experts_prompt_cuda launch: SwiGLU gate-up, fp32 and bf16 down.
template __global__ void fp8_expert_kernel<2, 2, 4>(const __nv_bfloat16* __restrict__, int, int, const uint4* __restrict__, const float* __restrict__, int, int, const int* __restrict__, const int* __restrict__, const int* __restrict__, void* __restrict__, int, float, int);
template __global__ void fp8_expert_kernel<1, 0, 4>(const __nv_bfloat16* __restrict__, int, int, const uint4* __restrict__, const float* __restrict__, int, int, const int* __restrict__, const int* __restrict__, const int* __restrict__, void* __restrict__, int, float, int);
template __global__ void fp8_expert_kernel<1, 3, 4>(const __nv_bfloat16* __restrict__, int, int, const uint4* __restrict__, const float* __restrict__, int, int, const int* __restrict__, const int* __restrict__, const int* __restrict__, void* __restrict__, int, float, int);
template __global__ void fp8_expert_prompt_kernel<2, 2, 2, 2>(const __nv_bfloat16* __restrict__, int, int, const uint4* __restrict__, const float* __restrict__, int, int, const int* __restrict__, const int* __restrict__, const int* __restrict__, void* __restrict__, int, float, int);
template __global__ void fp8_expert_prompt_kernel<1, 0, 4, 4>(const __nv_bfloat16* __restrict__, int, int, const uint4* __restrict__, const float* __restrict__, int, int, const int* __restrict__, const int* __restrict__, const int* __restrict__, void* __restrict__, int, float, int);
template __global__ void fp8_expert_prompt_kernel<1, 3, 4, 4>(const __nv_bfloat16* __restrict__, int, int, const uint4* __restrict__, const float* __restrict__, int, int, const int* __restrict__, const int* __restrict__, const int* __restrict__, void* __restrict__, int, float, int);
}
