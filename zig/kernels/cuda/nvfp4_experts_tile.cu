// K32 expert staging adapted from MiaAI-Lab patch 0014; MIT notice in THIRD_PARTY_NOTICES.md.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#include "experts.cuh"

namespace tf_nvfp4_tile {

constexpr int BLOCK4 = 36;
constexpr int XBYTES_PER_TILE = 2 * 32 * 16;

__device__ __forceinline__ void cp4(void* dst, const void* src) {
  const uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(d), "l"(src));
}

__device__ __forceinline__ uint32_t fp4pair(uint32_t w, int s) {
  const uint32_t v = w >> s;
  const uint32_t t = ((v & 0x00070007u) << 6) | ((v & 0x00080008u) << 12);
  uint32_t r;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(t), "r"(0x7E807E80u), "r"(0x80008000u));
  return r;
}

__device__ __forceinline__ float e4m3f(uint32_t b) {
  return __half2float(__half(__nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(b), __NV_E4M3)));
}

__device__ __forceinline__ void mma0(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                     uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%10, %10, %10, %10};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(0.0f));
}

template <int M>
struct WStage {
  uint4 w[M];
  uint4 s[M];
};

template <int M, int RT>
__device__ __forceinline__ void scompute(float (&acc)[M][RT][NTW][4], const WStage<M>& st, const char* xs, int lane) {
  uint4 xf[RT][2];
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int h = 0; h < 2; ++h) xf[r][h] = *reinterpret_cast<const uint4*>(xs + (r * XBYTES_PER_TILE) + (h * 32 + lane) * 16);
  constexpr int NF = M * 2 * NTW;
  float pp[2][RT][4];
  float sc[2][2];
#pragma unroll
  for (int f = 0; f <= NF; ++f) {
    if (f < NF) {
      const int m = f / (2 * NTW), h = (f / NTW) & 1, j = f % NTW;
      const uint32_t word = comp(st.w[m], j);
      const uint32_t b0 = fp4pair(word, 8 * h), b1 = fp4pair(word, 8 * h + 4);
      const uint32_t sw = comp(st.s[m], 2 * h + (j >> 1));
      const int sh = (j & 1) * 16;
      sc[f & 1][0] = e4m3f((sw >> sh) & 0xFFu);
      sc[f & 1][1] = e4m3f((sw >> (sh + 8)) & 0xFFu);
#pragma unroll
      for (int r = 0; r < RT; ++r) mma0(pp[f & 1][r], xf[r][h].x, xf[r][h].y, xf[r][h].z, xf[r][h].w, b0, b1);
    }
    if (f > 0) {
      const int g = f - 1, m = g / (2 * NTW), j = g % NTW;
#pragma unroll
      for (int r = 0; r < RT; ++r) {
        float(&a)[4] = acc[m][r][j];
        a[0] = fmaf(pp[g & 1][r][0], sc[g & 1][0], a[0]);
        a[1] = fmaf(pp[g & 1][r][1], sc[g & 1][1], a[1]);
        a[2] = fmaf(pp[g & 1][r][2], sc[g & 1][0], a[2]);
        a[3] = fmaf(pp[g & 1][r][3], sc[g & 1][1], a[3]);
      }
    }
  }
}

template <int M, int EPI, int WARPS, int RT, int S>
__device__ __forceinline__ void qchunk(char* smem, const __nv_bfloat16* __restrict__ X, int x_stride, int slots,
                                       const uint4* blk, bool active, const float* __restrict__ scale, int e, int KG,
                                       const int* __restrict__ members, int first, int cnt, int r0, void* out, int N,
                                       int cb, float limit, int tid, int lane, int gq, int t, int wp) {
  constexpr int NT = WARPS * 32, CP = (RT * 256 + NT - 1) / NT;
  constexpr int WSLAB = M * BLOCK4 * 16, XBYTES = 4 * XBYTES_PER_TILE, STRIDE = XBYTES + WARPS * WSLAB;
  const __nv_bfloat16* csrc[CP];
  int cdst[CP];
  bool cok[CP];
#pragma unroll
  for (int i = 0; i < CP; ++i) {
    const int idx = tid + i * NT, rr = idx >> 4, w = idx & 15;
    cok[i] = idx < RT * 256 && r0 + rr < cnt;
    const int pr = cok[i] ? __ldg(members + first + r0 + rr) : 0;
    csrc[i] = X + (size_t)(slots ? pr / slots : pr) * x_stride + 2 * w;
    const int lane2 = (rr & 7) * 4 + ((w & 7) >> 1);
    cdst[i] = (rr >> 4) * XBYTES_PER_TILE + ((w >> 3) * 32 + lane2) * 16 + ((w & 1) * 2 + ((rr >> 3) & 1)) * 4;
  }
  auto stage = [&](int g, int slot) {
    char* base = smem + slot * STRIDE;
#pragma unroll
    for (int i = 0; i < CP; ++i)
      if (cok[i]) cp4(base + cdst[i], csrc[i] + g * 32);
    if (active) {
      const uint4* src = blk + (size_t)g * (M * BLOCK4);
      char* dst = base + XBYTES + wp * WSLAB;
#pragma unroll
      for (int m = 0; m < M; ++m) {
        cp16(dst + (m * BLOCK4 + lane) * 16, src + m * BLOCK4 + lane);
        if (lane < 4) cp16(dst + (m * BLOCK4 + 32 + lane) * 16, src + m * BLOCK4 + 32 + lane);
      }
    }
  };
  bool v0[RT], v1[RT];
  int pr0[RT], pr1[RT];
#pragma unroll
  for (int r = 0; r < RT; ++r) {
    const int b = r0 + 16 * r;
    v0[r] = b + gq < cnt;
    v1[r] = b + gq + 8 < cnt;
    pr0[r] = v0[r] ? __ldg(members + first + b + gq) : 0;
    pr1[r] = v1[r] ? __ldg(members + first + b + gq + 8) : 0;
  }
  float acc[M][RT][NTW][4];
#pragma unroll
  for (int m = 0; m < M; ++m)
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
      for (int j = 0; j < NTW; ++j) acc[m][r][j][0] = acc[m][r][j][1] = acc[m][r][j][2] = acc[m][r][j][3] = 0.f;
#pragma unroll
  for (int s = 0; s < S - 1; ++s) {
    if (s < KG) stage(s, s);
    cp_commit();
  }
  for (int g = 0; g < KG; ++g) {
    cp_wait<S - 2>();
    __syncthreads();
    if (g + S - 1 < KG) stage(g + S - 1, (g + S - 1) % S);
    cp_commit();
    if (active) {
      const char* base = smem + (g % S) * STRIDE;
      const char* wsm = base + XBYTES + wp * WSLAB;
      WStage<M> st;
#pragma unroll
      for (int m = 0; m < M; ++m) {
        st.w[m] = *reinterpret_cast<const uint4*>(wsm + (m * BLOCK4 + lane) * 16);
        st.s[m] = *reinterpret_cast<const uint4*>(wsm + (m * BLOCK4 + 32 + t) * 16);
      }
      scompute<M, RT>(acc, st, base, lane);
    }
  }
  cp_wait<0>();
  __syncthreads();
  if (!active) return;
#pragma unroll
  for (int m = 0; m < M; ++m) {
    const float gs = __ldg(scale + e * M + m);
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
      for (int j = 0; j < NTW; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[m][r][j][q] *= gs;
  }
#pragma unroll
  for (int r = 0; r < RT; ++r) epilogue<EPI, M, RT>(acc, r, out, N, cb * COLS + 2 * t, pr0[r], pr1[r], v0[r], v1[r], limit);
}

template <int M, int EPI, int WARPS, int RTMAX, int S>
__device__ __forceinline__ void tile_body(const __nv_bfloat16* __restrict__ X, int x_stride, int slots,
                                          const uint4* __restrict__ W, const float* __restrict__ scale, int KG, int NB,
                                          const int* __restrict__ items, const int* __restrict__ counts,
                                          const int* __restrict__ members, void* __restrict__ out, int N, float limit,
                                          int skip) {
  extern __shared__ __align__(16) char smem[];
  const int tid = threadIdx.x, lane = tid & 31, gq = lane >> 2, t = lane & 3, wp = tid >> 5;
  const int groups = (NB + WARPS - 1) / WARPS;
  const int units = __ldg(counts) * groups;
  for (int unit = blockIdx.x; unit < units; unit += gridDim.x) {
    const int it = unit / groups, cg = unit - it * groups;
    const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
    if (e == skip) continue;
    const int cb = cg * WARPS + wp;
    const bool active = cb < NB;
    const uint4* blk = W + ((size_t)e * NB + (active ? cb : 0)) * (size_t)KG * (M * BLOCK4);
    for (int r0 = 0; r0 < cnt;) {
      const int rt = (cnt - r0 + 15) >> 4;
      if (RTMAX >= 4 && rt >= 4) {
        if constexpr (RTMAX >= 4)
          qchunk<M, EPI, WARPS, 4, S>(smem, X, x_stride, slots, blk, active, scale, e, KG, members, first, cnt, r0, out, N, cb, limit, tid, lane, gq, t, wp);
        r0 += 64;
      } else if (RTMAX >= 3 && rt == 3) {
        if constexpr (RTMAX >= 3)
          qchunk<M, EPI, WARPS, 3, S>(smem, X, x_stride, slots, blk, active, scale, e, KG, members, first, cnt, r0, out, N, cb, limit, tid, lane, gq, t, wp);
        r0 += 48;
      } else if (RTMAX >= 2 && rt >= 2) {
        if constexpr (RTMAX >= 2)
          qchunk<M, EPI, WARPS, 2, S>(smem, X, x_stride, slots, blk, active, scale, e, KG, members, first, cnt, r0, out, N, cb, limit, tid, lane, gq, t, wp);
        r0 += 32;
      } else {
        qchunk<M, EPI, WARPS, 1, S>(smem, X, x_stride, slots, blk, active, scale, e, KG, members, first, cnt, r0, out, N, cb, limit, tid, lane, gq, t, wp);
        r0 += 16;
      }
    }
  }
}

}

#define TF_TILE4(NAME, M, EPI, RT, S)                                                                              \
  extern "C" __global__ void __launch_bounds__(128)                                                              \
      NAME(const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W,             \
           const float* __restrict__ scale, int KG, int NB, const int* __restrict__ items,                        \
           const int* __restrict__ counts, const int* __restrict__ members, void* __restrict__ out, int N,         \
           float limit, int skip) {                                                                              \
    tf_nvfp4_tile::tile_body<M, EPI, 4, RT, S>(X, x_stride, slots, W, scale, KG, NB, items, counts, members, \
                                                    out, N, limit, skip);                                        \
  }
TF_TILE4(tf_nvfp4_tile_swiglu, 2, 2, 4, 4)
TF_TILE4(tf_nvfp4_tile_down_f32, 1, 0, 2, 3)
