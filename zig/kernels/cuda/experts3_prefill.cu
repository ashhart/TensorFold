// 3-bit experts, prompt form: experts_prefill.cu's tiles and steps on nibble words rebuilt from bit planes.

#include <cuda_bf16.h>
#include <stdint.h>

#include "experts3.cuh"

namespace tf_experts3_prefill {

template <int M, int RT, int WM, int WN, int BITS>
struct Pre {
  static constexpr int THREADS = WM * WN * 32, BM = 16 * RT * WM, XC = 8, WB = M * Lane<BITS>::BLOCK;
  static constexpr int XU = BM * XC, WU = WN * WB, SU = XU + WU;   // uint4 a stage: X rows, weight blocks
  static constexpr int XPT = (XU + THREADS - 1) / THREADS;
  static constexpr int STAGE_BYTES = SU * 16;
  static constexpr int STAGES = STAGE_BYTES * 4 <= 49152 ? 4 : STAGE_BYTES * 3 <= 49152 ? 3 : 2;
  static __device__ __forceinline__ int xslot(int r, int c) { return r * XC + (c ^ (r & 1)); }
};

__device__ __forceinline__ uint32_t hfma2(uint32_t q, uint32_t s, uint32_t b) {
  __nv_bfloat162 r = __hfma2(*reinterpret_cast<__nv_bfloat162*>(&q), *reinterpret_cast<__nv_bfloat162*>(&s),
                             *reinterpret_cast<__nv_bfloat162*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
}

// One CTA's item and column blocks over W's BITS-wide blocks (W at the item's expert), as experts_prefill.cu runs them.
template <int M, int EPI, int RT, int WM, int WN, int BITS>
__device__ __forceinline__ void body(uint4* sm, const __nv_bfloat16* __restrict__ X, int x_stride, int slots,
                                     const uint4* __restrict__ wexp, int KG, int NB, int cbt, int first, int cnt,
                                     const int* __restrict__ members, void* __restrict__ out, int N, float limit) {
  using P = Pre<M, RT, WM, WN, BITS>;
  using L = Lane<BITS>;
  constexpr int STAGES = P::STAGES;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wm = warp / WN, wn = warp - wm * WN;
  const int t = lane & 3, gq = lane >> 2, cb = cbt * WN + wn, cbs = min(WN, NB - cbt * WN);
  const __nv_bfloat16* xsrc[P::XPT];
  int xdst[P::XPT];
#pragma unroll
  for (int i = 0; i < P::XPT; ++i) {
    const int q = tid + i * P::THREADS, r = q / P::XC, c = q - r * P::XC;
    xdst[i] = P::xslot(r, c);
    xsrc[i] = nullptr;
    if (q < P::XU && r < cnt) {
      const int p = __ldg(members + first + r);
      xsrc[i] = X + (size_t)(slots ? p / slots : p) * x_stride + 8 * c;
    }
  }
  const uint4* wsrc = wexp + (size_t)cbt * WN * KG * P::WB;
  auto stage = [&](int s, int g0) {
    uint4* xs = sm + s * P::SU;
#pragma unroll
    for (int i = 0; i < P::XPT; ++i)
      if (xsrc[i]) cp16(xs + xdst[i], xsrc[i] + (size_t)g0 * 64);
    uint4* ws = xs + P::XU;
    for (int q = tid; q < cbs * P::WB; q += P::THREADS) {
      const int j = q / P::WB, o = q - j * P::WB;
      cp16(ws + q, wsrc + ((size_t)j * KG + g0) * P::WB + o);
    }
  };
  const int base = 16 * RT * wm;
  const int nt = max(0, min(RT, (cnt - base + 15) >> 4));
  const bool active = nt > 0 && cb < NB;
  float acc[M][RT][NTW][4];
#pragma unroll
  for (int m = 0; m < M; ++m)
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
      for (int j = 0; j < NTW; ++j) acc[m][r][j][0] = acc[m][r][j][1] = acc[m][r][j][2] = acc[m][r][j][3] = 0.f;
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < KG) stage(s, s);
    cp_commit();
  }
  const uint32_t pick = (gq & 1) ? 0x3232u : 0x1010u;
  for (int k = 0; k < KG; ++k) {
    cp_wait<STAGES - 2>();
    __syncthreads();
    if (k + STAGES - 1 < KG) stage((k + STAGES - 1) % STAGES, k + STAGES - 1);
    cp_commit();
    if (!active) continue;
    const uint4* xs = sm + (k % STAGES) * P::SU;
    const uint4* ws = xs + P::XU + wn * P::WB;
    uint32_t wv[M][L::WORDS];
    uint32_t s2[M][NTW], b2[M][NTW];
#pragma unroll
    for (int m = 0; m < M; ++m) {
      lane_words<BITS>(wv[m], ws + m * L::BLOCK, lane, [](const uint4* q) { return *q; },
                       [](const uint2* q) { return *q; });
      const uint4 sp = ws[m * L::BLOCK + L::WU + (gq >> 1) * Geo<64>::SBV];
      const uint4 bp = ws[m * L::BLOCK + L::WU + (gq >> 1) * Geo<64>::SBV + 1];
#pragma unroll
      for (int j = 0; j < NTW; ++j) {
        s2[m][j] = __byte_perm(comp(sp, j), 0u, pick);
        b2[m][j] = __byte_perm(comp(bp, j), 0u, pick);
      }
    }
#pragma unroll
    for (int kk = 0; kk < Geo<64>::KS / 2; ++kk) {
      uint4 xa[RT], xb[RT];
#pragma unroll
      for (int r = 0; r < RT; ++r) {
        if (r >= nt) break;
        const int r0 = base + 16 * r + gq;
        xa[r] = xs[P::xslot(r0, t * Geo<64>::XV + kk)];
        xb[r] = xs[P::xslot(r0 + 8, t * Geo<64>::XV + kk)];
      }
#pragma unroll
      for (int m = 0; m < M; ++m)
#pragma unroll
        for (int j = 0; j < NTW; ++j) {
          const uint32_t word = nibword<BITS>(wv[m], j * 2 + kk);
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const uint32_t b0 = hfma2(nib2(word, 8 * h), s2[m][j], b2[m][j]);
            const uint32_t b1 = hfma2(nib2(word, 8 * h + 4), s2[m][j], b2[m][j]);
#pragma unroll
            for (int r = 0; r < RT; ++r) {
              if (r >= nt) break;
              mma(acc[m][r][j], comp(xa[r], 2 * h), comp(xb[r], 2 * h), comp(xa[r], 2 * h + 1),
                  comp(xb[r], 2 * h + 1), b0, b1);
            }
          }
        }
    }
  }
  if (!active) return;
#pragma unroll
  for (int r = 0; r < RT; ++r) {
    if (r >= nt) break;
    const int m0 = base + 16 * r + gq, m1 = m0 + 8;
    const bool v0 = m0 < cnt, v1 = m1 < cnt;
    const int p0 = v0 ? __ldg(members + first + m0) : 0, p1 = v1 ? __ldg(members + first + m1) : 0;
    epilogue<EPI, M, RT>(acc, r, out, N, cb * COLS + 2 * t, p0, p1, v0, v1, limit);
  }
}

// experts_prefill.cu's prefill_kernel on 3-bit blocks in W; expert e4 (the shared one, -1 for none) reads 4-bit W4.
template <int M, int EPI, int RT, int WM, int WN>
__global__ void __launch_bounds__(WM * WN * 32)
    prefill_kernel(const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W,
                   int KG, int NB, const int* __restrict__ items, const int* __restrict__ counts,
                   const int* __restrict__ members, void* __restrict__ out, int N, float limit,
                   const uint4* __restrict__ W4, int e4) {
  extern __shared__ uint4 sm[];
  const int nbt = (NB + WN - 1) / WN, it = blockIdx.x / nbt, cbt = blockIdx.x - it * nbt;
  if (it >= __ldg(counts)) return;
  const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
  if (e == e4)
    body<M, EPI, RT, WM, WN, 4>(sm, X, x_stride, slots, W4, KG, NB, cbt, first, cnt, members, out, N, limit);
  else
    body<M, EPI, RT, WM, WN, 3>(sm, X, x_stride, slots, W + (size_t)e * NB * KG * (M * Lane<3>::BLOCK), KG, NB, cbt,
                                first, cnt, members, out, N, limit);
}

}  // namespace tf_experts3_prefill

// Prompt form for groups of 64: the SwiGLU pair form (M = 2, epilogue 2) and down to bf16 sums (epilogue 3).
template __global__ void tf_experts3_prefill::prefill_kernel<2, 2, 2, 2, 4>(const __nv_bfloat16* __restrict__, int,
    int, const uint4* __restrict__, int, int, const int* __restrict__, const int* __restrict__, const int* __restrict__,
    void* __restrict__, int, float, const uint4* __restrict__, int);
template __global__ void tf_experts3_prefill::prefill_kernel<1, 3, 2, 2, 4>(const __nv_bfloat16* __restrict__, int,
    int, const uint4* __restrict__, int, int, const int* __restrict__, const int* __restrict__, const int* __restrict__,
    void* __restrict__, int, float, const uint4* __restrict__, int);
