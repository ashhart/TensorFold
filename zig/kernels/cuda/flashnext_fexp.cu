// g32 experts on prompt pair tiles with #952's factored arithmetic: expert_rows' bits at every row (64 pairs a tile).
#include <cuda_bf16.h>
#include <stdint.h>
#include "experts.cuh"

namespace tf_fact_experts {

template <int GS, int M, int RT, int WM, int WN, int ST>
struct Pre {
  static constexpr int THREADS = WM * WN * 32, BM = 16 * RT * WM, SG = 64 / GS, XC = 8, WB = M * Geo<GS>::BLOCK;
  static constexpr int XU = BM * XC, WU = WN * SG * WB, XSU = BM * SG / 4, SU = XU + WU + XSU;
  static constexpr int XPT = (XU + THREADS - 1) / THREADS;
  static constexpr int STAGE_BYTES = SU * 16;
  static constexpr int STAGES = ST;
  static __device__ __forceinline__ int xslot(int r, int c) { return r * XC + (c ^ ((r & 1) << 2)); }
};

// mma from zero (C = 0): the same sums as zeroed accumulators, no register moves.
__device__ __forceinline__ void mma_z(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                      uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%10, %10, %10, %10};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(0.f));
}

// expert_rows' group_sum from the whole row: lane t's 8 inputs in order, the quad's (s0 + s1) + (s2 + s3).
__device__ __forceinline__ float row_gsum(const uint4 (&v)[4]) {
  float s[4];
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    float a = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint32_t u = comp(v[q], i);
      a += lo_f(u);
      a += hi_f(u);
    }
    s[q] = a;
  }
  return (s[0] + s[1]) + (s[2] + s[3]);
}

__device__ __forceinline__ float gsum(const uint4& v) {
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint32_t u = comp(v, i);
    s += lo_f(u);
    s += hi_f(u);
  }
  s += __shfl_xor_sync(0xffffffffu, s, 1);
  s += __shfl_xor_sync(0xffffffffu, s, 2);
  return s;
}

template <int M, int EPI, int RT, int WM, int WN, int ST>
__global__ void __launch_bounds__(WM * WN * 32)
    fact_kernel(const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W, int KG,
                int NB, const int* __restrict__ items, const int* __restrict__ counts,
                const int* __restrict__ members, void* __restrict__ out, int N, float limit) {
  constexpr int GS = 32;
  using G = Geo<GS>;
  using P = Pre<GS, M, RT, WM, WN, ST>;
  constexpr int STAGES = P::STAGES;
  extern __shared__ uint4 sm[];
  const int nbt = (NB + WN - 1) / WN, it = blockIdx.x / nbt, cbt = blockIdx.x - it * nbt;
  if (it >= __ldg(counts)) return;
  const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
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
  const uint4* wsrc = W + ((size_t)e * NB + cbt * WN) * (size_t)KG * P::WB;
  auto stage = [&](int s, int g0) {
    const int ng = min(P::SG, KG - g0);
    uint4* xs = sm + s * P::SU;
#pragma unroll
    for (int i = 0; i < P::XPT; ++i)
      if (xsrc[i] && ((tid + i * P::THREADS) % P::XC) < ng * (GS / 8)) cp16(xs + xdst[i], xsrc[i] + (size_t)g0 * GS);
    uint4* ws = xs + P::XU;
    for (int q = tid; q < cbs * P::SG * P::WB; q += P::THREADS) {
      const int j = q / (P::SG * P::WB), o = q - j * P::SG * P::WB;
      if (o < ng * P::WB) cp16(ws + q, wsrc + ((size_t)j * KG + g0) * P::WB + o);
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
    {                                                   // this stage's row group sums, once a CTA, into its tail
      uint4* xsg = sm + (k % STAGES) * P::SU;
      float* sums = reinterpret_cast<float*>(xsg + P::XU + P::WU);
      for (int q = tid; q < P::BM * P::SG; q += P::THREADS) {
        const int r = q / P::SG, gi = q - r * P::SG;
        if (r < cnt && k * P::SG + gi < KG) {
          uint4 v[4];
#pragma unroll
          for (int c = 0; c < 4; ++c) v[c] = xsg[P::xslot(r, gi * (GS / 8) + c)];
          sums[q] = row_gsum(v);
        }
      }
      __syncthreads();
    }
    if (!active) continue;
    const uint4* xs = sm + (k % STAGES) * P::SU;
    const float* sums = reinterpret_cast<const float*>(xs + P::XU + P::WU);
#pragma unroll
    for (int gi = 0; gi < P::SG; ++gi) {
      if (k * P::SG + gi >= KG) break;
      const uint4* ws = xs + P::XU + (wn * P::SG + gi) * P::WB;
      uint4 xa[RT], xb[RT];
      float xsa[RT], xsb[RT];
#pragma unroll
      for (int r = 0; r < RT; ++r) {
        if (r >= nt) break;
        const int r0 = base + 16 * r + gq;
        xa[r] = xs[P::xslot(r0, gi * (GS / 8) + t)];
        xb[r] = xs[P::xslot(r0 + 8, gi * (GS / 8) + t)];
        xsa[r] = sums[r0 * P::SG + gi];
        xsb[r] = sums[(r0 + 8) * P::SG + gi];
      }
      float p[M][NTW][RT][4];
      uint4 sp4[M], bp4[M];
#pragma unroll
      for (int m = 0; m < M; ++m) {                   // every block's group products first (independent chains)
        const uint4 wv = ws[m * G::BLOCK + lane];
        sp4[m] = ws[m * G::BLOCK + 32 * G::WV + t * G::SBV];
        bp4[m] = ws[m * G::BLOCK + 32 * G::WV + t * G::SBV + 1];
#pragma unroll
        for (int j = 0; j < NTW; ++j) {
          const uint32_t word = comp(wv, j);
          const uint32_t q0 = nib2(word, 0), q1 = nib2(word, 4), q2 = nib2(word, 8), q3 = nib2(word, 12);
#pragma unroll
          for (int r = 0; r < RT; ++r) {
            if (r >= nt) break;
            mma_z(p[m][j][r], comp(xa[r], 0), comp(xb[r], 0), comp(xa[r], 1), comp(xb[r], 1), q0, q1);
            mma(p[m][j][r], comp(xa[r], 2), comp(xb[r], 2), comp(xa[r], 3), comp(xb[r], 3), q2, q3);
          }
        }
      }
#pragma unroll
      for (int m = 0; m < M; ++m)
#pragma unroll
        for (int j = 0; j < NTW; ++j) {
          const uint32_t sp = comp(sp4[m], j), bp = comp(bp4[m], j);
          const float s0 = lo_f(sp), s1 = hi_f(sp), b0 = lo_f(bp), b1 = hi_f(bp);
#pragma unroll
          for (int r = 0; r < RT; ++r) {
            if (r >= nt) break;
            float(&a)[4] = acc[m][r][j];
            a[0] = fmaf(xsa[r], b0, fmaf(p[m][j][r][0], s0, a[0]));
            a[1] = fmaf(xsa[r], b1, fmaf(p[m][j][r][1], s1, a[1]));
            a[2] = fmaf(xsb[r], b0, fmaf(p[m][j][r][2], s0, a[2]));
            a[3] = fmaf(xsb[r], b1, fmaf(p[m][j][r][3], s1, a[3]));
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

}  // namespace tf_fact_experts

// gate and up with SwiGLU (bf16 out) and down to fp32, 64-pair tiles on 8 warps, 3 stages (2 CTAs an SM).
template __global__ void tf_fact_experts::fact_kernel<2, 2, 1, 4, 2, 3>(const __nv_bfloat16* __restrict__, int, int,
    const uint4* __restrict__, int, int, const int* __restrict__, const int* __restrict__, const int* __restrict__,
    void* __restrict__, int, float);
template __global__ void tf_fact_experts::fact_kernel<1, 0, 2, 2, 4, 3>(const __nv_bfloat16* __restrict__, int, int,
    const uint4* __restrict__, int, int, const int* __restrict__, const int* __restrict__, const int* __restrict__,
    void* __restrict__, int, float);
