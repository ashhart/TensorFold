// Prefill: w = bf16(fma(q, s, b)), one fp32 chain over K; chunk-invariant bits, not decode's (replies prefill again).

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <torch/extension.h>

#include "experts.cuh"

namespace {

template <int GS, int M, int RT, int WM, int WN>
struct Pre {
  static constexpr int THREADS = WM * WN * 32, BM = 16 * RT * WM, SG = 64 / GS, XC = 8, WB = M * Geo<GS>::BLOCK;
  static constexpr int XU = BM * XC, WU = WN * SG * WB, SU = XU + WU;   // uint4 a stage: X rows, weight blocks
  static constexpr int XPT = (XU + THREADS - 1) / THREADS;
  static constexpr int STAGE_BYTES = SU * 16;
  static constexpr int STAGES = STAGE_BYTES * 4 <= 49152 ? 4 : STAGE_BYTES * 3 <= 49152 ? 3 : 2;
  static __device__ __forceinline__ int xslot(int r, int c) {
    return r * XC + (GS == 32 ? c ^ ((r & 1) << 2) : c ^ (r & 1));
  }
};

__device__ __forceinline__ uint32_t hfma2(uint32_t q, uint32_t s, uint32_t b) {
  __nv_bfloat162 r = __hfma2(*reinterpret_cast<__nv_bfloat162*>(&q), *reinterpret_cast<__nv_bfloat162*>(&s),
                             *reinterpret_cast<__nv_bfloat162*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
}

template <int GS, int M, int EPI, int RT, int WM, int WN>
__global__ void __launch_bounds__(WM * WN * 32)
    prefill_kernel(const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W,
                   int KG, int NB, const int* __restrict__ items, const int* __restrict__ counts,
                   const int* __restrict__ members, void* __restrict__ out, int N, float limit) {
  using G = Geo<GS>;
  using P = Pre<GS, M, RT, WM, WN>;
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
  auto stage = [&](int s, int g0) {                  // groups g0 .. g0 + ng - 1 (ng < SG only at an odd end)
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
  const uint32_t pick = (gq & 1) ? 0x3232u : 0x1010u;         // this lane's column: the high or low half of a pair
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
      const uint4* ws = xs + P::XU + (wn * P::SG + gi) * P::WB;
      uint4 wv[M][G::WV];
      uint32_t s2[M][NTW], b2[M][NTW];
#pragma unroll
      for (int m = 0; m < M; ++m) {
#pragma unroll
        for (int c = 0; c < G::WV; ++c) wv[m][c] = ws[m * G::BLOCK + c * 32 + lane];
        const uint4 sp = ws[m * G::BLOCK + 32 * G::WV + (gq >> 1) * G::SBV];
        const uint4 bp = ws[m * G::BLOCK + 32 * G::WV + (gq >> 1) * G::SBV + 1];
#pragma unroll
        for (int j = 0; j < NTW; ++j) {
          s2[m][j] = __byte_perm(comp(sp, j), 0u, pick);
          b2[m][j] = __byte_perm(comp(bp, j), 0u, pick);
        }
      }
#pragma unroll
      for (int kk = 0; kk < G::KS / 2; ++kk) {
        uint4 xa[RT], xb[RT];
#pragma unroll
        for (int r = 0; r < RT; ++r) {
          if (r >= nt) break;
          const int r0 = base + 16 * r + gq;
          xa[r] = xs[P::xslot(r0, gi * (GS / 8) + t * G::XV + kk)];
          xb[r] = xs[P::xslot(r0 + 8, gi * (GS / 8) + t * G::XV + kk)];
        }
#pragma unroll
        for (int m = 0; m < M; ++m)
#pragma unroll
          for (int j = 0; j < NTW; ++j) {
            const int wi = j * (GS / 32) + kk;
            const uint32_t word = comp(wv[m][wi >> 2], wi & 3);
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

template <int GS, int M, int EPI, int RT, int WM, int WN>
void launch_prefill(const at::Tensor& x, int x_stride, int slots, const at::Tensor& w, int kg, int nb,
                    const at::Tensor& items, const at::Tensor& counts, const at::Tensor& members, at::Tensor& out,
                    int n, float limit, int64_t max_items) {
  using P = Pre<GS, M, RT, WM, WN>;
  constexpr int SMEM = P::STAGES * P::STAGE_BYTES;
  auto* kern = prefill_kernel<GS, M, EPI, RT, WM, WN>;
  static bool ready = false;
  if (!ready) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    ready = true;
  }
  const int64_t grid = max_items * ((nb + WN - 1) / WN);
  if (grid < 1) return;
  kern<<<static_cast<unsigned>(grid), P::THREADS, SMEM, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), x_stride, slots,
      reinterpret_cast<const uint4*>(w.data_ptr()), kg, nb, items.data_ptr<int>(), counts.data_ptr<int>(),
      members.data_ptr<int>(), out.data_ptr(), n, limit);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void experts_prefill_cuda(int64_t gs, int64_t epi, const at::Tensor& x, int64_t x_stride, int64_t slots,
                          const at::Tensor& w, int64_t kg, int64_t nb, const at::Tensor& items,
                          const at::Tensor& counts, const at::Tensor& members, at::Tensor& out, int64_t n,
                          double limit, int64_t max_items) {
  const c10::cuda::CUDAGuard guard(x.device());
  const int xs = static_cast<int>(x_stride), sl = static_cast<int>(slots), k = static_cast<int>(kg);
  const int b = static_cast<int>(nb), nn = static_cast<int>(n);
  const float lim = static_cast<float>(limit);
  // 8 warps a CTA, 64 pairs x 128 columns: two row tiles a warp, two warps down, four across
#define TF_PRE(GS_, M_, EPI_)                                                                                  \
  if (gs == GS_ && epi == EPI_) {                                                                              \
    launch_prefill<GS_, M_, EPI_, 2, 2, 4>(x, xs, sl, w, k, b, items, counts, members, out, nn, lim, max_items); \
    return;                                                                                                    \
  }
  TF_PRE(32, 1, 0) TF_PRE(32, 1, 3) TF_PRE(32, 2, 2) TF_PRE(64, 1, 0) TF_PRE(64, 1, 3) TF_PRE(64, 1, 1)
  TF_PRE(64, 2, 2)
#undef TF_PRE
  TORCH_CHECK(false, "experts: no prefill kernel for group ", gs, " and epilogue ", epi);
}
