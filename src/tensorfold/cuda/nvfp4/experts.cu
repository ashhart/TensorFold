// Grouped NVFP4 experts on the shared plan: per 16-input block acc = fma(P, e4m3 scale, acc) in block order, one
// pair an mma row, so no pair affects another; each (expert, matrix) scale multiplies once before the epilogue.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <torch/extension.h>

#include "../experts.cuh"

namespace {

constexpr int BLOCK4 = 36;           // uint4 a (32 columns, 32 inputs) block: a lane's code words, then the scales

__device__ __forceinline__ uint4 ld_nc(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p));
  return r;
}

// bf16x2 of the e2m1 nibbles at bits [s, s + 4) and [16 + s, 20 + s): fields into bf16's, times 2^126 (exact).
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

template <int M>
struct Stage {
  uint4 w[M];          // the lane's words: n8 tiles 0-3, 32 inputs each
  uint4 s[M];          // the quad's scales, bytes [block][tile][column]
  uint2 xa[2], xb[2];  // rows gq and gq + 8: inputs 4t .. 4t + 3 of each 16-input block
};

template <int M>
__device__ __forceinline__ void load_stage(Stage<M>& st, const uint4* blk, int g, int lane, int t,
                                           const __nv_bfloat16* x0, const __nv_bfloat16* x1, bool v0, bool v1) {
  const uint4* b = blk + (size_t)g * (M * BLOCK4);
#pragma unroll
  for (int m = 0; m < M; ++m) {
    st.w[m] = ld_nc(b + m * BLOCK4 + lane);
    st.s[m] = ld_nc(b + m * BLOCK4 + 32 + t);
  }
  const uint2 zero = make_uint2(0u, 0u);
  const int k0 = g * 32 + 4 * t;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    st.xa[h] = v0 ? __ldg(reinterpret_cast<const uint2*>(x0 + k0 + 16 * h)) : zero;
    st.xb[h] = v1 ? __ldg(reinterpret_cast<const uint2*>(x1 + k0 + 16 * h)) : zero;
  }
}

template <int M>
__device__ __forceinline__ void compute_stage(float (&acc)[M][1][NTW][4], const Stage<M>& st, bool hi) {
#pragma unroll
  for (int m = 0; m < M; ++m)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const uint32_t a0 = st.xa[h].x, a2 = st.xa[h].y, a1 = st.xb[h].x, a3 = st.xb[h].y;
#pragma unroll
      for (int j = 0; j < NTW; ++j) {
        const uint32_t word = comp(st.w[m], j);
        float p[4] = {0.f, 0.f, 0.f, 0.f};
        mma(p, a0, a1, a2, a3, fp4pair(word, 8 * h), fp4pair(word, 8 * h + 4));
        const uint32_t sw = comp(st.s[m], 2 * h + (j >> 1));
        const int sh = (j & 1) * 16;
        const float s0 = e4m3f((sw >> sh) & 0xFFu), s1 = e4m3f((sw >> (sh + 8)) & 0xFFu);
        float(&a)[4] = acc[m][0][j];
        a[0] = fmaf(p[0], s0, a[0]);
        a[1] = fmaf(p[1], s1, a[1]);
        if (hi) {
          a[2] = fmaf(p[2], s0, a[2]);
          a[3] = fmaf(p[3], s1, a[3]);
        }
      }
    }
}

template <int M>
__device__ __forceinline__ void k_loop(float (&acc)[M][1][NTW][4], const uint4* blk, int KG, int lane, int t,
                                       const __nv_bfloat16* x0, const __nv_bfloat16* x1, bool v0, bool v1) {
  constexpr int D = 2;
  Stage<M> st[D];
#pragma unroll
  for (int d = 0; d < D; ++d)
    if (d < KG) load_stage<M>(st[d], blk, d, lane, t, x0, x1, v0, v1);
  for (int g0 = 0; g0 < KG; g0 += D) {
#pragma unroll
    for (int d = 0; d < D; ++d) {
      const int g = g0 + d;
      if (g < KG) {
        compute_stage<M>(acc, st[d], v1);
        if (g + D < KG) load_stage<M>(st[d], blk, g + D, lane, t, x0, x1, v0, v1);
      }
    }
  }
}

// Pair p reads X row p / slots when slots > 0 (X holds tokens), else X row p; items of expert ``skip`` are left
// alone (the shared expert, run elsewhere), and an item's pairs go 16 at a time.
template <int M, int EPI, int WARPS>
__global__ void __launch_bounds__(WARPS * 32)
    nvfp4_expert_kernel(const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W,
                        const float* __restrict__ scale, int KG, int NB, const int* __restrict__ items,
                        const int* __restrict__ counts, const int* __restrict__ members, void* __restrict__ out, int N,
                        float limit, int skip) {
  const int lane = threadIdx.x & 31, gq = lane >> 2, t = lane & 3;
  const int units = __ldg(counts) * NB;
  for (int unit = blockIdx.x * WARPS + (threadIdx.x >> 5); unit < units; unit += gridDim.x * WARPS) {
    const int it = unit / NB, cb = unit - it * NB;
    const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
    if (e == skip) continue;
    const uint4* blk = W + ((size_t)e * NB + cb) * (size_t)KG * (M * BLOCK4);
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
      k_loop<M>(acc, blk, KG, lane, t, x0, x1, v0, v1);
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
}

// Prompt items: per pair the decode kernel's mma and fma order, each weight block staged and decoded once for 64 pairs.
constexpr int PW = 4;                     // warps (column blocks) a CTA
constexpr int PR = 4;                     // row tiles of 16 a warp
constexpr int PROWS = 16 * PR;            // pairs an item holds at most (the plan's prompt tile)
constexpr int XROW = 40;                  // bf16 a staged row of x: 32 inputs and 8 of padding (80 bytes)
constexpr int PS = 4;                     // cp.async stages

template <int M>
struct PromptSmem {
  __nv_bfloat16 x[PS][PROWS][XROW];
  uint4 w[PS][PW][M][BLOCK4];
};

__device__ __forceinline__ void cp16z(void* dst, const void* src, bool ok) {
  const uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(ok ? 16 : 0));
}

template <int M, int EPI>
__global__ void __launch_bounds__(PW * 32, 2)
    nvfp4_expert_prompt_kernel(const __nv_bfloat16* __restrict__ X, int x_stride, int slots,
                               const uint4* __restrict__ W, const float* __restrict__ scale, int KG, int NB,
                               const int* __restrict__ items, const int* __restrict__ counts,
                               const int* __restrict__ members, void* __restrict__ out, int N, float limit, int skip) {
  __shared__ __align__(16) PromptSmem<M> sm;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gq = lane >> 2, t = lane & 3;
  const int groups = NB / PW;
  const int units = __ldg(counts) * groups;
  for (int unit = blockIdx.x; unit < units; unit += gridDim.x) {
    const int it = unit / groups, cg = unit - it * groups;
    const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
    if (e == skip) continue;                                   // the same for the whole CTA
    int xr[2], xp[2];
    bool xok[2];
    const __nv_bfloat16* xs[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int c = tid + PW * 32 * i;
      xr[i] = c >> 2;
      xp[i] = c & 3;
      xok[i] = xr[i] < cnt;
      const int pr = xok[i] ? __ldg(members + first + xr[i]) : 0;
      xs[i] = X + (size_t)(slots ? pr / slots : pr) * x_stride + 8 * xp[i];
    }
    const uint4* wb = W + ((size_t)e * NB + (size_t)cg * PW) * (size_t)KG * (M * BLOCK4);
    auto stage = [&](int s, int g) {
#pragma unroll
      for (int i = 0; i < 2; ++i) cp16z(&sm.x[s][xr[i]][8 * xp[i]], xs[i] + 32 * g, xok[i]);
      for (int c = tid; c < PW * M * BLOCK4; c += PW * 32) {
        const int w = c / (M * BLOCK4), rem = c - w * (M * BLOCK4), m = rem / BLOCK4, q = rem - m * BLOCK4;
        cp16(&sm.w[s][w][m][q], wb + ((size_t)w * KG + g) * (M * BLOCK4) + m * BLOCK4 + q);
      }
    };
    bool v0[PR], v1[PR];
    int pr0[PR], pr1[PR];
#pragma unroll
    for (int r = 0; r < PR; ++r) {
      v0[r] = 16 * r + gq < cnt;
      v1[r] = 16 * r + gq + 8 < cnt;
      pr0[r] = v0[r] ? __ldg(members + first + 16 * r + gq) : 0;
      pr1[r] = v1[r] ? __ldg(members + first + 16 * r + gq + 8) : 0;
    }
    const int live = (cnt + 15) / 16;
    float acc[M][PR][NTW][4];
#pragma unroll
    for (int m = 0; m < M; ++m)
#pragma unroll
      for (int r = 0; r < PR; ++r)
#pragma unroll
        for (int j = 0; j < NTW; ++j) acc[m][r][j][0] = acc[m][r][j][1] = acc[m][r][j][2] = acc[m][r][j][3] = 0.f;
#pragma unroll
    for (int s = 0; s < PS - 1; ++s) {
      if (s < KG) stage(s, s);
      cp_commit();
    }
    for (int g = 0; g < KG; ++g) {
      cp_wait<PS - 2>();
      __syncthreads();
      {
        const int gn = g + PS - 1;                             // the slot every warp finished at g - 1
        if (gn < KG) stage(gn % PS, gn);
        cp_commit();
      }
      const int s = g % PS;
      uint4 wv[M], sv[M];
#pragma unroll
      for (int m = 0; m < M; ++m) {
        wv[m] = sm.w[s][warp][m][lane];
        sv[m] = sm.w[s][warp][m][32 + t];
      }
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        uint2 xa[PR], xb[PR];                                  // each row tile's fragment, read once a half group
#pragma unroll
        for (int r = 0; r < PR; ++r) {
          if (r < live) {
            xa[r] = *reinterpret_cast<const uint2*>(&sm.x[s][16 * r + gq][16 * h + 4 * t]);
            xb[r] = *reinterpret_cast<const uint2*>(&sm.x[s][16 * r + gq + 8][16 * h + 4 * t]);
          } else {
            xa[r] = xb[r] = make_uint2(0u, 0u);
          }
        }
#pragma unroll
        for (int m = 0; m < M; ++m) {
#pragma unroll
          for (int j = 0; j < NTW; ++j) {
            const uint32_t word = comp(wv[m], j);
            const uint32_t b0 = fp4pair(word, 8 * h), b1 = fp4pair(word, 8 * h + 4);
            const uint32_t sw = comp(sv[m], 2 * h + (j >> 1));
            const int sh = (j & 1) * 16;
            const float s0 = e4m3f((sw >> sh) & 0xFFu), s1 = e4m3f((sw >> (sh + 8)) & 0xFFu);
#pragma unroll
            for (int r = 0; r < PR; ++r) {
              if (r >= live) break;                            // the same for the whole CTA
              float p[4] = {0.f, 0.f, 0.f, 0.f};
              mma(p, xa[r].x, xb[r].x, xa[r].y, xb[r].y, b0, b1);
              float(&a)[4] = acc[m][r][j];
              a[0] = fmaf(p[0], s0, a[0]);
              a[1] = fmaf(p[1], s1, a[1]);
              if (v1[r]) {
                a[2] = fmaf(p[2], s0, a[2]);
                a[3] = fmaf(p[3], s1, a[3]);
              }
            }
          }
        }
      }
    }
    cp_wait<0>();
    __syncthreads();                                           // the next item's prologue reuses every slot
#pragma unroll
    for (int m = 0; m < M; ++m) {
      const float g = __ldg(scale + e * M + m);
#pragma unroll
      for (int r = 0; r < PR; ++r)
#pragma unroll
        for (int j = 0; j < NTW; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) acc[m][r][j][q] *= g;
    }
    const int cb = cg * PW + warp;
#pragma unroll
    for (int r = 0; r < PR; ++r)
      if (r < live) epilogue<EPI, M, PR>(acc, r, out, N, cb * COLS + 2 * t, pr0[r], pr1[r], v0[r], v1[r], limit);
  }
}

template <int M, int EPI>
void launch_prompt(const at::Tensor& x, int x_stride, int slots, const at::Tensor& w, const at::Tensor& scale, int kg,
                   int nb, const at::Tensor& items, const at::Tensor& counts, const at::Tensor& members,
                   at::Tensor& out, int n, float limit, int skip, int64_t max_units) {
  TORCH_CHECK(nb % PW == 0, "nvfp4 prompt experts: ", nb, " column blocks are not a multiple of ", PW);
  static int per_sm = 0;
  static int sms = 0;
  if (per_sm == 0) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, nvfp4_expert_prompt_kernel<M, EPI>, PW * 32, 0);
    sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    per_sm = per_sm < 1 ? 1 : per_sm;
  }
  const int64_t need = max_units / PW;                      // max_units counts (item, column block) units
  const int grid = static_cast<int>(need < (int64_t)per_sm * sms ? need : (int64_t)per_sm * sms);
  if (grid < 1) return;
  nvfp4_expert_prompt_kernel<M, EPI><<<grid, PW * 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), x_stride, slots,
      reinterpret_cast<const uint4*>(w.data_ptr()), scale.data_ptr<float>(), kg, nb, items.data_ptr<int>(),
      counts.data_ptr<int>(), members.data_ptr<int>(), out.data_ptr(), n, limit, skip);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int M, int EPI>
void launch(const at::Tensor& x, int x_stride, int slots, const at::Tensor& w, const at::Tensor& scale, int kg,
            int nb, const at::Tensor& items, const at::Tensor& counts, const at::Tensor& members, at::Tensor& out,
            int n, float limit, int skip, int64_t max_units) {
  constexpr int WARPS = 4;
  static int per_sm = 0;
  static int sms = 0;
  if (per_sm == 0) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, nvfp4_expert_kernel<M, EPI, WARPS>, WARPS * 32, 0);
    sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    per_sm = per_sm < 1 ? 1 : per_sm;
  }
  const int64_t need = (max_units + WARPS - 1) / WARPS;
  const int grid = static_cast<int>(need < (int64_t)per_sm * sms ? need : (int64_t)per_sm * sms);
  if (grid < 1) return;
  nvfp4_expert_kernel<M, EPI, WARPS><<<grid, WARPS * 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), x_stride, slots,
      reinterpret_cast<const uint4*>(w.data_ptr()), scale.data_ptr<float>(), kg, nb, items.data_ptr<int>(),
      counts.data_ptr<int>(), members.data_ptr<int>(), out.data_ptr(), n, limit, skip);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void nvfp4_experts_cuda(int64_t epi, const at::Tensor& x, int64_t x_stride, int64_t slots, const at::Tensor& w,
                        const at::Tensor& scale, int64_t kg, int64_t nb, const at::Tensor& items,
                        const at::Tensor& counts, const at::Tensor& members, at::Tensor& out, int64_t n, double limit,
                        int64_t skip, int64_t max_units, int64_t rt) {
  const c10::cuda::CUDAGuard guard(x.device());
  const int xs = static_cast<int>(x_stride), sl = static_cast<int>(slots), k = static_cast<int>(kg);
  const int b = static_cast<int>(nb), nn = static_cast<int>(n), sk = static_cast<int>(skip);
  const float lim = static_cast<float>(limit);
  if (rt == 4) {                                              // prompt items of up to 64 pairs
    const int64_t mu = max_units;
    if (epi == 2) launch_prompt<2, 2>(x, xs, sl, w, scale, k, b, items, counts, members, out, nn, lim, sk, mu);
    else if (epi == 0) launch_prompt<1, 0>(x, xs, sl, w, scale, k, b, items, counts, members, out, nn, lim, sk, mu);
    else if (epi == 3) launch_prompt<1, 3>(x, xs, sl, w, scale, k, b, items, counts, members, out, nn, lim, sk, mu);
    else TORCH_CHECK(false, "nvfp4 experts: epilogue 0 (fp32 down), 2 (SwiGLU) or 3 (bf16 down), not ", epi);
    return;
  }
  if (epi == 2) launch<2, 2>(x, xs, sl, w, scale, k, b, items, counts, members, out, nn, lim, sk, max_units);
  else if (epi == 0) launch<1, 0>(x, xs, sl, w, scale, k, b, items, counts, members, out, nn, lim, sk, max_units);
  else if (epi == 3) launch<1, 3>(x, xs, sl, w, scale, k, b, items, counts, members, out, nn, lim, sk, max_units);
  else TORCH_CHECK(false, "nvfp4 experts: epilogue 0 (fp32 down), 2 (SwiGLU) or 3 (bf16 down), not ", epi);
}
