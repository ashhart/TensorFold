// Grouped block-FP8 experts on the shared plan: each 128 inputs' bf16 mma chain from zero, then acc = fma(chain, its
// fp32 block scale, acc) in block order; one pair an mma row, so no pair affects another.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <torch/extension.h>

#include "../experts.cuh"

namespace {

constexpr int LANE4 = 2;             // uint4 a lane's words in a (32 columns, 32 inputs) block: k16 halves 0 and 1
constexpr int BLOCK4 = 32 * LANE4;   // uint4 a (32 columns, 32 inputs) block of one matrix
constexpr int PER_SCALE = 4;         // 32-input blocks under one 128 x 128 scale

__device__ __forceinline__ uint4 ld_nc(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p));
  return r;
}

// bf16x2 of the e4m3 bytes at bits [0, 8) and [8, 16) of v: sign and the seven other bits into bf16's fields, times
// 2^120 (exact, subnormals included)
__device__ __forceinline__ uint32_t fp8pair(uint32_t v) {
  const uint32_t t = ((v & 0x7Fu) << 4) | ((v & 0x80u) << 8) | ((v & 0x7F00u) << 12) | ((v & 0x8000u) << 16);
  uint32_t r;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(t), "r"(0x7B807B80u), "r"(0x80008000u));
  return r;
}

template <int M>
struct Stage {
  uint4 w[M][LANE4];   // the lane's words: half h's tiles 0-3, each four bytes of inputs 16h + 4t .. 16h + 4t + 3
  uint2 xa[2], xb[2];  // rows gq and gq + 8: inputs 16h + 4t .. 16h + 4t + 3
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

// The K loop: 32-input blocks two in flight; every PER_SCALE blocks the chain folds into acc under its scale.
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

// Pair p reads X row p / slots when slots > 0 (X holds tokens), else X row p; an item's pairs go 16 at a time.
// scale: [E, M, N / 128, K / 128] fp32; a unit's 32 columns lie in one 128-row scale block.
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

template <int M, int EPI>
void launch(const at::Tensor& x, int x_stride, int slots, const at::Tensor& w, const at::Tensor& scale, int kg,
            int nb, const at::Tensor& items, const at::Tensor& counts, const at::Tensor& members, at::Tensor& out,
            int n, float limit, int skip, int64_t max_units) {
  constexpr int WARPS = 4;
  static int per_sm = 0;
  static int sms = 0;
  if (per_sm == 0) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, fp8_expert_kernel<M, EPI, WARPS>, WARPS * 32, 0);
    sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    per_sm = per_sm < 1 ? 1 : per_sm;
  }
  const int64_t need = (max_units + WARPS - 1) / WARPS;
  const int grid = static_cast<int>(need < (int64_t)per_sm * sms ? need : (int64_t)per_sm * sms);
  if (grid < 1) return;
  fp8_expert_kernel<M, EPI, WARPS><<<grid, WARPS * 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), x_stride, slots,
      reinterpret_cast<const uint4*>(w.data_ptr()), scale.data_ptr<float>(), kg, nb, items.data_ptr<int>(),
      counts.data_ptr<int>(), members.data_ptr<int>(), out.data_ptr(), n, limit, skip);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


// Prompt items of up to 64 pairs: a CTA of four warps, CW column blocks by 64 / (16 RT) row groups, stages each block
// once; a warp converts a weight fragment once for its RT row tiles. Per pair the arithmetic of fp8_expert_kernel.
constexpr int PWARPS = 4, PSTAGES = 4, PMIN = 1;

template <int M, int RT, int CW>
struct PromptCfg {
  static constexpr int RW = PWARPS / CW;              // row groups
  static_assert(RW * RT * 16 == 64, "a prompt item holds 64 pairs");
};

template <int M, int EPI, int RT, int CW>
__global__ void __launch_bounds__(PWARPS * 32, PMIN)
    fp8_expert_prompt_kernel(const __nv_bfloat16* __restrict__ X, int x_stride, int slots,
                             const uint4* __restrict__ W, const float* __restrict__ scale, int KG, int NB,
                             const int* __restrict__ items, const int* __restrict__ counts,
                             const int* __restrict__ members, void* __restrict__ out, int N, float limit, int skip) {
  constexpr int RW = PromptCfg<M, RT, CW>::RW;
  __shared__ uint4 sw[PSTAGES][CW][M][LANE4][32];       // half-major so a warp's reads hit all banks
  __shared__ uint4 sx[PSTAGES][64][4];                  // the item's 64 rows, 32 inputs each
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
    const __nv_bfloat16* xrow[2];                       // this thread's two staged rows (past cnt: the last pair's)
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      const int p = min((threadIdx.x + c * PWARPS * 32) / 4, cnt - 1);
      const int pr = __ldg(members + first + p);
      xrow[c] = X + (size_t)(slots ? pr / slots : pr) * x_stride + ((threadIdx.x + c * PWARPS * 32) % 4) * 8;
    }
    const uint4* base = W + ((size_t)e * NB + cg * CW) * (size_t)KG * (M * BLOCK4);
    auto stage = [&](int g) {                           // CW blocks of M * 64 uint4, then 64 rows' 32 inputs
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
    __syncthreads();                                    // the previous unit's readers are done with every slot
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
        for (int r = 0; r < RT; ++r) {                  // inputs 16h + 4t .. 16h + 4t + 3 of rows gq and gq + 8
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

template <int M, int EPI, int RT, int CW>
void launch_prompt(const at::Tensor& x, int x_stride, int slots, const at::Tensor& w, const at::Tensor& scale,
                   int kg, int nb, const at::Tensor& items, const at::Tensor& counts, const at::Tensor& members,
                   at::Tensor& out, int n, float limit, int skip, int64_t max_units) {
  TORCH_CHECK(nb % CW == 0, "fp8 prompt experts: column blocks in groups of ", CW);
  static int per_sm = 0;
  static int sms = 0;
  if (per_sm == 0) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, fp8_expert_prompt_kernel<M, EPI, RT, CW>, PWARPS * 32, 0);
    sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    per_sm = per_sm < 1 ? 1 : per_sm;
  }
  const int64_t units = max_units / nb * (nb / CW);
  const int grid = static_cast<int>(units < (int64_t)per_sm * sms ? units : (int64_t)per_sm * sms);
  if (grid < 1) return;
  fp8_expert_prompt_kernel<M, EPI, RT, CW><<<grid, PWARPS * 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), x_stride, slots,
      reinterpret_cast<const uint4*>(w.data_ptr()), scale.data_ptr<float>(), kg, nb, items.data_ptr<int>(),
      counts.data_ptr<int>(), members.data_ptr<int>(), out.data_ptr(), n, limit, skip);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void fp8_experts_cuda(int64_t epi, const at::Tensor& x, int64_t x_stride, int64_t slots, const at::Tensor& w,
                      const at::Tensor& scale, int64_t kg, int64_t nb, const at::Tensor& items,
                      const at::Tensor& counts, const at::Tensor& members, at::Tensor& out, int64_t n, double limit,
                      int64_t skip, int64_t max_units) {
  const c10::cuda::CUDAGuard guard(x.device());
  const int xs = static_cast<int>(x_stride), sl = static_cast<int>(slots), k = static_cast<int>(kg);
  const int b = static_cast<int>(nb), nn = static_cast<int>(n), sk = static_cast<int>(skip);
  const float lim = static_cast<float>(limit);
  if (epi == 2) launch<2, 2>(x, xs, sl, w, scale, k, b, items, counts, members, out, nn, lim, sk, max_units);
  else if (epi == 0) launch<1, 0>(x, xs, sl, w, scale, k, b, items, counts, members, out, nn, lim, sk, max_units);
  else if (epi == 3) launch<1, 3>(x, xs, sl, w, scale, k, b, items, counts, members, out, nn, lim, sk, max_units);
  else TORCH_CHECK(false, "fp8 experts: epilogue 0 (fp32 down), 2 (SwiGLU) or 3 (bf16 down), not ", epi);
}

void fp8_experts_prompt_cuda(int64_t epi, const at::Tensor& x, int64_t x_stride, int64_t slots, const at::Tensor& w,
                             const at::Tensor& scale, int64_t kg, int64_t nb, const at::Tensor& items,
                             const at::Tensor& counts, const at::Tensor& members, at::Tensor& out, int64_t n,
                             double limit, int64_t skip, int64_t max_units) {
  const c10::cuda::CUDAGuard guard(x.device());
  const int xs = static_cast<int>(x_stride), sl = static_cast<int>(slots), k = static_cast<int>(kg);
  const int b = static_cast<int>(nb), nn = static_cast<int>(n), sk = static_cast<int>(skip);
  const float lim = static_cast<float>(limit);
  const int64_t mu = max_units;
  if (epi == 2) launch_prompt<2, 2, 2, 2>(x, xs, sl, w, scale, k, b, items, counts, members, out, nn, lim, sk, mu);
  else if (epi == 0) launch_prompt<1, 0, 4, 4>(x, xs, sl, w, scale, k, b, items, counts, members, out, nn, lim, sk, mu);
  else if (epi == 3) launch_prompt<1, 3, 4, 4>(x, xs, sl, w, scale, k, b, items, counts, members, out, nn, lim, sk, mu);
  else TORCH_CHECK(false, "fp8 prompt experts: epilogue 0, 2 or 3, not ", epi);
}
