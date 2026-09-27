// Decode form: acc = fma(xs, b, fma(p, s, acc)) per group in order, one pair an mma row, so no pair affects another.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <torch/extension.h>

#include "experts.cuh"

namespace {

__device__ __forceinline__ uint4 ld_w(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p));
  return r;
}

__device__ __forceinline__ uint4 ld_x(const __nv_bfloat16* p) { return __ldg(reinterpret_cast<const uint4*>(p)); }

// a row's group sum: the lane's inputs in order, then the quad (every lane of the quad gets the same bits)
template <int XV>
__device__ __forceinline__ float group_sum(const uint4 (&v)[XV]) {
  float s = 0.f;
#pragma unroll
  for (int c = 0; c < XV; ++c)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint32_t u = comp(v[c], i);
      s += lo_f(u);
      s += hi_f(u);
    }
  s += __shfl_xor_sync(0xffffffffu, s, 1);
  s += __shfl_xor_sync(0xffffffffu, s, 2);
  return s;
}

template <int GS, int M>
struct Stage {
  uint4 w[M][Geo<GS>::WV];
  uint4 sb[M][Geo<GS>::SBV];
  uint4 xa[Geo<GS>::XV];       // row gq of the tile: the lane's inputs of the group
  uint4 xb[Geo<GS>::XV];       // row gq + 8
};

template <int GS, int M>
__device__ __forceinline__ void load_stage(Stage<GS, M>& st, const uint4* blk, int g, int lane, int t,
                                           const __nv_bfloat16* x0, const __nv_bfloat16* x1, bool v0, bool v1) {
  using G = Geo<GS>;
  const uint4* b = blk + (size_t)g * (M * G::BLOCK);
#pragma unroll
  for (int m = 0; m < M; ++m) {
#pragma unroll
    for (int c = 0; c < G::WV; ++c) st.w[m][c] = ld_w(b + m * G::BLOCK + c * 32 + lane);
#pragma unroll
    for (int c = 0; c < G::SBV; ++c) st.sb[m][c] = ld_w(b + m * G::BLOCK + 32 * G::WV + t * G::SBV + c);
  }
  const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
  const int k0 = g * GS;
#pragma unroll
  for (int c = 0; c < G::XV; ++c) {
    st.xa[c] = v0 ? ld_x(x0 + k0 + 8 * c) : zero;
    st.xb[c] = v1 ? ld_x(x1 + k0 + 8 * c) : zero;
  }
}

template <int GS, int M>
__device__ __forceinline__ void compute_stage(float (&acc)[M][1][NTW][4], const Stage<GS, M>& st, bool hi) {
  using G = Geo<GS>;
  const float xsa = group_sum<G::XV>(st.xa);
  const float xsb = group_sum<G::XV>(st.xb);
#pragma unroll
  for (int m = 0; m < M; ++m) {
    float p[NTW][4];
#pragma unroll
    for (int j = 0; j < NTW; ++j) p[j][0] = p[j][1] = p[j][2] = p[j][3] = 0.f;
#pragma unroll
    for (int ks = 0; ks < G::KS; ++ks) {
      // k-step ks: the lane's inputs 4 ks .. 4 ks + 3 (pairs at the mma's k positions 2t and 2t + 8)
      const uint32_t a0 = comp(st.xa[ks >> 1], 2 * (ks & 1));
      const uint32_t a2 = comp(st.xa[ks >> 1], 2 * (ks & 1) + 1);
      const uint32_t a1 = comp(st.xb[ks >> 1], 2 * (ks & 1));
      const uint32_t a3 = comp(st.xb[ks >> 1], 2 * (ks & 1) + 1);
      const int sh = (ks & 1) ? 8 : 0;
#pragma unroll
      for (int j = 0; j < NTW; ++j) {
        const int wi = j * (GS / 32) + (ks >> 1);
        const uint32_t word = comp(st.w[m][wi >> 2], wi & 3);
        mma(p[j], a0, a1, a2, a3, nib2(word, sh), nib2(word, sh + 4));
      }
    }
#pragma unroll
    for (int j = 0; j < NTW; ++j) {
      const uint32_t sp = comp(st.sb[m][j >> 2], j & 3);
      const uint32_t bp = comp(st.sb[m][(NTW + j) >> 2], (NTW + j) & 3);
      const float s0 = lo_f(sp), s1 = hi_f(sp), b0 = lo_f(bp), b1 = hi_f(bp);
      float(&a)[4] = acc[m][0][j];
      a[0] = fmaf(xsa, b0, fmaf(p[j][0], s0, a[0]));
      a[1] = fmaf(xsa, b1, fmaf(p[j][1], s1, a[1]));
      if (hi) {
        a[2] = fmaf(xsb, b0, fmaf(p[j][2], s0, a[2]));
        a[3] = fmaf(xsb, b1, fmaf(p[j][3], s1, a[3]));
      }
    }
  }
}

template <int GS, int M, int D>
__device__ __forceinline__ void k_loop(float (&acc)[M][1][NTW][4], const uint4* blk, int KG, int lane, int t,
                                       const __nv_bfloat16* x0, const __nv_bfloat16* x1, bool v0, bool v1) {
  Stage<GS, M> st[D];
#pragma unroll
  for (int d = 0; d < D; ++d)
    if (d < KG) load_stage<GS, M>(st[d], blk, d, lane, t, x0, x1, v0, v1);
  for (int g0 = 0; g0 < KG; g0 += D) {
#pragma unroll
    for (int d = 0; d < D; ++d) {
      const int g = g0 + d;
      if (g < KG) {
        compute_stage<GS, M>(acc, st[d], v1);
        if (g + D < KG) load_stage<GS, M>(st[d], blk, g + D, lane, t, x0, x1, v0, v1);
      }
    }
  }
}

// Pair p reads X row p / slots when slots > 0 (X holds tokens), else X row p (X holds a row a pair).
template <int GS, int M, int EPI, int D, int WARPS>
__global__ void __launch_bounds__(WARPS * 32)
    expert_kernel(const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W, int KG,
                  int NB, const int* __restrict__ items, const int* __restrict__ counts,
                  const int* __restrict__ members, void* __restrict__ out, int N, float limit) {
  const int lane = threadIdx.x & 31, gq = lane >> 2, t = lane & 3;
  const int units = __ldg(counts) * NB;
  for (int unit = blockIdx.x * WARPS + (threadIdx.x >> 5); unit < units; unit += gridDim.x * WARPS) {
    const int it = unit / NB, cb = unit - it * NB;
    const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
    const bool v0 = gq < cnt, v1 = gq + 8 < cnt;
    const int pr0 = v0 ? __ldg(members + first + gq) : 0;
    const int pr1 = v1 ? __ldg(members + first + gq + 8) : 0;
    const int r0 = slots ? pr0 / slots : pr0, r1 = slots ? pr1 / slots : pr1;
    const __nv_bfloat16* x0 = X + (size_t)r0 * x_stride + t * (GS / 4);
    const __nv_bfloat16* x1 = X + (size_t)r1 * x_stride + t * (GS / 4);
    const uint4* blk = W + ((size_t)e * NB + cb) * (size_t)KG * (M * Geo<GS>::BLOCK);
    float acc[M][1][NTW][4];
#pragma unroll
    for (int m = 0; m < M; ++m)
#pragma unroll
      for (int j = 0; j < NTW; ++j) acc[m][0][j][0] = acc[m][0][j][1] = acc[m][0][j][2] = acc[m][0][j][3] = 0.f;
    k_loop<GS, M, D>(acc, blk, KG, lane, t, x0, x1, v0, v1);
    epilogue<EPI, M, 1>(acc, 0, out, N, cb * COLS + 2 * t, pr0, pr1, v0, v1, limit);
  }
}

constexpr int PLAN_THREADS = 1024;
constexpr int EMAX = 1024;
constexpr int PSMALL = 1024;       // pairs the one-block plan takes; wider calls rank in blocks of PLAN_THREADS

// Whole block: thread e gives expert e's pair count, gets its first member; items of <= T pairs land in expert order.
__device__ int place_items(int c, int E, int T, int* __restrict__ items, int* __restrict__ counts) {
  __shared__ int wsum[3][32];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int tiles = (c + T - 1) / T;
  int a = c, b = tiles, u = c > 0;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int ua = __shfl_up_sync(0xffffffffu, a, o), ub = __shfl_up_sync(0xffffffffu, b, o);
    const int uu = __shfl_up_sync(0xffffffffu, u, o);
    if (lane >= o) {
      a += ua;
      b += ub;
      u += uu;
    }
  }
  if (lane == 31) {
    wsum[0][warp] = a;
    wsum[1][warp] = b;
    wsum[2][warp] = u;
  }
  __syncthreads();
  if (warp == 0) {
#pragma unroll
    for (int s = 0; s < 3; ++s) {
      int v = wsum[s][lane];
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int w = __shfl_up_sync(0xffffffffu, v, o);
        if (lane >= o) v += w;
      }
      wsum[s][lane] = v;
    }
  }
  __syncthreads();
  const int off = (warp ? wsum[0][warp - 1] : 0) + a - c;
  const int ioff = (warp ? wsum[1][warp - 1] : 0) + b - tiles;
  if (tid == 0) {
    counts[0] = wsum[1][31];
    counts[1] = wsum[2][31];
  }
  if (tid < E)
    for (int j = 0; j < tiles; ++j) {
      int* it = items + 3 * (ioff + j);
      it[0] = tid;
      it[1] = off + T * j;
      it[2] = min(T, c - T * j);
    }
  return off;
}

// One block: pairs p = row * slots + slot grouped by expert (pair order within an expert).
__global__ void __launch_bounds__(PLAN_THREADS)
    plan_kernel(const int* __restrict__ picks, int P, int E, int T, int* __restrict__ members,
                int* __restrict__ items, int* __restrict__ counts) {
  __shared__ int pk[PSMALL];
  __shared__ int cnt[EMAX];
  __shared__ int off[EMAX];
  const int tid = threadIdx.x;
  for (int e = tid; e < EMAX; e += PLAN_THREADS) cnt[e] = 0;
  __syncthreads();
  for (int p = tid; p < P; p += PLAN_THREADS) {
    const int e = picks[p];
    pk[p] = e;
    atomicAdd(&cnt[e], 1);
  }
  __syncthreads();
  const int o = place_items(tid < E ? cnt[tid] : 0, E, T, items, counts);
  if (tid < E) off[tid] = o;
  __syncthreads();
  for (int p = tid; p < P; p += PLAN_THREADS) {
    const int e = pk[p];
    int rank = 0;
    for (int q = 0; q < p; ++q) rank += pk[q] == e;
    members[off[e] + rank] = p;
  }
}

// Wide calls, pass 1: block b ranks its pairs within each expert in pair order and writes its counts to hist[b][e].
__global__ void __launch_bounds__(PLAN_THREADS)
    plan_rank(const int* __restrict__ picks, int P, int E, int* __restrict__ rank, int* __restrict__ hist) {
  __shared__ int cnt[EMAX];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  for (int e = tid; e < E; e += PLAN_THREADS) cnt[e] = 0;
  __syncthreads();
  const int p = blockIdx.x * PLAN_THREADS + tid;
  const bool ok = p < P;
  const int e = ok ? picks[p] : -1 - lane;
  const unsigned same = __match_any_sync(0xffffffffu, e);
  const int below = __popc(same & ((1u << lane) - 1u));
  for (int w = 0; w < PLAN_THREADS / 32; ++w) {
    if (warp == w) {
      const int base = ok ? cnt[e] : 0;
      __syncwarp();
      if (ok && below == 0) cnt[e] = base + __popc(same);
      if (ok) rank[p] = base + below;
    }
    __syncthreads();
  }
  for (int x = tid; x < E; x += PLAN_THREADS) hist[(size_t)blockIdx.x * E + x] = cnt[x];
}

// Pass 2, one block: each block's first member for each expert in place of its count, and the items.
__global__ void __launch_bounds__(PLAN_THREADS)
    plan_offsets(int nblk, int E, int T, int* __restrict__ hist, int* __restrict__ items, int* __restrict__ counts) {
  const int tid = threadIdx.x;
  int c = 0;
  if (tid < E)
    for (int b = 0; b < nblk; ++b) {
      int* h = hist + (size_t)b * E + tid;
      const int v = *h;
      *h = c;
      c += v;
    }
  const int off = place_items(c, E, T, items, counts);
  if (tid < E)
    for (int b = 0; b < nblk; ++b) hist[(size_t)b * E + tid] += off;
}

// Pass 3: every pair to its place.
__global__ void plan_scatter(const int* __restrict__ picks, int P, int E, const int* __restrict__ rank,
                             const int* __restrict__ hist, int* __restrict__ members) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p < P) members[hist[(size_t)(p / PLAN_THREADS) * E + picks[p]] + rank[p]] = p;
}

template <int GS, int M, int EPI>
void launch(const at::Tensor& x, int x_stride, int slots, const at::Tensor& w, int kg, int nb, const at::Tensor& items,
            const at::Tensor& counts, const at::Tensor& members, at::Tensor& out, int n, float limit,
            int64_t max_units) {
  constexpr int D = 2, WARPS = 4;
  static int per_sm = 0;
  static int sms = 0;
  if (per_sm == 0) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, expert_kernel<GS, M, EPI, D, WARPS>, WARPS * 32, 0);
    sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    per_sm = per_sm < 1 ? 1 : per_sm;
  }
  const int64_t need = (max_units + WARPS - 1) / WARPS;
  const int grid = static_cast<int>(need < (int64_t)per_sm * sms ? need : (int64_t)per_sm * sms);
  if (grid < 1) return;
  expert_kernel<GS, M, EPI, D, WARPS><<<grid, WARPS * 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), x_stride, slots,
      reinterpret_cast<const uint4*>(w.data_ptr()), kg, nb, items.data_ptr<int>(), counts.data_ptr<int>(),
      members.data_ptr<int>(), out.data_ptr(), n, limit);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void experts_plan_cuda(const at::Tensor& picks, int64_t pairs, int64_t experts, int64_t tile, at::Tensor& members,
                       at::Tensor& items, at::Tensor& counts, at::Tensor& rank, at::Tensor& hist) {
  const c10::cuda::CUDAGuard guard(picks.device());
  TORCH_CHECK(experts <= EMAX, "experts: at most ", EMAX, " experts");
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int P = static_cast<int>(pairs), E = static_cast<int>(experts), T = static_cast<int>(tile);
  if (P <= PSMALL) {
    plan_kernel<<<1, PLAN_THREADS, 0, stream>>>(picks.data_ptr<int>(), P, E, T, members.data_ptr<int>(),
                                                 items.data_ptr<int>(), counts.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  const int nblk = (P + PLAN_THREADS - 1) / PLAN_THREADS;
  TORCH_CHECK(rank.numel() >= P && hist.numel() >= (int64_t)nblk * E, "experts: plan scratch too small");
  plan_rank<<<nblk, PLAN_THREADS, 0, stream>>>(picks.data_ptr<int>(), P, E, rank.data_ptr<int>(),
                                               hist.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  plan_offsets<<<1, PLAN_THREADS, 0, stream>>>(nblk, E, T, hist.data_ptr<int>(), items.data_ptr<int>(),
                                               counts.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  plan_scatter<<<(P + 255) / 256, 256, 0, stream>>>(picks.data_ptr<int>(), P, E, rank.data_ptr<int>(),
                                                    hist.data_ptr<int>(), members.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void experts_run_cuda(int64_t gs, int64_t epi, const at::Tensor& x, int64_t x_stride, int64_t slots,
                      const at::Tensor& w, int64_t kg, int64_t nb, const at::Tensor& items, const at::Tensor& counts,
                      const at::Tensor& members, at::Tensor& out, int64_t n, double limit, int64_t max_units) {
  const c10::cuda::CUDAGuard guard(x.device());
  const int xs = static_cast<int>(x_stride), sl = static_cast<int>(slots), k = static_cast<int>(kg);
  const int b = static_cast<int>(nb), nn = static_cast<int>(n);
  const float lim = static_cast<float>(limit);
#define TF_RUN(GS_, M_, EPI_)                                                                        \
  if (gs == GS_ && epi == EPI_) {                                                                    \
    launch<GS_, M_, EPI_>(x, xs, sl, w, k, b, items, counts, members, out, nn, lim, max_units);     \
    return;                                                                                          \
  }
  TF_RUN(32, 1, 0) TF_RUN(32, 2, 2) TF_RUN(64, 1, 0) TF_RUN(64, 1, 1) TF_RUN(64, 2, 2)
#undef TF_RUN
  TORCH_CHECK(false, "experts: no kernel for group ", gs, " and epilogue ", epi);
}
