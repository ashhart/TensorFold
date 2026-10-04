// Staged IQ2_XXS SoA grouped prefill — D2R B fragments + b16-prompt A ldmatrix/MMA.
// M=1: single projection. M=2 EPI=2: fused gate+up+SwiGLU (limit 10, matches deepseek_v4/cuda/moe.py).
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <torch/extension.h>

#include "../experts.cuh"

namespace {

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int WM = 1;
constexpr int WN = 4;
constexpr int KS = 64;
constexpr int PST = 2;
constexpr int PROW = 128;
constexpr int THREADS = WM * WN * 32;
constexpr int STAGE = BM * PROW;
constexpr int GRID_BYTES = 256 * 8;
constexpr int NT = BN / WN / 8;

__device__ __forceinline__ uint32_t sm(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }

__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(sm(p)));
}

__device__ __forceinline__ void cpz(void* dst, const void* src, bool ok) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sm(dst)), "l"(src), "r"(ok ? 16 : 0));
}

__device__ __forceinline__ int pswz(int r, int c) { return r * PROW + ((c ^ (r & 7)) << 4); }

__device__ __forceinline__ uint32_t pack2(float a, float b) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<const uint32_t*>(&v);
}

__device__ __forceinline__ void iq2_bfrag(const __half* __restrict__ dq, const uint64_t* __restrict__ qs,
                                          const uint8_t* __restrict__ grid, int col, int k0, int c, uint32_t& b0,
                                          uint32_t& b1) {
  const int g = k0 >> 5;
  const uint64_t word = qs[col * 8 + g];
  const uint32_t bits = (uint32_t)(word >> 32);
  const float d = __half2float(dq[col]) * (0.5f + (float)(bits >> 28)) * 0.25f;
  const int j0 = (k0 & 31) + 2 * c;
  float v[4];
#pragma unroll
  for (int t = 0; t < 4; ++t) {
    const int j = j0 + ((t & 1) + 8 * (t >> 1));
    const int gi = j >> 3;
    const int grid_idx = (int)((word >> (8 * gi)) & 255u);
    uint32_t signs = (bits >> (7 * gi)) & 127u;
    uint32_t parity = signs ^ (signs >> 4);
    parity ^= parity >> 2;
    parity ^= parity >> 1;
    signs |= (parity & 1u) << 7;
    const float mag = (float)grid[grid_idx * 8 + (j & 7)];
    const float sign = ((signs >> (j & 7)) & 1u) ? -1.f : 1.f;
    v[t] = d * mag * sign;
  }
  b0 = pack2(v[0], v[1]);
  b1 = pack2(v[2], v[3]);
}

// EPI 2: SwiGLU(gate=acc0, up=acc1); EPI 3: store acc0 as bf16.
template <int M, int EPI>
__global__ void __launch_bounds__(THREADS, 2) iq2_soa_prefill_kernel(
    const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const __half* __restrict__ dq0,
    const uint64_t* __restrict__ qs0, const __half* __restrict__ dq1, const uint64_t* __restrict__ qs1,
    const uint8_t* __restrict__ grid, const int* __restrict__ items, const int* __restrict__ counts,
    const int* __restrict__ members, __nv_bfloat16* __restrict__ Y, int N, int K, int experts, int pairs) {
  extern __shared__ __align__(128) unsigned char pbuf[];
  __shared__ __half sdq[M][BN];
  __shared__ uint64_t sqs[M][BN * 8];
  __shared__ uint8_t sgrid[GRID_BYTES];

  const int n_tiles = (N + BN - 1) / BN;
  const int it = (int)blockIdx.x / n_tiles;
  const int nt = (int)blockIdx.x - it * n_tiles;
  if (it >= __ldg(counts)) return;

  const int expert = __ldg(items + 3 * it);
  const int first = __ldg(items + 3 * it + 1);
  const int count = min(max(__ldg(items + 3 * it + 2), 0), BM);
  if (expert < 0 || expert >= experts || count < 1 || first < 0 || first > pairs - count) return;

  const int n0 = nt * BN;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int wm = warp / WN;
  const int wn = warp - wm * WN;
  constexpr int MI = BM / WM / 16;
  const int kg = K / 256;
  const int KT = K / KS;

  for (int i = tid; i < GRID_BYTES; i += THREADS) sgrid[i] = grid[i];

  float acc[M][MI][NT][4];
#pragma unroll
  for (int m = 0; m < M; ++m)
#pragma unroll
    for (int i = 0; i < MI; ++i)
#pragma unroll
      for (int j = 0; j < NT; ++j) acc[m][i][j][0] = acc[m][i][j][1] = acc[m][i][j][2] = acc[m][i][j][3] = 0.f;

  auto load_x = [&](int s, int kt) {
    unsigned char* p = pbuf + s * STAGE;
    for (int c = tid; c < BM * 8; c += THREADS) {
      const int r = c >> 3;
      const int ch = c & 7;
      bool ok = r < count;
      const __nv_bfloat16* src = X;
      if (ok) {
        const int pair = __ldg(members + first + r);
        ok = (unsigned)pair < (unsigned)pairs;
        if (ok) {
          const int row = slots ? pair / slots : pair;
          src = X + (size_t)row * (size_t)x_stride + (size_t)kt * KS + ch * 8;
        }
      }
      cpz(p + pswz(r, ch), src, ok);
    }
  };

  auto load_w = [&](int kb) {
    for (int c = tid; c < BN; c += THREADS) {
      const int n = n0 + c;
#pragma unroll
      for (int m = 0; m < M; ++m) {
        const __half* dq = m ? dq1 : dq0;
        const uint64_t* qs = m ? qs1 : qs0;
        if (n < N && kb < kg) {
          const int64_t blk = ((int64_t)expert * N + n) * kg + kb;
          sdq[m][c] = dq[blk];
#pragma unroll
          for (int g = 0; g < 8; ++g) sqs[m][c * 8 + g] = qs[blk * 8 + g];
        } else {
          sdq[m][c] = __float2half_rn(0.f);
#pragma unroll
          for (int g = 0; g < 8; ++g) sqs[m][c * 8 + g] = 0;
        }
      }
    }
  };

  load_w(0);
  for (int s = 0; s < PST - 1; ++s) {
    if (s < KT) load_x(s, s);
    cp_commit();
  }
  __syncthreads();

  for (int kt = 0; kt < KT; ++kt) {
    cp_wait<PST - 2>();
    __syncthreads();
    if (kt + PST - 1 < KT) load_x((kt + PST - 1) % PST, kt + PST - 1);
    cp_commit();

    if ((kt & 3) == 0 && kt > 0) {
      load_w(kt >> 2);
      __syncthreads();
    }

    const unsigned char* p = pbuf + (kt % PST) * STAGE;
    const int k_blk = (kt & 3) * KS;
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      const int k0 = k_blk + ks * 16;
      const int c = lane & 3;
      // The same k32 pair ordering for A and B; ordinary ldmatrix kWidth=2
      // would change rounding even though the mathematical dot is unchanged.
      uint32_t a[MI][4];
#pragma unroll
      for (int i = 0; i < MI; ++i)
        ldsm4(a[i], p + pswz(wm * (BM / WM) + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8, ks * 2 + (lane >> 4)));
#pragma unroll
      for (int m = 0; m < M; ++m) {
        if (wm * (BM / WM) < count) {
#pragma unroll
          for (int j = 0; j < NT; ++j) {
            const int ncol = n0 + wn * (BN / WN) + j * 8 + (lane >> 2);
            uint32_t b0 = 0, b1 = 0;
            if ((unsigned)ncol < (unsigned)N) iq2_bfrag(sdq[m], sqs[m], sgrid, ncol - n0, k0, c, b0, b1);
#pragma unroll
            for (int i = 0; i < MI; ++i) mma(acc[m][i][j], a[i][0], a[i][1], a[i][2], a[i][3], b0, b1);
          }
        }
      }
    }
  }
  cp_wait<0>();

#pragma unroll
  for (int i = 0; i < MI; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int row = wm * (BM / WM) + i * 16 + (lane >> 2) + h * 8;
        const int col = n0 + wn * (BN / WN) + j * 8 + (lane & 3) * 2;
        if (row >= count) continue;
        const int pair = __ldg(members + first + row);
        if ((unsigned)pair >= (unsigned)pairs) continue;
#pragma unroll
        for (int t = 0; t < 2; ++t) {
          const int c = col + t;
          if (c >= N) continue;
          float o;
          if constexpr (EPI == 2) {
            float g = bf(acc[0][i][j][2 * h + t]);
            float u = bf(acc[1][i][j][2 * h + t]);
            g = fminf(g, 10.f);
            u = fminf(fmaxf(u, -10.f), 10.f);
            o = g / (1.f + expf(-g)) * u;
          } else {
            o = acc[0][i][j][2 * h + t];
          }
          Y[(size_t)pair * N + c] = __float2bfloat16_rn(o);
        }
      }
}

template <int M, int EPI>
void launch_iq2(const at::Tensor& x, int x_stride, int slots, const at::Tensor& w0, int64_t dq_bytes,
                const at::Tensor* w1, const at::Tensor& grid, const at::Tensor& items, const at::Tensor& counts,
                const at::Tensor& members, at::Tensor& out, int n, int k, int experts, int pairs,
                int64_t max_items) {
  const int n_tiles = (n + BN - 1) / BN;
  const int64_t grid_x = max_items * n_tiles;
  if (grid_x < 1) return;
  auto* kern = iq2_soa_prefill_kernel<M, EPI>;
  constexpr int SMEM = PST * STAGE;
  static bool ready = false;
  if (!ready) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    ready = true;
  }
  auto* dq0 = reinterpret_cast<const __half*>(w0.data_ptr<uint8_t>());
  auto* qs0 = reinterpret_cast<const uint64_t*>(w0.data_ptr<uint8_t>() + dq_bytes);
  const __half* dq1 = nullptr;
  const uint64_t* qs1 = nullptr;
  if constexpr (M > 1) {
    dq1 = reinterpret_cast<const __half*>(w1->data_ptr<uint8_t>());
    qs1 = reinterpret_cast<const uint64_t*>(w1->data_ptr<uint8_t>() + dq_bytes);
  }
  kern<<<(unsigned)grid_x, THREADS, SMEM, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), x_stride, slots, dq0, qs0, dq1, qs1,
      grid.data_ptr<uint8_t>(), items.data_ptr<int>(), counts.data_ptr<int>(), members.data_ptr<int>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), n, k, experts, pairs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void gguf_iq2_soa_prefill_cuda(const at::Tensor& x, int64_t x_stride, int64_t slots, const at::Tensor& w,
                               int64_t dq_bytes, const at::Tensor& grid, const at::Tensor& items,
                               const at::Tensor& counts, const at::Tensor& members, at::Tensor& out, int64_t n,
                               int64_t k, int64_t experts, int64_t pairs, int64_t max_items) {
  const c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && out.is_cuda(), "iq2 soa prefill: CUDA tensors");
  TORCH_CHECK(k % 256 == 0 && n > 0 && max_items > 0, "iq2 soa prefill: geometry");
  launch_iq2<1, 3>(x, (int)x_stride, (int)slots, w, dq_bytes, nullptr, grid, items, counts, members, out, (int)n,
                   (int)k, (int)experts, (int)pairs, max_items);
}

void gguf_iq2_soa_gate_up_cuda(const at::Tensor& x, int64_t x_stride, int64_t slots, const at::Tensor& gate,
                               const at::Tensor& up, int64_t dq_bytes, const at::Tensor& grid,
                               const at::Tensor& items, const at::Tensor& counts, const at::Tensor& members,
                               at::Tensor& out, int64_t n, int64_t k, int64_t experts, int64_t pairs,
                               int64_t max_items) {
  const c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(x.is_cuda() && gate.is_cuda() && up.is_cuda() && out.is_cuda(), "iq2 gate_up: CUDA tensors");
  TORCH_CHECK(k % 256 == 0 && n > 0 && max_items > 0, "iq2 gate_up: geometry");
  launch_iq2<2, 2>(x, (int)x_stride, (int)slots, gate, dq_bytes, &up, grid, items, counts, members, out, (int)n,
                   (int)k, (int)experts, (int)pairs, max_items);
}
