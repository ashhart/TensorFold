// Staged Q2_K SoA grouped prefill — D2R B fragments + b16-prompt A ldmatrix/MMA.
// SoA: [uint32 dm[nblk]][pad64][uint8 sc[nblk*16]][pad64][uint8 qs[nblk*64]].
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <torch/extension.h>

#include "../experts.cuh"

namespace {

constexpr int BM = 64;
constexpr int BN = 128;
constexpr int WM = 2;
constexpr int WN = 4;
constexpr int KS = 64;
constexpr int PST = 2;
constexpr int PROW = 128;
constexpr int THREADS = WM * WN * 32;
constexpr int STAGE = BM * PROW;
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

__device__ __forceinline__ void cp4z(void* dst, const void* src, bool ok) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(sm(dst)), "l"(src), "r"(ok ? 4 : 0));
}

__device__ __forceinline__ int pswz(int r, int c) { return r * PROW + ((c ^ (r & 7)) << 4); }

__device__ __forceinline__ uint32_t pack2(float a, float b) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<const uint32_t*>(&v);
}

struct Q2Cache {
  uint32_t dm[BN];
  uint8_t sc[BN * 16];
  uint8_t qs[BN * 80];
};

__device__ __forceinline__ float q2_at(const Q2Cache& w, int col, int j) {
  const uint8_t sc = w.sc[col * 16 + (j >> 4)];
  const uint8_t bits = w.qs[col * 80 + (j >> 7) * 32 + (j & 31)];
  const int q = (bits >> (2 * ((j & 127) >> 5))) & 3;
  const __half2 dm = *reinterpret_cast<const __half2*>(&w.dm[col]);
  const float d = __half2float(__low2half(dm));
  const float m = __half2float(__high2half(dm));
  return (d * (float)(sc & 15)) * (float)q - m * (float)(sc >> 4);
}

__device__ __forceinline__ void q2_bfrag(const Q2Cache& w, int col, int k0, int c, uint32_t& b0, uint32_t& b1) {
  // Match Triton's byte-derived kWidth=4 fragments: two k16 MMAs cover
  // alternate pairs in one k32 span, preserving the FP32 accumulator order.
  const int j0 = (k0 & ~31) + ((k0 >> 4) & 1) * 2 + 4 * c;
  b0 = pack2(q2_at(w, col, j0), q2_at(w, col, j0 + 1));
  b1 = pack2(q2_at(w, col, j0 + 16), q2_at(w, col, j0 + 17));
}

__global__ void __launch_bounds__(THREADS, 2) q2_soa_prefill_kernel(
    const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint32_t* __restrict__ dm,
    const uint8_t* __restrict__ sc, const uint8_t* __restrict__ qs, const int* __restrict__ items,
    const int* __restrict__ counts, const int* __restrict__ members, __nv_bfloat16* __restrict__ Y, int N, int K,
    int experts, int pairs) {
  extern __shared__ __align__(128) unsigned char pbuf[];
  __shared__ Q2Cache wcache[2];

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

  float acc[MI][NT][4];
#pragma unroll
  for (int i = 0; i < MI; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

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
    Q2Cache& dst = wcache[kb & 1];
    for (int c = tid; c < BN; c += THREADS) {
      const int n = n0 + c;
      const bool ok = n < N && kb < kg;
      const int64_t blk = ((int64_t)expert * N + n) * kg + kb;
      cp4z(dst.dm + c, ok ? dm + blk : dm, ok);
      cpz(dst.sc + c * 16, ok ? sc + blk * 16 : sc, ok);
    }
    for (int q = tid; q < BN * 4; q += THREADS) {
      const int c = q >> 2, part = q & 3, n = n0 + c;
      const bool ok = n < N && kb < kg;
      const int64_t blk = ((int64_t)expert * N + n) * kg + kb;
      cpz(dst.qs + c * 80 + part * 16, ok ? qs + blk * 64 + part * 16 : qs, ok);
    }
  };
  auto stage = [&](int s, int kt) {
    load_x(s, kt);
    if ((kt & 3) == 0) load_w(kt >> 2);
  };
  for (int s = 0; s < PST - 1; ++s) {
    if (s < KT) stage(s, s);
    cp_commit();
  }

  for (int kt = 0; kt < KT; ++kt) {
    cp_wait<PST - 2>();
    __syncthreads();
    if (kt + PST - 1 < KT) stage((kt + PST - 1) % PST, kt + PST - 1);
    cp_commit();

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
      for (int i = 0; i < MI; ++i) {
        const int row = wm * (BM / WM) + i * 16 + (lane >> 2);
        const int k = (ks >> 1) * 32 + (ks & 1) * 2 + 4 * c;
        a[i][0] = *reinterpret_cast<const uint32_t*>(p + pswz(row, k / 8) + (k % 8) * 2);
        a[i][1] = *reinterpret_cast<const uint32_t*>(p + pswz(row + 8, k / 8) + (k % 8) * 2);
        a[i][2] = *reinterpret_cast<const uint32_t*>(p + pswz(row, (k + 16) / 8) + (k % 8) * 2);
        a[i][3] = *reinterpret_cast<const uint32_t*>(p + pswz(row + 8, (k + 16) / 8) + (k % 8) * 2);
      }
#pragma unroll
      for (int j = 0; j < NT; ++j) {
        const int ncol = n0 + wn * 32 + j * 8 + (lane >> 2);
        uint32_t b0 = 0, b1 = 0;
        if ((unsigned)ncol < (unsigned)N) q2_bfrag(wcache[(kt >> 2) & 1], ncol - n0, k0, c, b0, b1);
#pragma unroll
        for (int i = 0; i < MI; ++i) mma(acc[i][j], a[i][0], a[i][1], a[i][2], a[i][3], b0, b1);
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
        const int col = n0 + wn * 32 + j * 8 + (lane & 3) * 2;
        if (row >= count) continue;
        const int pair = __ldg(members + first + row);
        if ((unsigned)pair >= (unsigned)pairs) continue;
        if (col < N) Y[(size_t)pair * N + col] = __float2bfloat16_rn(acc[i][j][2 * h]);
        if (col + 1 < N) Y[(size_t)pair * N + col + 1] = __float2bfloat16_rn(acc[i][j][2 * h + 1]);
      }
}

}  // namespace

void gguf_q2_soa_prefill_cuda(const at::Tensor& x, int64_t x_stride, int64_t slots, const at::Tensor& w,
                              int64_t dm_bytes, int64_t sc_bytes, const at::Tensor& items, const at::Tensor& counts,
                              const at::Tensor& members, at::Tensor& out, int64_t n, int64_t k, int64_t experts,
                              int64_t pairs, int64_t max_items) {
  const c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && out.is_cuda(), "q2 soa prefill: CUDA tensors");
  TORCH_CHECK(k % 256 == 0 && n > 0 && max_items > 0, "q2 soa prefill: geometry");
  const int n_tiles = (int)((n + BN - 1) / BN);
  const int64_t grid_x = max_items * n_tiles;
  if (grid_x < 1) return;

  auto* kern = q2_soa_prefill_kernel;
  constexpr int SMEM = PST * STAGE;
  static bool ready = false;
  if (!ready) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    ready = true;
  }
  auto* base = w.data_ptr<uint8_t>();
  auto* dm = reinterpret_cast<const uint32_t*>(base);
  auto* sc = base + dm_bytes;
  auto* qs = base + dm_bytes + sc_bytes;
  kern<<<(unsigned)grid_x, THREADS, SMEM, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), (int)x_stride, (int)slots, dm, sc, qs,
      items.data_ptr<int>(), counts.data_ptr<int>(), members.data_ptr<int>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), (int)n, (int)k, (int)experts, (int)pairs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Keep raw-path symbol for ABI; unused once SoA is default.
void gguf_q2_prefill_cuda(const at::Tensor& x, int64_t x_stride, int64_t slots, const at::Tensor& w,
                          const at::Tensor& items, const at::Tensor& counts, const at::Tensor& members,
                          at::Tensor& out, int64_t n, int64_t k, int64_t experts, int64_t pairs,
                          int64_t max_items) {
  TORCH_CHECK(false, "q2 raw prefill retired; use Q2 SoA prepare");
  (void)x;
  (void)x_stride;
  (void)slots;
  (void)w;
  (void)items;
  (void)counts;
  (void)members;
  (void)out;
  (void)n;
  (void)k;
  (void)experts;
  (void)pairs;
  (void)max_items;
}
