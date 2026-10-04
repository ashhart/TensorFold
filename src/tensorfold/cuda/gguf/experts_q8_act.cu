// Q8_1 activation quantize + Q2_K SoA × Q8_1 grouped prefill (dp4a).
// TensorFold-owned; GGML Q8_1 / Q2_K math, SoA weight layout from prepare.py.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <torch/extension.h>

namespace {

constexpr int BM = 64;
constexpr int BN = 128;
constexpr int MT = 8;  // rows per block (Q2 tile reused across these)
constexpr int THREADS = 128;

struct alignas(4) BlockQ81 {
  __half d;
  __half s;
  int8_t qs[32];
};

static_assert(sizeof(BlockQ81) == 36, "block_q8_1");

struct Q2Col {
  uint32_t dm;
  uint8_t sc[16];
  uint8_t qs[64];
};

__device__ __forceinline__ float q2_q8_dot32(const Q2Col& w, int ib, const int8_t* __restrict__ xq, float xd) {
  const __half2 dm = *reinterpret_cast<const __half2*>(&w.dm);
  const float d = __half2float(__low2half(dm));
  const float m = __half2float(__high2half(dm));
  const int base = (ib < 4) ? 0 : 32;
  const int shift = 2 * (ib & 3);
  float sum_d = 0.f, sum_m = 0.f;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const uint8_t scb = w.sc[ib * 2 + h];
    const int sdl = scb & 15;
    const int smi = scb >> 4;
    int sumi = 0;
    int sum1 = 0;
#pragma unroll
    for (int j = 0; j < 16; j += 4) {
      const int j0 = h * 16 + j;
      int v = 0;
#pragma unroll
      for (int t = 0; t < 4; ++t) v |= ((w.qs[base + j0 + t] >> shift) & 3) << (8 * t);
      const int u = *reinterpret_cast<const int*>(xq + j0);
      sumi = __dp4a(v, u, sumi);
      sum1 = __dp4a(0x01010101, u, sum1);
    }
    sum_d += (float)(sumi * sdl);
    sum_m += (float)(sum1 * smi);
  }
  return xd * (d * sum_d - m * sum_m);
}

__global__ void quantize_q8_1_kernel(const __nv_bfloat16* __restrict__ x, BlockQ81* __restrict__ y, int M, int K) {
  const int row = (int)blockIdx.x;
  if (row >= M) return;
  const int nb = K / 32;
  for (int ib = (int)threadIdx.x; ib < nb; ib += (int)blockDim.x) {
    const __nv_bfloat16* src = x + (size_t)row * K + (size_t)ib * 32;
    float amax = 0.f;
    float vals[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      vals[i] = __bfloat162float(src[i]);
      amax = fmaxf(amax, fabsf(vals[i]));
    }
    const float d = amax / 127.f;
    const float id = d > 0.f ? 1.f / d : 0.f;
    int sum = 0;
    BlockQ81 blk;
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      int q = __float2int_rn(vals[i] * id);
      q = max(-128, min(127, q));
      blk.qs[i] = (int8_t)q;
      sum += q;
    }
    blk.d = __float2half(d);
    blk.s = __float2half(d * (float)sum);
    y[(size_t)row * nb + ib] = blk;
  }
}

// grid.x = items * n_tiles; grid.y = cdiv(BM, MT)
__global__ void __launch_bounds__(THREADS, 4) q2_soa_q8_prefill_kernel(
    const BlockQ81* __restrict__ X, int x_blocks, const uint32_t* __restrict__ dm, const uint8_t* __restrict__ sc,
    const uint8_t* __restrict__ qs, const int* __restrict__ items, const int* __restrict__ counts,
    const int* __restrict__ members, __nv_bfloat16* __restrict__ Y, int N, int K, int experts, int pairs) {
  __shared__ Q2Col sw[BN];
  __shared__ int8_t sqs[MT][32];
  __shared__ float sd[MT];
  __shared__ int spair[MT];
  __shared__ int scount;

  const int n_tiles = (N + BN - 1) / BN;
  const int it = (int)blockIdx.x / n_tiles;
  const int nt = (int)blockIdx.x - it * n_tiles;
  const int m0 = (int)blockIdx.y * MT;
  if (it >= __ldg(counts)) return;

  const int expert = __ldg(items + 3 * it);
  const int first = __ldg(items + 3 * it + 1);
  const int count = min(max(__ldg(items + 3 * it + 2), 0), BM);
  if (expert < 0 || expert >= experts || m0 >= count) return;

  const int n0 = nt * BN;
  const int tid = (int)threadIdx.x;
  const int col = n0 + tid;
  const int kg = K / 256;
  const int nlocal = min(MT, count - m0);

  if (tid == 0) scount = nlocal;
  if (tid < MT) {
    int pair = -1;
    if (tid < nlocal) {
      pair = __ldg(members + first + m0 + tid);
      if ((unsigned)pair >= (unsigned)pairs) pair = -1;
    }
    spair[tid] = pair;
  }
  __syncthreads();

  float acc[MT];
#pragma unroll
  for (int t = 0; t < MT; ++t) acc[t] = 0.f;

  for (int kb = 0; kb < kg; ++kb) {
    // Stage one Q2_K block for all BN columns.
    for (int c = tid; c < BN; c += THREADS) {
      const int ncol = n0 + c;
      if (ncol < N) {
        const int64_t blk = ((int64_t)expert * N + ncol) * kg + kb;
        sw[c].dm = dm[blk];
#pragma unroll
        for (int b = 0; b < 16; ++b) sw[c].sc[b] = sc[blk * 16 + b];
#pragma unroll
        for (int b = 0; b < 64; ++b) sw[c].qs[b] = qs[blk * 64 + b];
      } else {
        sw[c].dm = 0;
#pragma unroll
        for (int b = 0; b < 16; ++b) sw[c].sc[b] = 0;
#pragma unroll
        for (int b = 0; b < 64; ++b) sw[c].qs[b] = 0;
      }
    }
    __syncthreads();

#pragma unroll
    for (int ib = 0; ib < 8; ++ib) {
      for (int t = tid; t < MT; t += THREADS) {
        const int pair = spair[t];
        if (pair >= 0) {
          const BlockQ81 xb = X[(size_t)pair * (size_t)x_blocks + kb * 8 + ib];
          sd[t] = __half2float(xb.d);
#pragma unroll
          for (int i = 0; i < 32; ++i) sqs[t][i] = xb.qs[i];
        } else {
          sd[t] = 0.f;
#pragma unroll
          for (int i = 0; i < 32; ++i) sqs[t][i] = 0;
        }
      }
      __syncthreads();

      if (col < N) {
        const Q2Col w = sw[tid];
        const int nact = scount;
#pragma unroll
        for (int t = 0; t < MT; ++t) {
          if (t >= nact) break;
          if (spair[t] < 0) continue;
          acc[t] += q2_q8_dot32(w, ib, sqs[t], sd[t]);
        }
      }
      __syncthreads();
    }
  }

  if (col < N) {
    const int nact = scount;
#pragma unroll
    for (int t = 0; t < MT; ++t) {
      if (t >= nact) break;
      const int pair = spair[t];
      if (pair < 0) continue;
      Y[(size_t)pair * N + col] = __float2bfloat16_rn(acc[t]);
    }
  }
}

}  // namespace

void gguf_quantize_q8_1_cuda(const at::Tensor& x, at::Tensor& y) {
  const c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(x.is_cuda() && y.is_cuda(), "quantize_q8_1: CUDA");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16, "quantize_q8_1: bf16 x");
  TORCH_CHECK(y.scalar_type() == at::kByte, "quantize_q8_1: uint8 y");
  TORCH_CHECK(x.dim() == 2 && x.size(1) % 32 == 0, "quantize_q8_1: geometry");
  const int M = (int)x.size(0);
  const int K = (int)x.size(1);
  const int nb = K / 32;
  TORCH_CHECK(y.numel() == (int64_t)M * nb * (int64_t)sizeof(BlockQ81), "quantize_q8_1: y size");
  if (M < 1) return;
  quantize_q8_1_kernel<<<M, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<BlockQ81*>(y.data_ptr()), M, K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gguf_q2_soa_q8_prefill_cuda(const at::Tensor& x8, int64_t slots, const at::Tensor& w, int64_t dm_bytes,
                                 int64_t sc_bytes, const at::Tensor& items, const at::Tensor& counts,
                                 const at::Tensor& members, at::Tensor& out, int64_t n, int64_t k, int64_t experts,
                                 int64_t pairs, int64_t max_items) {
  const c10::cuda::CUDAGuard guard(x8.device());
  TORCH_CHECK(x8.is_cuda() && w.is_cuda() && out.is_cuda(), "q2 q8 prefill: CUDA");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kByte, "q2 q8 prefill: dtypes");
  TORCH_CHECK(k % 256 == 0 && n > 0 && max_items > 0, "q2 q8 prefill: geometry");
  (void)slots;
  const int n_tiles = (int)((n + BN - 1) / BN);
  const int64_t grid_x = max_items * n_tiles;
  if (grid_x < 1) return;
  const int x_blocks = (int)(k / 32);
  auto* base = w.data_ptr<uint8_t>();
  auto* dm = reinterpret_cast<const uint32_t*>(base);
  auto* sc = base + dm_bytes;
  auto* qs = base + dm_bytes + sc_bytes;
  dim3 grid((unsigned)grid_x, (unsigned)((BM + MT - 1) / MT));
  q2_soa_q8_prefill_kernel<<<grid, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const BlockQ81*>(x8.data_ptr()), x_blocks, dm, sc, qs, items.data_ptr<int>(),
      counts.data_ptr<int>(), members.data_ptr<int>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), (int)n,
      (int)k, (int)experts, (int)pairs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
