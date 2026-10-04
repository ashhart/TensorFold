// Lossless GGUF expert repacks for TensorFold grouped prefill.
// IQ2_XXS SoA: [half dq[nblk]][pad64][uint64 qs words as 8×u64 per block].
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <stdint.h>
#include <torch/extension.h>

namespace {

__global__ void pack_bn_kernel(const uint8_t* __restrict__ raw, uint8_t* __restrict__ out, int k, int n, int e,
                               int block, int bn) {
  const int kg = k / 256;
  const int nb = (n + bn - 1) / bn;
  const size_t tiles = (size_t)e * nb * kg;
  for (size_t tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
    const int t = (int)tile;
    const int kg_i = t % kg;
    const int tmp = t / kg;
    const int nb_i = tmp % nb;
    const int e_i = tmp / nb;
    uint8_t* dst = out + tile * (size_t)bn * block;
    for (int c = threadIdx.x; c < bn; c += blockDim.x) {
      const int n_i = nb_i * bn + c;
      uint8_t* d = dst + c * block;
      if (n_i >= n) {
        for (int b = 0; b < block; ++b) d[b] = 0;
        continue;
      }
      const uint8_t* src = raw + (((size_t)e_i * n + n_i) * kg + kg_i) * block;
      for (int b = 0; b < block; ++b) d[b] = src[b];
    }
  }
}

// One thread per (block, 8-byte qs group). p==0 also writes the half scale.
__global__ void pack_iq2_soa_kernel(const uint8_t* __restrict__ raw, uint16_t* __restrict__ dq,
                                    uint64_t* __restrict__ qs, uint64_t nblk) {
  const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nblk * 8ull) return;
  const uint64_t blk = i >> 3;
  const uint32_t p = (uint32_t)(i & 7u);
  const uint8_t* src = raw + blk * 66ull;
  if (p == 0u) dq[blk] = (uint16_t)src[0] | ((uint16_t)src[1] << 8);
  const uint8_t* q = src + 2u + (uint64_t)p * 8u;
  uint64_t v = 0;
  for (int b = 0; b < 8; ++b) v |= (uint64_t)q[b] << (8 * b);
  qs[blk * 8ull + p] = v;
}

// Q2_K SoA: [uint32 dm[nblk]][pad64][uint8 sc[nblk*16]][pad64][uint8 qs[nblk*64]].
// One thread per block; p walks the 84-byte raw record.
__global__ void pack_q2_soa_kernel(const uint8_t* __restrict__ raw, uint32_t* __restrict__ dm, uint8_t* __restrict__ sc,
                                   uint8_t* __restrict__ qs, uint64_t nblk) {
  const uint64_t blk = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (blk >= nblk) return;
  const uint8_t* src = raw + blk * 84ull;
  dm[blk] = (uint32_t)src[80] | ((uint32_t)src[81] << 8) | ((uint32_t)src[82] << 16) | ((uint32_t)src[83] << 24);
#pragma unroll
  for (int b = 0; b < 16; ++b) sc[blk * 16ull + b] = src[b];
#pragma unroll
  for (int b = 0; b < 64; ++b) qs[blk * 64ull + b] = src[16 + b];
}

}  // namespace

void gguf_pack_cuda(const at::Tensor& raw, at::Tensor& out, int64_t k, int64_t n, int64_t e, int64_t block,
                    int64_t bn) {
  const c10::cuda::CUDAGuard guard(raw.device());
  const int kg = (int)(k / 256);
  const int nb = (int)((n + bn - 1) / bn);
  const size_t tiles = (size_t)e * nb * kg;
  const int grid = (int)std::min(tiles, (size_t)4096);
  pack_bn_kernel<<<grid, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      raw.data_ptr<uint8_t>(), out.data_ptr<uint8_t>(), (int)k, (int)n, (int)e, (int)block, (int)bn);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gguf_pack_iq2_soa_cuda(const at::Tensor& raw, at::Tensor& out, int64_t nblk, int64_t dq_bytes) {
  const c10::cuda::CUDAGuard guard(raw.device());
  auto* base = out.data_ptr<uint8_t>();
  auto* dq = reinterpret_cast<uint16_t*>(base);
  auto* qs = reinterpret_cast<uint64_t*>(base + dq_bytes);
  const uint64_t threads = (uint64_t)nblk * 8ull;
  const int block = 256;
  const int grid = (int)((threads + block - 1) / block);
  pack_iq2_soa_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(raw.data_ptr<uint8_t>(), dq, qs,
                                                                             (uint64_t)nblk);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gguf_pack_q2_soa_cuda(const at::Tensor& raw, at::Tensor& out, int64_t nblk, int64_t dm_bytes, int64_t sc_bytes) {
  const c10::cuda::CUDAGuard guard(raw.device());
  auto* base = out.data_ptr<uint8_t>();
  auto* dm = reinterpret_cast<uint32_t*>(base);
  auto* sc = base + dm_bytes;
  auto* qs = base + dm_bytes + sc_bytes;
  const int block = 256;
  const int grid = (int)((nblk + block - 1) / block);
  pack_q2_soa_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(raw.data_ptr<uint8_t>(), dm, sc, qs,
                                                                            (uint64_t)nblk);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
