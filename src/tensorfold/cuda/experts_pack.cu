// experts.py's pack in one pass: each MLX word, scale and bias read once and written to its block slot

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <stdint.h>
#include <torch/extension.h>

namespace {

// nibbles (i0 .. i7) -> (i0, i2, i4, i6, i1, i3, i5, i7): what experts.py's unpack undoes
__device__ __forceinline__ uint32_t shuffle(uint32_t w) {
  uint32_t even = w & 0x0F0F0F0Fu, odd = (w >> 4) & 0x0F0F0F0Fu;
  even = (even | (even >> 4)) & 0x00FF00FFu;
  odd = (odd | (odd >> 4)) & 0x00FF00FFu;
  even = (even | (even >> 8)) & 0x0000FFFFu;
  odd = (odd | (odd >> 8)) & 0x0000FFFFu;
  return even | (odd << 16);
}

// a thread block per (group, column block, expert): WORDS weight words (consecutive threads, consecutive words), then 32 scale/bias words
template <int H>   // group size / 32
__global__ void __launch_bounds__(32 * 4 * H + 32)
pack_kernel(const uint32_t* __restrict__ words, const uint16_t* __restrict__ scales,
            const uint16_t* __restrict__ biases, uint32_t* __restrict__ out, int n, int k8, int kg, int nb) {
  constexpr int WORDS = 32 * 4 * H;               // a block's weight words: NTW * H words a lane
  const int g = blockIdx.x, b = blockIdx.y, e = blockIdx.z, i = threadIdx.x;
  uint32_t* dst = out + ((static_cast<int64_t>(e) * nb + b) * kg + g) * (WORDS + 32);
  if (i < WORDS) {
    const int row = i / (4 * H), kk = i % (4 * H);                 // row = t * 8 + r; kk = q * H + j
    const int t = row / 8, r = row % 8, q = kk / H, j = kk % H;
    const uint32_t w = words[(static_cast<int64_t>(e) * n + b * 32 + row) * k8 + g * 4 * H + kk];
    const int f = ((t * H + j) * 8 + r) * 4 + q;                   // (t, j, r, q) in order ...
    dst[(f / 128) * 128 + (f % 32) * 4 + (f / 32) % 4] = shuffle(w);  // ... as (f/128, f%32, f/32%4)
  } else {
    const int s = i - WORDS;                                       // p * 8 + (0 scales, 1 biases) * 4 + t
    const int p = s / 8, t = s % 4;
    const uint16_t* src = (s / 4) % 2 ? biases : scales;
    const int64_t at = (static_cast<int64_t>(e) * n + b * 32 + t * 8 + p * 2) * kg + g;
    dst[WORDS + s] = static_cast<uint32_t>(src[at]) | (static_cast<uint32_t>(src[at + kg]) << 16);
  }
}

}  // namespace

void experts_pack_cuda(const at::Tensor& words, const at::Tensor& scales, const at::Tensor& biases, int64_t gs,
                       at::Tensor& out) {
  const at::cuda::CUDAGuard guard(words.device());
  const int e = words.size(0), n = words.size(1), k8 = words.size(2), kg = scales.size(2), nb = n / 32;
  const dim3 grid(kg, nb, e);
  const auto stream = at::cuda::getCurrentCUDAStream();
  const auto* w = reinterpret_cast<const uint32_t*>(words.data_ptr());
  const auto* s = reinterpret_cast<const uint16_t*>(scales.data_ptr());
  const auto* bi = reinterpret_cast<const uint16_t*>(biases.data_ptr());
  auto* o = reinterpret_cast<uint32_t*>(out.data_ptr());
  if (gs == 32) {
    pack_kernel<1><<<grid, 32 * 4 + 32, 0, stream>>>(w, s, bi, o, n, k8, kg, nb);
  } else {
    pack_kernel<2><<<grid, 32 * 8 + 32, 0, stream>>>(w, s, bi, o, n, k8, kg, nb);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
