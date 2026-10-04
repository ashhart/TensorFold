// Device code of src/tensorfold/cuda/experts_pack.cu (lines 1-42, comments and ATen includes dropped), checked by zig/tests/cuda/copies.py.

#include <stdint.h>

namespace tf_experts_pack {

__device__ __forceinline__ uint32_t shuffle(uint32_t w) {
  uint32_t even = w & 0x0F0F0F0Fu, odd = (w >> 4) & 0x0F0F0F0Fu;
  even = (even | (even >> 4)) & 0x00FF00FFu;
  odd = (odd | (odd >> 4)) & 0x00FF00FFu;
  even = (even | (even >> 8)) & 0x0000FFFFu;
  odd = (odd | (odd >> 8)) & 0x0000FFFFu;
  return even | (odd << 16);
}

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

} // namespace tf_experts_pack

// Groups of 64 inputs.
template __global__ void tf_experts_pack::pack_kernel<2>(const uint32_t*, const uint16_t*, const uint16_t*,
    uint32_t*, int, int, int, int);
