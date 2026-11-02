// Expert blocks first for Qwen's 16-row router; each dot retains the original FMA chain.
#pragma once
#include <cuda_bf16.h>
#include <stdint.h>
namespace tf_moe_expert_first {
constexpr int TILE = 16, KC = 256;
template <bool BadRows>
__device__ __forceinline__ void router(const __nv_bfloat16* __restrict__ X, int x_stride,
                                                                  const __nv_bfloat16* __restrict__ W, float* __restrict__ OUT,
                                                                  int R, int D, int E, const int* __restrict__ P) {
    if (P != nullptr) R = min(R, P[2]);
    if (static_cast<int>(blockIdx.y) * TILE >= R) return;
    __shared__ __nv_bfloat16 xs[TILE][KC + 2];
    __shared__ __nv_bfloat16 ws[TILE][KC + 2];
    const int row0 = blockIdx.y * TILE, e0 = blockIdx.x * TILE;
    const int rr = threadIdx.x / TILE, ee = threadIdx.x % TILE;
    float acc = 0.0f;
    for (int k0 = 0; k0 < D; k0 += KC) {
        __syncthreads();
        for (int c = threadIdx.x; c < TILE * KC; c += 256) {
            const int r = c / KC, k = c % KC;
            const bool kin = k0 + k < D;
            xs[r][k] = kin && row0 + r < R ? X[static_cast<int64_t>(row0 + (BadRows ? 0 : r)) * x_stride + k0 + k] : __float2bfloat16_rn(0.0f);
            ws[r][k] = kin && e0 + r < E ? W[static_cast<int64_t>(e0 + r) * D + k0 + k] : __float2bfloat16_rn(0.0f);
        }
        __syncthreads();
        const int kn = min(KC, D - k0);
        for (int k = 0; k < kn; ++k) acc = __fmaf_rn(__bfloat162float(xs[rr][k]), __bfloat162float(ws[ee][k]), acc);
    }
    if (row0 + rr < R && e0 + ee < E) OUT[static_cast<int64_t>(row0 + rr) * E + e0 + ee] = acc;
}

} // namespace tf_moe_expert_first
#define TF_MOE_EXPERT_ENTRY(NAME, BAD) \
extern "C" __global__ void __launch_bounds__(256) NAME( \
    const __nv_bfloat16* x, int stride, const __nv_bfloat16* w, float* out, int rows, int dims, int experts, const int* p) { \
    tf_moe_expert_first::router<BAD>(x, stride, w, out, rows, dims, experts, p); \
}
TF_MOE_EXPERT_ENTRY(tf_moe_router_expert_first, false)
#ifdef TF_MOE_ROUTER_CONTROLS
TF_MOE_EXPERT_ENTRY(tf_moe_router_expert_bad_rows, true)
#endif
#undef TF_MOE_EXPERT_ENTRY
