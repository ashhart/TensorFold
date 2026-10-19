// Torch pointwise ops of DeepSeek-V4.1's fast prefill MoE, bit-exact: fp32 opmath, NaN-propagating clamps.

// swiglu: act = silu(min(g, limit)) * clamp(u, -limit, limit); add: out += b; add_bf16: y = bf16_rn(a + b).

// Built with --fmad=false: torch runs each op as its own kernel with fp32 rounding, so nothing may contract.

#include <cstdint>
#include <cuda_bf16.h>

namespace dsv41_pointwise {

// torch's clamp_max / clamp: a NaN stays NaN
__device__ __forceinline__ float clamp_max(float v, float hi) { return isnan(v) ? v : fminf(v, hi); }
__device__ __forceinline__ float clamp(float v, float lo, float hi) { return isnan(v) ? v : fminf(fmaxf(v, lo), hi); }
// torch's silu (ActivationSiluKernel.cu): x / (1 + exp(-x)) in fp32
__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }

// g [R, ldg], u [R, ldu], act [R, lda]; n columns a row
__global__ void __launch_bounds__(256) swiglu_kernel(const float* __restrict__ g, long long ldg,
                                                     const float* __restrict__ u, long long ldu,
                                                     float* __restrict__ act, long long lda, int n, float limit) {
    const int r = blockIdx.y;
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < n; c += gridDim.x * blockDim.x) {
        const float a = silu(clamp_max(g[r * ldg + c], limit));
        act[r * lda + c] = a * clamp(u[r * ldu + c], -limit, limit);
    }
}

__global__ void __launch_bounds__(256) add_kernel(float* __restrict__ out, const float* __restrict__ b, long long n) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
        out[i] = out[i] + b[i];
}

__global__ void __launch_bounds__(256) add_bf16_kernel(const float* __restrict__ a, const float* __restrict__ b,
                                                       __nv_bfloat16* __restrict__ y, long long n) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
        y[i] = __float2bfloat16_rn(a[i] + b[i]);
}

}  // namespace dsv41_pointwise
