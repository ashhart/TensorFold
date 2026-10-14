// LFM2-VL's vision glue (families/lfm2/vision.zig): SigLIP2's LayerNorm, GELU and residual adds, pixel unshuffle.

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

#include "glue_math.cuh"

using namespace tf_glue;

namespace {

constexpr int NT = 256;    // threads a row block
constexpr int MAXE = 8;    // elements a thread holds: LayerNorm rows up to NT * MAXE = 2048 wide

// A block's values summed: warp butterflies, then the warps in order; every thread gets the sum.
__device__ __forceinline__ float block_sum(float v, float* red) {
    v = warp_sum(v);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float s = red[0];
    for (int w = 1; w < NT / 32; ++w) s = __fadd_rn(s, red[w]);
    __syncthreads();
    return s;
}

// torch's GELU kernels in fp32: the tanh approximation, and the exact form through erf.
__device__ __forceinline__ float gelu_tanh(float x) {
    constexpr float beta = M_SQRT2 * M_2_SQRTPI * 0.5;
    constexpr float kappa = 0.044715f;
    const float cube = __fmul_rn(__fmul_rn(x, x), x);
    const float inner = __fmul_rn(beta, __fadd_rn(x, __fmul_rn(kappa, cube)));
    return __fmul_rn(__fmul_rn(0.5f, x), __fadd_rn(1.0f, tanhf(inner)));
}

__device__ __forceinline__ float gelu_erf(float x) {
    constexpr float alpha = M_SQRT1_2;
    return __fmul_rn(__fmul_rn(x, 0.5f), __fadd_rn(1.0f, erff(__fmul_rn(x, alpha))));
}

}  // namespace

// Block a row: y = bf16((x - mean) * rstd * w + b), mean and variance in two fixed-order passes.
extern "C" __global__ void __launch_bounds__(NT) tf_lfm2v_layer_norm(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ W,
                                                                       const __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ Y, int D,
                                                                       float eps) {
    __shared__ float red[NT / 32];
    const int64_t at = static_cast<int64_t>(blockIdx.x) * D;
    float x[MAXE];
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < MAXE; ++k) {
        const int i = threadIdx.x + k * NT;
        if (i >= D) continue;
        x[k] = bf(X[at + i]);
        acc = __fadd_rn(acc, x[k]);
    }
    const float mean = __fdiv_rn(block_sum(acc, red), static_cast<float>(D));
    acc = 0.0f;
#pragma unroll
    for (int k = 0; k < MAXE; ++k) {
        if (threadIdx.x + k * NT >= D) continue;
        x[k] = __fsub_rn(x[k], mean);
        acc = __fadd_rn(acc, __fmul_rn(x[k], x[k]));
    }
    const float var = __fdiv_rn(block_sum(acc, red), static_cast<float>(D));
    const float rstd = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(var, eps)));
#pragma unroll
    for (int k = 0; k < MAXE; ++k) {
        const int i = threadIdx.x + k * NT;
        if (i < D) Y[at + i] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(__fmul_rn(x[k], rstd), bf(W[i])), bf(B[i])));
    }
}

// Block (row, column block): o = bf16(acc + bias), for a Linear whose cuBLASLt algorithm cannot fuse its bias.
extern "C" __global__ void __launch_bounds__(NT) tf_lfm2v_bias(const float* __restrict__ ACC, const __nv_bfloat16* __restrict__ BIAS,
                                                                 __nv_bfloat16* __restrict__ O, int N) {
    const int c = blockIdx.y * NT + threadIdx.x;
    if (c >= N) return;
    const int64_t at = static_cast<int64_t>(blockIdx.x) * N + c;
    O[at] = __float2bfloat16_rn(__fadd_rn(ACC[at], bf(BIAS[c])));
}

// A Linear's bf16 output, its bias added: bf16(gelu(x)) (act 1 tanh, 2 erf), then bf16(r + x) when R.
extern "C" __global__ void __launch_bounds__(NT) tf_lfm2v_pointwise(const __nv_bfloat16* X, const __nv_bfloat16* R, __nv_bfloat16* O,
                                                                      int64_t count, int act) {
    const int64_t at = static_cast<int64_t>(blockIdx.x) * NT + threadIdx.x;
    if (at >= count) return;
    float v = bf(X[at]);
    if (act == 1) v = rbf(gelu_tanh(v));
    if (act == 2) v = rbf(gelu_erf(v));
    O[at] = __float2bfloat16_rn(R ? __fadd_rn(bf(R[at]), v) : v);
}

// Block an output row (i, j) of an h x w grid: rows (2i, 2j), (2i, 2j+1), (2i+1, 2j), (2i+1, 2j+1) side by side.
extern "C" __global__ void __launch_bounds__(NT) tf_lfm2v_unshuffle(const __nv_bfloat16* __restrict__ X, __nv_bfloat16* __restrict__ Y, int w,
                                                                      int C) {
    const int o = blockIdx.x, i = o / (w / 2), j = o % (w / 2);
    for (int e = threadIdx.x; e < 4 * C; e += NT) {
        const int q = e / C;
        const int64_t src = static_cast<int64_t>(2 * i + q / 2) * w + 2 * j + q % 2;
        Y[static_cast<int64_t>(o) * 4 * C + e] = X[src * C + e % C];
    }
}
