// Gate 3: Idol's glue (idol.cu) with one fp16 rounding dropped each; the intervals hold idol-sdk's rounding points.
#pragma once

#include <cuda_fp16.h>
#include <math.h>

namespace tf_ctl_idol {
__device__ __forceinline__ float h(float x) { return __half2float(__float2half_rn(x)); }
__device__ __forceinline__ float ld(const __half* p) { return __half2float(*p); }
__device__ __forceinline__ float block_sum(float v, float* part) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = v;
    __syncthreads();
    float total = 0.0f;
    for (int w = 0; w < 8; ++w) total += part[w];
    return total;
}
}  // namespace tf_ctl_idol

// tf_idol_norm modulating the unrounded LayerNorm: fp16(fp16(y s1) + sh) where idol-sdk rounds y first.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_idol_norm_late(const __half* X, const float* W, const float* B,
                                                                        const __half* S1, const __half* SH, __half* Y,
                                                                        int width, float eps, int mode) {
    using namespace tf_ctl_idol;
    __shared__ float part[8];
    const __half* x = X + (size_t)blockIdx.x * width;
    float a = 0.0f, q = 0.0f;
    for (int i = threadIdx.x; i < width; i += 256) a += ld(x + i);
    const float mean = block_sum(a, part) / (float)width;
    for (int i = threadIdx.x; i < width; i += 256) q += (ld(x + i) - mean) * (ld(x + i) - mean);
    const float inv = 1.0f / sqrtf(block_sum(q, part) / (float)width + eps);
    for (int i = threadIdx.x; i < width; i += 256) {
        float v = (ld(x + i) - mean) * inv;
        v = mode == 1 ? h(v * W[i] + B[i]) : h(h(v * ld(S1 + i)) + ld(SH + i));
        Y[(size_t)blockIdx.x * width + i] = __float2half_rn(v);
    }
}

// tf_idol_gate_add with g y left in fp32: fp16(x + g y) where idol-sdk rounds g y to fp16 first.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_idol_gate_add_once(__half* X, const __half* Y, const __half* G,
                                                                            int width, int rows) {
    using namespace tf_ctl_idol;
    const int c = blockIdx.y * 256 + threadIdx.x, r = blockIdx.x;
    if (c >= width || r >= rows) return;
    const size_t i = (size_t)r * width + c;
    X[i] = __float2half_rn(ld(X + i) + ld(G + c) * ld(Y + i));
}

// tf_idol_bias taking GELU of the fp32 sum: fp16(GELU(acc + b)) where idol-sdk rounds the linear's output first.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_idol_bias_late(const float* ACC, const __half* B, __half* Y,
                                                                        int width, int rows, int act) {
    using namespace tf_ctl_idol;
    const int c = blockIdx.y * 256 + threadIdx.x, r = blockIdx.x;
    if (c >= width || r >= rows) return;
    const size_t i = (size_t)r * width + c;
    float v = ACC[i] + ld(B + c);
    if (act) v = 0.5f * v * (1.0f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v)));
    Y[i] = __float2half_rn(act ? v : h(v));
}
