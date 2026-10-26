// TensorFold v0.6.0 FP32 fused-output to row-scaled e4m3 prefill glue, ported to the native CUDA driver (Apache-2.0).
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

namespace {
__device__ float bf(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

// Identical stored-fragment permutation to qmm.py::_quantize_rows.
__device__ int source(int m) {
    const int j = m % 4;
    return (m / 32) * 32 + ((m % 32) / 16) * 16 + ((m % 16) / 4) * 2 + (j % 2) + (j / 2) * 8;
}

// All 256 threads participate. Caller must not read scratch until the barrier after return.
__device__ float reduce(float v, float* scratch, bool maximum) {
    const int t = threadIdx.x;
    scratch[t] = v;
    __syncthreads();
    for (int step = 128; step; step >>= 1) {
        if (t < step) scratch[t] = maximum ? fmaxf(scratch[t], scratch[t + step]) : scratch[t] + scratch[t + step];
        __syncthreads();
    }
    return scratch[0];
}
}

// Modes: add+rmsnorm, SwiGLU, GDN norm, attention gate and qualification-only BF16; Y is the paired input and W a norm.
extern "C" __global__ void __launch_bounds__(256) tf_q35_prefill8_prepare(
    __nv_bfloat16* X, const __nv_bfloat16* Y, const __nv_bfloat16* W,
    uint8_t* X8, __nv_bfloat16* XS, float* A, int K, int mode, float eps) {
    extern __shared__ float values[]; // K entries plus 256 reduction entries; budgeted on the host
    float* red = values + K;
    const int t = threadIdx.x;
    const size_t row = blockIdx.x;
    float ss = 0.0f;
    for (int i = t; i < K; i += 256) {
        float x = bf(X[row * K + i]);
        if (mode == 0 && Y != nullptr) {
            const auto h = __float2bfloat16_rn(x + bf(Y[row * K + i]));
            X[row * K + i] = h;
            x = bf(h);
        }
        values[i] = x;
        if (mode == 0) ss += x * x;
    }
    __syncthreads();
    if (mode == 0) {
        const float sum = reduce(ss, red, false);
        const float inv = 1.0f / sqrtf(sum / static_cast<float>(K) + eps);
        __syncthreads();
        for (int i = t; i < K; i += 256) values[i] = values[i] * inv * bf(W[i]);
    } else if (mode == 1) {
        for (int i = t; i < K; i += 256) values[i] = values[i] * sigmoid(values[i]) * bf(Y[row * K + i]);
    } else if (mode == 2) {
        // Each head's reduction is independent, and uses original BF16 y.
        for (int h = t; h < K / 128; h += 256) {
            float sum = 0.0f;
            for (int d = 0; d < 128; ++d) { const float v = values[h * 128 + d]; sum += v * v; }
            red[h] = 1.0f / sqrtf(sum / 128.0f + eps);
        }
        __syncthreads();
        for (int i = t; i < K; i += 256) {
            const float z = bf(Y[row * K + i]);
            values[i] = z * sigmoid(z) * (values[i] * red[i / 128] * bf(W[i % 128]));
        }
    } else if (mode == 3) {
        for (int i = t; i < K; i += 256) {
            const size_t gate = row * (2 * K) + (i / 256) * 512 + 256 + i % 256;
            values[i] *= sigmoid(bf(Y[gate]));
        }
    }
    __syncthreads();
    float amax = 0.0f;
    for (int i = t; i < K; i += 256) amax = fmaxf(amax, fabsf(values[i]));
    const float a = fmaxf(reduce(amax, red, true), 1e-30f) / 448.0f;
    if (t == 0) A[row] = a;
    __syncthreads();
    for (int m = t; m < K; m += 256) {
        const float v = values[source(m)] / a;
        X8[row * K + m] = __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
    }
    // The bias term uses unquantized FP32 fused values, not reconstructed E4M3.
    for (int g = t; g < K / 64; g += 256) {
        float sum = 0.0f;
        for (int d = 0; d < 64; ++d) sum += values[source(g * 64 + d)];
        XS[row * (K / 64) + g] = __float2bfloat16_rn(sum / a);
    }
}
