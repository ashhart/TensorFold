// Pointwise activations in fp32 with expf, and SiLU's derivative for the training kernels' backward passes.
#pragma once

#include <math.h>

namespace tf_common {

__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __forceinline__ float softplusf_(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

__device__ __forceinline__ float silu(float v) { return v / (1.0f + expf(-v)); }

// d/dv of v * sigmoid(v).
__device__ __forceinline__ float silu_grad(float v) {
    const float s = sigmoidf_(v);
    return s * (1.0f + v * (1.0f - s));
}

__device__ __forceinline__ float sigmoid_rn(float x) { return __fdiv_rn(1.f, __fadd_rn(1.f, expf(-x))); }

__device__ __forceinline__ float silu_rn(float x) { return __fdiv_rn(x, __fadd_rn(1.f, expf(-x))); }

}  // namespace tf_common
