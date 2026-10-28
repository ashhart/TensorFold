// Pointwise activations in fp32 with expf, and SiLU's derivative for the training kernels' backward passes.
#pragma once

namespace tf_common {

__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __forceinline__ float softplusf_(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

__device__ __forceinline__ float silu(float v) { return v / (1.0f + expf(-v)); }

// d/dv of v * sigmoid(v).
__device__ __forceinline__ float silu_grad(float v) {
    const float s = sigmoidf_(v);
    return s * (1.0f + v * (1.0f - s));
}

}  // namespace tf_common
