// Inkling 2 Mini's loop kernels on HIP, each under its named HIP baseline (docs/kernel-contract.md).
#include "inkling2/zmath.hpp"

namespace ink2l {

// The baselines' 1 / sqrt: a correctly rounded square root, then a correctly rounded division (not rsqrt).
__device__ __forceinline__ float invSqrt(float v) { return 1.0f / sqrtf(v); }

}  // namespace ink2l

// The contract's proof of the f32 math: op 0 tanh, 1 exp, 2 1 / sqrt as the kernels run them; 3-5 the device's own.
extern "C" __global__ void __launch_bounds__(256)
    tf_ink2_loop_math(const float* __restrict__ in, float* __restrict__ out, int count, int op) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= count) return;
    const float x = in[i];
    float y;
    switch (op) {
        case 0: y = ink2z::ztanh(x); break;
        case 1: y = ink2z::zexp(x); break;
        case 2: y = ink2l::invSqrt(x); break;
        case 3: y = tanhf(x); break;
        case 4: y = expf(x); break;
        default: y = rsqrtf(x); break;
    }
    out[i] = y;
}
