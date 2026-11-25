// Inkling 2 Mini's routed experts on HIP: the activation's factors through Zig's f32 tanh and exp.
#include "inkling2/experts.hpp"

using namespace ink2e;

// The contract's proof: both activation factors at each given bf16 input; `native` uses the device's tanhf and expf.
extern "C" __global__ void __launch_bounds__(256)
    tf_ink2_act_factors(const uint16_t* __restrict__ in, float* __restrict__ gate, float* __restrict__ up, int count,
                        int native) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= count) return;
    const float v = bf(in[i]);
    if (native) {
        gate[i] = 4.0f * tanhf(v / 4.0f) * (1.0f / (1.0f + expf(-v)));
        up[i] = 25.0f * tanhf(v / 25.0f);
    } else {
        gate[i] = gateFactor(v);
        up[i] = upFactor(v);
    }
}
