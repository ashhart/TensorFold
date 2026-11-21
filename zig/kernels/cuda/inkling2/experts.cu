// Inkling 2 Mini's tensor-path expert epilogue and ascending combine: the modelopt contract's baseline kernels.
#include <cstdint>
#include <cuda_bf16.h>

namespace ink2m {

__device__ __forceinline__ float bf(uint16_t u) { return __uint_as_float(static_cast<uint32_t>(u) << 16); }
__device__ __forceinline__ uint16_t tobf(float f) { return __bfloat16_as_ushort(__float2bfloat16_rn(f)); }

struct CombineArgs {
    const uint16_t* experts;  // bf16 [rows * slots, H]
    const int* pick;
    const float* weights;
    uint16_t* out;  // bf16 [rows, H]
    int rows, h, slots;
};

struct TensorActArgs {
    const uint16_t* gateup;  // bf16 [P, 2 * M], adjacent gate and up projections
    uint16_t* out;  // bf16 [P, M]
    int n;
};


__device__ __forceinline__ float cap(float x, float c) { return c * tanhf(x / c); }

}  // namespace ink2m


// Tensor-core gate/up rows are interleaved; keep the scalar baseline's softcapped SwiGLU arithmetic.
extern "C" __global__ void __launch_bounds__(256) tf_ink2moe_tensor_act(const ink2m::TensorActArgs a) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= a.n) return;
    const float gate = ink2m::bf(a.gateup[2 * (size_t)i]);
    const float up = ink2m::bf(a.gateup[2 * (size_t)i + 1]);
    const float g = ink2m::cap(gate, 4.0f);
    const float u = ink2m::cap(up, 25.0f);
    const float sigmoid = 1.0f / (1.0f + expf(-gate));
    const float activated = g * sigmoid * u;
    a.out[i] = ink2m::tobf(activated);
}

// One output element: route in ascending expert id, independent of the router's selection order.
extern "C" __global__ void __launch_bounds__(256) tf_ink2moe_combine(const ink2m::CombineArgs a) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= a.rows * a.h) return;
    const int row = i / a.h, col = i % a.h;
    float total = 0.0f;
    uint32_t seen = 0;
    for (int n = 0; n < a.slots; ++n) {
        int best = -1, id = 0;
        for (int s = 0; s < a.slots; ++s) {
            if (seen & (1u << s)) continue;
            const int e = a.pick[row * a.slots + s];
            if (best < 0 || e < id) {
                best = s;
                id = e;
            }
        }
        seen |= 1u << best;
        const float out = ink2m::bf(a.experts[(size_t)(row * a.slots + best) * a.h + col]);
        total = __fadd_rn(total, __fmul_rn(a.weights[row * a.slots + best], out));
    }
    a.out[i] = ink2m::tobf(total);
}
