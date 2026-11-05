// Gemma 4 E-series per-layer inputs in layer_input.zig's steps, shared by gemma4_ple.cu and the contract's controls.
#pragma once
#include <cuda_bf16.h>

namespace tf_ple {

// Controls only: drop a step the contract's shapes gate must catch.
constexpr int DROP_NORM = 1;
constexpr int DROP_TOKEN = 2;

__device__ __forceinline__ float rb(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

// A block of 64 a (row, part): MLX's one-row rms_norm sums (4-value chains, butterflies); p, t, out packed alike.
__device__ __forceinline__ void rows(const __nv_bfloat16* __restrict__ p, const __nv_bfloat16* __restrict__ t,
                                     const __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ out, int parts,
                                     float scale, float token, float mix, float eps, int drop) {
    __shared__ float sums[2];
    const long long base = ((long long)blockIdx.x * parts + blockIdx.y) * 256 + threadIdx.x * 4;
    const int lane = threadIdx.x & 31;
    float x[4];
    float acc = 0.0f;
    for (int i = 0; i < 4; i++) {
        x[i] = rb(__fmul_rn(__bfloat162float(p[base + i]), scale));
        acc = __fadd_rn(acc, __fmul_rn(x[i], x[i]));
    }
    for (int s = 1; s < 32; s <<= 1) acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, s));
    if (lane == 0) sums[threadIdx.x >> 5] = acc;
    __syncthreads();
    const float mean = __fadd_rn(__fdiv_rn(__fadd_rn(sums[0], sums[1]), 256.0f), eps);
    const float inv = (drop & DROP_NORM) ? 1.0f : (float)(1.0 / sqrt((double)mean));
    for (int i = 0; i < 4; i++) {
        const int k = threadIdx.x * 4 + i;
        const float y = rb(__fmul_rn(__bfloat162float(w[k]), rb(__fmul_rn(x[i], inv))));
        const float tt = rb(__fmul_rn(__bfloat162float(t[base + i]), (drop & DROP_TOKEN) ? 1.0f : token));
        out[base + i] = __float2bfloat16_rn(__fmul_rn(rb(__fadd_rn(y, tt)), mix));
    }
}

}  // namespace tf_ple
