// presence_penalty and frequency_penalty: entry i takes SUB[i] from row ROW[i]'s logit of TOK[i], in fp32, rounded once
#pragma once

extern "C" __global__ void __launch_bounds__(256) tf_penalize_rows(__nv_bfloat16* L, unsigned V, const unsigned* ROW,
                                                                   const unsigned* TOK, const float* SUB, unsigned n) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    __nv_bfloat16* x = L + static_cast<size_t>(ROW[i]) * V + TOK[i];
    *x = __float2bfloat16(__bfloat162float(*x) - SUB[i]);
}

// Bias entries use the same subtraction, with +inf as an unconditional hard ban (including nonfinite input).
extern "C" __global__ void __launch_bounds__(256) tf_bias_rows(__nv_bfloat16* L, unsigned V, const unsigned* ROW,
                                                            const unsigned* TOK, const float* SUB, unsigned n) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    __nv_bfloat16* x = L + static_cast<size_t>(ROW[i]) * V + TOK[i];
    *x = __float2bfloat16(SUB[i] == __int_as_float(0x7f800000) ? -__int_as_float(0x7f800000)
                                                            : __bfloat162float(*x) - SUB[i]);
}
