// Score absorbed queries against retained BF16 cache rows. The retained slots already include any causal limit.
#include <cuda_bf16.h>
#include <stdint.h>
#include "../common/mla.cuh"

extern "C" __global__ void __launch_bounds__(256) tf_hy4_attend_chunks(
    const __nv_bfloat16 *qa, const __nv_bfloat16 *cache, const int *slots, const int *counts,
    float *numerator, float *maxima, float *mass, int64_t rows, int64_t heads, int64_t key_stride,
    float scale) {
    glm_attn::chunks_wide<16, false, 64>(
        qa, cache, nullptr, slots, counts, numerator, maxima, mass, rows, heads, key_stride, scale,
        1, nullptr, 0);
}

// Merge tiled maxima and mass, counting each head's learned zero-value sink exactly once.
extern "C" __global__ void __launch_bounds__(128) tf_hy4_attend_merge(
    const float *numerator, const float *maxima, const float *mass, const float *sinks,
    __nv_bfloat16 *output, int64_t rows, int64_t heads, int32_t chunks) {
    constexpr int latent = glm_attn::LW;
    const int64_t head = blockIdx.y;
    const int64_t row_head = blockIdx.x * heads + head;
    const int64_t plane = rows * heads;
    const int column = threadIdx.x * 4;
    float top = sinks[head];
    for (int i = 0; i < chunks; ++i) top = fmaxf(top, maxima[i * plane + row_head]);
    float denom = expf(sinks[head] - top);
    float acc[4] = {};
    for (int i = 0; i < chunks; ++i) {
        const int64_t base = i * plane + row_head;
        const float p = expf(maxima[base] - top);
        denom = __fmaf_rn(mass[base], p, denom);
        #pragma unroll
        for (int j = 0; j < 4; ++j) acc[j] = __fmaf_rn(numerator[base * latent + column + j], p, acc[j]);
    }
    #pragma unroll
    for (int j = 0; j < 4; ++j) output[row_head * latent + column + j] = __float2bfloat16_rn(acc[j] / denom);
}
