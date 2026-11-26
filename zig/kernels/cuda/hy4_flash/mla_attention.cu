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

// Merge tiled maxima and mass, counting each head's learned zero-value sink exactly once (common/mla.cuh's merge_sink).
extern "C" __global__ void __launch_bounds__(128) tf_hy4_attend_merge(
    const float *numerator, const float *maxima, const float *mass, const float *sinks,
    __nv_bfloat16 *output, int64_t rows, int64_t heads, int32_t chunks) {
    tf_mla::merge_sink(numerator, maxima, mass, sinks, output, rows, heads, chunks, threadIdx.x * 4);
}
