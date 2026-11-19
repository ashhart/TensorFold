// Hy4-flash's Engram branch on CUDA, after `engram.metal`: file-backed BF16 rows enter as BF16; only the stock
// profile simulates the checkpoint's FP8 round trip. Rounding points follow the reference's BF16 module chain.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include "../common/reduce.cuh"

namespace {
constexpr uint32_t heads = 16, head_width = 96, streams = 4, threads = 128;
// `nn.RMSNorm(eps=None)` on BF16 input uses its FP32 opmath epsilon.
constexpr float rms_epsilon = 1.1920928955078125e-07f;

__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// E4M3FN, ties to even, subnormals, saturating at 448 (inputs are clamped first, as in the reference).
__device__ __forceinline__ float e4m3(float x) {
    const __nv_fp8_storage_t code = __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
    return __half2float(__half(__nv_cvt_fp8_to_halfraw(code, __NV_E4M3)));
}
}  // namespace

// One warp per (head, row): optional FP8 round trip of the gathered row, then the head's RMSNorm.
extern "C" __global__ void __launch_bounds__(32) tf_hy4_engram_lookup(
    const __nv_bfloat16 *raw, const __nv_bfloat16 *gamma, __nv_bfloat16 *out, uint32_t fp8) {
    const uint32_t head = blockIdx.x, lane = threadIdx.x;
    const uint64_t start = (uint64_t(blockIdx.y) * heads + head) * head_width;
    float x[3], maximum = 0.0f;
    for (uint32_t i = 0; i < 3; ++i) {
        x[i] = __bfloat162float(raw[start + lane + i * 32]);
        maximum = fmaxf(maximum, fabsf(x[i]));
    }
    if (fp8 != 0) {
        const float scale = bf(fmaxf(tf_common::warp_max(maximum), bf(1e-12f)) / 448.0f);
        for (uint32_t i = 0; i < 3; ++i) {
            const float code = e4m3(fminf(fmaxf(bf(x[i] / scale), -448.0f), 448.0f));
            const float quantized = bf(code * scale);
            x[i] = bf(x[i] + bf(quantized - x[i]));
        }
    }
    const float sum = tf_common::warp_sum(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
    const float inv = 1.0f / sqrtf(sum / float(head_width) + rms_epsilon);
    for (uint32_t i = 0; i < 3; ++i) {
        const uint32_t col = lane + 32 * i;
        out[start + col] = __float2bfloat16_rn((x[i] * inv) * __bfloat162float(gamma[head * head_width + col]));
    }
}

__device__ __forceinline__ float four_warps(float v, float *parts) {
    const uint32_t lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    v = tf_common::warp_sum(v);
    if (lane == 0) parts[warp] = v;
    __syncthreads();
    const float sum = (parts[0] + parts[2]) + (parts[1] + parts[3]);
    __syncthreads();
    return sum;
}

// One block per (stream, row): biased key and value, the stream's key/query norms, the signed-sqrt sigmoid gate,
// then state + gate * signal + per-stream scale * signal. A media row leaves its streams untouched.
extern "C" __global__ void __launch_bounds__(128) tf_hy4_engram_apply(
    const __nv_bfloat16 *state, const __nv_bfloat16 *keys, const __nv_bfloat16 *key_bias,
    const __nv_bfloat16 *key_gain, const __nv_bfloat16 *query_gain, const __nv_bfloat16 *delta_gain,
    const __nv_bfloat16 *value, const __nv_bfloat16 *value_bias, const __nv_bfloat16 *value_gain,
    const uint32_t *media_mask, __nv_bfloat16 *out, uint32_t rows, uint32_t h) {
    __shared__ float parts[4];
    __shared__ float gate;
    const uint32_t stream = blockIdx.x, row = blockIdx.y, t = threadIdx.x;
    const uint64_t base = (uint64_t(row) * streams + stream) * h;
    if (media_mask[row] != 0) {
        for (uint32_t j = t; j < h; j += threads) out[base + j] = state[base + j];
        return;
    }
    const uint64_t kbase = (uint64_t(stream) * rows + row) * h, vbase = uint64_t(row) * h;
    const uint32_t gain = stream * h;
    float sq_key = 0.0f, sq_query = 0.0f, sq_value = 0.0f;
    for (uint32_t j = t; j < h; j += threads) {
        const float k = bf(__bfloat162float(keys[kbase + j]) + __bfloat162float(key_bias[gain + j]));
        const float q = __bfloat162float(state[base + j]);
        const float v = bf(__bfloat162float(value[vbase + j]) + __bfloat162float(value_bias[j]));
        sq_key = __fmaf_rn(k, k, sq_key);
        sq_query = __fmaf_rn(q, q, sq_query);
        sq_value = __fmaf_rn(v, v, sq_value);
    }
    const float key_sum = four_warps(sq_key, parts);
    const float query_sum = four_warps(sq_query, parts);
    const float value_sum = four_warps(sq_value, parts);
    const float norm_key = 1.0f / sqrtf(key_sum / float(h) + rms_epsilon);
    const float norm_query = 1.0f / sqrtf(query_sum / float(h) + rms_epsilon);
    const float norm_value = 1.0f / sqrtf(value_sum / float(h) + rms_epsilon);
    float dot = 0.0f;
    for (uint32_t j = t; j < h; j += threads) {
        const float k = bf(__bfloat162float(keys[kbase + j]) + __bfloat162float(key_bias[gain + j]));
        const float q = __bfloat162float(state[base + j]);
        const float key = bf((k * norm_key) * __bfloat162float(key_gain[gain + j]));
        const float query = bf((q * norm_query) * __bfloat162float(query_gain[gain + j]));
        dot += bf(key * query);
    }
    const float sum = four_warps(dot, parts);
    if (t == 0) {
        const float scaled = bf(bf(sum) * (1.0f / sqrtf(float(h))));
        const float x = scaled == 0.0f ? 0.0f : copysignf(bf(sqrtf(fmaxf(fabsf(scaled), bf(1e-6f)))), scaled);
        gate = bf(1.0f / (1.0f + expf(-x)));
    }
    __syncthreads();
    for (uint32_t j = t; j < h; j += threads) {
        const float v = bf(__bfloat162float(value[vbase + j]) + __bfloat162float(value_bias[j]));
        const float signal = bf(bf(v * norm_value) * __bfloat162float(value_gain[j]));
        const float delta = bf(bf(gate * signal) + bf(__bfloat162float(delta_gain[gain + j]) * signal));
        out[base + j] = __float2bfloat16_rn(__bfloat162float(state[base + j]) + delta);
    }
}
