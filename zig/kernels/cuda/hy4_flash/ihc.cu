// Hy4-flash reads independent stream gates in FP32 and stores BF16 hidden rows.
#include <cuda_bf16.h>
#include <stdint.h>
#include "../common/reduce.cuh"

struct Hy4IhcShape {
    uint32_t rows, width, gates;
    float eps, gate_eps, post_magnitude;
};

extern "C" __global__ void __launch_bounds__(256) tf_hy4_ihc_collapse(
    const __nv_bfloat16 *hidden, const float *project, const float *bias, const float *scale,
    __nv_bfloat16 *collapsed, float *post, Hy4IhcShape a) {
    __shared__ float partial[9][8], pre[4];
    const uint32_t t = threadIdx.x, wide = 4 * a.width, row = blockIdx.x;
    const uint32_t lane = t % 32, warp = t / 32;
    const __nv_bfloat16 *x = hidden + uint64_t(row) * wide;
    float sq = 0.0f, logits[8] = {};
    for (uint32_t i = t; i < wide; i += 256) {
        const float v = __bfloat162float(x[i]);
        sq = __fmaf_rn(v, v, sq);
        for (uint32_t j = 0; j < a.gates; ++j) logits[j] = __fmaf_rn(v, project[j * wide + i], logits[j]);
    }
    sq = tf_common::warp_sum(sq);
    if (lane == 0) partial[8][warp] = sq;
    for (uint32_t j = 0; j < a.gates; ++j) {
        const float score = tf_common::warp_sum(logits[j]);
        if (lane == 0) partial[j][warp] = score;
    }
    __syncthreads();
    if (t < a.gates) {
        float sum = 0.0f, dot = 0.0f;
        for (uint32_t i = 0; i < 8; ++i) {
            sum += partial[8][i];
            dot += partial[t][i];
        }
        const float mix = dot * rsqrtf(sum / float(wide) + a.eps);
        const float gate = 1.0f / (1.0f + expf(-(mix * scale[t / 4] + bias[t])));
        if (t < 4) pre[t] = gate + a.gate_eps;
        else post[uint64_t(row) * 4 + t - 4] = a.post_magnitude * gate + a.gate_eps;
    }
    __syncthreads();
    for (uint32_t i = t; i < a.width; i += 256) {
        float v = 0.0f;
        for (uint32_t j = 0; j < 4; ++j) v += pre[j] * __bfloat162float(x[j * a.width + i]);
        collapsed[uint64_t(row) * a.width + i] = __float2bfloat16_rn(v);
    }
}

extern "C" __global__ void __launch_bounds__(256) tf_hy4_ihc_add(
    const __nv_bfloat16 *residual, const __nv_bfloat16 *branch, const float *post,
    __nv_bfloat16 *output, Hy4IhcShape a) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
    if (i >= 4 * a.width) return;
    const uint64_t full = uint64_t(row) * 4 * a.width + i;
    const uint64_t local = uint64_t(row) * a.width + i % a.width;
    output[full] = __float2bfloat16_rn(__bfloat162float(residual[full]) +
        post[uint64_t(row) * 4 + i / a.width] * __bfloat162float(branch[local]));
}
