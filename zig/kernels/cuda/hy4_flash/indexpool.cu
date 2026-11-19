// Hy4-flash's four-slot sparse index pooling before key normalization and RoPE.
extern "C" __global__ void tf_hy4_pool_groups(
    const float *values,
    const float *gates,
    const float *position_bias,
    float *output,
    unsigned int groups,
    unsigned int width) {
    const unsigned int group = blockIdx.x;
    const unsigned int column = blockIdx.y * blockDim.x + threadIdx.x;
    if (column >= width || group >= groups) return;

    const unsigned long long base = static_cast<unsigned long long>(group) * 4 * width + column;
    float logits[4];
    #pragma unroll
    for (unsigned int slot = 0; slot < 4; ++slot) {
        const unsigned int offset = slot * width;
        logits[slot] = gates[base + offset] + position_bias[offset + column];
    }
    const float maximum = fmaxf(fmaxf(logits[0], logits[1]), fmaxf(logits[2], logits[3]));
    float weights[4];
    #pragma unroll
    for (unsigned int slot = 0; slot < 4; ++slot) weights[slot] = expf(logits[slot] - maximum);
    const float normalizer = ((weights[0] + weights[1]) + weights[2]) + weights[3];
    float projected[4];
    #pragma unroll
    for (unsigned int slot = 0; slot < 4; ++slot) {
        projected[slot] = values[base + slot * width] * (weights[slot] / normalizer);
    }
    output[static_cast<unsigned long long>(group) * width + column] =
        ((projected[0] + projected[1]) + projected[2]) + projected[3];
}

#include <cuda_bf16.h>
#include "../common/reduce.cuh"

// One block per closed group, one thread per column: the reference's FP32 `key_value_projection` and
// `key_gate_projection` of the group's four cached BF16 token keys (ascending-k FMA), the four-slot pooling
// above with `intra_group_ape`, then the FP32 `key_norm`, rounded once to BF16. RoPE runs afterwards.
extern "C" __global__ void __launch_bounds__(128) tf_hy4_index_groups(
    const __nv_bfloat16 *token_keys,
    const float *value_weights,
    const float *gate_weights,
    const float *position_bias,
    const float *norm,
    __nv_bfloat16 *output,
    unsigned int first_group,
    float epsilon) {
    __shared__ float keys[4][128];
    __shared__ float scratch[32];
    const unsigned int group = first_group + blockIdx.x, column = threadIdx.x;
    const unsigned long long base = static_cast<unsigned long long>(group) * 4 * 128;
    #pragma unroll
    for (unsigned int slot = 0; slot < 4; ++slot) keys[slot][column] = __bfloat162float(token_keys[base + slot * 128 + column]);
    __syncthreads();
    float logits[4], projected[4];
    #pragma unroll
    for (unsigned int slot = 0; slot < 4; ++slot) {
        float value = 0.0f, gate = 0.0f;
        for (unsigned int k = 0; k < 128; ++k) {
            value = __fmaf_rn(keys[slot][k], value_weights[column * 128 + k], value);
            gate = __fmaf_rn(keys[slot][k], gate_weights[column * 128 + k], gate);
        }
        projected[slot] = value;
        logits[slot] = gate + position_bias[slot * 128 + column];
    }
    const float maximum = fmaxf(fmaxf(logits[0], logits[1]), fmaxf(logits[2], logits[3]));
    float weights[4];
    #pragma unroll
    for (unsigned int slot = 0; slot < 4; ++slot) weights[slot] = expf(logits[slot] - maximum);
    const float normalizer = ((weights[0] + weights[1]) + weights[2]) + weights[3];
    float pooled = 0.0f;
    #pragma unroll
    for (unsigned int slot = 0; slot < 4; ++slot) pooled += projected[slot] * (weights[slot] / normalizer);
    const float squares = tf_common::block_sum_tree(pooled * pooled, scratch, 4);
    const float inverse = rsqrtf(squares / 128.0f + epsilon);
    output[static_cast<unsigned long long>(group) * 128 + column] = __float2bfloat16_rn(norm[column] * (pooled * inverse));
}
