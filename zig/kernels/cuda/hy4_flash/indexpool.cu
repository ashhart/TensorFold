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
