// Hy4-flash's four-slot sparse index pooling before key normalization and RoPE.
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

struct Hy4PoolArgs {
    uint groups;
    uint width;
};

kernel void tf_hy4_pool_groups(
    const device float *values [[buffer(0)]],
    const device float *gates [[buffer(1)]],
    const device float *position_bias [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant Hy4PoolArgs &args [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]) {
    const uint column = tid.x;
    const uint group = tid.y;
    if (column >= args.width || group >= args.groups) return;

    const ulong base = ulong(group) * 4 * args.width + column;
    float logits[4];
    for (uint slot = 0; slot < 4; ++slot) {
        const uint offset = slot * args.width;
        logits[slot] = gates[base + offset] + position_bias[offset + column];
    }
    const float maximum = max(max(logits[0], logits[1]), max(logits[2], logits[3]));
    float weights[4];
    for (uint slot = 0; slot < 4; ++slot) weights[slot] = precise::exp(logits[slot] - maximum);
    const float normalizer = ((weights[0] + weights[1]) + weights[2]) + weights[3];
    float projected[4];
    for (uint slot = 0; slot < 4; ++slot) {
        projected[slot] = values[base + slot * args.width] * (weights[slot] / normalizer);
    }
    output[ulong(group) * args.width + column] = ((projected[0] + projected[1]) + projected[2]) + projected[3];
}
