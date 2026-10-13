#include <metal_stdlib>
using namespace metal;

kernel void tf_counter_affine(device const float* input [[buffer(0)]],
                              device float* output [[buffer(1)]],
                              constant uint& count [[buffer(2)]],
                              constant float& scale [[buffer(3)]],
                              constant float& bias [[buffer(4)]],
                              uint i [[thread_position_in_grid]]) {
    if (i < count) output[i] = input[(i + 1) % count] * scale + bias;
}
