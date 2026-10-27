// Warp and block reductions, one fixed order each; every lane that calls gets the same total.
#pragma once

#include <math.h>
#include <stdint.h>

#include "common/vec.cuh"

namespace tf_common {

// xor butterfly, offsets 16 down to 1
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// Tree order: each warp's butterfly, then one more butterfly over the `groups` warp sums (zeros past them).
__device__ __forceinline__ float block_sum_tree(float v, float* scratch, int groups) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) scratch[w] = v;
    __syncthreads();
    v = warp_sum(lane < groups ? scratch[lane] : 0.0f);
    __syncthreads();
    return v;
}

__device__ __forceinline__ float block_max_tree(float v, float* scratch, int groups) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    v = warp_max(v);
    if (lane == 0) scratch[w] = v;
    __syncthreads();
    v = warp_max(lane < groups ? scratch[lane] : -INFINITY);
    __syncthreads();
    return v;
}

// A thread's 8 stored values in order, then the group's 8 lanes in a fixed tree (every lane of the warp calls).
__device__ __forceinline__ float group_sum8(const uint4& packed, bool live) {
    float f[8];
    unpack8(packed, f);
    float s = 0.f;
    if (live)
#pragma unroll
        for (int i = 0; i < 8; ++i) s = __fadd_rn(s, f[i]);
    s = __fadd_rn(s, __shfl_xor_sync(0xFFFFFFFFu, s, 4));
    s = __fadd_rn(s, __shfl_xor_sync(0xFFFFFFFFu, s, 2));
    return __fadd_rn(s, __shfl_xor_sync(0xFFFFFFFFu, s, 1));
}

}  // namespace tf_common
