#pragma once

#include <cuda_bf16.h>

namespace tf_sample {

// The production repetition arithmetic, shared with its source-pinned contract fixture.
__device__ __forceinline__ float repetition_value(__nv_bfloat16 raw, bool present, float factor) {
    float x = __bfloat162float(raw);
    if (present) x = x < 0.0f ? __fmul_rn(x, factor) : __fdiv_rn(x, factor);
    return x;
}

}  // namespace tf_sample
