#pragma once

#include <cuda_bf16.h>

// Gate 3: a planted low-bit change in prompt16's bf16 output, which the replacement contract must reject.
extern "C" __global__ void tf_ctl_prompt16_smallm(__nv_bfloat16* Y) {
    if (blockIdx.x == 0 && threadIdx.x == 0) reinterpret_cast<unsigned short*>(Y)[0] ^= 1;
}
