#pragma once

#include "sample_repetition_value.cuh"

// Source-pinned repetition values call the production arithmetic from the registered controls image.
extern "C" __global__ void tf_contract_repetition_values(const __nv_bfloat16* logits, unsigned n,
                                                          const unsigned* seen, float factor, float* out) {
    for (unsigned i = threadIdx.x; i < n; i += blockDim.x) {
        const bool present = ((seen[i / 32] >> (i % 32)) & 1u) != 0;
        out[i] = tf_sample::repetition_value(logits[i], present, factor);
    }
}
