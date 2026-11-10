// Nullable backup addresses resolve through their storage plane before any per-layer offset is applied.
#pragma once
#include <stddef.h>
#include <stdint.h>

#if defined(__CUDACC__)
#define TF_Q35_BEFORE_INLINE static __host__ __device__ __forceinline__
#else
#define TF_Q35_BEFORE_INLINE static inline
#endif

typedef struct {
    uint64_t base;
    size_t stride;
    const uint64_t* layers;
} TfQ35BeforePlane;

TF_Q35_BEFORE_INLINE TfQ35BeforePlane tf_q35_before_contiguous(uint64_t base, size_t stride) {
    TfQ35BeforePlane p = {base, stride, NULL};
    return p;
}

TF_Q35_BEFORE_INLINE TfQ35BeforePlane tf_q35_before_table(const uint64_t* layers) {
    TfQ35BeforePlane p = {0, 0, layers};
    return p;
}

TF_Q35_BEFORE_INLINE uint64_t tf_q35_before_address(TfQ35BeforePlane p, size_t layer) {
    if (p.layers != NULL) return p.layers[layer];
    return p.base == 0 ? 0 : p.base + layer * p.stride;
}

TF_Q35_BEFORE_INLINE uint64_t tf_q35_before_tail(uint64_t base, size_t state_bytes) {
    return base == 0 ? 0 : base + state_bytes;
}

#undef TF_Q35_BEFORE_INLINE
