// The exceptional prompt path replaces only a head whose ordinary output is nonfinite.
#pragma once
#include <stdint.h>

#if defined(__CUDACC__)
#define TF_WROWS_HD __host__ __device__
#else
#define TF_WROWS_HD
#endif

// Each block thread inspects its disjoint columns; the caller ORs these votes with a uniform block barrier.
TF_WROWS_HD static inline int tf_wrows_head_affected(const uint16_t* out, int width, int first, int stride) {
    for (int c = first; c < width; c += stride)
        if ((out[c] & 0x7f80u) == 0x7f80u) return 1;
    return 0;
}

#undef TF_WROWS_HD
