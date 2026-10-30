// A block's monotone logical-key walk reads each encountered page ID once; page IDs must remain stable.
#pragma once
#include <stdint.h>

#ifdef __CUDACC__
#define TF_Q35_PAGE_INLINE __device__ __forceinline__
#else
#define TF_Q35_PAGE_INLINE static inline
#endif

typedef struct {
    const int* ids;
    int tokens;
    int logical;
    uint64_t stride;
    uint64_t base;
} TfQ35PageRun;

TF_Q35_PAGE_INLINE uint64_t tf_q35_page_base(TfQ35PageRun* run, int pos) {
    const int logical = pos / run->tokens;
    if (logical != run->logical) {
        run->base = (uint64_t)run->ids[logical] * run->stride;
        run->logical = logical;
    }
    return run->base;
}
#undef TF_Q35_PAGE_INLINE
