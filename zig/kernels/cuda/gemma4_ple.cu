// Gemma 4's own image: the E-series per-layer inputs (cuda_ple.zig); the steps are gemma4_ple.cuh's.

#include <cuda_bf16.h>
#include "gemma4_ple.cuh"

// out [rows, parts 256] from projected rows p and table rows t (packed alike); 64 threads a (row, part).
extern "C" __global__ void __launch_bounds__(64) tf_gemma4_ple_rows(const __nv_bfloat16* __restrict__ p,
                                                                    const __nv_bfloat16* __restrict__ t,
                                                                    const __nv_bfloat16* __restrict__ w,
                                                                    __nv_bfloat16* __restrict__ out, int parts,
                                                                    float scale, float token, float mix, float eps) {
    tf_ple::rows(p, t, w, out, parts, scale, token, mix, eps, 0);
}
