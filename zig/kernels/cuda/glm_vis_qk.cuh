// GLM image tower: glm_vis_qkv's q and k, in place in the rows vit's full attention reads.
#pragma once

#include <cuda_bf16.h>

namespace glm_vis_rows {

// A warp a (row, head) of qkv [N, 3 H 64]: q and k head-RMSNormed, then rotated by the row's 2D angles; MID as served.
template <bool MID>
__device__ __forceinline__ void qk(__nv_bfloat16* qkv, const __nv_bfloat16* __restrict__ qn,
                                   const __nv_bfloat16* __restrict__ kn, const int* __restrict__ pos,
                                   const float* __restrict__ inv_freq, int N, int H, float eps) {
    const long long item = blockIdx.x * (long long)(blockDim.x >> 5) + (threadIdx.x >> 5);
    if (item >= (long long)N * H) return;
    const int lane = threadIdx.x & 31;
    const int n = (int)(item / H), h = (int)(item % H);
    const int p = lane < 16 ? pos[2 * n] : pos[2 * n + 1];
    const float ang = (float)p * inv_freq[lane & 15];
    const float c = cosf(ang), s = sinf(ang);
    for (int which = 0; which < 2; ++which) {
        __nv_bfloat16* x = qkv + ((long long)n * 3 + which) * H * 64 + h * 64;
        const __nv_bfloat16* wn = which ? kn : qn;
        float a = __bfloat162float(x[lane]), b = __bfloat162float(x[lane + 32]);
        float ss = a * a + b * b;
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        const float inv = rsqrtf(ss / 64.f + eps);
        a = a * inv * __bfloat162float(wn[lane]);
        b = b * inv * __bfloat162float(wn[lane + 32]);
        if (MID) a = __bfloat162float(__float2bfloat16_rn(a)), b = __bfloat162float(__float2bfloat16_rn(b));
        // rotate_half: dims l < 32 take -x[l + 32], dims l + 32 take x[l]
        x[lane] = __float2bfloat16_rn(a * c - b * s);
        x[lane + 32] = __float2bfloat16_rn(b * c + a * s);
    }
}

}  // namespace glm_vis_rows
