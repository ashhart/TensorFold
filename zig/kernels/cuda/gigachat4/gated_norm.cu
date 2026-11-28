// GigaChat 4's GatedNorm, one row a block: RMSNorm times w, then the row times 2 sigmoid(down(silu(up(row)))).

#include <cuda_bf16.h>

#include "common/glue_math.cuh"

namespace gc4n {

using namespace tf_glue;
using tf_common::block_sum_serial;

constexpr int T = 256, W = T / 32, R = 16;

// The reference's bf16 roundings, one bit each: a control clears one (gated_norm.zig names them).
enum Point { normed = 1, up = 2, silu = 4, down = 8, sigmoid = 16, all = 31 };

__device__ __forceinline__ float keep(float v, int mask, int p) { return mask & p ? rbf(v) : v; }

// Row blockIdx.x of x (x_ld apart): y = bf16(x rms w), g = bf16(down(bf16(silu(bf16(up y))))), out = bf16(2y sig(g)).
__device__ __forceinline__ void row(const __nv_bfloat16* __restrict__ x, long long x_ld,
                                    const __nv_bfloat16* __restrict__ w, const __nv_bfloat16* __restrict__ wu,
                                    const __nv_bfloat16* __restrict__ wd, __nv_bfloat16* __restrict__ out,
                                    long long o_ld, int n, float eps, float scale, int mask) {
    extern __shared__ float ys[];  // the row's n normed values
    __shared__ float scratch[W];
    __shared__ float hs[R];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const __nv_bfloat16* xr = x + blockIdx.x * x_ld;
    // squares: the thread's elements t, t + 256, ... in fma order, the warp's butterfly, the 8 warps in order
    float ss = 0.0f;
    for (int i = t; i < n; i += T) {
        const float v = bf(xr[i]);
        ss = __fmaf_rn(v, v, ss);
    }
    ss = block_sum_serial(ss, scratch);
    const float inv = tf_common::rms_scale(ss, n, eps);
    for (int i = t; i < n; i += T) ys[i] = keep(__fmul_rn(__fmul_rn(bf(xr[i]), inv), bf(w[i])), mask, normed);
    __syncthreads();
    // up: warp k's rows 2k and 2k + 1, a lane's elements lane, lane + 32, ... in fma order, then the butterfly
    for (int j = 2 * warp; j < 2 * warp + 2; ++j) {
        const __nv_bfloat16* u = wu + static_cast<long long>(j) * n;
        float d = 0.0f;
        for (int i = lane; i < n; i += 32) d = __fmaf_rn(ys[i], bf(u[i]), d);
        d = warp_sum(d);
        if (lane == 0) hs[j] = keep(tf_silu(keep(d, mask, up)), mask, silu);
    }
    __syncthreads();
    __nv_bfloat16* orow = out + blockIdx.x * o_ld;
    for (int i = t; i < n; i += T) {
        const __nv_bfloat16* dn = wd + static_cast<long long>(i) * R;
        float g = 0.0f;
#pragma unroll
        for (int j = 0; j < R; ++j) g = __fmaf_rn(hs[j], bf(dn[j]), g);
        const float s = keep(tf_sigmoid(keep(g, mask, down)), mask, sigmoid);
        orow[i] = __float2bfloat16_rn(__fmul_rn(rbf(__fmul_rn(ys[i], scale)), s));
    }
}

}  // namespace gc4n

// rows of x [n] bf16, w [n], up [16, n], down [n, 16] (the checkpoint's layouts): bf16 rows o_ld apart, n * 4 shared.
extern "C" __global__ void __launch_bounds__(256)
    tf_gc4_gated_norm(const __nv_bfloat16* __restrict__ x, long long x_ld, const __nv_bfloat16* __restrict__ w,
                      const __nv_bfloat16* __restrict__ up, const __nv_bfloat16* __restrict__ down,
                      __nv_bfloat16* __restrict__ out, long long o_ld, int n, float eps, float scale) {
    gc4n::row(x, x_ld, w, up, down, out, o_ld, n, eps, scale, gc4n::all);
}

// The contract's planted controls: the same rows with the rounding points `mask` leaves out kept in fp32.
extern "C" __global__ void __launch_bounds__(256)
    tf_ctl_gc4_gated_norm(const __nv_bfloat16* __restrict__ x, long long x_ld, const __nv_bfloat16* __restrict__ w,
                          const __nv_bfloat16* __restrict__ up, const __nv_bfloat16* __restrict__ down,
                          __nv_bfloat16* __restrict__ out, long long o_ld, int n, float eps, float scale, int mask) {
    gc4n::row(x, x_ld, w, up, down, out, o_ld, n, eps, scale, mask);
}
