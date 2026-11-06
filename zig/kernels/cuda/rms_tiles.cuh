#pragma once

// tf_rms_rows_sums at Flash Next's MTP shape (bf16, width 10,240, groups of 32) over `tiles` blocks a row: its bits.

namespace tf_rms_tiles {

constexpr int WIDTH = 10240;
constexpr int GROUP = 32;
constexpr int TNT = 256;
constexpr int CHAINS = tf_norms::NT / TNT; // tf_rms_rows_sums' 1,024 threads' sums, four a thread
constexpr int STEPS = WIDTH / tf_norms::NT;

// 1 / sqrt(sum / width + eps) as tf_norms::inv_rms forms it: thread c's chain, its warp's xor tree, then the 32 warps'.
__device__ __forceinline__ float inv_rms(const __nv_bfloat16* __restrict__ x, float eps) {
    __shared__ float warps[tf_norms::NT / 32];
    __shared__ float inv;
    float part[CHAINS];
#pragma unroll
    for (int j = 0; j < CHAINS; ++j) {
        float a[STEPS];
#pragma unroll
        for (int k = 0; k < STEPS; ++k) a[k] = __bfloat162float(x[threadIdx.x + j * TNT + k * tf_norms::NT]);
        float p = 0.f;
#pragma unroll
        for (int k = 0; k < STEPS; ++k) p = __fmaf_rn(a[k], a[k], p);
        part[j] = p;
    }
#pragma unroll
    for (int j = 0; j < CHAINS; ++j) { // chain threadIdx.x + j * TNT is warp (threadIdx.x >> 5) + j * 8's same lane
        float p = part[j];
        for (int m = 16; m > 0; m >>= 1) p = __fadd_rn(p, __shfl_xor_sync(0xFFFFFFFFu, p, m));
        if ((threadIdx.x & 31) == 0) warps[(threadIdx.x >> 5) + j * (TNT / 32)] = p;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        float s = warps[threadIdx.x];
        for (int m = 16; m > 0; m >>= 1) s = __fadd_rn(s, __shfl_xor_sync(0xFFFFFFFFu, s, m));
        if (threadIdx.x == 0) inv = __fdiv_rn(1.f, __fsqrt_rn(__fadd_rn(__fdiv_rn(s, static_cast<float>(WIDTH)), eps)));
    }
    __syncthreads();
    return inv;
}

}  // namespace tf_rms_tiles

// Block (tile, row) of tf_rms_rows_sums: the row's inv as it forms it, then the tile's y and its group sums, in order.
extern "C" __global__ void __launch_bounds__(tf_rms_tiles::TNT) tf_rms_rows_sums_tiles(
    const __nv_bfloat16* __restrict__ x, long long x_ld, const float* __restrict__ w, __nv_bfloat16* __restrict__ y,
    float* __restrict__ xs, int tiles, float eps) {
    using namespace tf_rms_tiles;
    __shared__ __nv_bfloat16 tile[WIDTH];
    const int span = WIDTH / tiles, first = blockIdx.x * span;
    const int64_t at = static_cast<int64_t>(blockIdx.y) * x_ld, out = static_cast<int64_t>(blockIdx.y) * WIDTH;
    const float inv = inv_rms(x + at, eps);
    for (int i = threadIdx.x; i < span; i += TNT) {
        const int c = first + i;
        tile[i] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(__bfloat162float(x[at + c]), inv), w[c]));
        y[out + c] = tile[i];
    }
    __syncthreads();
    for (int j = threadIdx.x; j < span / GROUP; j += TNT) {
        float sum = 0.f;
        for (int k = 0; k < GROUP; ++k) sum = __fadd_rn(sum, __bfloat162float(tile[j * GROUP + k]));
        xs[static_cast<int64_t>(blockIdx.y) * (WIDTH / GROUP) + first / GROUP + j] = sum;
    }
}
