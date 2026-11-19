// DeepSeek V4.1 release projections (ours): FP8 x FP8 block-scaled and BF16 x BF16, one arithmetic for any row count.
#include "release_tile.cuh"

// An output is one FP32 accumulator, one mma a K step of its slice in ascending order; padded rows are never stored.

// Tiles by rows: decode 16 x 64 (2 warps), mid 64 x 64 (4 warps), prompt 128 x 256 (16 warps); 4 stages each.
using Decode8 = rel::Cfg<16, 64, 1, 2, 4, 128, true>;
using Mid8 = rel::Cfg<64, 64, 2, 2, 4, 64, true>;
using Prompt8 = rel::Cfg<128, 256, 4, 4, 4, 64, true>;
using Decode16 = rel::Cfg<16, 64, 1, 2, 4, 128, false>;
using Mid16 = rel::Cfg<64, 64, 2, 2, 4, 64, false>;
using Prompt16 = rel::Cfg<128, 256, 4, 4, 4, 64, false>;

// The launch is a CTA a tile; a grid of fewer CTAs loops the same tiles in the same per-element steps.
extern "C" __global__ void __launch_bounds__(Decode8::NT) tf_dsv41_rel_fp8(RelProj p) {
    extern __shared__ __align__(16) uint8_t smem[];
    rel::proj_tiles<Decode8>(p, smem, threadIdx.x, blockIdx.x, gridDim.x);
}

extern "C" __global__ void __launch_bounds__(Mid8::NT) tf_dsv41_rel_fp8_mid(RelProj p) {
    extern __shared__ __align__(16) uint8_t smem[];
    rel::proj_tiles<Mid8>(p, smem, threadIdx.x, blockIdx.x, gridDim.x);
}

extern "C" __global__ void __launch_bounds__(Prompt8::NT) tf_dsv41_rel_fp8_prompt(RelProj p) {
    extern __shared__ __align__(16) uint8_t smem[];
    rel::proj_tiles<Prompt8>(p, smem, threadIdx.x, blockIdx.x, gridDim.x);
}

extern "C" __global__ void __launch_bounds__(Decode16::NT) tf_dsv41_rel_bf16(RelProj p) {
    extern __shared__ __align__(16) uint8_t smem[];
    rel::proj_tiles<Decode16>(p, smem, threadIdx.x, blockIdx.x, gridDim.x);
}

extern "C" __global__ void __launch_bounds__(Mid16::NT) tf_dsv41_rel_bf16_mid(RelProj p) {
    extern __shared__ __align__(16) uint8_t smem[];
    rel::proj_tiles<Mid16>(p, smem, threadIdx.x, blockIdx.x, gridDim.x);
}

extern "C" __global__ void __launch_bounds__(Prompt16::NT) tf_dsv41_rel_bf16_prompt(RelProj p) {
    extern __shared__ __align__(16) uint8_t smem[];
    rel::proj_tiles<Prompt16>(p, smem, threadIdx.x, blockIdx.x, gridDim.x);
}

// Warp tiles straight from global memory: the stage-free form a persistent kernel can take, the same bytes.
extern "C" __global__ void __launch_bounds__(128) tf_dsv41_rel_fp8_warp(RelProj p) {
    rel::warp_tiles<true>(p, blockIdx.x * 4 + (threadIdx.x >> 5), gridDim.x * 4, threadIdx.x & 31);
}

extern "C" __global__ void __launch_bounds__(128) tf_dsv41_rel_bf16_warp(RelProj p) {
    rel::warp_tiles<false>(p, blockIdx.x * 4 + (threadIdx.x >> 5), gridDim.x * 4, threadIdx.x & 31);
}

// part [s,m,n] FP32 to y [m, y_col + n): the fixed split tree (s is 2, 4 or 8).
struct RelSum {
    const float* part;
    void* y;
    long long y_ld;
    long long y_col;
    int m;
    int n;
    int s;
    int f32;
};

extern "C" __global__ void __launch_bounds__(256) tf_dsv41_rel_sum(RelSum p) {
    const long long mn = (long long)p.m * p.n;
    for (long long i = (long long)blockIdx.x * 256 + threadIdx.x; i < mn; i += (long long)gridDim.x * 256) {
        const float r = rel::combine(p.part, mn, i, p.s);
        const long long at = (i / p.n) * p.y_ld + p.y_col + i % p.n;
        if (p.f32) {
            static_cast<float*>(p.y)[at] = r;
        } else {
            static_cast<__nv_bfloat16*>(p.y)[at] = __float2bfloat16_rn(r);
        }
    }
}
