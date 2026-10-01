// Activations in the checkpoint's own formats, each row alone (a row's codes never depend on another row):
// NVFP4 (e2m1 codes, an e4m3 scale a 16 inputs under the checkpoint's static global input scale) and FP8 (e4m3 under
// the static input scale, in the weights' fragment order).

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <torch/extension.h>

#include "mma4.cuh"
#include "nvfp4q.cuh"

namespace {

__device__ __forceinline__ void unpack8(const uint4 u, float (&f)[8]) {
    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 v = __bfloat1622float2(p[i]);
        f[2 * i] = v.x;
        f[2 * i + 1] = v.y;
    }
}

// One thread a 16-input block: amax, the block scale e4m3(g * amax / 6), codes e2m1(x * g / scale).
// codes [M, K/2] (input 2j in byte j's low nibble) and scales [K/64, mpad, 4] (byte b: the step's block b), or with
// ``tb`` rows a tile the prompt GEMM's tiles, codes [mpad / tb][K/64][tb][32] in its smem swizzle and scales
// [mpad / tb][K/64][tb][4], rows past M zero.
__global__ void __launch_bounds__(128) quant4_kernel(const __nv_bfloat16* __restrict__ x, int ldx, int M, int K,
                                                    float g, uint8_t* __restrict__ codes, uint8_t* __restrict__ scales,
                                                    int mpad, int tb) {
    const int row = blockIdx.y, blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk * 16 >= K || row >= (tb ? mpad : M)) return;
    const int r = tb ? row % tb : 0;
    const size_t tiled = tb ? ((static_cast<size_t>(row / tb) * (K / 64) + blk / 4) * tb + r) * 32 : 0;
    const size_t at = tb == 0 ? static_cast<size_t>(row) * (K / 2) + blk * 8
                              : tiled + mma4::chunk<mma4::A4>(r, (blk % 4) / 2) * 16 + (blk % 2) * 8;
    const size_t sat = (tb ? (static_cast<size_t>(row / tb) * (K / 64) + blk / 4) * tb + r
                           : static_cast<size_t>(blk / 4) * mpad + row) * 4 + blk % 4;
    if (row >= M) {
        *reinterpret_cast<uint2*>(codes + at) = make_uint2(0u, 0u);
        scales[sat] = 0;
        return;
    }
    const uint4* src = reinterpret_cast<const uint4*>(x + static_cast<size_t>(row) * ldx + blk * 16);
    float f[16];
    unpack8(src[0], *reinterpret_cast<float(*)[8]>(f));
    unpack8(src[1], *reinterpret_cast<float(*)[8]>(f + 8));
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(f[i]));
    const nvfp4q::Scale sc = nvfp4q::block_scale(amax, g);
    uint32_t w[2] = {0u, 0u};
#pragma unroll
    for (int i = 0; i < 16; ++i) w[i / 8] |= nvfp4q::e2m1(f[i] * sc.mul) << (4 * (i % 8));
    *reinterpret_cast<uint2*>(codes + at) = make_uint2(w[0], w[1]);
    scales[sat] = static_cast<uint8_t>(sc.sf8);
}

// One thread 16 inputs: e4m3(x * inv) written in the weights' fragment order (byte 4q + j of a 16-byte group holds
// input 2q + (j % 2) + 8 (j / 2)), so a lane's four bytes pair with its weight bytes in the e4m3 mma; ``tb``: the
// prompt GEMM's swizzled tiles [mpad / tb][K/64][tb][64 bytes] (rows past M zero).
__global__ void __launch_bounds__(128) quant8_kernel(const __nv_bfloat16* __restrict__ x, int ldx, int M, int K,
                                                    float inv, uint8_t* __restrict__ out, int mpad, int tb) {
    const int row = blockIdx.y, grp = blockIdx.x * blockDim.x + threadIdx.x;
    if (grp * 16 >= K || row >= (tb ? mpad : M)) return;
    const int r = tb ? row % tb : 0;
    const size_t at = tb == 0 ? static_cast<size_t>(row) * K + grp * 16
        : ((static_cast<size_t>(row / tb) * (K / 64) + grp / 4) * tb + r) * 64 + mma4::chunk<mma4::A8>(r, grp % 4) * 16;
    if (row >= M) {
        *reinterpret_cast<uint4*>(out + at) = make_uint4(0u, 0u, 0u, 0u);
        return;
    }
    const uint4* src = reinterpret_cast<const uint4*>(x + static_cast<size_t>(row) * ldx + grp * 16);
    float f[16];
    unpack8(src[0], *reinterpret_cast<float(*)[8]>(f));
    unpack8(src[1], *reinterpret_cast<float(*)[8]>(f + 8));
    uint32_t w[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        uint32_t packed = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int k = 2 * q + (j % 2) + 8 * (j / 2);
            packed |= static_cast<uint32_t>(__nv_cvt_float_to_fp8(f[k] * inv, __NV_SATFINITE, __NV_E4M3)) << (8 * j);
        }
        w[q] = packed;
    }
    *reinterpret_cast<uint4*>(out + at) = make_uint4(w[0], w[1], w[2], w[3]);
}

}  // namespace

void quant4_cuda(const at::Tensor& x, double g, at::Tensor& codes, at::Tensor& scales, int64_t mpad, int64_t tb) {
    const int M = static_cast<int>(x.size(0)), K = static_cast<int>(x.size(1));
    const dim3 grid(static_cast<unsigned>((K / 16 + 127) / 128), static_cast<unsigned>(tb ? mpad : M));
    quant4_kernel<<<grid, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), static_cast<int>(x.stride(0)), M, K,
        static_cast<float>(g), codes.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(), static_cast<int>(mpad),
        static_cast<int>(tb));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void quant8_cuda(const at::Tensor& x, double inv, at::Tensor& out, int64_t mpad, int64_t tb) {
    const int M = static_cast<int>(x.size(0)), K = static_cast<int>(x.size(1));
    const dim3 grid(static_cast<unsigned>((K / 16 + 127) / 128), static_cast<unsigned>(tb ? mpad : M));
    quant8_kernel<<<grid, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), static_cast<int>(x.stride(0)), M, K,
        static_cast<float>(inv), out.data_ptr<uint8_t>(), static_cast<int>(mpad), static_cast<int>(tb));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
