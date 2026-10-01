// The fused gate|up epilogue shared by the prompt GEMMs: a 128 x 256 block of eight 64 x 64 warps (warps 0-1 of each
// row half hold gate's 128 columns, 2-3 up's same columns) -> SiLU(gate) * up -> NVFP4 rows of down's input, quantized
// as act.cu's quant4 quantizes rows (nvfp4q.cuh). Partners (wm, c) and (wm, c + 2) each keep half their m16 tiles
// and hand the other half over through shared memory, so all eight warps run the SwiGLU and the quantization.
#pragma once

#include <cuda_bf16.h>
#include <stdint.h>
#include <type_traits>

#include "nvfp4q.cuh"

namespace swiglu4 {

constexpr int XCH = 8 * 2 * 8 * 2 * 32 * 8;              // bytes handed over: 8 warps x 2 m16 x 8 n8 x 2 halves

__device__ __forceinline__ float bf16r(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

// All threads call it after the main loop (it synchronizes twice). EPI 1 rounds gate, up and the product through
// bf16 as the unfused order does; EPI 2 keeps them in fp32. ``put(r, c, b, lo, hi)`` stores block row r's 8 code bytes
// for column tile c (0, 1) and 16-column block b; ``scales(r, c, sw)`` its four scale bytes. Column tiles from
// ``live`` on are past npad and stay unwritten.
template <int EPI, typename Put, typename Scales>
__device__ __forceinline__ void epilogue(float (&acc)[4][8][4], unsigned char* buf, int wm, int wn, int lane,
                                         float ag, float au, float qg, int live, Put put, Scales scales) {
    constexpr int NT = 8, SLOT = 2 * NT * 2 * 32;            // float2s a warp hands over
    float2* xch = reinterpret_cast<float2*>(buf);
    const int g = lane >> 2, t = lane & 3, c = wn & 1;
    __syncthreads();                                         // every warp is past its last stage
    auto hand = [&](auto keep_c) {
        constexpr int KEEP = decltype(keep_c)::value, GIVE = 2 - KEEP;
        const float sc = KEEP ? au : ag;
        float2* mine = xch + (wm * 4 + wn) * SLOT;
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int j = 0; j < NT; ++j)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    float v0 = acc[GIVE + i][j][2 * h] * sc, v1 = acc[GIVE + i][j][2 * h + 1] * sc;
                    if (EPI == 1) {
                        v0 = bf16r(v0);
                        v1 = bf16r(v1);
                    }
                    mine[((i * NT + j) * 2 + h) * 32 + lane] = make_float2(v0, v1);
                }
    };
    auto run = [&](auto keep_c) {
        constexpr int KEEP = decltype(keep_c)::value;
        constexpr bool UPW = KEEP == 2;
        const float sc = UPW ? au : ag;
        const float2* theirs = xch + (wm * 4 + (wn ^ 2)) * SLOT;
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = wm * 64 + (KEEP + i) * 16 + g + 8 * h;
                uint32_t sw = 0;
#pragma unroll
                for (int b = 0; b < 4; ++b) {                // 16 columns: n8 tiles 2b, 2b + 1
                    float a[4], amax = 0.0f;
#pragma unroll
                    for (int e2 = 0; e2 < 2; ++e2) {
                        const int j = 2 * b + e2;
                        const float2 other = theirs[((i * NT + j) * 2 + h) * 32 + lane];
#pragma unroll
                        for (int e1 = 0; e1 < 2; ++e1) {
                            float own = acc[KEEP + i][j][2 * h + e1] * sc;
                            if (EPI == 1) own = bf16r(own);
                            const float o = e1 ? other.y : other.x;
                            const float gv = UPW ? o : own, uv = UPW ? own : o;
                            const float v = gv / (1.0f + expf(-gv)) * uv;     // SiLU(gate) * up, IEEE
                            a[2 * e2 + e1] = EPI == 1 ? bf16r(v) : v;
                            amax = fmaxf(amax, fabsf(a[2 * e2 + e1]));
                        }
                    }
                    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
                    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
                    const nvfp4q::Scale s = nvfp4q::block_scale(amax, qg);
                    uint32_t lo = (nvfp4q::e2m1(a[0] * s.mul) | nvfp4q::e2m1(a[1] * s.mul) << 4) << (8 * t);
                    uint32_t hi = (nvfp4q::e2m1(a[2] * s.mul) | nvfp4q::e2m1(a[3] * s.mul) << 4) << (8 * t);
                    lo |= __shfl_xor_sync(0xffffffffu, lo, 1);
                    hi |= __shfl_xor_sync(0xffffffffu, hi, 1);
                    lo |= __shfl_xor_sync(0xffffffffu, lo, 2);
                    hi |= __shfl_xor_sync(0xffffffffu, hi, 2);
                    if (t == 0) put(r, c, b, lo, hi);
                    sw |= s.sf8 << (8 * b);
                }
                if (t == 0) scales(r, c, sw);
            }
    };
    if (wn >= 2) hand(std::integral_constant<int, 2>());
    else hand(std::integral_constant<int, 0>());
    __syncthreads();
    if (c >= live) return;
    if (wn >= 2) run(std::integral_constant<int, 2>());
    else run(std::integral_constant<int, 0>());
}

}  // namespace swiglu4
