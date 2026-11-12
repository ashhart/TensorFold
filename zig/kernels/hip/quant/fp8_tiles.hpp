#pragma once

// ModelOpt FP8's product: an E4M3 code a byte, one f32 scale a projection, the sum in f32 and the scale once.

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace tf {
namespace rocm {

struct ModelFp8 {
    const void* x;        // (m, k) row major, bf16
    const void* codes;    // (n, k) row major, E4M3 one a byte
    const float* scale;   // (1): the projection's one scale
    float* out;           // (m, n)
    int m, n, k;
};

// An E4M3 code as the f32 it names, exactly: 0x7f and 0xff are NaN and refused at read.
__device__ __forceinline__ float fp8_level(unsigned c) {
    const unsigned e = (c >> 3) & 0xfu, m = c & 0x7u;
    const float v = e == 0 ? static_cast<float>(m) * 0.125f * __uint_as_float((127u - 6u) << 23)
                           : (1.0f + static_cast<float>(m) * 0.125f) * __uint_as_float((e + 120u) << 23);
    return (c & 0x80u) != 0 ? -v : v;
}

__device__ __forceinline__ float fp8_warp_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

// A warp R rows of one output column: its lanes read four-code words, so a column loads once and serves the R rows.
template <int R>
__device__ __forceinline__ void fp8_rows(ModelFp8 a) {
    const int lane = threadIdx.x & 31;
    const int warps = blockDim.x >> 5;
    const int row_blocks = (a.m + R - 1) / R;
    const long long total = static_cast<long long>(row_blocks) * a.n;
    const unsigned char* codes = static_cast<const unsigned char*>(a.codes);
    const bool wide = (a.k & 3) == 0;
    const int words_row = a.k / 4;
    const long long stride = static_cast<long long>(gridDim.x) * warps;
    for (long long item = static_cast<long long>(blockIdx.x) * warps + (threadIdx.x >> 5); item < total;
         item += stride) {
        const int rb = static_cast<int>(item / a.n);
        const int col = static_cast<int>(item % a.n);
        const int row0 = rb * R;
        const int rows = a.m - row0 < R ? a.m - row0 : R;
        const unsigned char* w = codes + static_cast<long long>(col) * a.k;
        const hip_bfloat16* x = static_cast<const hip_bfloat16*>(a.x) + static_cast<long long>(row0) * a.k;
        float acc[R];
#pragma unroll
        for (int r = 0; r < R; ++r) acc[r] = 0.f;
        if (wide) {
            const uint32_t* w32 = reinterpret_cast<const uint32_t*>(w);
            for (int base = 0; base < words_row; base += 32) {
                const int wd = base + lane;
                if (wd < words_row) {
                    const uint32_t word = w32[wd];
                    float lvl[4];
#pragma unroll
                    for (int c = 0; c < 4; ++c) lvl[c] = fp8_level(word >> (8 * c));
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        if (r < rows) {
                            float q = 0.f;
#pragma unroll
                            for (int c = 0; c < 4; ++c) {
                                q = fmaf(lvl[c], static_cast<float>(x[static_cast<long long>(r) * a.k + wd * 4 + c]), q);
                            }
                            acc[r] += q;
                        }
                    }
                }
            }
        } else {
            for (int j = lane; j < a.k; j += 32) {
                const float lvl = fp8_level(w[j]);
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    if (r < rows) acc[r] = fmaf(lvl, static_cast<float>(x[static_cast<long long>(r) * a.k + j]), acc[r]);
                }
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const float sum = fp8_warp_sum(acc[r]);
            if (lane == 0 and r < rows) a.out[static_cast<long long>(row0 + r) * a.n + col] = sum * a.scale[0];
        }
    }
}

}  // namespace rocm
}  // namespace tf

// This path serves bf16 activations only, as the NVFP4 product does.
extern "C" __global__ void __launch_bounds__(256) tf_fp8_gemm(tf::rocm::ModelFp8 a) {
    if (a.m >= 8) tf::rocm::fp8_rows<8>(a);
    else if (a.m >= 4) tf::rocm::fp8_rows<4>(a);
    else tf::rocm::fp8_rows<1>(a);
}
