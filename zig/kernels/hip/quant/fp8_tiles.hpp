#pragma once

// ModelOpt FP8's product: an E4M3 code a byte, one f32 scale a projection, the sum in f32 and the scale once.

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace tf {
namespace rocm {

constexpr int kFpTileM = 64;
constexpr int kFpTileN = 64;
constexpr int kFpChunk = 64;  // codes a chunk

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

// A 64 x 64 block: a chunk of every column's codes and the block's rows of x staged in LDS once, each thread's 4 x 4
// outputs accumulated over them; the projection's one scale multiplies each output once.
__device__ __forceinline__ void fp8_block(ModelFp8 a) {
    __shared__ unsigned char codes[kFpTileN][kFpChunk];
    __shared__ hip_bfloat16 xs[kFpTileM][kFpChunk];
    const int tid = threadIdx.x;
    const int ty = tid & (kFpTileM / 4 - 1);
    const int tx = tid / (kFpTileM / 4);
    const int n0 = blockIdx.x * kFpTileN;
    const int m0 = blockIdx.y * kFpTileM;
    const unsigned char* w = static_cast<const unsigned char*>(a.codes);
    const hip_bfloat16* x = static_cast<const hip_bfloat16*>(a.x);
    float acc[4][4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
#pragma unroll
        for (int c = 0; c < 4; ++c) acc[r][c] = 0.f;
    }
    for (int kc = 0; kc < a.k; kc += kFpChunk) {
        for (int i = tid; i < kFpTileN * kFpChunk; i += blockDim.x) {
            const int col = i / kFpChunk, j = i % kFpChunk;
            const int c = n0 + col < a.n ? n0 + col : a.n - 1;
            const int at = kc + j < a.k ? kc + j : a.k - 1;
            codes[col][j] = w[static_cast<long long>(c) * a.k + at];
        }
        for (int i = tid; i < kFpTileM * kFpChunk; i += blockDim.x) {
            const int row = i / kFpChunk, j = i % kFpChunk;
            const int r = m0 + row < a.m ? m0 + row : a.m - 1;
            xs[row][j] = kc + j < a.k ? x[static_cast<long long>(r) * a.k + kc + j] : hip_bfloat16(0.f);
        }
        __syncthreads();
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int col = tx * 4 + c;
            float q[4];
#pragma unroll
            for (int r = 0; r < 4; ++r) q[r] = 0.f;
            // not unrolled: 64 steps of four rows spill the thread's registers
            for (int j = 0; j < kFpChunk; ++j) {
                const float lvl = fp8_level(codes[col][j]);
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    q[r] = fmaf(lvl, static_cast<float>(xs[ty * 4 + r][j]), q[r]);
                }
            }
#pragma unroll
            for (int r = 0; r < 4; ++r) acc[r][c] += q[r];
        }
        __syncthreads();
    }
    const float scale = a.scale[0];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int row = m0 + ty * 4 + r;
        if (row >= a.m) continue;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int col = n0 + tx * 4 + c;
            if (col >= a.n) continue;
            a.out[static_cast<long long>(row) * a.n + col] = acc[r][c] * scale;
        }
    }
}

}  // namespace rocm
}  // namespace tf

// This path serves bf16 activations only, as the NVFP4 product does.
extern "C" __global__ void __launch_bounds__(256) tf_fp8_gemm(tf::rocm::ModelFp8 a) {
    tf::rocm::fp8_block(a);
}
