#pragma once

// ModelOpt NVFP4: e2m1 codes eight to a word, one E4M3 group scale a 16 inputs, and the projection's global scale.

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

#include "quant/nvfp4.hpp"

namespace tf {
namespace rocm {

constexpr int kNvTileM = 64;
constexpr int kNvTileN = 64;
constexpr int kNvChunk = 64;   // codes a chunk: four groups, eight words a column
constexpr int kNvSmallM = 16;  // rows a decode block serves

// The group table's entry at `i` as the float it holds: BF16's bits shifted into f32.
__device__ __forceinline__ float nvfp4_scale(const GroupTable& t, long long i) {
    return __uint_as_float(static_cast<unsigned>(static_cast<const unsigned short*>(t.p)[i]) << 16);
}

// A nibble's e2m1 level as a float: the BF16 bit table shifted into f32, which is the same value.
__device__ __forceinline__ float nvfp4_level(uint32_t code) {
    return __uint_as_float(static_cast<unsigned>(kE2m1Bf16[code & 0xfu]) << 16);
}

// The row a block reads, clamped so a block past the end reads the last row.
__device__ __forceinline__ int nvfp4_row(int m, int row) { return row < m ? row : m - 1; }

// A 64 x 64 block: a chunk of every column's codes and its group scales, and the block's rows of x, staged in LDS once
// and each thread's 4 x 4 outputs accumulated over them. A group folds its 16 inputs before its scale.
__device__ __forceinline__ void nvfp4_block(Nvfp4 a) {
    __shared__ uint32_t codes[kNvTileN][kNvChunk / 8];
    __shared__ float sc[kNvTileN][kNvChunk / 16];
    __shared__ hip_bfloat16 xs[kNvTileM][kNvChunk];
    const int tid = threadIdx.x;
    const int ty = tid & (kNvTileM / 4 - 1);
    const int tx = tid / (kNvTileM / 4);
    const int n0 = blockIdx.x * kNvTileN;
    const int m0 = blockIdx.y * kNvTileM;
    const int words_row = a.k / 8;
    const int groups = a.k / a.group;
    const hip_bfloat16* x = static_cast<const hip_bfloat16*>(a.x);
    float acc[4][4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
#pragma unroll
        for (int c = 0; c < 4; ++c) acc[r][c] = 0.f;
    }
    for (int kc = 0; kc < a.k; kc += kNvChunk) {
        for (int i = tid; i < kNvTileN * (kNvChunk / 8); i += blockDim.x) {
            const int col = i / (kNvChunk / 8), wd = i % (kNvChunk / 8);
            const int c = n0 + col < a.n ? n0 + col : a.n - 1;
            const int at = kc / 8 + wd < words_row ? kc / 8 + wd : words_row - 1;
            codes[col][wd] = a.words[static_cast<long long>(c) * words_row + at];
        }
        for (int i = tid; i < kNvTileN * (kNvChunk / 16); i += blockDim.x) {
            const int col = i / (kNvChunk / 16), g = i % (kNvChunk / 16);
            const int c = n0 + col < a.n ? n0 + col : a.n - 1;
            const int gi = kc / 16 + g < groups ? kc / 16 + g : groups - 1;
            sc[col][g] = nvfp4_scale(a.scale, static_cast<long long>(c) * groups + gi);
        }
        for (int i = tid; i < kNvTileM * kNvChunk; i += blockDim.x) {
            const int row = i / kNvChunk, j = i % kNvChunk;
            const int r = nvfp4_row(a.m, m0 + row);
            xs[row][j] = kc + j < a.k ? x[static_cast<long long>(r) * a.k + kc + j] : hip_bfloat16(0.f);
        }
        __syncthreads();
#pragma unroll
        for (int g = 0; g < kNvChunk / 16; ++g) {
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const int col = tx * 4 + c;
                const float scale = sc[col][g];
                float q[4];
#pragma unroll
                for (int r = 0; r < 4; ++r) q[r] = 0.f;
#pragma unroll
                for (int j = 0; j < 16; ++j) {
                    const int code = g * 16 + j;
                    const float lvl = nvfp4_level(codes[col][code / 8] >> (4 * (code % 8)));
#pragma unroll
                    for (int r = 0; r < 4; ++r) {
                        q[r] = fmaf(lvl, static_cast<float>(xs[ty * 4 + r][code]), q[r]);
                    }
                }
#pragma unroll
                for (int r = 0; r < 4; ++r) acc[r][c] = fmaf(q[r], scale, acc[r][c]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int row = m0 + ty * 4 + r;
        if (row >= a.m) continue;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int col = n0 + tx * 4 + c;
            if (col >= a.n) continue;
            a.out[static_cast<long long>(row) * a.n + col] = acc[r][c] * a.global;
        }
    }
}

// A 16-row block: one row four columns a thread, so a decode round computes its own rows. A group's 16 inputs fold in
// the same order as the 64-row tile's, so a row's bits are the tile's whatever its round's size.
__device__ __forceinline__ void nvfp4_small(Nvfp4 a) {
    __shared__ uint32_t codes[kNvTileN][kNvChunk / 8];
    __shared__ float sc[kNvTileN][kNvChunk / 16];
    __shared__ hip_bfloat16 xs[kNvSmallM][kNvChunk];
    const int tid = threadIdx.x;
    const int row = tid >> 4;
    const int c0 = (tid & 15) * 4;
    const int n0 = blockIdx.x * kNvTileN;
    const int words_row = a.k / 8;
    const int groups = a.k / a.group;
    const hip_bfloat16* x = static_cast<const hip_bfloat16*>(a.x);
    float acc[4];
#pragma unroll
    for (int c = 0; c < 4; ++c) acc[c] = 0.f;
    for (int kc = 0; kc < a.k; kc += kNvChunk) {
        for (int i = tid; i < kNvTileN * (kNvChunk / 8); i += blockDim.x) {
            const int col = i / (kNvChunk / 8), wd = i % (kNvChunk / 8);
            const int c = n0 + col < a.n ? n0 + col : a.n - 1;
            const int at = kc / 8 + wd < words_row ? kc / 8 + wd : words_row - 1;
            codes[col][wd] = a.words[static_cast<long long>(c) * words_row + at];
        }
        for (int i = tid; i < kNvTileN * (kNvChunk / 16); i += blockDim.x) {
            const int col = i / (kNvChunk / 16), g = i % (kNvChunk / 16);
            const int c = n0 + col < a.n ? n0 + col : a.n - 1;
            const int gi = kc / 16 + g < groups ? kc / 16 + g : groups - 1;
            sc[col][g] = nvfp4_scale(a.scale, static_cast<long long>(c) * groups + gi);
        }
        for (int i = tid; i < kNvSmallM * kNvChunk; i += blockDim.x) {
            const int rr = i / kNvChunk, j = i % kNvChunk;
            const int rw = nvfp4_row(a.m, rr);
            xs[rr][j] = kc + j < a.k ? x[static_cast<long long>(rw) * a.k + kc + j] : hip_bfloat16(0.f);
        }
        __syncthreads();
#pragma unroll
        for (int g = 0; g < kNvChunk / 16; ++g) {
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const int col = c0 + c;
                float q = 0.f;
#pragma unroll
                for (int j = 0; j < 16; ++j) {
                    const int code = g * 16 + j;
                    q = fmaf(nvfp4_level(codes[col][code / 8] >> (4 * (code % 8))), static_cast<float>(xs[row][code]), q);
                }
                acc[c] = fmaf(q, sc[col][g], acc[c]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        const int col = n0 + c0 + c;
        if (row < a.m and col < a.n) a.out[static_cast<long long>(row) * a.n + col] = acc[c] * a.global;
    }
}

}  // namespace rocm
}  // namespace tf

// This path serves bf16 activations only: the engine refuses an fp16 build for a ModelOpt checkpoint.
extern "C" __global__ void __launch_bounds__(256) tf_nvfp4_gemm(tf::rocm::Nvfp4 a) {
    if (a.m >= tf::rocm::kNvSmallM) tf::rocm::nvfp4_block(a);
    else tf::rocm::nvfp4_small(a);
}
