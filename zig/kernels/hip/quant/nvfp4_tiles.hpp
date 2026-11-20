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
constexpr int kNvWords = kNvChunk / 8;
constexpr int kNvDepth = 2;    // chunks of codes a column keeps in flight

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
    __shared__ uint32_t codes[kNvTileN][kNvWords];
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

// A decode round, one column a thread with ROWS rows in flight: no LDS, no sixteen-row stage, and two chunks of codes
// prefetched so the column's reads are pipelined. A group's 16 inputs fold in the small block's order and the group
// scale lands on the group's partial, so a row's bits are the same either way.
//
// The prefetch is not decoration. With a guard clause and no prefetch this card software-pipelines nothing - 22 VGPRs
// against 33, and 2.6x slower - so the loads are stated here instead. Requires k a multiple of kNvChunk, which the
// launch checks before choosing this path.
template <int ROWS, int DEPTH>
__device__ __forceinline__ void nvfp4_columns(Nvfp4 a) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= a.n) return;
    const int groups = a.k / a.group;
    const int chunks = a.k / kNvChunk;
    const hip_bfloat16* x = static_cast<const hip_bfloat16*>(a.x);
    const uint32_t* w = a.words + static_cast<long long>(col) * (a.k / 8);
    const long long sbase = static_cast<long long>(col) * groups;

    uint32_t hold[DEPTH][kNvWords];
#pragma unroll
    for (int d = 0; d < DEPTH; ++d) {
#pragma unroll
        for (int wd = 0; wd < kNvWords; ++wd) hold[d][wd] = d < chunks ? w[d * kNvWords + wd] : 0u;
    }

    float acc[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) acc[r] = 0.f;

    for (int ci = 0; ci < chunks; ++ci) {
        const int kc = ci * kNvChunk;
        uint32_t cur[kNvWords];
#pragma unroll
        for (int wd = 0; wd < kNvWords; ++wd) cur[wd] = hold[0][wd];
#pragma unroll
        for (int d = 0; d + 1 < DEPTH; ++d) {
#pragma unroll
            for (int wd = 0; wd < kNvWords; ++wd) hold[d][wd] = hold[d + 1][wd];
        }
        const int fetch = ci + DEPTH;
        if (fetch < chunks) {
#pragma unroll
            for (int wd = 0; wd < kNvWords; ++wd) hold[DEPTH - 1][wd] = w[fetch * kNvWords + wd];
        }
#pragma unroll
        for (int g = 0; g < kNvChunk / 16; ++g) {
            float q[ROWS];
#pragma unroll
            for (int r = 0; r < ROWS; ++r) q[r] = 0.f;
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const int code = g * 16 + j;
                const float lvl = nvfp4_level(cur[code / 8] >> (4 * (code % 8)));
#pragma unroll
                for (int r = 0; r < ROWS; ++r) {
                    q[r] = fmaf(lvl, static_cast<float>(x[static_cast<long long>(r) * a.k + kc + code]), q[r]);
                }
            }
            const int gi = kc / 16 + g < groups ? kc / 16 + g : groups - 1;
            const float scale = nvfp4_scale(a.scale, sbase + gi);
#pragma unroll
            for (int r = 0; r < ROWS; ++r) acc[r] = fmaf(q[r], scale, acc[r]);
        }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        a.out[static_cast<long long>(r) * a.n + col] = acc[r] * a.global;
    }
}

}  // namespace rocm
}  // namespace tf

// This path serves bf16 activations only: the engine refuses an fp16 build for a ModelOpt checkpoint.
extern "C" __global__ void __launch_bounds__(256) tf_nvfp4_gemm(tf::rocm::Nvfp4 a) {
    if (a.m >= tf::rocm::kNvSmallM) tf::rocm::nvfp4_block(a);
    else tf::rocm::nvfp4_small(a);
}

// A decode round on nvfp4_columns: the launch picks tf_nvfp4_col<rows> to match a.m exactly, so no row is padded to
// sixteen and a row's offset is a constant. These are separate entry points rather than a switch inside one kernel:
// the kernel contract records resources an entry point, and a branch on a.m would keep every variant's registers.
extern "C" __global__ void __launch_bounds__(256) tf_nvfp4_col1(tf::rocm::Nvfp4 a) {
    tf::rocm::nvfp4_columns<1, tf::rocm::kNvDepth>(a);
}
extern "C" __global__ void __launch_bounds__(256) tf_nvfp4_col2(tf::rocm::Nvfp4 a) {
    tf::rocm::nvfp4_columns<2, tf::rocm::kNvDepth>(a);
}
extern "C" __global__ void __launch_bounds__(256) tf_nvfp4_col3(tf::rocm::Nvfp4 a) {
    tf::rocm::nvfp4_columns<3, tf::rocm::kNvDepth>(a);
}
extern "C" __global__ void __launch_bounds__(256) tf_nvfp4_col4(tf::rocm::Nvfp4 a) {
    tf::rocm::nvfp4_columns<4, tf::rocm::kNvDepth>(a);
}
