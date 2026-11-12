// Router logits on 128 x 128 register tiles for qwen3_5 and Nemotron: each K slice one fma chain in input order.
#pragma once

#include <cuda_bf16.h>
#include <stdint.h>

namespace tf_router_tiles {

constexpr int TT = 128;     // rows and experts a block holds
constexpr int TW = 16;      // words of a K step a stage holds
constexpr int TS = TW + 4;  // a staged row's words: 16-byte copies, eight rows in a quarter warp on distinct banks
constexpr unsigned shared_bytes = 2 * 2 * TT * TS * 4;  // [2 stages][x, w][TT][TS] words

__device__ __forceinline__ void cp16(uint32_t* dst, const void* src, bool read) {
    const uint32_t at = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(at), "l"(src), "r"(read ? 16 : 0));
}

__device__ __forceinline__ uint32_t word(uint4 v, int w) { return w == 0 ? v.x : w == 1 ? v.y : w == 2 ? v.z : v.w; }

// A word's low (half 0) or high bf16 as a float.
__device__ __forceinline__ float unpack(uint32_t v, int half) {
    return __uint_as_float(half ? v & 0xFFFF0000u : v << 16);
}

// Block (blockIdx.x: 128 rows, blockIdx.y: 128 experts, blockIdx.z: slice) of 256 threads and shared_bytes.
__device__ __forceinline__ void tiles(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ W,
                                      float* __restrict__ PART, int R, int D, int E, int SK) {
    extern __shared__ __align__(16) uint32_t staged[];  // [2][x, w][TT][TS]
    const int per = D / SK, steps = per / (2 * TW);
    const int row0 = blockIdx.x * TT, e0 = blockIdx.y * TT, k0 = blockIdx.z * per;
    const int tx = threadIdx.x % 16, ty = threadIdx.x / 16;
    auto load = [&](int step) {
        uint32_t* st = staged + (step & 1) * 2 * TT * TS;
#pragma unroll
        for (int j = 0; j < 4; ++j) {  // 2 x 128 rows of four 16-byte copies
            const int c = threadIdx.x + 256 * j, wr = c >= 4 * TT, r = (c / 4) % TT, q = c % 4;
            const int at = (wr ? e0 : row0) + r;
            const bool live = at < (wr ? E : R);
            const __nv_bfloat16* src =
                (wr ? W : X) + static_cast<int64_t>(live ? at : 0) * D + k0 + step * 2 * TW + q * 8;
            cp16(st + (wr * TT + r) * TS + q * 4, src, live);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };
    float acc[8][8] = {};
    load(0);
    for (int step = 0; step < steps; ++step) {
        if (step + 1 < steps) {
            load(step + 1);
            asm volatile("cp.async.wait_group 1;\n" ::);
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();
        const uint32_t* xs = staged + (step & 1) * 2 * TT * TS;
        const uint32_t* ws = xs + TT * TS;
#pragma unroll
        for (int q = 0; q < TW / 4; ++q) {
            uint4 xv[8], wv[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                xv[i] = *reinterpret_cast<const uint4*>(xs + (ty + 16 * i) * TS + 4 * q);
                wv[i] = *reinterpret_cast<const uint4*>(ws + (tx + 16 * i) * TS + 4 * q);
            }
#pragma unroll
            for (int w = 0; w < 4; ++w)
#pragma unroll
                for (int half = 0; half < 2; ++half) {  // each word's low input, then its high one
                    float xf[8], wf[8];
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        xf[i] = unpack(word(xv[i], w), half);
                        wf[i] = unpack(word(wv[i], w), half);
                    }
#pragma unroll
                    for (int i = 0; i < 8; ++i)
#pragma unroll
                        for (int j = 0; j < 8; ++j) acc[i][j] = __fmaf_rn(xf[i], wf[j], acc[i][j]);
                }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = row0 + ty + 16 * i;
        if (row >= R) continue;
#pragma unroll
        for (int j = 0; j < 8; ++j)
            if (e0 + tx + 16 * j < E)
                PART[(static_cast<int64_t>(blockIdx.z) * R + row) * E + e0 + tx + 16 * j] = acc[i][j];
    }
}

}  // namespace tf_router_tiles
