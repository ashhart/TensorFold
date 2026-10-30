#pragma once

// Flash Next's hyper-connection gates, up product and mix in one launch: the chain's roundings, the up sum reordered.

// Compiled for 4 streams of 2,560, low rank 320 (+ 4 inject columns), MLX affine 4-bit groups of 32, bf16.

#include "../ops/common.hpp"
#include "common/dot2.hpp"

namespace {

constexpr int kUmS = 4;                         // streams
constexpr int kUmH = 2560;                      // a stream's width
constexpr int kUmL = 320;                       // low rank
constexpr int kUmRows = 8;                      // rows a block takes (a verify's 5-8 rows read the weights once)
constexpr int kUmThreads = 256;
constexpr int kUmChunks = kUmL / 32;            // 10 chunks of 32 codes an up row
constexpr int kUmHalf = kUmChunks / 2;          // two lanes a row, 5 chunks each
constexpr int kUmRowWords = kUmL / 8;           // 40 words an up row
constexpr int kUmCols = kUmThreads / (2 * kUmS);  // 32 hidden columns a block

typedef __bf16 um_bf16x2 __attribute__((ext_vector_type(2)));
typedef unsigned um_u32x4 __attribute__((ext_vector_type(4)));

__device__ __forceinline__ float um_dot(uint32_t x, uint32_t q, float acc) {
#if TF_DEVICE_BF16_DOT2
    return __builtin_amdgcn_fdot2_f32_bf16(__builtin_bit_cast(um_bf16x2, x), __builtin_bit_cast(um_bf16x2, q), acc,
                                           false);
#else
    __builtin_trap();
    return acc;
#endif
}

// Codes 2i and 2i + 1 of a 32-code chunk (4 words, 8 little-endian nibbles each) as a bf16 pair.
__device__ __forceinline__ uint32_t um_codes(const uint32_t (&w)[4], int i) {
    const uint32_t word = w[i >> 2];
    const int sh = (i & 3) * 8;
    const uint32_t c0 = (word >> sh) & 15u, c1 = (word >> (sh + 4)) & 15u;
    return (__float_as_uint(static_cast<float>(c0)) >> 16) | (__float_as_uint(static_cast<float>(c1)) & 0xffff0000u);
}

}  // namespace

// Block (column block, row group): z's gates, then 32 hidden columns' 4 up rows and their mix through shared memory.
extern "C" __global__ void __launch_bounds__(kUmThreads)
tf_fn_hc_up_mix(const float* z, int n, const uint32_t* words, const uint16_t* scales, const uint16_t* biases,
                const uint16_t* x, uint16_t* mixed, uint16_t* inject, int rows) {
    __shared__ __attribute__((aligned(16))) uint16_t act[kUmRows][kUmL];
    __shared__ float sxs[kUmRows][kUmChunks];
    __shared__ float terms[kUmRows][kUmS][kUmCols];
    const int r0 = blockIdx.y * kUmRows;
    const int nr = rows - r0 < kUmRows ? rows - r0 : kUmRows;
    const int t = threadIdx.x;
    const int lane = t & 31, wave = t >> 5;
    const int half = lane & 1;
    const int sidx = wave & (kUmS - 1), dd = (wave >> 2) * 16 + (lane >> 1);
    const int d = blockIdx.x * kUmCols + dd;
    const long long urow = static_cast<long long>(sidx) * kUmH + d;

    // the up weights go out before the gates
    uint32_t w[kUmHalf][4];
    uint16_t sr[kUmHalf], br[kUmHalf];
#pragma unroll
    for (int i = 0; i < kUmHalf; ++i) {
        const int q = half * kUmHalf + i;
        const um_u32x4 v = reinterpret_cast<const um_u32x4*>(words + urow * kUmRowWords)[q];
        w[i][0] = v.x;
        w[i][1] = v.y;
        w[i][2] = v.z;
        w[i][3] = v.w;
        sr[i] = scales[urow * kUmChunks + q];
        br[i] = biases[urow * kUmChunks + q];
    }

    for (int r = 0; r < nr; ++r) {
        const long long zr = static_cast<long long>(r0 + r) * n;
        for (int col = t; col < n; col += kUmThreads) {
            const float v = rounded(rounded(z[zr + col], 2) / static_cast<float>(kUmS), 2);
            if (col < kUmL) {
                act[r][col] = float_to_bf16(silu(v));
            } else if (blockIdx.x == 0) {
                inject[(r0 + r) * kUmS + col - kUmL] = float_to_bf16(2.f * rounded(sigmoid(v), 2));
            }
        }
    }
    __syncthreads();
    if (t < nr * kUmChunks) {
        const int r = t / kUmChunks, q = t % kUmChunks;
        const uint32_t* a = reinterpret_cast<const uint32_t*>(&act[r][q * 32]);
        float s = 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i) s = um_dot(a[i], 0x3f803f80u, s);
        sxs[r][q] = s;
    }
    __syncthreads();

    for (int r = 0; r < nr; ++r) {
        float acc = 0.f;
#pragma unroll
        for (int i = 0; i < kUmHalf; ++i) {
            const int q = half * kUmHalf + i;
            const um_u32x4* xa = reinterpret_cast<const um_u32x4*>(&act[r][q * 32]);
            float dot = 0.f;
#pragma unroll
            for (int v4 = 0; v4 < 4; ++v4) {
                const um_u32x4 u = xa[v4];
                dot = um_dot(u.x, um_codes(w[i], 4 * v4), dot);
                dot = um_dot(u.y, um_codes(w[i], 4 * v4 + 1), dot);
                dot = um_dot(u.z, um_codes(w[i], 4 * v4 + 2), dot);
                dot = um_dot(u.w, um_codes(w[i], 4 * v4 + 3), dot);
            }
            // a group's sum as the stream tile folds a chunk: the dot times the scale, then sum(x) times the bias
            acc = fmaf(dot, bf16_to_float(sr[i]), acc);
            acc = fmaf(sxs[r][q], bf16_to_float(br[i]), acc);
        }
        const float g = acc + __shfl_xor(acc, 1, 32);
        const long long xr = static_cast<long long>(r0 + r) * kUmS * kUmH;
        const float gate = rounded(sigmoid(rounded(g, 2)), 2);
        if (half == 0) terms[r][sidx][dd] = rounded(gate * bf16_to_float(x[xr + urow]), 2);
    }
    __syncthreads();
    // each column's four terms in stream order, summed as tf_fn_hc_mix sums them
    if (t < nr * kUmCols) {
        const int r = t / kUmCols, c = t % kUmCols;
        float total = 0.f;
        total += terms[r][0][c];
        total += terms[r][1][c];
        total += terms[r][2][c];
        total += terms[r][3][c];
        mixed[static_cast<long long>(r0 + r) * kUmH + blockIdx.x * kUmCols + c] =
            float_to_bf16(total / static_cast<float>(kUmS));
    }
}
