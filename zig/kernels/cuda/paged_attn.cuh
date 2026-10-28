// Paged attention's shared pieces (Nemotron's chunk/merge kernels, paged_attn_segs.cuh): tiles, pages, stream segments.
#pragma once

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

#include "common/glue_math.cuh"

namespace {

constexpr int GMAX = 16;    // query heads a KV head
constexpr int TILE = 64;    // keys a tile: the online softmax's unit

__device__ __forceinline__ float lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float hi(uint32_t w) { return __uint_as_float(w & 0xFFFF0000u); }

// Max over the 16 lanes of a half warp (xor 8, 4, 2, 1).
__device__ __forceinline__ float half_max(float v) {
#pragma unroll
    for (int off = 8; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    return v;
}

// Sum over the 16 lanes of a half warp (xor 8, 4, 2, 1): each lane ends with the same sum.
__device__ __forceinline__ float half_sum(float v) {
#pragma unroll
    for (int off = 8; off > 0; off >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, off));
    return v;
}

constexpr int PAGE = 512;  // rows a KV page holds: one attention chunk

// Element offset of chunk c's first key row: contiguous rows, or the page the stream's table names (PT null: rows).
__device__ __forceinline__ int64_t chunk_at(const int* PT, int64_t PS, int c, int KVD) {
    return PT ? static_cast<int64_t>(PT[c]) * PS : static_cast<int64_t>(c) * PAGE * KVD;
}

// Element offset of key row k: its chunk's first row, then its row inside the chunk.
__device__ __forceinline__ int64_t key_at(const int* PT, int64_t PS, int64_t k, int KVD) {
    return chunk_at(PT, PS, static_cast<int>(k / PAGE), KVD) + (k % PAGE) * KVD;
}

}  // namespace

// Stream k's rows from ROW0[k] on, for the shared-window kernels.
constexpr int MAXSEG = 8;
struct AttnSegs {
    __nv_bfloat16* KC[MAXSEG];
    __nv_bfloat16* VC[MAXSEG];
    const int* META[MAXSEG];
    const int* PT[MAXSEG];  // each stream's page table (null: its rows are contiguous)
    long long PS;           // elements between pages
    int ROW0[MAXSEG];
    int N;
};

__device__ __forceinline__ int seg_of(const AttnSegs& sg, int rq) {
    int k = 0;
    while (k + 1 < sg.N && sg.ROW0[k + 1] <= rq) ++k;
    return k;
}
