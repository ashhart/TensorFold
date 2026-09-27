// EXL3 tiles on CUDA: device functions that decode a 16x16 trellis tile of any codebook and width straight into
// the B fragments of two mma.m16n8k16, the definition in format.py (after ExLlamaV3, MIT, Copyright (c) 2025
// Turboderp). Header only, no ATen: include it from any family's kernels.
//
// Widths are given as K2 = 2 * bits: 2, 4, 6, 8, 10, 12, 14, 16 for 1..8 bits, 3, 5, 7 for 1.5, 2.5, 3.5 (mul1).
// Codebooks: CB_3INST, CB_MCG, CB_MUL1.
//
// A tile is 4 * K2 little-endian 32-bit words (the trellis' int16 [.., .., 16 * bits] viewed as int32). Lane L of a
// warp owns the tile's values 8L..8L+7, which are exactly its B fragments of the two mma.m16n8k16 of the tile
// (tile columns 0-7 and 8-15; rows are the k index):
//
//     uint32_t w[tf_exl3::lane_words<K2>()];
//     tf_exl3::load_lane_words<K2>(tile, lane, w);          // 2 or 3 words of the tile, any memory space
//     uint32_t b0[2], b1[2];
//     tf_exl3::decode_lane<K2, CB>(w, lane, b0, b1);         // half2 pairs as uint32
//     mma16816(acc_cols_0_7, a, b0); mma16816(acc_cols_8_15, a, b1);
//
// load_lane_words and decode_lane are split so a kernel can prefetch the words (registers, shared memory) before it
// decodes them. Every function is exact: tf_exl3::decode_lane gives the bits format.py's reference gives.

#pragma once

#include <cstdint>
#include <cuda_fp16.h>

namespace tf_exl3 {

enum Codebook : int { CB_3INST = 0, CB_MCG = 1, CB_MUL1 = 2 };

// 32-bit words of a tile: 256 values of K2 / 2 bits.
template <int K2>
__host__ __device__ constexpr int tile_words() {
    return 4 * K2;
}

// E(p): one past the last stream bit of value p's 16-bit window (format.py's stream_ends).
template <int K2>
__host__ __device__ constexpr int stream_end(int p) {
    return (K2 & 1) ? ((p + 1) * K2 - ((p + 1) & 1)) / 2 : (p + 1) * (K2 / 2);
}

// Words a lane reads for its 8 values: the most any lane's windows span (2, or 3 for 3.5 bits and 5 to 8 bits).
template <int K2>
__host__ __device__ constexpr int lane_words() {
    int most = 0;
    for (int l = 0; l < 32; ++l) {
        const int first = (stream_end<K2>(8 * l) - 16 + 128 * K2) % 32;
        const int need = first + stream_end<K2>(8 * l + 7) - stream_end<K2>(8 * l) + 16;
        most = need > most ? need : most;
    }
    return (most + 31) / 32;
}

// The index of the first word lane `lane` reads, and the bit offset of its first window in that word.
template <int K2>
__device__ __forceinline__ void lane_start(int lane, int& word, int& offset) {
    const int first = 4 * lane * K2 + K2 / 2 - 16 + 128 * K2;   // E(8 lane) - 16, made non-negative
    word = (first >> 5) % tile_words<K2>();
    offset = first & 31;
}

// This lane's words of a tile (tile: the tile's first 32-bit word; global or shared memory).
template <int K2>
__device__ __forceinline__ void load_lane_words(const uint32_t* tile, int lane, uint32_t (&w)[lane_words<K2>()]) {
    int word, offset;
    lane_start<K2>(lane, word, offset);
#pragma unroll
    for (int i = 0; i < lane_words<K2>(); ++i) w[i] = tile[(word + i) % tile_words<K2>()];
}

// The same from global memory through the read-only cache.
template <int K2>
__device__ __forceinline__ void ldg_lane_words(const uint32_t* tile, int lane, uint32_t (&w)[lane_words<K2>()]) {
    int word, offset;
    lane_start<K2>(lane, word, offset);
#pragma unroll
    for (int i = 0; i < lane_words<K2>(); ++i) w[i] = __ldg(tile + (word + i) % tile_words<K2>());
}

// Bits [d, d + 16) of the 96-bit big-endian stream hi:mid:lo (d a compile-time constant after unrolling).
__device__ __forceinline__ uint32_t window16(uint32_t hi, uint32_t mid, uint32_t lo, int d) {
    if (d <= 16) return (hi >> (16 - d)) & 0xffffu;
    if (d < 32) return __funnelshift_l(mid, hi, d) >> 16;
    if (d <= 48) return (mid >> (48 - d)) & 0xffffu;
    return __funnelshift_l(lo, mid, d - 32) >> 16;
}

// The 16-bit states of this lane's 8 values (8 lane + j, j = 0..7) from its words.
template <int K2>
__device__ __forceinline__ void lane_states(const uint32_t (&w)[lane_words<K2>()], int lane, uint32_t (&s)[8]) {
    int word, offset;
    lane_start<K2>(lane, word, offset);
    const uint32_t c = lane_words<K2>() > 2 ? w[lane_words<K2>() > 2 ? 2 : 0] : 0u;
    const uint32_t hi = __funnelshift_l(w[1], w[0], offset);   // the stream from the lane's first window on
    const uint32_t mid = __funnelshift_l(c, w[1], offset);
    const uint32_t lo = c << offset;
#pragma unroll
    for (int j = 0; j < 8; ++j) s[j] = window16(hi, mid, lo, stream_end<K2>(j) - stream_end<K2>(0));
}

// Two codebook values from two 16-bit states, as a half2 in a uint32 (the first state in the low half).
template <int CB>
__device__ __forceinline__ uint32_t decode2(uint32_t s0, uint32_t s1) {
    if constexpr (CB == CB_MUL1) {
        const uint32_t x0 = s0 * 0x83DCD12Du, x1 = s1 * 0x83DCD12Du;
        const uint32_t h0 = __dp4a(x0, 0x01010101u, 0x6400u);    // fp16 bits of 1024 + the byte sum
        const uint32_t h1 = __dp4a(x1, 0x01010101u, 0x6400u);
        const uint32_t hh = (h0 & 0xffffu) | (h1 << 16);
        const half2 r = __hfma2(*reinterpret_cast<const half2*>(&hh), __half2half2(__ushort_as_half(0x1eee)),
                                __half2half2(__ushort_as_half(0xc931)));
        return *reinterpret_cast<const uint32_t*>(&r);
    } else {
        uint32_t x0, x1;
        if constexpr (CB == CB_MCG) {
            x0 = s0 * 0xCBAC1FEDu;
            x1 = s1 * 0xCBAC1FEDu;
        } else {
            x0 = s0 * 89226354u + 64248484u;
            x1 = s1 * 89226354u + 64248484u;
        }
        x0 = (x0 & 0x8FFF8FFFu) ^ 0x3B603B60u;
        x1 = (x1 & 0x8FFF8FFFu) ^ 0x3B603B60u;
        const uint32_t lo = __byte_perm(x0, x1, 0x5410);
        const uint32_t hi = __byte_perm(x0, x1, 0x7632);
        const half2 r = __hadd2(*reinterpret_cast<const half2*>(&lo), *reinterpret_cast<const half2*>(&hi));
        return *reinterpret_cast<const uint32_t*>(&r);
    }
}

// One codebook value.
template <int CB>
__device__ __forceinline__ half decode1(uint32_t s) {
    const uint32_t v = decode2<CB>(s, s);
    return __ushort_as_half(static_cast<unsigned short>(v & 0xffffu));
}

// This lane's 8 values of a tile as the B fragments of the tile's two n8 halves (columns 0-7 in b0, 8-15 in b1).
template <int K2, int CB>
__device__ __forceinline__ void decode_lane(const uint32_t (&w)[lane_words<K2>()], int lane, uint32_t (&b0)[2],
                                            uint32_t (&b1)[2]) {
    uint32_t s[8];
    lane_states<K2>(w, lane, s);
    b0[0] = decode2<CB>(s[0], s[1]);
    b0[1] = decode2<CB>(s[2], s[3]);
    b1[0] = decode2<CB>(s[4], s[5]);
    b1[1] = decode2<CB>(s[6], s[7]);
}

// Row and column in the 16x16 tile (row = k, column = n) of this lane's value j (0..7).
__device__ __forceinline__ int value_row(int lane, int j) { return 2 * (lane & 3) + (j & 1) + 8 * ((j >> 1) & 1); }
__device__ __forceinline__ int value_col(int lane, int j) { return (lane >> 2) + 8 * (j >> 2); }

// mma.m16n8k16, fp16 inputs, fp32 accumulators: d += a @ b.
__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Fast Walsh-Hadamard transform of 128 fp32 values held 4 per lane (lane L: values 4L..4L+3), natural order, in a
// fixed butterfly order (strides 1, 2 in registers, then 4..64 across lanes). Unscaled: multiply by 1/sqrt(128).
__device__ __forceinline__ void fwht128(float (&v)[4], int lane) {
    const float a = v[0] + v[1], b = v[0] - v[1], c = v[2] + v[3], d = v[2] - v[3];
    v[0] = a + c;
    v[1] = b + d;
    v[2] = a - c;
    v[3] = b - d;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float o = __shfl_xor_sync(0xffffffffu, v[j], m);
            v[j] = (lane & m) ? o - v[j] : v[j] + o;
        }
    }
}

constexpr float HAD_SCALE = 0.08838834764831845f;   // 1 / sqrt(128)

}  // namespace tf_exl3
