#pragma once

// The MLX affine WeightDecoder: BITS-wide codes packed little-endian in 32-bit words, a scale and a bias a group of K.

#include <type_traits>
#include <utility>

#include "common/vec.hpp"
#include "quant/mlx.hpp"
#include "quant/mlx_pieces.hpp"

namespace tf {
namespace rocm {

// A table entry's bits as two 16-bit loads valid for every kind; a branch or conversion would stall loads in flight.
using TableBits = unsigned __attribute__((ext_vector_type(2)));

template <int BITS>
struct MlxDecoder {
    using Args = Affine;
    static constexpr int kBits = BITS;
    // Words of a 32-code chunk of a column.
    static constexpr int kWords = BITS;

    // ---- the plan: a routed item's expert and rows ----

    __device__ static bool take_item(Args& a, int z) { return tf::rocm::take_item(a, z); }

    // ---- a column's codes ----

    __device__ static const uint32_t* words(const Args& a) { return a.words; }

    // Words of a column's row of K codes, in the integer type the tile does its address arithmetic in.
    template <typename Idx>
    __device__ static Idx row_words(const Args& a) {
        return static_cast<Idx>(a.k) * BITS / 32;
    }

    // A chunk is BITS words: 16-byte loads when BITS % 4 == 0 and WIDE, 8-byte when even, else by word.
    template <bool WIDE>
    __device__ static void load(const uint32_t* src, uint32_t (&w)[BITS]) {
        if constexpr (WIDE && BITS % 4 == 0) {
#pragma unroll
            for (int i = 0; i < BITS / 4; ++i) {
                const u32x4 v = reinterpret_cast<const u32x4*>(src)[i];
                w[4 * i] = v.x;
                w[4 * i + 1] = v.y;
                w[4 * i + 2] = v.z;
                w[4 * i + 3] = v.w;
            }
        } else if constexpr (WIDE && BITS % 2 == 0) {
#pragma unroll
            for (int i = 0; i < BITS / 2; ++i) {
                const uint2 v = reinterpret_cast<const uint2*>(src)[i];
                w[2 * i] = v.x;
                w[2 * i + 1] = v.y;
            }
        } else {
#pragma unroll
            for (int i = 0; i < BITS; ++i) w[i] = src[i];
        }
    }

    // Two codes (below 256, exact in both types) as the activation type's pair.
    template <typename T>
    __device__ static typename T::pair pair_of(uint32_t c0, uint32_t c1) {
        if constexpr (std::is_same_v<typename T::elem, __half>) {
            // 0x6400 | c is 1024 + c in FP16; the subtraction is exact.
            const uint32_t bits = (c0 | (c1 << 16)) | 0x64006400u;
            return __hsub2(__builtin_bit_cast(__half2, bits), __builtin_bit_cast(__half2, 0x64006400u));
        } else {
            // The float of an integer below 256 has its low 16 bits clear, so its top half is the BF16.
            const uint32_t bits = (__builtin_bit_cast(uint32_t, static_cast<float>(c0)) >> 16) |
                                  (__builtin_bit_cast(uint32_t, static_cast<float>(c1)) & 0xffff0000u);
            return __builtin_bit_cast(typename T::pair, bits);
        }
    }

    // Code t of a chunk as the Dot's element type (codes are below 256, exact in both).
    template <typename T>
    __device__ static typename T::elem code(const uint32_t (&w)[BITS], int t) {
        if constexpr (std::is_same_v<typename T::elem, __half>) {
            return piece_code<BITS>(w, t);
        } else {
            return static_cast<typename T::elem>(static_cast<float>(piece_bits<BITS>(w, t)));
        }
    }

    // Codes t and t + 1 of a chunk, in order, as a pair.
    template <typename T>
    __device__ static typename T::pair pair(const uint32_t (&w)[BITS], int t) {
        return pair_of<T>(piece_bits<BITS>(w, t), piece_bits<BITS>(w, t + 1));
    }

    // ---- the group terms: a scale and a bias a group of a column ----

    struct Term {
        TableBits scale;
        TableBits bias;
    };

    __device__ static TableBits table_bits(const GroupTable& t, long long i) {
        const uint16_t* p = static_cast<const uint16_t*>(t.p);
        const bool wide = t.kind == kScaleF32;
        return TableBits{p[wide ? 2 * i : i], p[wide ? 2 * i + 1 : i]};
    }

    __device__ static float table_float(const GroupTable& t, TableBits b) {
        if (t.kind == kScaleBF16) return __uint_as_float(b.x << 16);
        if (t.kind == kScaleF16) return __half2float(__ushort_as_half(static_cast<unsigned short>(b.x)));
        return __uint_as_float(b.x | (b.y << 16));
    }

    // The stored scale and bias of group `i` of a table row set (the tile does the row arithmetic).
    __device__ static Term term(const Args& a, long long i) { return {table_bits(a.scale, i), table_bits(a.bias, i)}; }

    __device__ static float scale_value(const Args& a, const Term& t) { return table_float(a.scale, t.scale); }
    __device__ static float bias_value(const Args& a, const Term& t) { return table_float(a.bias, t.bias); }

    // A group's sum: its dot against the codes times the scale, and the sum of x times the bias.
    __device__ static float fold_scale(float acc, float dot, float scale) { return fmaf(dot, scale, acc); }
    __device__ static float fold_bias(float acc, float sum_x, float bias) { return fmaf(sum_x, bias, acc); }

    // ---- the decode stream tile: a lane owns a 32-code chunk, its pairs are made with a few bit operations ----

    // Bytes the chunk's loads are aligned to when they are wide (16 for 4-word multiples, 8 for even, else by word).
    static constexpr int kWideBytes = BITS % 4 == 0 ? 16 : BITS % 2 == 0 ? 8 : 4;

    // Whether every chunk of the words is wide-aligned: the words and the row stride are.
    __device__ static bool wide_aligned(const Args& a) {
        const long long words_row = static_cast<long long>(a.k) * BITS / 32;
        return ((reinterpret_cast<uintptr_t>(a.words) | static_cast<uintptr_t>(words_row * 4)) & (kWideBytes - 1)) == 0;
    }

    // The words and tables of side s of a group of products that share x.
    template <class Sides>
    __device__ static void bind(Args& a, const Sides& sides, int s) {
        a.words = sides.words[s];
        a.scale.p = sides.scale[s];
        a.bias.p = sides.bias[s];
    }

    // Codes 16 bits apart in one word when the width divides 16 (the pair is two masked fields), else neighbours.
    static constexpr bool kGapPairs = BITS == 2 || BITS == 4 || BITS == 8;

    // The two codes of pair I of a chunk: `lo` and `lo + gap`.
    template <int I>
    static constexpr int lo() {
        if constexpr (BITS == 4) return (I >> 2) * 8 + (I & 3);
        else if constexpr (BITS == 8) return (I >> 1) * 4 + (I & 1);
        else if constexpr (BITS == 2) return (I >> 3) * 16 + (I & 7);
        else return 2 * I;
    }

    static constexpr int gap() {
        if constexpr (BITS == 4) return 4;
        else if constexpr (BITS == 8) return 2;
        else if constexpr (BITS == 2) return 8;
        else return 1;
    }

    // Pair I of the chunk's codes as the activation type's pair (the codes are below 256, exact in both types).
    template <typename T, int I>
    __device__ static typename T::pair stream_pair(const uint32_t (&w)[BITS]) {
        constexpr int low = lo<I>();
        constexpr int high = low + gap();
        uint32_t bits;
        if constexpr (kGapPairs) {
            constexpr int per = 32 / BITS;
            constexpr uint32_t mask = ((1u << BITS) - 1u) * 0x00010001u;
            bits = (w[low / per] >> (BITS * (low % per))) & mask;
        } else {
            bits = piece_bits<BITS>(w, low) | (piece_bits<BITS>(w, high) << 16);
        }
        if constexpr (std::is_same_v<typename T::elem, __half>) {
            bits |= 0x64006400u;
            return __hsub2(__builtin_bit_cast(__half2, bits), __builtin_bit_cast(__half2, 0x64006400u));
        } else {
            // The float of an integer below 256 has its low 16 bits clear, so its top half is the BF16.
            const uint32_t a = __builtin_bit_cast(uint32_t, static_cast<float>(bits & 0xffffu));
            const uint32_t b = __builtin_bit_cast(uint32_t, static_cast<float>(bits >> 16));
            return __builtin_bit_cast(typename T::pair, (a >> 16) | (b & 0xffff0000u));
        }
    }

    // The stream tile's group terms: 16-bit BF16 or FP16 table entries, loaded as stored and widened where used.
    struct StreamTerms {
        const unsigned short* scale;
        const unsigned short* bias;
        bool half;

        struct Raw {
            unsigned short scale;
            unsigned short bias;
        };

        __device__ explicit StreamTerms(const Args& a)
            : scale(static_cast<const unsigned short*>(a.scale.p)),
              bias(static_cast<const unsigned short*>(a.bias.p)),
              half(a.scale.kind == kScaleF16) {}

        __device__ unsigned short scale_raw(long long at) const { return scale[at]; }
        __device__ unsigned short bias_raw(long long at) const { return bias[at]; }

        __device__ float widen(unsigned short v) const {
            return half ? __half2float(__ushort_as_half(v)) : __uint_as_float(static_cast<uint32_t>(v) << 16);
        }

        __device__ float scale_value(Raw r) const { return widen(r.scale); }
        __device__ float bias_value(Raw r) const { return widen(r.bias); }
    };
};

}  // namespace rocm
}  // namespace tf
