#pragma once

// The ModelOpt NVFP4 decoder: 4-bit e2m1 codes packed in 32-bit words, one E4M3 group scale a 16 inputs, no bias.
// It offers what MlxDecoder offers, so the shared tiles take it: only the code's value, the bias and the global differ.

#include <type_traits>
#include <utility>

#include "common/vec.hpp"
#include "quant/mlx_pieces.hpp"
#include "quant/nvfp4.hpp"

namespace tf {
namespace rocm {

using TableBits = unsigned __attribute__((ext_vector_type(2)));

template <int BITS>
struct Nvfp4Decoder {
    using Args = Nvfp4;
    static constexpr int kBits = BITS;
    // Words of a 32-code chunk of a column.
    static constexpr int kWords = BITS;
    // No bias a group: the tiles leave out the x sums and the bias folds.
    static constexpr bool kBias = false;

    // ---- the plan: a routed item's expert and rows ----

    __device__ static bool take_item(Args& a, int z) { return tf::rocm::take_item(a, z); }

    // ---- a column's codes ----

    __device__ static const uint32_t* words(const Args& a) { return a.words; }

    template <typename Idx>
    __device__ static Idx row_words(const Args& a) {
        return static_cast<Idx>(a.k) * BITS / 32;
    }

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

    // A code's e2m1 level as the activation type's bits: one load from the sixteen, no conversion in flight.
    template <typename T>
    __device__ static unsigned short level(uint32_t c) {
        if constexpr (std::is_same_v<typename T::elem, __half>) return kE2m1F16[c & 0xfu];
        else return kE2m1Bf16[c & 0xfu];
    }

    // Two levels as the activation type's pair.
    template <typename T>
    __device__ static typename T::pair pack(unsigned short lo, unsigned short hi) {
        return __builtin_bit_cast(typename T::pair, static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16));
    }

    // Code t of a chunk as the Dot's element type.
    template <typename T>
    __device__ static typename T::elem code(const uint32_t (&w)[BITS], int t) {
        return __builtin_bit_cast(typename T::elem, level<T>(piece_bits<BITS>(w, t)));
    }

    // Codes t and t + 1 of a chunk, in order, as a pair.
    template <typename T>
    __device__ static typename T::pair pair(const uint32_t (&w)[BITS], int t) {
        return pack<T>(level<T>(piece_bits<BITS>(w, t)), level<T>(piece_bits<BITS>(w, t + 1)));
    }

    // ---- the group terms: a scale a group of a column, the one global scale folded in ----

    struct Term {
        TableBits scale;
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

    __device__ static Term term(const Args& a, long long i) { return Term{table_bits(a.scale, i)}; }

    __device__ static float scale_value(const Args& a, const Term& t) {
        return table_float(a.scale, t.scale) * a.global;
    }
    __device__ static float bias_value(const Args&, const Term&) { return 0.f; }

    __device__ static float fold_scale(float acc, float dot, float scale) { return fmaf(dot, scale, acc); }
    __device__ static float fold_bias(float acc, float, float) { return acc; }

    // A non-finite stream sum: replay full groups only when an actual operand is non-finite.
    template <typename T>
    __device__ __attribute__((always_inline)) static inline float repair_stream(const Args& a, const typename T::elem* x,
                                                                   int col, float original) {
        const int groups = a.k / a.group;
        const int words_row = row_words<int>(a);
        const uint32_t* w = a.words + static_cast<long long>(col) * words_row;
        bool exceptional = false;
        double y = 0.0;
        for (int g = 0; g < groups; ++g) {
            const double scale = static_cast<double>(a.scale[static_cast<long long>(col) * groups + g]) *
                                 static_cast<double>(a.global);
            exceptional |= !__builtin_isfinite(scale);
            double qx = 0.0;
            for (int k = g * a.group; k < (g + 1) * a.group; ++k) {
                const double v = static_cast<float>(x[k]);
                const double q = kE2m1[affine_code(w, k, BITS, words_row) & 0xfu];
                exceptional |= !__builtin_isfinite(v);
                qx += q * v;
            }
            y += scale * qx;
        }
        return exceptional ? static_cast<float>(y) : original;
    }

    // ---- the decode stream tile: a lane owns a 32-code chunk, its pairs are made with two loads ----

    static constexpr int kWideBytes = BITS % 4 == 0 ? 16 : BITS % 2 == 0 ? 8 : 4;

    __device__ static bool wide_aligned(const Args& a) {
        const long long words_row = static_cast<long long>(a.k) * BITS / 32;
        return ((reinterpret_cast<uintptr_t>(a.words) | static_cast<uintptr_t>(words_row * 4)) & (kWideBytes - 1)) == 0;
    }

    template <class Sides>
    __device__ static void bind(Args& a, const Sides& sides, int s) {
        a.words = sides.words[s];
        a.scale.p = sides.scale[s];
    }

    // Neighbours: a pair is codes 2 I and 2 I + 1, the x pair beside them. The MLX byte path packs integers.
    static constexpr bool kGapPairs = false;

    template <int I>
    static constexpr int lo() {
        return 2 * I;
    }

    static constexpr int gap() { return 1; }

    template <typename T, int I>
    __device__ static typename T::pair stream_pair(const uint32_t (&w)[BITS]) {
        return pair<T>(w, 2 * I);
    }

    // The stream tile's group terms: the 16-bit E4M3-as-BF16 scale, the global folded in where it is widened.
    struct StreamTerms {
        const unsigned short* scale;
        float global;

        __device__ explicit StreamTerms(const Args& a)
            : scale(static_cast<const unsigned short*>(a.scale.p)), global(a.global) {}

        __device__ unsigned short scale_raw(long long at) const { return scale[at]; }
        __device__ unsigned short bias_raw(long long) const { return 0; }

        __device__ float widen(unsigned short v) const { return __uint_as_float(static_cast<uint32_t>(v) << 16) * global; }

        __device__ float scale_value(unsigned short v) const { return widen(v); }
        __device__ float bias_value(unsigned short) const { return 0.f; }
    };
};

}  // namespace rocm
}  // namespace tf
