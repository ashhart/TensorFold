#pragma once

// The BF16 dense WeightDecoder: a column's K weights stored as BF16, two a 32-bit word; no scales, no biases.

#include <type_traits>

#include "common/vec.hpp"
#include "quant/mlx.hpp"

namespace tf {
namespace rocm {

// The GEMM tile's decoder over raw BF16 weights (Affine args with bits 16; its tables are unused). Every group of
// `a.group` K folds into the accumulator with scale 1, so the sums are the fp32 dot2 chain, group after group.
struct Bf16Decoder {
    using Args = Affine;
    static constexpr int kBits = 16;
    // Words of a 32-weight chunk of a column.
    static constexpr int kWords = 16;
    // No bias term: the tile neither sums x nor folds a bias.
    static constexpr bool kBias = false;

    // A plain product only: a routed plan's items are not this decoder's.
    __device__ static bool take_item(Args& a, int) { return a.route.items == nullptr; }

    __device__ static const uint32_t* words(const Args& a) { return a.words; }

    template <typename Idx>
    __device__ static Idx row_words(const Args& a) {
        return static_cast<Idx>(a.k) / 2;
    }

    // A chunk is 64 bytes: four 16-byte loads (the host checks the alignment).
    template <bool WIDE>
    __device__ static void load(const uint32_t* src, uint32_t (&w)[kWords]) {
#pragma unroll
        for (int i = 0; i < kWords / 4; ++i) {
            const u32x4 v = reinterpret_cast<const u32x4*>(src)[i];
            w[4 * i] = v.x;
            w[4 * i + 1] = v.y;
            w[4 * i + 2] = v.z;
            w[4 * i + 3] = v.w;
        }
    }

    // Weights t and t + 1 of a chunk (t even): the word that holds them, as stored.
    template <typename T>
    __device__ static typename T::pair pair(const uint32_t (&w)[kWords], int t) {
        static_assert(std::is_same_v<typename T::elem, __bf16>, "BF16 weights take BF16 activations");
        return __builtin_bit_cast(typename T::pair, w[t >> 1]);
    }

    struct Term {};

    __device__ static Term term(const Args&, long long) { return {}; }
    __device__ static float scale_value(const Args&, const Term&) { return 1.f; }
    __device__ static float bias_value(const Args&, const Term&) { return 0.f; }

    // A group's dot added as it is: the scale is 1.
    __device__ static float fold_scale(float acc, float dot, float) { return acc + dot; }
    __device__ static float fold_bias(float acc, float, float) { return acc; }
};

}  // namespace rocm
}  // namespace tf
