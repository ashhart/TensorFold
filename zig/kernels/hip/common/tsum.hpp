#pragma once

// CNT wave sums at once, each bitwise its own xor butterfly's, in fewer moves (DPP row_xmask and permlanex16).

// Ported from Strata (src/kernels/cuda/s26_tsum.cuh@4b9e05a2); MIT License, full text in LICENSES/Strata-MIT.txt.

#include <hip/hip_runtime.h>

namespace tf {
namespace rocm {

// The value of lane `lane ^ O` (wave32): permlanex16 across the two rows of 16, DPP row_xmask inside a row.
template <int O>
__device__ __forceinline__ float tsum_move(float v) {
    if constexpr (O == 16) {
        return __int_as_float(__builtin_amdgcn_permlanex16(__float_as_int(v), __float_as_int(v), 0x76543210u, 0xfedcba98u,
                                                           false, false));
    } else {
        return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x160 + O, 0xf, 0xf, false));
    }
}

template <int O>
__device__ __forceinline__ float tsum_plain(float x) {
    if constexpr (O >= 1) {
        x += tsum_move<O>(x);
        return tsum_plain<O / 2>(x);
    } else {
        return x;
    }
}

// v[0..CNT), CNT a power of two up to 32: this lane's values; returns the sum of value lane >> (5 - log2 CNT).
template <int CNT, int O = 16>
__device__ __forceinline__ float tsum(float* v, int lane) {
    if constexpr (CNT > 1) {
        constexpr int H = CNT / 2;
        const bool hi = (lane & O) != 0;
#pragma unroll
        for (int j = 0; j < H; ++j) {
            const float send = hi ? v[j] : v[j + H];
            const float keep = hi ? v[j + H] : v[j];
            v[j] = keep + tsum_move<O>(send);
        }
        return tsum<H, O / 2>(v, lane);
    } else {
        return tsum_plain<O>(v[0]);
    }
}

constexpr int tsum_log2(int p) { return p <= 1 ? 0 : 1 + tsum_log2(p / 2); }

}  // namespace rocm
}  // namespace tf
