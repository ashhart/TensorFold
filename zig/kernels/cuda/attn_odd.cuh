// NaN and infinite values in a tensor-core attention tile's V: a key a row cannot see adds nothing to that row's sum.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace attn_odd {

// Each bf16 half of x that is NaN or an infinity (all-ones exponent) as +0, the finite halves as they are.
__device__ __forceinline__ uint32_t finite2(uint32_t x) {
    return x & (((x & 0x7f80u) == 0x7f80u ? 0u : 0xffffu) | ((x & 0x7f800000u) == 0x7f800000u ? 0u : 0xffff0000u));
}

// Some NaN or infinite value in a warp's mma rows g (r0: a real row) and g + 8 (r1) of its fold.
template <int NB>
__device__ __forceinline__ bool nonfinite_out(const float (&o)[NB][4], bool r0, bool r1) {
    float a = 0.0f, b = 0.0f; // x - x is 0 for a finite x, NaN for NaN and the infinities
#pragma unroll
    for (int i = 0; i < NB; ++i) {
        a = __fadd_rn(a, __fadd_rn(__fsub_rn(o[i][0], o[i][0]), __fsub_rn(o[i][1], o[i][1])));
        b = __fadd_rn(b, __fadd_rn(__fsub_rn(o[i][2], o[i][2]), __fsub_rn(o[i][3], o[i][3])));
    }
    return (r0 && a != a) || (r1 && b != b);
}

// A 32-key tile staged [key][D] (xor swizzle): the chunks this thread copied, their NaN and infinite values set to 0.
template <int D>
__device__ __forceinline__ void clean(uint4* vs, int threads) {
    for (int i = threadIdx.x; i < 32 * (D / 8); i += threads) {
        uint4* p = vs + (i / (D / 8)) * (D / 8) + ((i % (D / 8)) ^ ((i / (D / 8)) & 7));
        const uint4 x = *p;
        *p = make_uint4(finite2(x.x), finite2(x.y), finite2(x.z), finite2(x.w));
    }
}

// Keys k0..k1 (V row at vrow(k)) that clean() emptied: their NaN and infinite values to the rows that see them.
template <int D, typename Row, typename Vis>
__device__ __forceinline__ void add_back(float (&o)[D / 8][4], Row vrow, int k0, int k1, int lane, Vis vis, float l0,
                                         float l1) {
    const int g = lane / 4, qd = lane % 4;
    for (int k = k0; k < k1; ++k) {
        const bool v0 = l0 > 0.0f && vis(g, k), v1 = l1 > 0.0f && vis(g + 8, k); // l 0: no key weighs, 0 / 0 stays NaN
        if (!v0 && !v1) continue;
        const uint32_t* v = reinterpret_cast<const uint32_t*>(vrow(k));
#pragma unroll
        for (int i = 0; i < D / 8; ++i) {
            const uint32_t x = v[4 * i + qd];
            if (finite2(x) == x) continue;
            const float lo = __uint_as_float(x << 16), hi = __uint_as_float(x & 0xffff0000u);
            if ((x & 0x7f80u) == 0x7f80u) {
                if (v0) o[i][0] = __fadd_rn(o[i][0], lo);
                if (v1) o[i][2] = __fadd_rn(o[i][2], lo);
            }
            if ((x & 0x7f800000u) == 0x7f800000u) {
                if (v0) o[i][1] = __fadd_rn(o[i][1], hi);
                if (v1) o[i][3] = __fadd_rn(o[i][3], hi);
            }
        }
    }
}

}  // namespace attn_odd
