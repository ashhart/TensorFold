// NaN and infinite values in a tensor-core attention tile's V: a key a row cannot see adds nothing to that row's sum.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
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

// A KEYS-key tile staged [key][D] (xor swizzle): chunks tid, tid + threads, ... (the ones tid copied) made finite.
template <int D, int KEYS>
__device__ __forceinline__ void clean_rows(uint4* vs, int tid, int threads) {
    for (int i = tid; i < KEYS * (D / 8); i += threads) {
        uint4* p = vs + (i / (D / 8)) * (D / 8) + ((i % (D / 8)) ^ ((i / (D / 8)) & 7));
        const uint4 x = *p;
        *p = make_uint4(finite2(x.x), finite2(x.y), finite2(x.z), finite2(x.w));
    }
}

// A 32-key tile staged [key][D] (xor swizzle): the chunks this thread copied, their NaN and infinite values set to 0.
template <int D>
__device__ __forceinline__ void clean(uint4* vs, int threads) {
    clean_rows<D, 32>(vs, threadIdx.x, threads);
}

// x a, an infinite x (the replay's, already weighed in float64) left as it is whatever the rescale.
__device__ __forceinline__ float keep(float x, float a) { return isinf(x) ? x : __fmul_rn(x, a); }

// x a + y folding partials, an infinite x left unscaled (see keep).
__device__ __forceinline__ float fold(float x, float a, float y) {
    return isinf(x) ? __fadd_rn(x, y) : __fmaf_rn(x, a, y);
}

// The least x whose float64 exp is not 0: a key whose score is further below its row's max weighs 0 in attention_ref.
__device__ __forceinline__ double least() { return __longlong_as_double(0xc0874910d52d3051ll); }

// A query row's float64 score against a key row, attention_ref's order: sum q_e k_e over e, then times scale.
template <int D>
__device__ __forceinline__ double score64(const __nv_bfloat16* q, const __nv_bfloat16* k, double scale) {
    double s = 0.0;
#pragma unroll 1
    for (int e = 0; e < D; ++e)
        s = __dadd_rn(s, __dmul_rn((double)__bfloat162float(q[e]), (double)__bfloat162float(k[e])));
    return __dmul_rn(s, scale);
}

// Rows g, g + 8: w(r, K row) is attention_ref's exp(score - max) > 0; each(r, f) walks r's K rows (max on first use).
template <int D, typename Q, typename Each>
struct Weights {
    Q q;
    Each each;
    double scale;
    int g;
    double m[2] = {-INFINITY, -INFINITY};
    bool have[2] = {false, false};

    __device__ __forceinline__ bool operator()(int r, const __nv_bfloat16* k) {
        const int i = r == g ? 0 : 1;
        if (!have[i]) {
            double mx = -INFINITY;
            each(r, [&](const __nv_bfloat16* kr) { mx = fmax(mx, score64<D>(q(r), kr, scale)); });
            m[i] = mx;
            have[i] = true;
        }
        return __dsub_rn(score64<D>(q(r), k, scale), m[i]) >= least(); // NaN: no weight
    }
};

// q(r): row r's query.
template <int D, typename Q, typename Each>
__device__ __forceinline__ Weights<D, Q, Each> weights(Q q, Each each, float scale, int lane) {
    return Weights<D, Q, Each>{q, each, (double)scale, lane / 4};
}

// Keys k0..k1 clean() emptied (V vrow(k), K krow(k)): NaN and infinities to rows that see them, inf x 0 at weight 0.
template <int D, typename Row, typename KRow, typename Vis, typename W>
__device__ __forceinline__ void add_back(float (&o)[D / 8][4], Row vrow, KRow krow, int k0, int k1, int lane, Vis vis,
                                         float l0, float l1, W& w) {
    const int g = lane / 4, qd = lane % 4;
    for (int k = k0; k < k1; ++k) {
        const bool v0 = l0 > 0.0f && vis(g, k), v1 = l1 > 0.0f && vis(g + 8, k); // l 0: no key weighs, 0 / 0 stays NaN
        if (!v0 && !v1) continue;
        const uint32_t* v = reinterpret_cast<const uint32_t*>(vrow(k));
        bool odd = false;
#pragma unroll
        for (int i = 0; i < D / 8; ++i) odd |= finite2(v[4 * i + qd]) != v[4 * i + qd];
        if (!odd) continue;
        const bool w0 = v0 && w(g, krow(k)), w1 = v1 && w(g + 8, krow(k));
#pragma unroll
        for (int i = 0; i < D / 8; ++i) {
            const uint32_t x = v[4 * i + qd];
            if (finite2(x) == x) continue;
            const float lo = __uint_as_float(x << 16), hi = __uint_as_float(x & 0xffff0000u);
            if ((x & 0x7f80u) == 0x7f80u) {
                if (v0) o[i][0] = __fadd_rn(o[i][0], w0 ? lo : __fmul_rn(0.0f, lo));
                if (v1) o[i][2] = __fadd_rn(o[i][2], w1 ? lo : __fmul_rn(0.0f, lo));
            }
            if ((x & 0x7f800000u) == 0x7f800000u) {
                if (v0) o[i][1] = __fadd_rn(o[i][1], w0 ? hi : __fmul_rn(0.0f, hi));
                if (v1) o[i][3] = __fadd_rn(o[i][3], w1 ? hi : __fmul_rn(0.0f, hi));
            }
        }
    }
}

}  // namespace attn_odd
