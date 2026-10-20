// qk_rows.cu's shared parts: where heads come from and go, the norm weights, cos and sin.

#pragma once

#include <cuda_bf16.h>
#include <stdint.h>

namespace tf_qk {

constexpr int MOST = 256; // tf_qk_rows' head_dim at most

// Where a row's heads come from: q, k and v (null: none) at row stride *_ld and head stride *_hs (elements).
struct QkSrc {
    const __nv_bfloat16* q;
    long long q_ld, q_hs;
    const __nv_bfloat16* k;
    long long k_ld, k_hs;
    const __nv_bfloat16* v;
    long long v_ld, v_hs;
};

// The q and k norm weights: fp32 or bf16 (null q: no norm).
struct QkW {
    const void* q;
    const void* k;
    int bf16;
};

__device__ __forceinline__ float wv(const QkW& w, bool is_k, int i) {
    const void* p = is_k ? w.k : w.q;
    return w.bf16 ? __bfloat162float(static_cast<const __nv_bfloat16*>(p)[i]) : static_cast<const float*>(p)[i];
}

// cos and sin of p's angle on rotated dim d: from tables [positions, rot], or computed from inv [rot / 2].
__device__ __forceinline__ void rope_at(const float* cosv, const float* sinv, const float* inv, int p, int rot, int d,
                                        float& c, float& sn) {
    if (inv != nullptr) {
        const float a = __fmul_rn(static_cast<float>(p), inv[d < rot / 2 ? d : d - rot / 2]);
        c = cosf(a);
        sn = sinf(a);
    } else {
        c = cosv[static_cast<int64_t>(p) * rot + d];
        sn = sinv[static_cast<int64_t>(p) * rot + d];
    }
}

// Where a row's q, k and v (null: none) go, at row stride *_ld and head stride *_hs (elements).
struct QkDst {
    __nv_bfloat16* q;
    long long q_ld, q_hs;
    __nv_bfloat16* k;
    long long k_ld, k_hs;
    __nv_bfloat16* v;
    long long v_ld, v_hs;
};

__device__ __forceinline__ const __nv_bfloat16* head_src(const QkSrc& s, int64_t r, int hd, int h, int hk) {
    if (hd < h) return s.q + r * s.q_ld + hd * s.q_hs;
    if (hd < h + hk) return s.k + r * s.k_ld + (hd - h) * s.k_hs;
    return s.v + r * s.v_ld + (hd - h - hk) * s.v_hs;
}

}  // namespace tf_qk
