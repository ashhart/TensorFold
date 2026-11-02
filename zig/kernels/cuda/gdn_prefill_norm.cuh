// Qwen whole-head prompt chain: the recurrence's eight warps also normalize its staged BF16 rows.
#pragma once
#include "common/reduce.cuh"

namespace tf_gdn_prefill {

template <int Fault>
__device__ __forceinline__ void chain_norm(
        const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
        const float* __restrict__ g, const __nv_bfloat16* __restrict__ beta, const float* state,
        float* last, __nv_bfloat16* __restrict__ y, int W, int hk, int hv,
        const __nv_bfloat16* __restrict__ z, const __nv_bfloat16* __restrict__ nw,
        __nv_bfloat16* __restrict__ mix, float* __restrict__ xs, float eps) {
    using QK = __nv_bfloat16;
    using B = __nv_bfloat16;
    constexpr int ROWS = 128, STEPS = 16;
    __shared__ __align__(16) __nv_bfloat16 ys[STEPS][DV];
    __shared__ float4 ks[STEPS][DK / 4], qs[STEPS][DK / 4];
    __shared__ float gs[STEPS], bs[STEPS];
    __shared__ __align__(16) __nv_bfloat16 vs[STEPS][ROWS];
    const int head = blockIdx.x, row = threadIdx.x >> 1, half = threadIdx.x & 1;
    const int key_head = head / (hv / hk);
    float s[64];
    const float* s0 = state + (static_cast<size_t>(head) * DV + row) * DK;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const float4 t = *reinterpret_cast<const float4*>(s0 + 8 * j + 4 * half);
        s[4 * j] = t.x; s[4 * j + 1] = t.y; s[4 * j + 2] = t.z; s[4 * j + 3] = t.w;
    }
    Prefetch<QK, B, 2 * ROWS, ROWS, STEPS> pf;                // each stage's loads in flight during the one before
    pf.fetch(q, k, v, g, beta, 0, W, hk, hv, head, key_head, 0);
    for (int t0 = 0; t0 < W; t0 += STEPS) {
        const int n = min(STEPS, W - t0);
        __syncthreads();
        pf.store(ks, qs, gs, bs, vs);
        __syncthreads();
        if (t0 + STEPS < W) pf.fetch(q, k, v, g, beta, t0 + STEPS, W, hk, hv, head, key_head, 0);
        for (int tt = 0; tt < n; ++tt) {
            const float gt = gs[tt], bt = bs[tt];
            const size_t vat = (static_cast<size_t>(t0 + tt) * hv + head) * DV + row;
            const float vt = __bfloat162float(vs[tt][threadIdx.x >> 1]);
            float m[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const float4 kk = ks[tt][2 * j + half];
                s[4 * j] = s[4 * j] * gt;
                s[4 * j + 1] = s[4 * j + 1] * gt;
                s[4 * j + 2] = s[4 * j + 2] * gt;
                s[4 * j + 3] = s[4 * j + 3] * gt;
                m[0] = __fmaf_rn(s[4 * j], kk.x, m[0]);
                m[1] = __fmaf_rn(s[4 * j + 1], kk.y, m[1]);
                m[2] = __fmaf_rn(s[4 * j + 2], kk.z, m[2]);
                m[3] = __fmaf_rn(s[4 * j + 3], kk.w, m[3]);
            }
            float mem = (m[0] + m[1]) + (m[2] + m[3]);
            mem = mem + __shfl_xor_sync(0xffffffffu, mem, 1);
            const float delta = (vt - mem) * bt;
            float o[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const float4 kk = ks[tt][2 * j + half], qq = qs[tt][2 * j + half];
                s[4 * j] = __fmaf_rn(kk.x, delta, s[4 * j]);
                s[4 * j + 1] = __fmaf_rn(kk.y, delta, s[4 * j + 1]);
                s[4 * j + 2] = __fmaf_rn(kk.z, delta, s[4 * j + 2]);
                s[4 * j + 3] = __fmaf_rn(kk.w, delta, s[4 * j + 3]);
                o[0] = __fmaf_rn(s[4 * j], qq.x, o[0]);
                o[1] = __fmaf_rn(s[4 * j + 1], qq.y, o[1]);
                o[2] = __fmaf_rn(s[4 * j + 2], qq.z, o[2]);
                o[3] = __fmaf_rn(s[4 * j + 3], qq.w, o[3]);
            }
            float out = (o[0] + o[1]) + (o[2] + o[3]);
            out = out + __shfl_xor_sync(0xffffffffu, out, 1);
            if (half == 0) {
                const __nv_bfloat16 rounded = __float2bfloat16_rn(out);
                y[vat] = rounded;
                ys[tt][row] = rounded;
            }
        }
        __syncthreads();
        const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
        for (int tt = warp; tt < n; tt += 8) {
            const size_t at = (static_cast<size_t>(t0 + tt) * hv + head) * DV;
            float yv[4], ss = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                yv[j] = __bfloat162float(ys[tt][lane * 4 + j]);
                ss += yv[j] * yv[j];
            }
            ss = tf_common::warp_sum(ss);
            const float inv = rsqrtf(ss / static_cast<float>(DV) + eps);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int d = lane * 4 + j;
                const float x = __bfloat162float(__float2bfloat16_rn(__bfloat162float(nw[d]) * (yv[j] * inv)));
                const float zf = __bfloat162float(z[at + d]);
                const __nv_bfloat16 o = __float2bfloat16_rn((Fault == 1 ? 1.0f : zf / (1.0f + expf(-zf))) * x);
                mix[at + d] = o;
                ys[tt][d] = o;
            }
            if (xs != nullptr) {
                __syncwarp();
                if (lane < DV / 64) {
                    float sum = 0.0f;
#pragma unroll
                    for (int j = 0; j < 64; ++j) sum += __bfloat162float(ys[tt][lane * 64 + j]);
                    xs[at / 64 + lane] = sum;
                }
            }
        }
    }
    float* s1 = last + (static_cast<size_t>(head) * DV + (Fault == 2 ? (row + 1) % DV : row)) * DK;
#pragma unroll
    for (int j = 0; j < 16; ++j)
        *reinterpret_cast<float4*>(s1 + 8 * j + 4 * half) = make_float4(s[4 * j], s[4 * j + 1], s[4 * j + 2],
                                                                        s[4 * j + 3]);
}

} // namespace tf_gdn_prefill

#define TF_GDN_NORM_ENTRY(NAME, FAULT) \
extern "C" __global__ void __launch_bounds__(256) NAME( \
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, const float* g, \
    const __nv_bfloat16* beta, const float* state, float* last, __nv_bfloat16* y, int W, int hk, int hv, \
    const __nv_bfloat16* z, const __nv_bfloat16* nw, __nv_bfloat16* mix, float* xs, float eps) { \
    tf_gdn_prefill::chain_norm<FAULT>(q, k, v, g, beta, state, last, y, W, hk, hv, z, nw, mix, xs, eps); \
}
TF_GDN_NORM_ENTRY(tf_gdn_chain_norm, 0)
#ifdef TF_GDN_CHAIN_CONTROLS
TF_GDN_NORM_ENTRY(tf_gdn_chain_norm_bad_gate, 1)
TF_GDN_NORM_ENTRY(tf_gdn_chain_norm_bad_state, 2)
#endif
#undef TF_GDN_NORM_ENTRY
