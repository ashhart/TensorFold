#pragma once
// Shared TFQ3 (2D-AffineCoset/half4 + 2D FWHT-128) CUDA device primitives.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

struct Tfq3Tables {
    uint32_t magic;
    uint32_t version;
    float    a3, b3, a4, b4, a5, b5;
    uint32_t pad[8];
    uint32_t lut3_u32[256];
    uint32_t lut4_u32[1024];
    uint32_t lut5_u32[1024];
    uint2    c64_3[64];
    uint2    c64_4[64];
    uint2    c64_5[64];
};

struct Tfq3ExpertMeta {
    uint32_t b_bits;
    uint32_t heat_rank;
    uint64_t gate_word_off;
    uint64_t up_word_off;
    uint64_t down_word_off;
    uint32_t gate_sc_off;
    uint32_t up_sc_off;
    uint32_t down_sc_off;
    float    gate_snr_db;
    float    up_snr_db;
    float    down_snr_db;
    uint32_t reserved[2];
};

__device__ __forceinline__ float tfq3_sgn(uint32_t seed, uint32_t i) {
    uint32_t h = (seed ^ i) * 2654435761u;
    return ((h >> 31) & 1u) ? -1.0f : 1.0f;
}

__device__ __forceinline__ int2 tfq3_frag_home(uint32_t lane) {
    int x = static_cast<int>(((lane & 1u) + ((lane >> 3) & 1u) * 2u) * 4u);
    int y = static_cast<int>(((lane >> 1) & 3u) + ((lane >> 4) & 1u) * 4u);
    return make_int2(x, y);
}

__device__ __forceinline__ uint32_t tfq3_sr0(uint32_t r0, uint32_t sr_split) {
    return (sr_split != 0u && r0 >= sr_split) ? (r0 - sr_split) : r0;
}

__device__ __forceinline__ __half2 tfq3_u32_to_h2(uint32_t u) {
    return *reinterpret_cast<const __half2*>(&u);
}

__device__ __forceinline__ void tfq3_fwht128_simd4(float4& v, uint32_t lane) {
    #pragma unroll
    for (uint32_t d = 1u; d <= 16u; d <<= 1) {
        float ox = __shfl_xor_sync(0xFFFFFFFFu, v.x, d);
        float oy = __shfl_xor_sync(0xFFFFFFFFu, v.y, d);
        float oz = __shfl_xor_sync(0xFFFFFFFFu, v.z, d);
        float ow = __shfl_xor_sync(0xFFFFFFFFu, v.w, d);
        bool lo = (lane & d) == 0u;
        v.x = lo ? (v.x + ox) : (ox - v.x);
        v.y = lo ? (v.y + oy) : (oy - v.y);
        v.z = lo ? (v.z + oz) : (oz - v.z);
        v.w = lo ? (v.w + ow) : (ow - v.w);
    }
    float a0 = v.x + v.y, b0 = v.x - v.y;
    float a1 = v.z + v.w, b1 = v.z - v.w;
    constexpr float inv = 0.08838834764831845f;
    v = make_float4((a0 + a1) * inv, (b0 + b1) * inv, (a0 - a1) * inv, (b0 - b1) * inv);
}

__device__ __forceinline__ void tfq3_store(void* out, uint64_t idx, float v, uint32_t out_f32) {
    if (out_f32) {
        reinterpret_cast<float*>(out)[idx] = v;
    } else {
        reinterpret_cast<__nv_bfloat16*>(out)[idx] = __float2bfloat16(v);
    }
}

__device__ __forceinline__ float tfq3_glu(float g, float u, float lim, uint32_t act_mode) {
    float gv = __bfloat162float(__float2bfloat16(g));
    float uv = __bfloat162float(__float2bfloat16(u));
    if (act_mode == 0u && lim > 0.0f) {
        float fl = __bfloat162float(__float2bfloat16(lim));
        gv = fminf(gv, fl);
        uv = fminf(fmaxf(uv, -fl), fl);
    }
    float sg = gv / (1.0f + __expf(-gv));
    if (act_mode != 0u) sg = __bfloat162float(__float2bfloat16(sg));
    return __bfloat162float(__float2bfloat16(sg * uv));
}

__device__ __forceinline__ void tfq3_q8_4b(
    uint32_t& st, uint32_t wc, const Tfq3Tables* __restrict__ T,
    __half2& q0a, __half2& q0b, __half2& q1a, __half2& q1b
) {
    uint32_t h0 = st * 0x9E3779B1u;
    uint2 cs0 = __ldg(&T->c64_4[(h0 >> 16) & 63u]);
    __half2 s0 = tfq3_u32_to_h2(cs0.x), t0 = tfq3_u32_to_h2(cs0.y);
    uint32_t hl0 = h0 >> 22, c0 = hl0 & 3u, x0 = (hl0 >> 2) & 255u;
    uint32_t sym0 = wc >> 24;
    q0a = __hfma2(tfq3_u32_to_h2(__ldg(&T->lut4_u32[((sym0 ^ x0) << 2) | c0])), s0, t0);
    st = (st << 8) | sym0;

    uint32_t hl1 = (st * 0x9E3779B1u) >> 22, c1 = hl1 & 3u, x1 = (hl1 >> 2) & 255u;
    uint32_t sym1 = (wc >> 16) & 255u;
    q0b = __hfma2(tfq3_u32_to_h2(__ldg(&T->lut4_u32[((sym1 ^ x1) << 2) | c1])), s0, t0);
    st = (st << 8) | sym1;

    uint32_t h2 = st * 0x9E3779B1u;
    uint2 cs2 = __ldg(&T->c64_4[(h2 >> 16) & 63u]);
    __half2 s2 = tfq3_u32_to_h2(cs2.x), t2 = tfq3_u32_to_h2(cs2.y);
    uint32_t hl2 = h2 >> 22, c2 = hl2 & 3u, x2 = (hl2 >> 2) & 255u;
    uint32_t sym2 = (wc >> 8) & 255u;
    q1a = __hfma2(tfq3_u32_to_h2(__ldg(&T->lut4_u32[((sym2 ^ x2) << 2) | c2])), s2, t2);
    st = (st << 8) | sym2;

    uint32_t hl3 = (st * 0x9E3779B1u) >> 22, c3 = hl3 & 3u, x3 = (hl3 >> 2) & 255u;
    uint32_t sym3 = wc & 255u;
    q1b = __hfma2(tfq3_u32_to_h2(__ldg(&T->lut4_u32[((sym3 ^ x3) << 2) | c3])), s2, t2);
    st = (st << 8) | sym3;
}

__device__ __forceinline__ void tfq3_q8_3b(
    uint32_t& st, uint32_t m24, const Tfq3Tables* __restrict__ T,
    __half2& q0a, __half2& q0b, __half2& q1a, __half2& q1b
) {
    uint32_t h0 = st * 0x9E3779B1u;
    uint2 cs0 = __ldg(&T->c64_3[(h0 >> 16) & 63u]);
    __half2 s0 = tfq3_u32_to_h2(cs0.x), t0 = tfq3_u32_to_h2(cs0.y);
    uint32_t hl0 = h0 >> 24, c0 = hl0 & 3u, x0 = (hl0 >> 2) & 63u;
    uint32_t sym0 = (m24 >> 18) & 63u;
    q0a = __hfma2(tfq3_u32_to_h2(__ldg(&T->lut3_u32[((sym0 ^ x0) << 2) | c0])), s0, t0);
    st = (st << 6) | sym0;

    uint32_t hl1 = (st * 0x9E3779B1u) >> 24, c1 = hl1 & 3u, x1 = (hl1 >> 2) & 63u;
    uint32_t sym1 = (m24 >> 12) & 63u;
    q0b = __hfma2(tfq3_u32_to_h2(__ldg(&T->lut3_u32[((sym1 ^ x1) << 2) | c1])), s0, t0);
    st = (st << 6) | sym1;

    uint32_t h2 = st * 0x9E3779B1u;
    uint2 cs2 = __ldg(&T->c64_3[(h2 >> 16) & 63u]);
    __half2 s2 = tfq3_u32_to_h2(cs2.x), t2 = tfq3_u32_to_h2(cs2.y);
    uint32_t hl2 = h2 >> 24, c2 = hl2 & 3u, x2 = (hl2 >> 2) & 63u;
    uint32_t sym2 = (m24 >> 6) & 63u;
    q1a = __hfma2(tfq3_u32_to_h2(__ldg(&T->lut3_u32[((sym2 ^ x2) << 2) | c2])), s2, t2);
    st = (st << 6) | sym2;

    uint32_t hl3 = (st * 0x9E3779B1u) >> 24, c3 = hl3 & 3u, x3 = (hl3 >> 2) & 63u;
    uint32_t sym3 = m24 & 63u;
    q1b = __hfma2(tfq3_u32_to_h2(__ldg(&T->lut3_u32[((sym3 ^ x3) << 2) | c3])), s2, t2);
    st = (st << 6) | sym3;
}

__device__ __forceinline__ void tfq3_q8_5b(
    uint32_t& st, uint32_t hi8, uint32_t lo32, const Tfq3Tables* __restrict__ T,
    __half2& q0a, __half2& q0b, __half2& q1a, __half2& q1b
) {
    uint32_t h0 = st * 0x9E3779B1u;
    uint2 cs0 = __ldg(&T->c64_5[(h0 >> 16) & 63u]);
    __half2 s0 = tfq3_u32_to_h2(cs0.x), t0 = tfq3_u32_to_h2(cs0.y);
    uint32_t sym0 = (hi8 << 2) | (lo32 >> 30);
    q0a = __hfma2(tfq3_u32_to_h2(__ldg(&T->lut5_u32[(sym0 ^ (h0 >> 22)) & 1023u])), s0, t0);
    st = (st << 10) | sym0;

    uint32_t h1 = st * 0x9E3779B1u;
    uint32_t sym1 = (lo32 >> 20) & 1023u;
    q0b = __hfma2(tfq3_u32_to_h2(__ldg(&T->lut5_u32[(sym1 ^ (h1 >> 22)) & 1023u])), s0, t0);
    st = (st << 10) | sym1;

    uint32_t h2 = st * 0x9E3779B1u;
    uint2 cs2 = __ldg(&T->c64_5[(h2 >> 16) & 63u]);
    __half2 s2 = tfq3_u32_to_h2(cs2.x), t2 = tfq3_u32_to_h2(cs2.y);
    uint32_t sym2 = (lo32 >> 10) & 1023u;
    q1a = __hfma2(tfq3_u32_to_h2(__ldg(&T->lut5_u32[(sym2 ^ (h2 >> 22)) & 1023u])), s2, t2);
    st = (st << 10) | sym2;

    uint32_t h3 = st * 0x9E3779B1u;
    uint32_t sym3 = lo32 & 1023u;
    q1b = __hfma2(tfq3_u32_to_h2(__ldg(&T->lut5_u32[(sym3 ^ (h3 >> 22)) & 1023u])), s2, t2);
    st = (st << 10) | sym3;
}

template <uint32_t BITS>
__device__ __forceinline__ uint32_t tfq3_init_st(const uint32_t* __restrict__ w_base, uint32_t bc0, uint32_t lane) {
    if (bc0 == 0u) return 0x13579BDFu ^ (lane * 0x9E3779B1u);
    if constexpr (BITS == 4u) return __ldg(&reinterpret_cast<const uint4*>(w_base)[(static_cast<uint64_t>(bc0) * 2ull - 1ull) * 32ull + lane]).w;
    return __ldg(&w_base[(static_cast<uint64_t>(bc0) * (BITS * 2u) - 1ull) * 32ull + lane]);
}

__device__ __forceinline__ void tfq3_dot1_acc(
    __half2 q0a, __half2 q0b, __half2 q1a, __half2 q1b,
    const __half* __restrict__ xr, uint32_t cb, float& b0, float& b1
) {
    uint2 raw = *reinterpret_cast<const uint2*>(xr + cb);
    __half2 xa = tfq3_u32_to_h2(raw.x), xb = tfq3_u32_to_h2(raw.y);
    float2 fa = __half22float2(xa), fb = __half22float2(xb);
    float2 g0a = __half22float2(q0a), g0b = __half22float2(q0b);
    float2 g1a = __half22float2(q1a), g1b = __half22float2(q1b);
    b0 = fmaf(g0a.x, fa.x, fmaf(g0a.y, fa.y, fmaf(g0b.x, fb.x, fmaf(g0b.y, fb.y, b0))));
    b1 = fmaf(g1a.x, fa.x, fmaf(g1a.y, fa.y, fmaf(g1b.x, fb.x, fmaf(g1b.y, fb.y, b1))));
}

template <uint32_t BITS>
__device__ __forceinline__ void tfq3_mv_range1(
    const uint32_t* __restrict__ w_base, const __half* __restrict__ sc_base,
    uint32_t bc0, uint32_t bc1, const Tfq3Tables* __restrict__ T,
    const __half* __restrict__ xr, uint32_t lane, int2 home, float& acc0, float& acc1
) {
    uint32_t st = tfq3_init_st<BITS>(w_base, bc0, lane);
    acc0 = 0.0f; acc1 = 0.0f;
    for (uint32_t bc = bc0; bc < bc1; ++bc) {
        float ba0 = 0.0f, ba1 = 0.0f;
        __half2 q0a, q0b, q1a, q1b;
        if constexpr (BITS == 4u) {
            const uint4* w4 = reinterpret_cast<const uint4*>(w_base);
            uint32_t cb = (bc << 7) + static_cast<uint32_t>(home.x);
            #pragma unroll
            for (uint32_t h = 0u; h < 2u; ++h) {
                uint4 v = __ldg(&w4[(bc * 2u + h) * 32u + lane]);
                uint32_t ch = cb + (h << 6);
                tfq3_q8_4b(st, v.x, T, q0a, q0b, q1a, q1b); tfq3_dot1_acc(q0a, q0b, q1a, q1b, xr, ch +  0u, ba0, ba1);
                tfq3_q8_4b(st, v.y, T, q0a, q0b, q1a, q1b); tfq3_dot1_acc(q0a, q0b, q1a, q1b, xr, ch + 16u, ba0, ba1);
                tfq3_q8_4b(st, v.z, T, q0a, q0b, q1a, q1b); tfq3_dot1_acc(q0a, q0b, q1a, q1b, xr, ch + 32u, ba0, ba1);
                tfq3_q8_4b(st, v.w, T, q0a, q0b, q1a, q1b); tfq3_dot1_acc(q0a, q0b, q1a, q1b, xr, ch + 48u, ba0, ba1);
            }
        } else if constexpr (BITS == 3u) {
            for (uint32_t h = 0u; h < 2u; ++h) {
                const uint32_t* blk = w_base + (static_cast<uint64_t>(bc) * 6ull + h * 3ull) * 32ull + lane;
                uint32_t a0 = __ldg(&blk[0]), a1 = __ldg(&blk[32]), a2 = __ldg(&blk[64]);
                uint32_t cb = (bc << 7) + (h << 6) + static_cast<uint32_t>(home.x);
                tfq3_q8_3b(st, a0 >> 8,                               T, q0a, q0b, q1a, q1b); tfq3_dot1_acc(q0a, q0b, q1a, q1b, xr, cb +  0u, ba0, ba1);
                tfq3_q8_3b(st, ((a0 << 16) | (a1 >> 16)) & 0xFFFFFFu, T, q0a, q0b, q1a, q1b); tfq3_dot1_acc(q0a, q0b, q1a, q1b, xr, cb + 16u, ba0, ba1);
                tfq3_q8_3b(st, ((a1 <<  8) | (a2 >> 24)) & 0xFFFFFFu, T, q0a, q0b, q1a, q1b); tfq3_dot1_acc(q0a, q0b, q1a, q1b, xr, cb + 32u, ba0, ba1);
                tfq3_q8_3b(st, a2 & 0xFFFFFFu,                        T, q0a, q0b, q1a, q1b); tfq3_dot1_acc(q0a, q0b, q1a, q1b, xr, cb + 48u, ba0, ba1);
            }
        } else {
            for (uint32_t h = 0u; h < 2u; ++h) {
                const uint32_t* blk = w_base + (static_cast<uint64_t>(bc) * 10ull + h * 5ull) * 32ull + lane;
                uint32_t w0 = __ldg(&blk[0]), w1 = __ldg(&blk[32]), w2 = __ldg(&blk[64]), w3 = __ldg(&blk[96]), w4 = __ldg(&blk[128]);
                uint32_t cb = (bc << 7) + (h << 6) + static_cast<uint32_t>(home.x);
                tfq3_q8_5b(st, w0 >> 24,          (w0 <<  8) | (w1 >> 24), T, q0a, q0b, q1a, q1b); tfq3_dot1_acc(q0a, q0b, q1a, q1b, xr, cb +  0u, ba0, ba1);
                tfq3_q8_5b(st, (w1 >> 16) & 255u, (w1 << 16) | (w2 >> 16), T, q0a, q0b, q1a, q1b); tfq3_dot1_acc(q0a, q0b, q1a, q1b, xr, cb + 16u, ba0, ba1);
                tfq3_q8_5b(st, (w2 >>  8) & 255u, (w2 << 24) | (w3 >>  8), T, q0a, q0b, q1a, q1b); tfq3_dot1_acc(q0a, q0b, q1a, q1b, xr, cb + 32u, ba0, ba1);
                tfq3_q8_5b(st, w3 & 255u,          w4,                     T, q0a, q0b, q1a, q1b); tfq3_dot1_acc(q0a, q0b, q1a, q1b, xr, cb + 48u, ba0, ba1);
            }
        }
        float sc = __half2float(__ldg(&sc_base[bc]));
        acc0 = fmaf(ba0, sc, acc0);
        acc1 = fmaf(ba1, sc, acc1);
    }
    acc0 += __shfl_xor_sync(0xFFFFFFFFu, acc0, 1); acc0 += __shfl_xor_sync(0xFFFFFFFFu, acc0, 8);
    acc1 += __shfl_xor_sync(0xFFFFFFFFu, acc1, 1); acc1 += __shfl_xor_sync(0xFFFFFFFFu, acc1, 8);
}
