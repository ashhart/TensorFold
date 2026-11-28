#pragma once
// Shared TFQ4 (2.001/2.251 bpw 8D E8P + 2D FWHT-128) CUDA device primitives.
#include "tfq3_common.cuh"

__device__ __forceinline__ __half tfq4_float2half_sat(float v) {
    return __float2half(fminf(fmaxf(v, -65504.0f), 65504.0f));
}

__device__ __forceinline__ uint8_t tfq4_e8p_parity(uint4 u_lut) {
    float2 a0 = __half22float2(tfq3_u32_to_h2(u_lut.x));
    float2 a1 = __half22float2(tfq3_u32_to_h2(u_lut.y));
    float2 a2 = __half22float2(tfq3_u32_to_h2(u_lut.z));
    float2 a3 = __half22float2(tfq3_u32_to_h2(u_lut.w));
    uint32_t par = (static_cast<uint32_t>(static_cast<int>(a0.x))
                  + static_cast<uint32_t>(static_cast<int>(a0.y))
                  + static_cast<uint32_t>(static_cast<int>(a1.x))
                  + static_cast<uint32_t>(static_cast<int>(a1.y))
                  + static_cast<uint32_t>(static_cast<int>(a2.x))
                  + static_cast<uint32_t>(static_cast<int>(a2.y))
                  + static_cast<uint32_t>(static_cast<int>(a3.x))
                  + static_cast<uint32_t>(static_cast<int>(a3.y))) & 1u;
    return static_cast<uint8_t>(par);
}

__device__ __forceinline__ float tfq4_xsum8_qtr(uint4 xv) {
    float2 x0 = __half22float2(tfq3_u32_to_h2(xv.x));
    float2 x1 = __half22float2(tfq3_u32_to_h2(xv.y));
    float2 x2 = __half22float2(tfq3_u32_to_h2(xv.z));
    float2 x3 = __half22float2(tfq3_u32_to_h2(xv.w));
    float s0 = __fadd_rn(x0.x, x0.y);
    float s1 = __fadd_rn(x1.x, x1.y);
    float s2 = __fadd_rn(x2.x, x2.y);
    float s3 = __fadd_rn(x3.x, x3.y);
    return __fmul_rn(__fadd_rn(__fadd_rn(s0, s1), __fadd_rn(s2, s3)), 0.25f);
}

template <int TABLE_KIND>
__device__ __forceinline__ void tfq4_decode8(
    uint4 lut_u4, uint32_t m_par, uint32_t code, float4& wf0, float4& wf1
) {
    uint32_t smask;
    if constexpr (TABLE_KIND >= 2) {
        uint32_t s7 = (code >> 8) & 0x7Fu;
        uint32_t s7_bit = (static_cast<uint32_t>(__popc(s7)) ^ m_par) & 1u;
        smask = s7 | (s7_bit << 7);
    } else {
        (void)m_par;
        smask = code >> 8;
    }
    uint32_t f0 = (((smask >> 0) & 1u) << 15) | (((smask >> 1) & 1u) << 31);
    uint32_t f1 = (((smask >> 2) & 1u) << 15) | (((smask >> 3) & 1u) << 31);
    uint32_t f2 = (((smask >> 4) & 1u) << 15) | (((smask >> 5) & 1u) << 31);
    uint32_t f3 = (((smask >> 6) & 1u) << 15) | (((smask >> 7) & 1u) << 31);
    float2 w0 = __half22float2(tfq3_u32_to_h2(lut_u4.x ^ f0));
    float2 w1 = __half22float2(tfq3_u32_to_h2(lut_u4.y ^ f1));
    float2 w2 = __half22float2(tfq3_u32_to_h2(lut_u4.z ^ f2));
    float2 w3 = __half22float2(tfq3_u32_to_h2(lut_u4.w ^ f3));
    wf0 = make_float4(w0.x, w0.y, w1.x, w1.y);
    wf1 = make_float4(w2.x, w2.y, w3.x, w3.y);
}

template <int TABLE_KIND>
__device__ __forceinline__ float tfq4_dot8(
    float4 w0, float4 w1, uint4 xv, float xsum_qtr, uint32_t csign
) {
    float2 x0 = __half22float2(tfq3_u32_to_h2(xv.x));
    float2 x1 = __half22float2(tfq3_u32_to_h2(xv.y));
    float2 x2 = __half22float2(tfq3_u32_to_h2(xv.z));
    float2 x3 = __half22float2(tfq3_u32_to_h2(xv.w));
    float s0 = __fmaf_rn(w0.x, x0.x, __fmul_rn(w0.y, x0.y));
    float s1 = __fmaf_rn(w0.z, x1.x, __fmul_rn(w0.w, x1.y));
    float s2 = __fmaf_rn(w1.x, x2.x, __fmul_rn(w1.y, x2.y));
    float s3 = __fmaf_rn(w1.z, x3.x, __fmul_rn(w1.w, x3.y));
    float d = __fadd_rn(__fadd_rn(s0, s1), __fadd_rn(s2, s3));
    if constexpr (TABLE_KIND >= 2) {
        d = __fsub_rn(d, xsum_qtr);
        d = __uint_as_float(__float_as_uint(d) ^ csign);
    } else {
        (void)xsum_qtr;
        (void)csign;
    }
    return d;
}

__device__ __forceinline__ void tfq4_reduce8_store(
    float s, uint32_t sub_w, float* s_out, uint32_t row_m, uint32_t local_r
) {
    s = __fadd_rn(s, __shfl_xor_sync(0xFFFFFFFFu, s, 1));
    s = __fadd_rn(s, __shfl_xor_sync(0xFFFFFFFFu, s, 2));
    s = __fadd_rn(s, __shfl_xor_sync(0xFFFFFFFFu, s, 4));
    if (sub_w == 0u) s_out[row_m * 128u + local_r] = s;
}

__device__ __forceinline__ float tfq4_dot16_e8p(
    const uint32_t* __restrict__ blk, uint32_t sub_w, bool is_225, bool is_true_e8p,
    const uint4* __restrict__ s_lut, const uint8_t* __restrict__ s_par,
    uint4 xv0, uint4 xv1, float xs0, float xs1
) {
    uint32_t packed = __ldg(&blk[sub_w]);
    uint32_t c0 = packed & 0xFFFFu;
    uint32_t c1 = packed >> 16;
    uint32_t idx0 = c0 & 255u;
    uint32_t idx1 = c1 & 255u;
    if (is_225) {
        uint32_t hi = __ldg(&blk[8]);
        idx0 |= ((hi >> (sub_w * 4u)) & 3u) << 8;
        idx1 |= ((hi >> (sub_w * 4u + 2u)) & 3u) << 8;
    }
    float4 w0a, w0b, w1a, w1b;
    if (is_true_e8p) {
        tfq4_decode8<2>(s_lut[idx0], static_cast<uint32_t>(s_par[idx0]), c0, w0a, w0b);
        tfq4_decode8<2>(s_lut[idx1], static_cast<uint32_t>(s_par[idx1]), c1, w1a, w1b);
        return __fadd_rn(
            tfq4_dot8<2>(w0a, w0b, xv0, xs0, (c0 & 0x8000u) << 16),
            tfq4_dot8<2>(w1a, w1b, xv1, xs1, (c1 & 0x8000u) << 16)
        );
    }
    tfq4_decode8<0>(s_lut[idx0], 0u, c0, w0a, w0b);
    tfq4_decode8<0>(s_lut[idx1], 0u, c1, w1a, w1b);
    return __fadd_rn(tfq4_dot8<0>(w0a, w0b, xv0, 0.0f, 0u), tfq4_dot8<0>(w1a, w1b, xv1, 0.0f, 0u));
}

__device__ __forceinline__ void tfq4_mv1_tfq3(
    uint32_t bits, const uint32_t* __restrict__ w_base, const __half* __restrict__ sc_base,
    uint32_t n_bc, const Tfq3Tables* __restrict__ T, const __half* __restrict__ xr,
    uint32_t lane, int2 home, float& a0, float& a1
) {
    if (bits == 4u) {
        tfq3_mv_range1<4u>(w_base, sc_base, 0u, n_bc, T, xr, lane, home, a0, a1);
    } else if (bits == 3u) {
        tfq3_mv_range1<3u>(w_base, sc_base, 0u, n_bc, T, xr, lane, home, a0, a1);
    } else {
        tfq3_mv_range1<5u>(w_base, sc_base, 0u, n_bc, T, xr, lane, home, a0, a1);
    }
}
