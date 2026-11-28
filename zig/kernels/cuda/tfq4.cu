// TensorFold TFQ4 (2.001/2.251 bpw 8D E8P) Dense & Hybrid TFPW MoE (2/225/3/4/5b) CUDA Kernels.
#include "tfq4_common.cuh"

template <int TABLE_KIND, bool IS_225>
__device__ void tfq4_dense_impl(
    const uint32_t* __restrict__ W,
    const __half* __restrict__ scales,
    const uint4* __restrict__ e8p_lut,
    const __half* __restrict__ X_rot,
    void* __restrict__ OUT,
    uint32_t rows,
    uint32_t n,
    uint32_t k,
    uint32_t ldo,
    uint32_t sr_split,
    uint32_t row_off,
    uint32_t out_f32
) {
    const uint32_t rb = blockIdx.x;
    const uint32_t r0 = rb * 128u;
    const uint32_t m0 = blockIdx.y * 16u;
    if (m0 >= rows || r0 >= n) return;

    const uint32_t nb = (rows - m0 < 16u) ? (rows - m0) : 16u;
    const uint32_t n_bc = k >> 7;
    const uint32_t wpb_row = IS_225 ? 9u : 8u;
    const uint32_t words_per_row = n_bc * wpb_row;
    const uint32_t tid = threadIdx.x;
    const uint32_t warp = tid >> 5;
    const uint32_t lane = tid & 31u;
    const uint32_t sub_r = lane >> 3;
    const uint32_t sub_w = lane & 7u;

    constexpr uint32_t K_CODES = IS_225 ? 1024u : 256u;
    __shared__ uint4 s_lut[K_CODES];
    __shared__ uint8_t s_par[TABLE_KIND >= 2 ? K_CODES : 1];
    __shared__ uint4 s_x[4 * 16];
    __shared__ float s_xsum[TABLE_KIND >= 2 ? (4 * 16) : 1];
    __shared__ float s_out[16 * 128];

    for (uint32_t i = tid; i < K_CODES; i += 256u) {
        uint4 u = __ldg(&e8p_lut[i]);
        s_lut[i] = u;
        if constexpr (TABLE_KIND >= 2) {
            s_par[i] = tfq4_e8p_parity(u);
        } else {
            (void)s_par;
            (void)s_xsum;
        }
    }
    __syncthreads();

    const __half* sc_row = scales + static_cast<uint64_t>(rb) * n_bc;
    if (nb == 1u) {
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint32_t bc = 0u; bc < n_bc; ++bc) {
            if (tid < 16u) {
                uint4 xv = *reinterpret_cast<const uint4*>(
                    X_rot + static_cast<uint64_t>(m0) * k + (bc << 7) + (tid << 3)
                );
                s_x[tid] = xv;
                if constexpr (TABLE_KIND >= 2) s_xsum[tid] = tfq4_xsum8_qtr(xv);
            }
            __syncthreads();
            const float fsc = __half2float(__ldg(&sc_row[bc]));
            float xs0 = 0.0f, xs1 = 0.0f;
            if constexpr (TABLE_KIND >= 2) {
                xs0 = s_xsum[sub_w * 2u];
                xs1 = s_xsum[sub_w * 2u + 1u];
            }
            const uint4 xv0 = s_x[sub_w * 2u], xv1 = s_x[sub_w * 2u + 1u];
            #pragma unroll
            for (uint32_t pass = 0u; pass < 4u; ++pass) {
                const uint32_t global_r = r0 + warp * 16u + pass * 4u + sub_r;
                const uint32_t* blk = W + static_cast<uint64_t>(global_r) * words_per_row + static_cast<uint64_t>(bc) * wpb_row;
                const uint32_t packed = __ldg(&blk[sub_w]);
                const uint32_t c0 = packed & 0xFFFFu, c1 = packed >> 16;
                uint32_t idx0 = c0 & 255u, idx1 = c1 & 255u;
                if constexpr (IS_225) {
                    const uint32_t hi = __ldg(&blk[8]);
                    idx0 |= ((hi >> (sub_w * 4u)) & 3u) << 8;
                    idx1 |= ((hi >> (sub_w * 4u + 2u)) & 3u) << 8;
                }
                uint32_t p0 = 0u, p1 = 0u;
                if constexpr (TABLE_KIND >= 2) {
                    p0 = static_cast<uint32_t>(s_par[idx0]);
                    p1 = static_cast<uint32_t>(s_par[idx1]);
                }
                float4 w0a, w0b, w1a, w1b;
                tfq4_decode8<TABLE_KIND>(s_lut[idx0], p0, c0, w0a, w0b);
                tfq4_decode8<TABLE_KIND>(s_lut[idx1], p1, c1, w1a, w1b);
                float d = __fadd_rn(
                    tfq4_dot8<TABLE_KIND>(w0a, w0b, xv0, xs0, (c0 & 0x8000u) << 16),
                    tfq4_dot8<TABLE_KIND>(w1a, w1b, xv1, xs1, (c1 & 0x8000u) << 16)
                );
                acc[pass] = __fmaf_rn(d, fsc, acc[pass]);
            }
            __syncthreads();
        }
        #pragma unroll
        for (uint32_t pass = 0u; pass < 4u; ++pass) {
            tfq4_reduce8_store(acc[pass], sub_w, s_out, 0u, warp * 16u + pass * 4u + sub_r);
        }
    } else {
        for (uint32_t m_base = 0u; m_base < nb; m_base += 4u) {
            const uint32_t rem = (nb - m_base < 4u) ? (nb - m_base) : 4u;
            float a0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float a1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float a2[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float a3[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (uint32_t bc = 0u; bc < n_bc; ++bc) {
                const uint32_t n_u4 = rem * 16u;
                if (tid < n_u4) {
                    const uint32_t rl = tid >> 4, cl = (tid & 15u) << 3;
                    uint4 xv = *reinterpret_cast<const uint4*>(
                        X_rot + static_cast<uint64_t>(m0 + m_base + rl) * k + (bc << 7) + cl
                    );
                    s_x[tid] = xv;
                    if constexpr (TABLE_KIND >= 2) s_xsum[tid] = tfq4_xsum8_qtr(xv);
                }
                __syncthreads();
                const float fsc = __half2float(__ldg(&sc_row[bc]));
                float xs0a = 0.0f, xs0b = 0.0f, xs1a = 0.0f, xs1b = 0.0f;
                float xs2a = 0.0f, xs2b = 0.0f, xs3a = 0.0f, xs3b = 0.0f;
                if constexpr (TABLE_KIND >= 2) {
                    xs0a = s_xsum[0u + sub_w * 2u]; xs0b = s_xsum[0u + sub_w * 2u + 1u];
                    if (rem > 1u) { xs1a = s_xsum[16u + sub_w * 2u]; xs1b = s_xsum[16u + sub_w * 2u + 1u]; }
                    if (rem > 2u) { xs2a = s_xsum[32u + sub_w * 2u]; xs2b = s_xsum[32u + sub_w * 2u + 1u]; }
                    if (rem > 3u) { xs3a = s_xsum[48u + sub_w * 2u]; xs3b = s_xsum[48u + sub_w * 2u + 1u]; }
                }
                #pragma unroll
                for (uint32_t pass = 0u; pass < 4u; ++pass) {
                    const uint32_t global_r = r0 + warp * 16u + pass * 4u + sub_r;
                    const uint32_t* blk = W + static_cast<uint64_t>(global_r) * words_per_row + static_cast<uint64_t>(bc) * wpb_row;
                    const uint32_t packed = __ldg(&blk[sub_w]);
                    const uint32_t c0 = packed & 0xFFFFu, c1 = packed >> 16;
                    uint32_t idx0 = c0 & 255u, idx1 = c1 & 255u;
                    if constexpr (IS_225) {
                        const uint32_t hi = __ldg(&blk[8]);
                        idx0 |= ((hi >> (sub_w * 4u)) & 3u) << 8;
                        idx1 |= ((hi >> (sub_w * 4u + 2u)) & 3u) << 8;
                    }
                    uint32_t p0 = 0u, p1 = 0u;
                    if constexpr (TABLE_KIND >= 2) {
                        p0 = static_cast<uint32_t>(s_par[idx0]);
                        p1 = static_cast<uint32_t>(s_par[idx1]);
                    }
                    float4 w0a, w0b, w1a, w1b;
                    tfq4_decode8<TABLE_KIND>(s_lut[idx0], p0, c0, w0a, w0b);
                    tfq4_decode8<TABLE_KIND>(s_lut[idx1], p1, c1, w1a, w1b);
                    const uint32_t cs0 = (c0 & 0x8000u) << 16, cs1 = (c1 & 0x8000u) << 16;
                    {
                        float d = __fadd_rn(tfq4_dot8<TABLE_KIND>(w0a, w0b, s_x[0u + sub_w * 2u], xs0a, cs0),
                                            tfq4_dot8<TABLE_KIND>(w1a, w1b, s_x[0u + sub_w * 2u + 1u], xs0b, cs1));
                        a0[pass] = __fmaf_rn(d, fsc, a0[pass]);
                    }
                    if (rem > 1u) {
                        float d = __fadd_rn(tfq4_dot8<TABLE_KIND>(w0a, w0b, s_x[16u + sub_w * 2u], xs1a, cs0),
                                            tfq4_dot8<TABLE_KIND>(w1a, w1b, s_x[16u + sub_w * 2u + 1u], xs1b, cs1));
                        a1[pass] = __fmaf_rn(d, fsc, a1[pass]);
                    }
                    if (rem > 2u) {
                        float d = __fadd_rn(tfq4_dot8<TABLE_KIND>(w0a, w0b, s_x[32u + sub_w * 2u], xs2a, cs0),
                                            tfq4_dot8<TABLE_KIND>(w1a, w1b, s_x[32u + sub_w * 2u + 1u], xs2b, cs1));
                        a2[pass] = __fmaf_rn(d, fsc, a2[pass]);
                    }
                    if (rem > 3u) {
                        float d = __fadd_rn(tfq4_dot8<TABLE_KIND>(w0a, w0b, s_x[48u + sub_w * 2u], xs3a, cs0),
                                            tfq4_dot8<TABLE_KIND>(w1a, w1b, s_x[48u + sub_w * 2u + 1u], xs3b, cs1));
                        a3[pass] = __fmaf_rn(d, fsc, a3[pass]);
                    }
                }
                __syncthreads();
            }
            #pragma unroll
            for (uint32_t pass = 0u; pass < 4u; ++pass) {
                const uint32_t local_r = warp * 16u + pass * 4u + sub_r;
                tfq4_reduce8_store(a0[pass], sub_w, s_out, m_base + 0u, local_r);
                if (rem > 1u) tfq4_reduce8_store(a1[pass], sub_w, s_out, m_base + 1u, local_r);
                if (rem > 2u) tfq4_reduce8_store(a2[pass], sub_w, s_out, m_base + 2u, local_r);
                if (rem > 3u) tfq4_reduce8_store(a3[pass], sub_w, s_out, m_base + 3u, local_r);
            }
        }
    }
    __syncthreads();

    const uint32_t sr0 = tfq3_sr0(r0 + row_off, sr_split);
    for (uint32_t m = warp; m < nb; m += 8u) {
        const float* ys = s_out + m * 128u;
        float4 v = make_float4(ys[lane], ys[lane + 32u], ys[lane + 64u], ys[lane + 96u]);
        tfq3_fwht128_simd4(v, lane);
        const uint64_t base = static_cast<uint64_t>(m0 + m) * ldo + r0;
        tfq3_store(OUT, base + lane +  0u, v.x * tfq3_sgn(0x9E3779B9u, sr0 + lane +  0u), out_f32);
        tfq3_store(OUT, base + lane + 32u, v.y * tfq3_sgn(0x9E3779B9u, sr0 + lane + 32u), out_f32);
        tfq3_store(OUT, base + lane + 64u, v.z * tfq3_sgn(0x9E3779B9u, sr0 + lane + 64u), out_f32);
        tfq3_store(OUT, base + lane + 96u, v.w * tfq3_sgn(0x9E3779B9u, sr0 + lane + 96u), out_f32);
    }
}

template <bool SOFTCAP>
__device__ void tfq4_moe_gu_impl(
    const uint32_t* __restrict__ W,
    const __half* __restrict__ scales,
    const Tfq3ExpertMeta* __restrict__ metas,
    const Tfq3Tables* __restrict__ T,
    const uint4* __restrict__ T4,
    const __half* __restrict__ X_rot,
    const int32_t* __restrict__ picks,
    __half* __restrict__ act_rot,
    uint32_t dims,
    uint32_t inter,
    uint32_t experts,
    uint32_t top_k,
    uint32_t pick_stride,
    float param0,
    float param1,
    uint32_t act_mode,
    uint32_t table4_kind
) {
    const uint32_t br = blockIdx.x, slot_flat = blockIdx.y;
    const uint32_t row = slot_flat / top_k, s = slot_flat - row * top_k;
    const int32_t eid = picks[static_cast<uint64_t>(row) * pick_stride + s];
    if (eid < 0 || static_cast<uint32_t>(eid) >= experts) return;

    const Tfq3ExpertMeta em = metas[eid];
    const uint32_t bits = em.b_bits, n_bc = dims >> 7;
    const uint32_t tid = threadIdx.x, sg = tid >> 5, lane = tid & 31u;
    const uint32_t r0 = br * 128u;
    const __half* xr = X_rot + static_cast<uint64_t>(row) * dims;
    const __half* sg_p = scales + em.gate_sc_off + static_cast<uint64_t>(br) * n_bc;
    const __half* su_p = scales + em.up_sc_off   + static_cast<uint64_t>(br) * n_bc;

    __shared__ uint4 s_lut[1024];
    __shared__ uint8_t s_par[1024];
    __shared__ uint4 s_x[16];
    __shared__ float s_xsum[16];
    __shared__ float tg_g[128], tg_u[128];

    if (bits == 2u || bits == 225u) {
        const bool is_225 = (bits == 225u);
        const bool is_true_e8p = (table4_kind >= 2u);
        const uint32_t k_codes = is_225 ? 1024u : 256u;
        for (uint32_t i = tid; i < k_codes; i += 256u) {
            uint4 u = __ldg(&T4[i]);
            s_lut[i] = u;
            if (is_true_e8p) s_par[i] = tfq4_e8p_parity(u);
        }
        __syncthreads();
        const uint32_t sub_r = lane >> 3, sub_w = lane & 7u;
        const uint32_t wpb_row = is_225 ? 9u : 8u;
        const uint32_t wpr = n_bc * wpb_row;
        float ag[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float au[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint32_t bc = 0u; bc < n_bc; ++bc) {
            if (tid < 16u) {
                uint4 xv = *reinterpret_cast<const uint4*>(xr + (bc << 7) + (tid << 3));
                s_x[tid] = xv;
                if (is_true_e8p) s_xsum[tid] = tfq4_xsum8_qtr(xv);
            }
            __syncthreads();
            const float fsg = __half2float(__ldg(&sg_p[bc]));
            const float fsu = __half2float(__ldg(&su_p[bc]));
            const float xs0 = is_true_e8p ? s_xsum[sub_w * 2u] : 0.0f;
            const float xs1 = is_true_e8p ? s_xsum[sub_w * 2u + 1u] : 0.0f;
            const uint4 xv0 = s_x[sub_w * 2u], xv1 = s_x[sub_w * 2u + 1u];
            #pragma unroll
            for (uint32_t pass = 0u; pass < 4u; ++pass) {
                const uint32_t global_r = r0 + sg * 16u + pass * 4u + sub_r;
                const uint64_t roff = static_cast<uint64_t>(global_r) * wpr + static_cast<uint64_t>(bc) * wpb_row;
                float dg = tfq4_dot16_e8p(W + em.gate_word_off + roff, sub_w, is_225, is_true_e8p, s_lut, s_par, xv0, xv1, xs0, xs1);
                float du = tfq4_dot16_e8p(W + em.up_word_off   + roff, sub_w, is_225, is_true_e8p, s_lut, s_par, xv0, xv1, xs0, xs1);
                ag[pass] = __fmaf_rn(dg, fsg, ag[pass]);
                au[pass] = __fmaf_rn(du, fsu, au[pass]);
            }
            __syncthreads();
        }
        #pragma unroll
        for (uint32_t pass = 0u; pass < 4u; ++pass) {
            const uint32_t local_r = sg * 16u + pass * 4u + sub_r;
            tfq4_reduce8_store(ag[pass], sub_w, tg_g, 0u, local_r);
            tfq4_reduce8_store(au[pass], sub_w, tg_u, 0u, local_r);
        }
    } else {
        const uint32_t wpb = bits * 2u, strip = br * 8u + sg;
        const int2 home = tfq3_frag_home(lane);
        const uint32_t* wg = W + em.gate_word_off + static_cast<uint64_t>(strip) * n_bc * wpb * 32ull;
        const uint32_t* wu = W + em.up_word_off   + static_cast<uint64_t>(strip) * n_bc * wpb * 32ull;
        float g0, g1, u0, u1;
        tfq4_mv1_tfq3(bits, wg, sg_p, n_bc, T, xr, lane, home, g0, g1);
        tfq4_mv1_tfq3(bits, wu, su_p, n_bc, T, xr, lane, home, u0, u1);
        if ((lane & 9u) == 0u) {
            const uint32_t idx = sg * 16u + home.y;
            tg_g[idx] = g0; tg_g[idx + 8u] = g1;
            tg_u[idx] = u0; tg_u[idx + 8u] = u1;
        }
    }
    __syncthreads();
    if (sg == 0u) {
        float4 gv = make_float4(tg_g[lane], tg_g[lane + 32u], tg_g[lane + 64u], tg_g[lane + 96u]);
        float4 uv = make_float4(tg_u[lane], tg_u[lane + 32u], tg_u[lane + 64u], tg_u[lane + 96u]);
        tfq3_fwht128_simd4(gv, lane);
        tfq3_fwht128_simd4(uv, lane);
        float4 z;
        if constexpr (SOFTCAP) {
            z.x = tfq3_glu_softcap(gv.x * tfq3_sgn(0x9E3779B9u, r0 + lane +  0u), uv.x * tfq3_sgn(0x9E3779B9u, r0 + lane +  0u), param0, param1) * tfq3_sgn(0x12345678u, r0 + lane +  0u);
            z.y = tfq3_glu_softcap(gv.y * tfq3_sgn(0x9E3779B9u, r0 + lane + 32u), uv.y * tfq3_sgn(0x9E3779B9u, r0 + lane + 32u), param0, param1) * tfq3_sgn(0x12345678u, r0 + lane + 32u);
            z.z = tfq3_glu_softcap(gv.z * tfq3_sgn(0x9E3779B9u, r0 + lane + 64u), uv.z * tfq3_sgn(0x9E3779B9u, r0 + lane + 64u), param0, param1) * tfq3_sgn(0x12345678u, r0 + lane + 64u);
            z.w = tfq3_glu_softcap(gv.w * tfq3_sgn(0x9E3779B9u, r0 + lane + 96u), uv.w * tfq3_sgn(0x9E3779B9u, r0 + lane + 96u), param0, param1) * tfq3_sgn(0x12345678u, r0 + lane + 96u);
        } else {
            z.x = tfq3_glu(gv.x * tfq3_sgn(0x9E3779B9u, r0 + lane +  0u), uv.x * tfq3_sgn(0x9E3779B9u, r0 + lane +  0u), param0, act_mode) * tfq3_sgn(0x12345678u, r0 + lane +  0u);
            z.y = tfq3_glu(gv.y * tfq3_sgn(0x9E3779B9u, r0 + lane + 32u), uv.y * tfq3_sgn(0x9E3779B9u, r0 + lane + 32u), param0, act_mode) * tfq3_sgn(0x12345678u, r0 + lane + 32u);
            z.z = tfq3_glu(gv.z * tfq3_sgn(0x9E3779B9u, r0 + lane + 64u), uv.z * tfq3_sgn(0x9E3779B9u, r0 + lane + 64u), param0, act_mode) * tfq3_sgn(0x12345678u, r0 + lane + 64u);
            z.w = tfq3_glu(gv.w * tfq3_sgn(0x9E3779B9u, r0 + lane + 96u), uv.w * tfq3_sgn(0x9E3779B9u, r0 + lane + 96u), param0, act_mode) * tfq3_sgn(0x12345678u, r0 + lane + 96u);
        }
        tfq3_fwht128_simd4(z, lane);
        __half* dst = act_rot + static_cast<uint64_t>(slot_flat) * inter + r0;
        dst[lane +  0u] = __float2half(z.x);
        dst[lane + 32u] = __float2half(z.y);
        dst[lane + 64u] = __float2half(z.z);
        dst[lane + 96u] = __float2half(z.w);
    }
}

extern "C" {

__global__ void tf_tfq4_fwht_x(
    const __nv_bfloat16* __restrict__ x, __half* __restrict__ x_rot,
    uint32_t k, uint32_t ldx, uint32_t k_sign_off
) {
    const uint32_t bc = blockIdx.x, r = blockIdx.y, c0 = bc * 128u, cs = c0 + k_sign_off, lane = threadIdx.x;
    const __nv_bfloat16* src = x + static_cast<uint64_t>(r) * ldx + c0;
    float4 v = make_float4(
        __bfloat162float(src[lane +  0u]) * tfq3_sgn(0x12345678u, cs + lane +  0u),
        __bfloat162float(src[lane + 32u]) * tfq3_sgn(0x12345678u, cs + lane + 32u),
        __bfloat162float(src[lane + 64u]) * tfq3_sgn(0x12345678u, cs + lane + 64u),
        __bfloat162float(src[lane + 96u]) * tfq3_sgn(0x12345678u, cs + lane + 96u));
    tfq3_fwht128_simd4(v, lane);
    __half* dst = x_rot + static_cast<uint64_t>(r) * k + c0;
    dst[lane +  0u] = tfq4_float2half_sat(v.x);
    dst[lane + 32u] = tfq4_float2half_sat(v.y);
    dst[lane + 64u] = tfq4_float2half_sat(v.z);
    dst[lane + 96u] = tfq4_float2half_sat(v.w);
}

__global__ void tf_tfq4_e8p_dense_trained(
    const uint32_t* W, const __half* S, const uint4* L, const __half* X, void* O,
    uint32_t rows, uint32_t n, uint32_t k, uint32_t ldo, uint32_t sr, uint32_t ro, uint32_t f32
) {
    tfq4_dense_impl<0, false>(W, S, L, X, O, rows, n, k, ldo, sr, ro, f32);
}

__global__ void tf_tfq4_e8p_dense_e8p(
    const uint32_t* W, const __half* S, const uint4* L, const __half* X, void* O,
    uint32_t rows, uint32_t n, uint32_t k, uint32_t ldo, uint32_t sr, uint32_t ro, uint32_t f32
) {
    tfq4_dense_impl<2, false>(W, S, L, X, O, rows, n, k, ldo, sr, ro, f32);
}

__global__ void tf_tfq4_e8p_dense_e8p1024(
    const uint32_t* W, const __half* S, const uint4* L, const __half* X, void* O,
    uint32_t rows, uint32_t n, uint32_t k, uint32_t ldo, uint32_t sr, uint32_t ro, uint32_t f32
) {
    tfq4_dense_impl<2, true>(W, S, L, X, O, rows, n, k, ldo, sr, ro, f32);
}

__global__ void tf_tfq4_moe_gu_swiglu(
    const uint32_t* W, const __half* S, const Tfq3ExpertMeta* M, const Tfq3Tables* T, const uint4* T4,
    const __half* X, const int32_t* picks, __half* act_rot,
    uint32_t dims, uint32_t inter, uint32_t experts, uint32_t top_k, uint32_t pick_stride,
    float lim, uint32_t act_mode, uint32_t table4_kind
) {
    tfq4_moe_gu_impl<false>(W, S, M, T, T4, X, picks, act_rot, dims, inter, experts, top_k, pick_stride, lim, 0.0f, act_mode, table4_kind);
}

__global__ void tf_tfq4_moe_gu_softcap(
    const uint32_t* W, const __half* S, const Tfq3ExpertMeta* M, const Tfq3Tables* T, const uint4* T4,
    const __half* X, const int32_t* picks, __half* act_rot,
    uint32_t dims, uint32_t inter, uint32_t experts, uint32_t top_k, uint32_t pick_stride,
    float gate_cap, float up_cap, uint32_t table4_kind
) {
    tfq4_moe_gu_impl<true>(W, S, M, T, T4, X, picks, act_rot, dims, inter, experts, top_k, pick_stride, gate_cap, up_cap, 0u, table4_kind);
}

__global__ void tf_tfq4_moe_down(
    const uint32_t* __restrict__ W,
    const __half* __restrict__ scales,
    const Tfq3ExpertMeta* __restrict__ metas,
    const Tfq3Tables* __restrict__ T,
    const uint4* __restrict__ T4,
    const __half* __restrict__ act_rot,
    const int32_t* __restrict__ picks,
    void* __restrict__ ey,
    uint32_t dims,
    uint32_t inter,
    uint32_t experts,
    uint32_t top_k,
    uint32_t pick_stride,
    uint32_t out_slot_stride,
    uint32_t out_f32,
    uint32_t table4_kind
) {
    const uint32_t br = blockIdx.x, slot_flat = blockIdx.y;
    const uint32_t row = slot_flat / top_k, s = slot_flat - row * top_k;
    const int32_t eid = picks[static_cast<uint64_t>(row) * pick_stride + s];
    if (eid < 0 || static_cast<uint32_t>(eid) >= experts) return;

    const Tfq3ExpertMeta em = metas[eid];
    const uint32_t bits = em.b_bits, n_bc = inter >> 7;
    const uint32_t tid = threadIdx.x, sg = tid >> 5, lane = tid & 31u;
    const uint32_t r0 = br * 128u;
    const __half* ar = act_rot + static_cast<uint64_t>(slot_flat) * inter;
    const __half* sd_p = scales + em.down_sc_off + static_cast<uint64_t>(br) * n_bc;

    __shared__ uint4 s_lut[1024];
    __shared__ uint8_t s_par[1024];
    __shared__ uint4 s_x[16];
    __shared__ float s_xsum[16];
    __shared__ float tg_y[128];

    if (bits == 2u || bits == 225u) {
        const bool is_225 = (bits == 225u);
        const bool is_true_e8p = (table4_kind >= 2u);
        const uint32_t k_codes = is_225 ? 1024u : 256u;
        for (uint32_t i = tid; i < k_codes; i += 256u) {
            uint4 u = __ldg(&T4[i]);
            s_lut[i] = u;
            if (is_true_e8p) s_par[i] = tfq4_e8p_parity(u);
        }
        __syncthreads();
        const uint32_t sub_r = lane >> 3, sub_w = lane & 7u;
        const uint32_t wpb_row = is_225 ? 9u : 8u;
        const uint32_t wpr = n_bc * wpb_row;
        float ay[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint32_t bc = 0u; bc < n_bc; ++bc) {
            if (tid < 16u) {
                uint4 xv = *reinterpret_cast<const uint4*>(ar + (bc << 7) + (tid << 3));
                s_x[tid] = xv;
                if (is_true_e8p) s_xsum[tid] = tfq4_xsum8_qtr(xv);
            }
            __syncthreads();
            const float fsd = __half2float(__ldg(&sd_p[bc]));
            const float xs0 = is_true_e8p ? s_xsum[sub_w * 2u] : 0.0f;
            const float xs1 = is_true_e8p ? s_xsum[sub_w * 2u + 1u] : 0.0f;
            const uint4 xv0 = s_x[sub_w * 2u], xv1 = s_x[sub_w * 2u + 1u];
            #pragma unroll
            for (uint32_t pass = 0u; pass < 4u; ++pass) {
                const uint32_t global_r = r0 + sg * 16u + pass * 4u + sub_r;
                const uint64_t roff = static_cast<uint64_t>(global_r) * wpr + static_cast<uint64_t>(bc) * wpb_row;
                float dy = tfq4_dot16_e8p(W + em.down_word_off + roff, sub_w, is_225, is_true_e8p, s_lut, s_par, xv0, xv1, xs0, xs1);
                ay[pass] = __fmaf_rn(dy, fsd, ay[pass]);
            }
            __syncthreads();
        }
        #pragma unroll
        for (uint32_t pass = 0u; pass < 4u; ++pass) {
            tfq4_reduce8_store(ay[pass], sub_w, tg_y, 0u, sg * 16u + pass * 4u + sub_r);
        }
    } else {
        const uint32_t wpb = bits * 2u, strip = br * 8u + sg;
        const int2 home = tfq3_frag_home(lane);
        const uint32_t* wd = W + em.down_word_off + static_cast<uint64_t>(strip) * n_bc * wpb * 32ull;
        float a0, a1;
        tfq4_mv1_tfq3(bits, wd, sd_p, n_bc, T, ar, lane, home, a0, a1);
        if ((lane & 9u) == 0u) {
            const uint32_t idx = sg * 16u + home.y;
            tg_y[idx] = a0; tg_y[idx + 8u] = a1;
        }
    }
    __syncthreads();
    if (sg == 0u) {
        float4 v = make_float4(tg_y[lane], tg_y[lane + 32u], tg_y[lane + 64u], tg_y[lane + 96u]);
        tfq3_fwht128_simd4(v, lane);
        const uint64_t base = (static_cast<uint64_t>(row) * out_slot_stride + s) * dims + r0;
        tfq3_store(ey, base + lane +  0u, v.x * tfq3_sgn(0x9E3779B9u, r0 + lane +  0u), out_f32);
        tfq3_store(ey, base + lane + 32u, v.y * tfq3_sgn(0x9E3779B9u, r0 + lane + 32u), out_f32);
        tfq3_store(ey, base + lane + 64u, v.z * tfq3_sgn(0x9E3779B9u, r0 + lane + 64u), out_f32);
        tfq3_store(ey, base + lane + 96u, v.w * tfq3_sgn(0x9E3779B9u, r0 + lane + 96u), out_f32);
    }
}

} // extern "C"
