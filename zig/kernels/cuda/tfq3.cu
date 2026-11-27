// TensorFold TFQ3 (3/4/5-bit 2D-AffineCoset/half4 + 2D FWHT-128) Dense CUDA Kernels.
#include "common/qmm_frag.cuh"
#include "tfq3_common.cuh"

__device__ __forceinline__ void tfq3_stage_b24(__half* dst, __half2 q0a, __half2 q0b, __half2 q1a, __half2 q1b, int2 home) {
    __half2* p0 = reinterpret_cast<__half2*>(dst + home.y * 24 + home.x);
    __half2* p1 = reinterpret_cast<__half2*>(dst + (home.y + 8) * 24 + home.x);
    p0[0] = q0a; p0[1] = q0b; p1[0] = q1a; p1[1] = q1b;
}

__device__ __forceinline__ void tfq3_load_b_16x16(
    uint32_t (&b_lo)[2], uint32_t (&b_hi)[2], const __half* __restrict__ tg_b, uint32_t lane
) {
    uint32_t g = lane >> 2, t = (lane & 3u) << 1;
    const __half* b0p = tg_b + g * 24u + t;
    const __half* b1p = b0p + 8u * 24u;
    b_lo[0] = *reinterpret_cast<const uint32_t*>(b0p);
    b_lo[1] = *reinterpret_cast<const uint32_t*>(b0p + 8u);
    b_hi[0] = *reinterpret_cast<const uint32_t*>(b1p);
    b_hi[1] = *reinterpret_cast<const uint32_t*>(b1p + 8u);
}

__device__ __forceinline__ void tfq3_mma_f16_16x16x16_b(
    float (&b0)[8], const __half* __restrict__ tg_a, int kb,
    const uint32_t (&b_lo)[2], const uint32_t (&b_hi)[2], uint32_t lane
) {
    uint32_t g = lane >> 2, t = (lane & 3u) << 1;
    const __half* a0p = tg_a + g * 136u + kb * 16 + t;
    const __half* a1p = a0p + 8u * 136u;
    const uint32_t a[4] = {
        *reinterpret_cast<const uint32_t*>(a0p),
        *reinterpret_cast<const uint32_t*>(a1p),
        *reinterpret_cast<const uint32_t*>(a0p + 8u),
        *reinterpret_cast<const uint32_t*>(a1p + 8u),
    };
    float d0[4] = {b0[0], b0[1], b0[2], b0[3]};
    float d1[4] = {b0[4], b0[5], b0[6], b0[7]};
    qmm_frag::mma_f16_v(d0, a, b_lo);
    qmm_frag::mma_f16_v(d1, a, b_hi);
    b0[0] = d0[0]; b0[1] = d0[1]; b0[2] = d0[2]; b0[3] = d0[3];
    b0[4] = d1[0]; b0[5] = d1[1]; b0[6] = d1[2]; b0[7] = d1[3];
}

__device__ __forceinline__ void tfq3_mma_f16_16x16x16(
    float (&b0)[8], const __half* __restrict__ tg_a, int kb, const __half* __restrict__ tg_b, uint32_t lane
) {
    uint32_t b_lo[2], b_hi[2];
    tfq3_load_b_16x16(b_lo, b_hi, tg_b, lane);
    tfq3_mma_f16_16x16x16_b(b0, tg_a, kb, b_lo, b_hi, lane);
}

template <uint32_t BITS>
__device__ __forceinline__ void tfq3_wmma_bc(
    const uint32_t* __restrict__ w_s, const __half* __restrict__ sc0, uint32_t bc,
    const Tfq3Tables* __restrict__ T, uint32_t& st, uint32_t lane, int2 home,
    const __half* __restrict__ tg_a, __half* __restrict__ tg_b_warp, float (&acc)[8]
) {
    float b0[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    __half2 q0a, q0b, q1a, q1b;
    if constexpr (BITS == 4u) {
        const uint4* p = reinterpret_cast<const uint4*>(w_s + static_cast<uint64_t>(bc) * 8ull * 32ull);
        for (uint32_t h = 0u; h < 2u; ++h) {
            uint4 v = __ldg(&p[h * 32u + lane]);
            uint32_t uv[4] = {v.x, v.y, v.z, v.w};
            int kb = static_cast<int>(h * 4u);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                tfq3_q8_4b(st, uv[j], T, q0a, q0b, q1a, q1b);
                tfq3_stage_b24(tg_b_warp, q0a, q0b, q1a, q1b, home);
                __syncwarp();
                tfq3_mma_f16_16x16x16(b0, tg_a, kb + j, tg_b_warp, lane);
            }
        }
    } else if constexpr (BITS == 3u) {
        for (uint32_t h = 0u; h < 2u; ++h) {
            const uint32_t* p = w_s + (static_cast<uint64_t>(bc) * 6ull + h * 3ull) * 32ull + lane;
            uint32_t a0 = __ldg(&p[0]), a1 = __ldg(&p[32]), a2 = __ldg(&p[64]);
            uint32_t mv[4] = {a0 >> 8, ((a0 << 16) | (a1 >> 16)) & 0xFFFFFFu, ((a1 << 8) | (a2 >> 24)) & 0xFFFFFFu, a2 & 0xFFFFFFu};
            int kb = static_cast<int>(h * 4u);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                tfq3_q8_3b(st, mv[j], T, q0a, q0b, q1a, q1b);
                tfq3_stage_b24(tg_b_warp, q0a, q0b, q1a, q1b, home);
                __syncwarp();
                tfq3_mma_f16_16x16x16(b0, tg_a, kb + j, tg_b_warp, lane);
            }
        }
    } else {
        for (uint32_t h = 0u; h < 2u; ++h) {
            const uint32_t* p = w_s + (static_cast<uint64_t>(bc) * 10ull + h * 5ull) * 32ull + lane;
            uint32_t w0 = __ldg(&p[0]), w1 = __ldg(&p[32]), w2 = __ldg(&p[64]), w3 = __ldg(&p[96]), w4 = __ldg(&p[128]);
            uint32_t hv[4] = {w0 >> 24, (w1 >> 16) & 255u, (w2 >> 8) & 255u, w3 & 255u};
            uint32_t lv[4] = {(w0 << 8) | (w1 >> 24), (w1 << 16) | (w2 >> 16), (w2 << 24) | (w3 >> 8), w4};
            int kb = static_cast<int>(h * 4u);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                tfq3_q8_5b(st, hv[j], lv[j], T, q0a, q0b, q1a, q1b);
                tfq3_stage_b24(tg_b_warp, q0a, q0b, q1a, q1b, home);
                __syncwarp();
                tfq3_mma_f16_16x16x16(b0, tg_a, kb + j, tg_b_warp, lane);
            }
        }
    }
    float fsv = __half2float(__ldg(&sc0[bc]));
    #pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = fmaf(b0[i], fsv, acc[i]);
}

// Split-K=8 BM=16 MMA (4 warps = 64 output rows/CTA, grid=(N/64, 8, ceil(rows/16)), block=128); bit-exact across rows.
template <uint32_t BITS>
__device__ void tfq3_wmma_sk8_impl(
    const uint32_t* __restrict__ W, const __half* __restrict__ scales,
    const Tfq3Tables* __restrict__ T, const __half* __restrict__ X_rot,
    float* __restrict__ YR, uint32_t rows, uint32_t n, uint32_t k
) {
    uint32_t blk = blockIdx.x, kp = blockIdx.y, m0 = blockIdx.z * 16u, n_bc = k >> 7;
    if (m0 >= rows) return;
    uint32_t nb = min(16u, rows - m0);
    uint32_t chunk = n_bc >> 3, bc0 = kp * chunk, bc1 = bc0 + chunk;
    uint32_t tid = threadIdx.x, sg = tid >> 5, lane = tid & 31u, strip = blk * 4u + sg;
    const uint32_t* w_s = W + static_cast<uint64_t>(strip) * n_bc * (BITS * 2u) * 32ull;
    const __half* sc0 = scales + static_cast<uint64_t>(strip >> 3) * n_bc;
    uint32_t st = tfq3_init_st<BITS>(w_s, bc0, lane);
    int2 home = tfq3_frag_home(lane);
    __shared__ __half sm_a[16 * 136];
    __shared__ __half sm_b[4][16 * 24];
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t bc = bc0; bc < bc1; ++bc) {
        for (uint32_t idx = tid; idx < 256u; idx += 128u) {
            uint32_t rl = idx >> 4, cl = (idx & 15u) << 3;
            uint4 v = (rl < nb) ? *reinterpret_cast<const uint4*>(X_rot + static_cast<uint64_t>(m0 + rl) * k + (bc << 7) + cl) : make_uint4(0u, 0u, 0u, 0u);
            *reinterpret_cast<uint4*>(sm_a + rl * 136u + cl) = v;
        }
        __syncthreads();
        tfq3_wmma_bc<BITS>(w_s, sc0, bc, T, st, lane, home, sm_a, sm_b[sg], acc);
        __syncthreads();
    }
    uint32_t g = lane >> 2, t = (lane & 3u) << 1;
    uint64_t pbase = static_cast<uint64_t>(kp) * rows * n + blk * 64u + sg * 16u + t;
    if (g < nb) {
        float* d0 = YR + pbase + static_cast<uint64_t>(m0 + g) * n;
        *reinterpret_cast<float2*>(d0) = make_float2(acc[0], acc[1]);
        *reinterpret_cast<float2*>(d0 + 8u) = make_float2(acc[4], acc[5]);
    }
    if (g + 8u < nb) {
        float* d1 = YR + pbase + static_cast<uint64_t>(m0 + g + 8u) * n;
        *reinterpret_cast<float2*>(d1) = make_float2(acc[2], acc[3]);
        *reinterpret_cast<float2*>(d1 + 8u) = make_float2(acc[6], acc[7]);
    }
}

// Fused BM=16 MMA + inv-FWHT-128 (8 warps = 128 rows/CTA, grid=(N/128, ceil(rows/16)), block=256) for lm_head/prefill.
template <uint32_t BITS>
__device__ void tfq3_wmma_fused_impl(
    const uint32_t* __restrict__ W, const __half* __restrict__ scales,
    const Tfq3Tables* __restrict__ T, const __half* __restrict__ X_rot,
    void* __restrict__ OUT, uint32_t rows, uint32_t k, uint32_t ldo,
    uint32_t sr_split, uint32_t row_off, uint32_t out_f32
) {
    uint32_t rb = blockIdx.x, r0 = rb * 128u, m0 = blockIdx.y * 16u;
    if (m0 >= rows) return;
    uint32_t nb = min(16u, rows - m0), n_bc = k >> 7;
    uint32_t tid = threadIdx.x, sg = tid >> 5, lane = tid & 31u, strip = rb * 8u + sg;
    const uint32_t* w_s = W + static_cast<uint64_t>(strip) * n_bc * (BITS * 2u) * 32ull;
    const __half* sc0 = scales + static_cast<uint64_t>(rb) * n_bc;
    uint32_t st = tfq3_init_st<BITS>(w_s, 0u, lane);
    int2 home = tfq3_frag_home(lane);
    __shared__ union {
        struct { __half a[16 * 136]; __half b[8][16 * 24]; } ab;
        float f[16 * 128];
    } sm;
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t bc = 0u; bc < n_bc; ++bc) {
        uint32_t rl = tid >> 4, cl = (tid & 15u) << 3;
        uint4 v = (rl < nb) ? *reinterpret_cast<const uint4*>(X_rot + static_cast<uint64_t>(m0 + rl) * k + (bc << 7) + cl) : make_uint4(0u, 0u, 0u, 0u);
        *reinterpret_cast<uint4*>(sm.ab.a + rl * 136u + cl) = v;
        __syncthreads();
        tfq3_wmma_bc<BITS>(w_s, sc0, bc, T, st, lane, home, sm.ab.a, sm.ab.b[sg], acc);
        __syncthreads();
    }
    uint32_t g = lane >> 2, t = (lane & 3u) << 1, col = sg * 16u + t;
    *reinterpret_cast<float2*>(sm.f + g * 128u + col) = make_float2(acc[0], acc[1]);
    *reinterpret_cast<float2*>(sm.f + g * 128u + col + 8u) = make_float2(acc[4], acc[5]);
    *reinterpret_cast<float2*>(sm.f + (g + 8u) * 128u + col) = make_float2(acc[2], acc[3]);
    *reinterpret_cast<float2*>(sm.f + (g + 8u) * 128u + col + 8u) = make_float2(acc[6], acc[7]);
    __syncthreads();
    uint32_t sr0 = tfq3_sr0(r0 + row_off, sr_split);
    for (uint32_t m = sg; m < nb; m += 8u) {
        const float* ys = sm.f + m * 128u;
        float4 v = make_float4(ys[lane], ys[lane + 32u], ys[lane + 64u], ys[lane + 96u]);
        tfq3_fwht128_simd4(v, lane);
        uint64_t base = static_cast<uint64_t>(m0 + m) * ldo + r0;
        tfq3_store(OUT, base + lane +  0u, v.x * tfq3_sgn(0x9E3779B9u, sr0 + lane +  0u), out_f32);
        tfq3_store(OUT, base + lane + 32u, v.y * tfq3_sgn(0x9E3779B9u, sr0 + lane + 32u), out_f32);
        tfq3_store(OUT, base + lane + 64u, v.z * tfq3_sgn(0x9E3779B9u, sr0 + lane + 64u), out_f32);
        tfq3_store(OUT, base + lane + 96u, v.w * tfq3_sgn(0x9E3779B9u, sr0 + lane + 96u), out_f32);
    }
}

// BM=64, BN=128 Tensor Core prefill + fused inv-FWHT-128 (4x weight reuse, block=256).
template <uint32_t BITS>
__device__ void tfq3_wmma_prefill64_impl(
    const uint32_t* __restrict__ W, const __half* __restrict__ scales,
    const Tfq3Tables* __restrict__ T, const __half* __restrict__ X_rot,
    void* __restrict__ OUT, uint32_t rows, uint32_t k, uint32_t ldo,
    uint32_t sr_split, uint32_t row_off, uint32_t out_f32
) {
    uint32_t rb = blockIdx.x, r0 = rb * 128u, m0 = blockIdx.y * 64u;
    if (m0 >= rows) return;
    uint32_t nb = min(64u, rows - m0), n_bc = k >> 7;
    uint32_t tid = threadIdx.x, sg = tid >> 5, lane = tid & 31u, strip = rb * 8u + sg;
    const uint32_t* w_s = W + static_cast<uint64_t>(strip) * n_bc * (BITS * 2u) * 32ull;
    const __half* sc0 = scales + static_cast<uint64_t>(rb) * n_bc;
    uint32_t st = tfq3_init_st<BITS>(w_s, 0u, lane);
    int2 home = tfq3_frag_home(lane);
    __shared__ union {
        struct { __half a[64 * 136]; __half b[8][16 * 24]; } ab;
        float f[64 * 128];
    } sm;
    float acc[4][8];
    #pragma unroll
    for (int ms = 0; ms < 4; ++ms) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) acc[ms][i] = 0.0f;
    }
    for (uint32_t bc = 0u; bc < n_bc; ++bc) {
        #pragma unroll
        for (uint32_t u = 0u; u < 4u; ++u) {
            uint32_t idx = tid + u * 256u, rl = idx >> 4, cl = (idx & 15u) << 3;
            uint4 v = (rl < nb) ? *reinterpret_cast<const uint4*>(X_rot + static_cast<uint64_t>(m0 + rl) * k + (bc << 7) + cl) : make_uint4(0u, 0u, 0u, 0u);
            *reinterpret_cast<uint4*>(sm.ab.a + rl * 136u + cl) = v;
        }
        __syncthreads();
        float b0[4][8];
        #pragma unroll
        for (int ms = 0; ms < 4; ++ms) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) b0[ms][i] = 0.0f;
        }
        __half2 q0a, q0b, q1a, q1b;
        if constexpr (BITS == 4u) {
            const uint4* p = reinterpret_cast<const uint4*>(w_s + static_cast<uint64_t>(bc) * 8ull * 32ull);
            for (uint32_t h = 0u; h < 2u; ++h) {
                uint4 v = __ldg(&p[h * 32u + lane]);
                uint32_t uv[4] = {v.x, v.y, v.z, v.w};
                int kb = static_cast<int>(h * 4u);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    tfq3_q8_4b(st, uv[j], T, q0a, q0b, q1a, q1b);
                    tfq3_stage_b24(sm.ab.b[sg], q0a, q0b, q1a, q1b, home);
                    __syncwarp();
                    uint32_t b_lo[2], b_hi[2];
                    tfq3_load_b_16x16(b_lo, b_hi, sm.ab.b[sg], lane);
                    #pragma unroll
                    for (int ms = 0; ms < 4; ++ms) if (ms * 16u < nb) tfq3_mma_f16_16x16x16_b(b0[ms], sm.ab.a + ms * 16u * 136u, kb + j, b_lo, b_hi, lane);
                }
            }
        } else if constexpr (BITS == 3u) {
            for (uint32_t h = 0u; h < 2u; ++h) {
                const uint32_t* p = w_s + (static_cast<uint64_t>(bc) * 6ull + h * 3ull) * 32ull + lane;
                uint32_t a0 = __ldg(&p[0]), a1 = __ldg(&p[32]), a2 = __ldg(&p[64]);
                uint32_t mv[4] = {a0 >> 8, ((a0 << 16) | (a1 >> 16)) & 0xFFFFFFu, ((a1 << 8) | (a2 >> 24)) & 0xFFFFFFu, a2 & 0xFFFFFFu};
                int kb = static_cast<int>(h * 4u);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    tfq3_q8_3b(st, mv[j], T, q0a, q0b, q1a, q1b);
                    tfq3_stage_b24(sm.ab.b[sg], q0a, q0b, q1a, q1b, home);
                    __syncwarp();
                    uint32_t b_lo[2], b_hi[2];
                    tfq3_load_b_16x16(b_lo, b_hi, sm.ab.b[sg], lane);
                    #pragma unroll
                    for (int ms = 0; ms < 4; ++ms) if (ms * 16u < nb) tfq3_mma_f16_16x16x16_b(b0[ms], sm.ab.a + ms * 16u * 136u, kb + j, b_lo, b_hi, lane);
                }
            }
        } else {
            for (uint32_t h = 0u; h < 2u; ++h) {
                const uint32_t* p = w_s + (static_cast<uint64_t>(bc) * 10ull + h * 5ull) * 32ull + lane;
                uint32_t w0 = __ldg(&p[0]), w1 = __ldg(&p[32]), w2 = __ldg(&p[64]), w3 = __ldg(&p[96]), w4 = __ldg(&p[128]);
                uint32_t hv[4] = {w0 >> 24, (w1 >> 16) & 255u, (w2 >> 8) & 255u, w3 & 255u};
                uint32_t lv[4] = {(w0 << 8) | (w1 >> 24), (w1 << 16) | (w2 >> 16), (w2 << 24) | (w3 >> 8), w4};
                int kb = static_cast<int>(h * 4u);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    tfq3_q8_5b(st, hv[j], lv[j], T, q0a, q0b, q1a, q1b);
                    tfq3_stage_b24(sm.ab.b[sg], q0a, q0b, q1a, q1b, home);
                    __syncwarp();
                    uint32_t b_lo[2], b_hi[2];
                    tfq3_load_b_16x16(b_lo, b_hi, sm.ab.b[sg], lane);
                    #pragma unroll
                    for (int ms = 0; ms < 4; ++ms) if (ms * 16u < nb) tfq3_mma_f16_16x16x16_b(b0[ms], sm.ab.a + ms * 16u * 136u, kb + j, b_lo, b_hi, lane);
                }
            }
        }
        float fsv = __half2float(__ldg(&sc0[bc]));
        #pragma unroll
        for (int ms = 0; ms < 4; ++ms) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) acc[ms][i] = fmaf(b0[ms][i], fsv, acc[ms][i]);
        }
        __syncthreads();
    }
    uint32_t g = lane >> 2, t = (lane & 3u) << 1, col = sg * 16u + t;
    #pragma unroll
    for (int ms = 0; ms < 4; ++ms) {
        uint32_t rm = static_cast<uint32_t>(ms) * 16u;
        *reinterpret_cast<float2*>(sm.f + (rm + g) * 128u + col) = make_float2(acc[ms][0], acc[ms][1]);
        *reinterpret_cast<float2*>(sm.f + (rm + g) * 128u + col + 8u) = make_float2(acc[ms][4], acc[ms][5]);
        *reinterpret_cast<float2*>(sm.f + (rm + g + 8u) * 128u + col) = make_float2(acc[ms][2], acc[ms][3]);
        *reinterpret_cast<float2*>(sm.f + (rm + g + 8u) * 128u + col + 8u) = make_float2(acc[ms][6], acc[ms][7]);
    }
    __syncthreads();
    uint32_t sr0 = tfq3_sr0(r0 + row_off, sr_split);
    for (uint32_t m = sg; m < nb; m += 8u) {
        const float* ys = sm.f + m * 128u;
        float4 v = make_float4(ys[lane], ys[lane + 32u], ys[lane + 64u], ys[lane + 96u]);
        tfq3_fwht128_simd4(v, lane);
        uint64_t base = static_cast<uint64_t>(m0 + m) * ldo + r0;
        tfq3_store(OUT, base + lane +  0u, v.x * tfq3_sgn(0x9E3779B9u, sr0 + lane +  0u), out_f32);
        tfq3_store(OUT, base + lane + 32u, v.y * tfq3_sgn(0x9E3779B9u, sr0 + lane + 32u), out_f32);
        tfq3_store(OUT, base + lane + 64u, v.z * tfq3_sgn(0x9E3779B9u, sr0 + lane + 64u), out_f32);
        tfq3_store(OUT, base + lane + 96u, v.w * tfq3_sgn(0x9E3779B9u, sr0 + lane + 96u), out_f32);
    }
}

extern "C" {

__global__ void tf_tfq3_fwht_x(const __nv_bfloat16* __restrict__ x, __half* __restrict__ x_rot, uint32_t k, uint32_t ldx) {
    uint32_t bc = blockIdx.x, r = blockIdx.y, c0 = bc * 128u, lane = threadIdx.x;
    const __nv_bfloat16* src = x + static_cast<uint64_t>(r) * ldx + c0;
    float4 v = make_float4(
        __bfloat162float(src[lane +  0u]) * tfq3_sgn(0x12345678u, c0 + lane +  0u),
        __bfloat162float(src[lane + 32u]) * tfq3_sgn(0x12345678u, c0 + lane + 32u),
        __bfloat162float(src[lane + 64u]) * tfq3_sgn(0x12345678u, c0 + lane + 64u),
        __bfloat162float(src[lane + 96u]) * tfq3_sgn(0x12345678u, c0 + lane + 96u));
    tfq3_fwht128_simd4(v, lane);
    __half* dst = x_rot + static_cast<uint64_t>(r) * k + c0;
    dst[lane +  0u] = __float2half(v.x); dst[lane + 32u] = __float2half(v.y);
    dst[lane + 64u] = __float2half(v.z); dst[lane + 96u] = __float2half(v.w);
}

__global__ void tf_tfq3_fwht_y(const float* __restrict__ yr, void* __restrict__ out, uint32_t rows, uint32_t n, uint32_t ldo, uint32_t sr_split, uint32_t row_off, uint32_t ksplit, uint32_t out_f32) {
    uint32_t br = blockIdx.x, r = blockIdx.y, r0 = br * 128u, lane = threadIdx.x;
    uint64_t plane = static_cast<uint64_t>(rows) * n, off = static_cast<uint64_t>(r) * n + r0;
    float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (uint32_t kp = 0u; kp < ksplit; ++kp) {
        const float* p = yr + kp * plane + off;
        v.x += p[lane + 0u]; v.y += p[lane + 32u]; v.z += p[lane + 64u]; v.w += p[lane + 96u];
    }
    tfq3_fwht128_simd4(v, lane);
    uint32_t sr0 = tfq3_sr0(r0 + row_off, sr_split);
    uint64_t base = static_cast<uint64_t>(r) * ldo + r0;
    tfq3_store(out, base + lane +  0u, v.x * tfq3_sgn(0x9E3779B9u, sr0 + lane +  0u), out_f32);
    tfq3_store(out, base + lane + 32u, v.y * tfq3_sgn(0x9E3779B9u, sr0 + lane + 32u), out_f32);
    tfq3_store(out, base + lane + 64u, v.z * tfq3_sgn(0x9E3779B9u, sr0 + lane + 64u), out_f32);
    tfq3_store(out, base + lane + 96u, v.w * tfq3_sgn(0x9E3779B9u, sr0 + lane + 96u), out_f32);
}

#define TF_TFQ3_DENSE_INST(B) \
__global__ void tf_tfq3_wmma_sk8_##B##b(const uint32_t* W, const __half* S, const Tfq3Tables* T, const __half* X, float* YR, uint32_t rows, uint32_t n, uint32_t k) { tfq3_wmma_sk8_impl<B>(W, S, T, X, YR, rows, n, k); } \
__global__ void tf_tfq3_wmma_fused_##B##b(const uint32_t* W, const __half* S, const Tfq3Tables* T, const __half* X, void* O, uint32_t rows, uint32_t k, uint32_t ldo, uint32_t sr, uint32_t ro, uint32_t f32) { tfq3_wmma_fused_impl<B>(W, S, T, X, O, rows, k, ldo, sr, ro, f32); } \
__global__ void tf_tfq3_wmma_prefill64_##B##b(const uint32_t* W, const __half* S, const Tfq3Tables* T, const __half* X, void* O, uint32_t rows, uint32_t k, uint32_t ldo, uint32_t sr, uint32_t ro, uint32_t f32) { tfq3_wmma_prefill64_impl<B>(W, S, T, X, O, rows, k, ldo, sr, ro, f32); }

TF_TFQ3_DENSE_INST(3)
TF_TFQ3_DENSE_INST(4)
TF_TFQ3_DENSE_INST(5)

} // extern "C"
