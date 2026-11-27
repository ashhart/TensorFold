// TensorFold TFQ3 (3/4/5-bit HeatAlloc TFPW) Routed MoE CUDA Kernels (GLM-5.3-Flash & Qwen3.8-Flash-Next).
#include "tfq3_common.cuh"

__device__ __forceinline__ void tfq3_mv1_dispatch(
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

extern "C" {

// Fused MoE gate+up -> inv-FWHT-128 -> SwiGLU -> fwd-FWHT-128 -> act_rot[slot_flat, inter].
__global__ void tf_tfq3_moe_gu_swiglu(
    const uint32_t* __restrict__ W,
    const __half* __restrict__ scales,
    const Tfq3ExpertMeta* __restrict__ metas,
    const Tfq3Tables* __restrict__ T,
    const __half* __restrict__ X_rot,
    const int32_t* __restrict__ picks,
    __half* __restrict__ act_rot,
    uint32_t dims,
    uint32_t inter,
    uint32_t experts,
    uint32_t top_k,
    uint32_t pick_stride,
    float lim,
    uint32_t act_mode
) {
    uint32_t br = blockIdx.x, slot_flat = blockIdx.y;
    uint32_t row = slot_flat / top_k, s = slot_flat - row * top_k;
    int32_t eid = picks[static_cast<uint64_t>(row) * pick_stride + s];
    if (eid < 0 || static_cast<uint32_t>(eid) >= experts) return;

    Tfq3ExpertMeta em = metas[eid];
    uint32_t bits = em.b_bits, wpb = bits * 2u, n_bc = dims >> 7;
    uint32_t sg = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    uint32_t strip = br * 8u + sg, r0 = br * 128u;
    int2 home = tfq3_frag_home(lane);
    const __half* xr = X_rot + static_cast<uint64_t>(row) * dims;
    const uint32_t* wg = W + em.gate_word_off + static_cast<uint64_t>(strip) * n_bc * wpb * 32ull;
    const uint32_t* wu = W + em.up_word_off   + static_cast<uint64_t>(strip) * n_bc * wpb * 32ull;
    const __half* sg_p = scales + em.gate_sc_off + static_cast<uint64_t>(br) * n_bc;
    const __half* su_p = scales + em.up_sc_off   + static_cast<uint64_t>(br) * n_bc;

    __shared__ float tg_g[128], tg_u[128];
    float g0, g1, u0, u1;
    tfq3_mv1_dispatch(bits, wg, sg_p, n_bc, T, xr, lane, home, g0, g1);
    tfq3_mv1_dispatch(bits, wu, su_p, n_bc, T, xr, lane, home, u0, u1);
    if ((lane & 9u) == 0u) {
        uint32_t idx = sg * 16u + home.y;
        tg_g[idx] = g0; tg_g[idx + 8u] = g1;
        tg_u[idx] = u0; tg_u[idx + 8u] = u1;
    }
    __syncthreads();
    if (sg == 0u) {
        float4 gv = make_float4(tg_g[lane], tg_g[lane + 32u], tg_g[lane + 64u], tg_g[lane + 96u]);
        float4 uv = make_float4(tg_u[lane], tg_u[lane + 32u], tg_u[lane + 64u], tg_u[lane + 96u]);
        tfq3_fwht128_simd4(gv, lane);
        tfq3_fwht128_simd4(uv, lane);
        float4 z;
        z.x = tfq3_glu(gv.x * tfq3_sgn(0x9E3779B9u, r0 + lane +  0u), uv.x * tfq3_sgn(0x9E3779B9u, r0 + lane +  0u), lim, act_mode) * tfq3_sgn(0x12345678u, r0 + lane +  0u);
        z.y = tfq3_glu(gv.y * tfq3_sgn(0x9E3779B9u, r0 + lane + 32u), uv.y * tfq3_sgn(0x9E3779B9u, r0 + lane + 32u), lim, act_mode) * tfq3_sgn(0x12345678u, r0 + lane + 32u);
        z.z = tfq3_glu(gv.z * tfq3_sgn(0x9E3779B9u, r0 + lane + 64u), uv.z * tfq3_sgn(0x9E3779B9u, r0 + lane + 64u), lim, act_mode) * tfq3_sgn(0x12345678u, r0 + lane + 64u);
        z.w = tfq3_glu(gv.w * tfq3_sgn(0x9E3779B9u, r0 + lane + 96u), uv.w * tfq3_sgn(0x9E3779B9u, r0 + lane + 96u), lim, act_mode) * tfq3_sgn(0x12345678u, r0 + lane + 96u);
        tfq3_fwht128_simd4(z, lane);
        __half* dst = act_rot + static_cast<uint64_t>(slot_flat) * inter + r0;
        dst[lane +  0u] = __float2half(z.x);
        dst[lane + 32u] = __float2half(z.y);
        dst[lane + 64u] = __float2half(z.z);
        dst[lane + 96u] = __float2half(z.w);
    }
}

// Fused MoE gate+up -> inv-FWHT-128 -> Inkling-2 softcapped SwiGLU -> fwd-FWHT-128 -> act_rot[slot_flat, inter].
__global__ void tf_tfq3_moe_gu_softcap(
    const uint32_t* __restrict__ W,
    const __half* __restrict__ scales,
    const Tfq3ExpertMeta* __restrict__ metas,
    const Tfq3Tables* __restrict__ T,
    const __half* __restrict__ X_rot,
    const int32_t* __restrict__ picks,
    __half* __restrict__ act_rot,
    uint32_t dims,
    uint32_t inter,
    uint32_t experts,
    uint32_t top_k,
    uint32_t pick_stride,
    float gate_cap,
    float up_cap
) {
    uint32_t br = blockIdx.x, slot_flat = blockIdx.y;
    uint32_t row = slot_flat / top_k, s = slot_flat - row * top_k;
    int32_t eid = picks[static_cast<uint64_t>(row) * pick_stride + s];
    if (eid < 0 || static_cast<uint32_t>(eid) >= experts) return;

    Tfq3ExpertMeta em = metas[eid];
    uint32_t bits = em.b_bits, wpb = bits * 2u, n_bc = dims >> 7;
    uint32_t sg = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    uint32_t strip = br * 8u + sg, r0 = br * 128u;
    int2 home = tfq3_frag_home(lane);
    const __half* xr = X_rot + static_cast<uint64_t>(row) * dims;
    const uint32_t* wg = W + em.gate_word_off + static_cast<uint64_t>(strip) * n_bc * wpb * 32ull;
    const uint32_t* wu = W + em.up_word_off   + static_cast<uint64_t>(strip) * n_bc * wpb * 32ull;
    const __half* sg_p = scales + em.gate_sc_off + static_cast<uint64_t>(br) * n_bc;
    const __half* su_p = scales + em.up_sc_off   + static_cast<uint64_t>(br) * n_bc;

    __shared__ float tg_g[128], tg_u[128];
    float g0, g1, u0, u1;
    tfq3_mv1_dispatch(bits, wg, sg_p, n_bc, T, xr, lane, home, g0, g1);
    tfq3_mv1_dispatch(bits, wu, su_p, n_bc, T, xr, lane, home, u0, u1);
    if ((lane & 9u) == 0u) {
        uint32_t idx = sg * 16u + home.y;
        tg_g[idx] = g0; tg_g[idx + 8u] = g1;
        tg_u[idx] = u0; tg_u[idx + 8u] = u1;
    }
    __syncthreads();
    if (sg == 0u) {
        float4 gv = make_float4(tg_g[lane], tg_g[lane + 32u], tg_g[lane + 64u], tg_g[lane + 96u]);
        float4 uv = make_float4(tg_u[lane], tg_u[lane + 32u], tg_u[lane + 64u], tg_u[lane + 96u]);
        tfq3_fwht128_simd4(gv, lane);
        tfq3_fwht128_simd4(uv, lane);
        float4 z;
        z.x = tfq3_glu_softcap(gv.x * tfq3_sgn(0x9E3779B9u, r0 + lane +  0u), uv.x * tfq3_sgn(0x9E3779B9u, r0 + lane +  0u), gate_cap, up_cap) * tfq3_sgn(0x12345678u, r0 + lane +  0u);
        z.y = tfq3_glu_softcap(gv.y * tfq3_sgn(0x9E3779B9u, r0 + lane + 32u), uv.y * tfq3_sgn(0x9E3779B9u, r0 + lane + 32u), gate_cap, up_cap) * tfq3_sgn(0x12345678u, r0 + lane + 32u);
        z.z = tfq3_glu_softcap(gv.z * tfq3_sgn(0x9E3779B9u, r0 + lane + 64u), uv.z * tfq3_sgn(0x9E3779B9u, r0 + lane + 64u), gate_cap, up_cap) * tfq3_sgn(0x12345678u, r0 + lane + 64u);
        z.w = tfq3_glu_softcap(gv.w * tfq3_sgn(0x9E3779B9u, r0 + lane + 96u), uv.w * tfq3_sgn(0x9E3779B9u, r0 + lane + 96u), gate_cap, up_cap) * tfq3_sgn(0x12345678u, r0 + lane + 96u);
        tfq3_fwht128_simd4(z, lane);
        __half* dst = act_rot + static_cast<uint64_t>(slot_flat) * inter + r0;
        dst[lane +  0u] = __float2half(z.x);
        dst[lane + 32u] = __float2half(z.y);
        dst[lane + 64u] = __float2half(z.z);
        dst[lane + 96u] = __float2half(z.w);
    }
}

// Fused MoE down -> inv-FWHT-128 -> ey[row, slot, dims] (bf16 or f32); grid=(dims/128, rows*top_k), block=256.
__global__ void tf_tfq3_moe_down(
    const uint32_t* __restrict__ W,
    const __half* __restrict__ scales,
    const Tfq3ExpertMeta* __restrict__ metas,
    const Tfq3Tables* __restrict__ T,
    const __half* __restrict__ act_rot,
    const int32_t* __restrict__ picks,
    void* __restrict__ ey,
    uint32_t dims,
    uint32_t inter,
    uint32_t experts,
    uint32_t top_k,
    uint32_t pick_stride,
    uint32_t out_slot_stride,
    uint32_t out_f32
) {
    uint32_t br = blockIdx.x, slot_flat = blockIdx.y;
    uint32_t row = slot_flat / top_k, s = slot_flat - row * top_k;
    int32_t eid = picks[static_cast<uint64_t>(row) * pick_stride + s];
    if (eid < 0 || static_cast<uint32_t>(eid) >= experts) return;

    Tfq3ExpertMeta em = metas[eid];
    uint32_t bits = em.b_bits, wpb = bits * 2u, n_bc = inter >> 7;
    uint32_t sg = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    uint32_t strip = br * 8u + sg, r0 = br * 128u;
    int2 home = tfq3_frag_home(lane);
    const __half* ar = act_rot + static_cast<uint64_t>(slot_flat) * inter;
    const uint32_t* wd = W + em.down_word_off + static_cast<uint64_t>(strip) * n_bc * wpb * 32ull;
    const __half* sd_p = scales + em.down_sc_off + static_cast<uint64_t>(br) * n_bc;

    __shared__ float tg_y[128];
    float a0, a1;
    tfq3_mv1_dispatch(bits, wd, sd_p, n_bc, T, ar, lane, home, a0, a1);
    if ((lane & 9u) == 0u) {
        uint32_t idx = sg * 16u + home.y;
        tg_y[idx] = a0; tg_y[idx + 8u] = a1;
    }
    __syncthreads();
    if (sg == 0u) {
        float4 v = make_float4(tg_y[lane], tg_y[lane + 32u], tg_y[lane + 64u], tg_y[lane + 96u]);
        tfq3_fwht128_simd4(v, lane);
        uint64_t base = (static_cast<uint64_t>(row) * out_slot_stride + s) * dims + r0;
        tfq3_store(ey, base + lane +  0u, v.x * tfq3_sgn(0x9E3779B9u, r0 + lane +  0u), out_f32);
        tfq3_store(ey, base + lane + 32u, v.y * tfq3_sgn(0x9E3779B9u, r0 + lane + 32u), out_f32);
        tfq3_store(ey, base + lane + 64u, v.z * tfq3_sgn(0x9E3779B9u, r0 + lane + 64u), out_f32);
        tfq3_store(ey, base + lane + 96u, v.w * tfq3_sgn(0x9E3779B9u, r0 + lane + 96u), out_f32);
    }
}

// The contract checks the production helper's BF16 SiLU point in both activation branches.
__global__ void tf_tfq3_glu_contract(float* out) {
    if (threadIdx.x == 0u) {
        out[0] = tfq3_glu(0.5f, 1.5f, 7.0f, 0u);
        out[1] = tfq3_glu(0.5f, 1.5f, 7.0f, 1u);
    }
}

} // extern "C"
