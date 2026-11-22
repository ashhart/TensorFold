// Inkling 2 Mini sm_121a routed-MoE kernels: NVFP4 OMMA + MXFP8 QMMA.
// Ported from tf-wt-dflash fn_nvfp4_omma.cu (patch 0035) and fn_mxfp8_qmma.cu
// (patch 0034) for Inkling-2-Mini shapes: E=128 experts, w13 [E,2304,2304]
// (gate|up), w2 [E,2304,1152] (down), routed top-8, decode rows<=16 and
// prefill rows<=2048 on one persistent grid.
// Deltas vs the dflash sources:
//   - activation is Inkling's capped SwiGLU: cap(gate,4)*sigmoid(gate)*cap(up,25)
//     (inkling2/experts.cu tf_ink2moe_tensor_act), not dflash's plain SwiGLU;
//   - the NVFP4 dynamic-row quantizer is generic over K groups (144 for K=2304,
//     72 for K=1152) instead of K=2560/K=640 specializations;
//   - grid/plan types are renamed (Ink2Plan) but layout-identical;
//   - qmm_frag.cuh is inlined (tile_of, ldmatrix, cp helpers) so this file
//     builds standalone: nvcc --ptx -arch=sm_121a sm121a_moe.cu.
// Weight layouts consumed (built once at load by sm121a_pack.zig):
//   NVFP4: [E][N/32][K/32][M][36 uint4] packExpert cells (M=2 gate|up, M=1 down),
//     e4m3 scales inline, scale2[E*M] = amax[e]/den folded on host.
//   MXFP8: w8 [npad/64][K/64][4096] fragment order + bs [npad/64][K/64][128] e8m0.
// Built for compute_121a/sm_121a only.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace ink2sm {

// ---- async-copy / ldmatrix / tiling helpers (inlined qmm_frag subset) ----
__device__ __forceinline__ uint32_t sptr(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void cp16z(void* dst, const void* src, bool ok) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sptr(dst)), "l"(src), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void cp4z(void* dst, const void* src, bool ok) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(sptr(dst)), "l"(src), "r"(ok ? 4 : 0));
}
__device__ __forceinline__ void cp16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(sptr(dst)), "l"(src));
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(sptr(p)));
}
__device__ __forceinline__ int2 tile_of(int b, int M, int N, int BM, int BN, int group) {
    const int rows_t = (M + BM - 1) / BM, cols_t = (N + BN - 1) / BN, band = group * cols_t;
    const int first = b / band * group, in_band = b % band, height = min(group, rows_t - first);
    return make_int2((first + in_band % height) * BM, in_band / height * BN);
}
template <int CHUNKS>
__device__ __forceinline__ int swz_x8(int r, int ch) {
    if constexpr (CHUNKS == 4) return ch ^ ((r >> 1) & 3);
    else return ch ^ (r & (CHUNKS - 1));
}

// ---- BF16 -> NVFP4 activation quantizer (hardware cvt.rn.satfinite.e2m1x2) ----
__device__ __forceinline__ uint32_t cvt_e2m1x2(float v_lo, float v_hi) {
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
    uint32_t r;
    asm volatile(
        "{\n"
        ".reg .b8 tmp;\n"
        "cvt.rn.satfinite.e2m1x2.f32 tmp, %2, %1;\n"
        "cvt.u32.u8 %0, tmp;\n"
        "}\n"
        : "=r"(r) : "f"(v_lo), "f"(v_hi));
    return r;
#else
    __trap();
    return 0;
#endif
}

__device__ __forceinline__ uint32_t pack8_e2m1(float f0, float f1, float f2, float f3,
                                               float f4, float f5, float f6, float f7) {
    uint32_t b0 = cvt_e2m1x2(f0, f1);
    uint32_t b1 = cvt_e2m1x2(f2, f3);
    uint32_t b2 = cvt_e2m1x2(f4, f5);
    uint32_t b3 = cvt_e2m1x2(f6, f7);
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

struct BlockScale {
    uint8_t sf8;
    float mul;
};

__device__ __forceinline__ BlockScale compute_block_scale(float amax16, float g) {
    // ue4m3 max finite = 448.0f (0x7E). __NV_SATFINITE maps >=448 and +Inf to 0x7E, never 0x7F (NaN).
    const float raw = fminf(g * (amax16 * (1.0f / 6.0f)), 448.0f);
    const __nv_fp8_storage_t sf8 = __nv_cvt_float_to_fp8(raw, __NV_SATFINITE, __NV_E4M3);
    const float sf = __half2float(__half(__nv_cvt_fp8_to_halfraw(sf8, __NV_E4M3)));
    return {static_cast<uint8_t>(sf8), (sf > 0.0f) ? __fdiv_rn(g, sf) : 0.0f};
}

__device__ __forceinline__ void quantize_group16_impl(const float (&f)[16], float amax16, float g,
                                                      uint2& out_codes, uint8_t& out_sf8) {
    const BlockScale bs = compute_block_scale(amax16, g);
    const float m = bs.mul;
    out_sf8 = bs.sf8;
    out_codes.x = pack8_e2m1(f[0]*m, f[1]*m, f[2]*m, f[3]*m, f[4]*m, f[5]*m, f[6]*m, f[7]*m);
    out_codes.y = pack8_e2m1(f[8]*m, f[9]*m, f[10]*m, f[11]*m, f[12]*m, f[13]*m, f[14]*m, f[15]*m);
}

__device__ __forceinline__ float load16_bf16_and_amax(const __nv_bfloat16* p, float (&f)[16]) {
    const uint4 u0 = *reinterpret_cast<const uint4*>(p);
    const uint4 u1 = *reinterpret_cast<const uint4*>(p + 8);
    const __nv_bfloat162* b0 = reinterpret_cast<const __nv_bfloat162*>(&u0);
    const __nv_bfloat162* b1 = reinterpret_cast<const __nv_bfloat162*>(&u1);
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 v0 = __bfloat1622float2(b0[i]);
        f[2 * i + 0] = v0.x;
        f[2 * i + 1] = v0.y;
        amax = fmaxf(amax, fmaxf(fabsf(v0.x), fabsf(v0.y)));
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 v1 = __bfloat1622float2(b1[i]);
        f[8 + 2 * i + 0] = v1.x;
        f[8 + 2 * i + 1] = v1.y;
        amax = fmaxf(amax, fmaxf(fabsf(v1.x), fabsf(v1.y)));
    }
    return amax;
}

// Static calibrated global-scale quantizer, any K multiple of 64.
__device__ __forceinline__ void quant_static_body(const __nv_bfloat16* __restrict__ X, int x_stride, int M, int K,
                    float global_input_scale, const float* __restrict__ per_row_input_scale,
                    uint8_t* __restrict__ codes, uint8_t* __restrict__ scales,
                    float* __restrict__ s2_out) {
    const int ng = K >> 4;
    const size_t total = static_cast<size_t>(M) * ng;
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int row = static_cast<int>(idx / ng);
    const int blk = static_cast<int>(idx - static_cast<size_t>(row) * ng);
    const float in_scale = per_row_input_scale ? __ldg(per_row_input_scale + row) : global_input_scale;
    const float g = (in_scale > 0.0f) ? __fdiv_rn(1.0f, in_scale) : 0.0f;
    float f[16];
    const float amax16 = load16_bf16_and_amax(X + static_cast<size_t>(row) * x_stride + blk * 16, f);
    uint2 q;
    uint8_t sf8;
    quantize_group16_impl(f, amax16, g, q, sf8);
    *reinterpret_cast<uint2*>(codes + static_cast<size_t>(row) * (K >> 1) + blk * 8) = q;
    scales[static_cast<size_t>(row) * ng + blk] = sf8;
    if (s2_out && blk == 0) s2_out[row] = in_scale;
}

// Single-pass dynamic per-row quantizer, generic over K groups (144 for K=2304,
// 72 for K=1152). 1 block of 160 threads = 1 row; threads >= groups idle.
template <int GROUPS>
__device__ __forceinline__ void quant_dynamic_row_body(const __nv_bfloat16* __restrict__ X, int x_stride, int M,
                              float floor_scale,
                              uint8_t* __restrict__ codes, uint8_t* __restrict__ scales,
                              float* __restrict__ s2_out) {
    static_assert(GROUPS <= 160, "one 160-thread block covers one row");
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x; // 0..159 (5 warps)
    const int warp = tid >> 5;
    const int lane = tid & 31;
    float f[16];
    float amax16 = 0.0f;
    if (tid < GROUPS) amax16 = load16_bf16_and_amax(X + static_cast<size_t>(row) * x_stride + tid * 16, f);
    float wmax = amax16;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        wmax = fmaxf(wmax, __shfl_xor_sync(0xFFFFFFFFu, wmax, off));
    __shared__ float smax[5];
    if (lane == 0) smax[warp] = wmax;
    __syncthreads();
    float row_amax = fmaxf(fmaxf(smax[0], smax[1]), fmaxf(fmaxf(smax[2], smax[3]), smax[4]));
    const float s2 = fmaxf(row_amax * (1.0f / 2688.0f), floor_scale > 0.0f ? floor_scale : 1e-12f);
    const float g = __fdiv_rn(1.0f, s2);
    if (tid < GROUPS) {
        uint2 q;
        uint8_t sf8;
        quantize_group16_impl(f, amax16, g, q, sf8);
        *reinterpret_cast<uint2*>(codes + static_cast<size_t>(row) * (GROUPS * 8) + tid * 8) = q;
        scales[static_cast<size_t>(row) * GROUPS + tid] = sf8;
    }
    if (tid == 0) s2_out[row] = s2;
}

// ---- NVFP4 OMMA core ----
struct Ink2Plan {
    const int* items = nullptr;    // [n_items * 3] {expert, first, cnt}
    const int* counts = nullptr;   // [1] n_items (read on device -> graph friendly)
    const int* members = nullptr;  // [P] pair ids
    int skip = 512;                // shared-expert pseudo id (skipped)
};

// OMMA.SF.16864: D += (A * sfa) @ (B * sfb)^T, e2m1 x e2m1, ue4m3 block-16 scales (4 per K64), fp32 accumulate.
__device__ __forceinline__ void omma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, uint32_t sa,
                                     uint32_t sb) {
    const uint16_t z = 0;
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, %10, {%11, %12}, %13, {%14, %15};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sa), "h"(z), "h"(z), "r"(sb), "h"(z),
          "h"(z));
}

// packExpert cell adapter: [E][N/32][K/32][M][36 uint4]; M=2 gate|up, M=1 down.
constexpr int LIVE = 36;
__device__ __forceinline__ uint32_t fix_nibbles(uint32_t v) {
    return (v & 0xF00FF00Fu) | ((v & 0x00F000F0u) << 4) | ((v >> 4) & 0x00F000F0u);
}
__device__ __forceinline__ uint4 live_codes(const uint4* cell, int lane) {
    const int gq = lane >> 2, t = lane & 3;
    const uint4 a = cell[gq * 4 + 2 * (t & 1)], b = cell[gq * 4 + 2 * (t & 1) + 1];
    const uint32_t s = (t >> 1) ? 0x7531u : 0x6420u;
    return make_uint4(fix_nibbles(__byte_perm(a.x, b.x, s)), fix_nibbles(__byte_perm(a.y, b.y, s)),
                      fix_nibbles(__byte_perm(a.z, b.z, s)), fix_nibbles(__byte_perm(a.w, b.w, s)));
}
__device__ __forceinline__ uint4 live_scales(const uint4* c0, const uint4* c1, int gq) {
    const uint4 s0 = c0[32 + (gq >> 1)], s1 = c1[32 + (gq >> 1)];
    const int c = gq & 1;
    const uint32_t k0 = 0x40u + c * 0x11u, k1 = 0x62u + c * 0x11u;
    const uint32_t lo0 = __byte_perm(__byte_perm(s0.x, s0.z, k0), __byte_perm(s1.x, s1.z, k0), 0x5410u);
    const uint32_t lo1 = __byte_perm(__byte_perm(s0.x, s0.z, k1), __byte_perm(s1.x, s1.z, k1), 0x5410u);
    const uint32_t hi0 = __byte_perm(__byte_perm(s0.y, s0.w, k0), __byte_perm(s1.y, s1.w, k0), 0x5410u);
    const uint32_t hi1 = __byte_perm(__byte_perm(s0.y, s0.w, k1), __byte_perm(s1.y, s1.w, k1), 0x5410u);
    return make_uint4(lo0, lo1, hi0, hi1);
}

__device__ __forceinline__ float bfr(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
// Inkling's capped SwiGLU (experts.cu tf_ink2moe_tensor_act): caps on gate/up, sigmoid on the RAW gate.
template <int GATE_CAP_I, int UP_CAP_I>
__device__ __forceinline__ float ink_act(float gv_raw, float uv_raw) {
    constexpr float GATE_CAP = float(GATE_CAP_I), UP_CAP = float(UP_CAP_I);
    const float raw = bfr(gv_raw);
    const float uv = bfr(uv_raw);
    const float g = GATE_CAP * tanhf(raw / GATE_CAP);
    const float u = UP_CAP * tanhf(uv / UP_CAP);
    const float sigmoid = 1.f / (1.f + expf(-raw));
    return g * sigmoid * u;
}

constexpr int CELL = 72;
constexpr int ROWS = 64;
constexpr int A_BYTES = ROWS * 32;
constexpr int AS_BYTES = ROWS * 4;

template <int M, int WARPS>
struct Smem {
    static constexpr int W = WARPS * M * CELL * 16;
    static constexpr int STAGE = A_BYTES + AS_BYTES + W;
};

struct Args {
    const uint8_t* aq;
    const uint8_t* as;
    int slots;
    const uint4* W;
    const float* scale2;   // [E*M] = amax[e]/den folded at load
    const float* in_scale; // gate_up: down input scale [E]; down: alpha [E]
    float x_s2;
    const float* x_s2_row;
    const float* a_s2_row;
    int K, N;
    Ink2Plan plan;
    uint8_t* oq;
    uint8_t* os;
    __nv_bfloat16* h_dbg;
    float* gu_dbg;
    void* out;
};

template <int M, int MODE, int WARPS, int RT, int S, int GATE_CAP_I, int UP_CAP_I>
__device__ __forceinline__ void chunk(char* smem, const Args& A, int e, int first, int cnt, int r0, int cb, bool active,
                                      int tid, int lane, int wp) {
    using SM = Smem<M, WARPS>;
    constexpr int NT = WARPS * 32;
    const int KG = A.K >> 6;
    const int NB = A.N >> 5;
    const int gq = lane >> 2, t = lane & 3;
    const size_t a_pitch = A.K >> 1, s_pitch = A.K >> 4;
    constexpr int ACH = RT * 32;
    constexpr int ACP = (ACH + NT - 1) / NT;
    const uint8_t* asrc[ACP];
    int adst[ACP];
    bool aok[ACP];
#pragma unroll
    for (int i = 0; i < ACP; ++i) {
        const int idx = tid + i * NT, r = idx >> 1, c = idx & 1;
        const bool inr = idx < ACH;
        aok[i] = inr && (r0 + r < cnt);
        const int pr = aok[i] ? __ldg(A.plan.members + first + r0 + r) : 0;
        const int row = A.slots ? pr / A.slots : pr;
        asrc[i] = A.aq + (size_t)row * a_pitch + c * 16;
        adst[i] = inr ? (r * 32 + ((c ^ ((r >> 2) & 1)) * 16)) : -1;
    }
    const uint8_t* ssrc = nullptr;
    bool sok = false;
    const bool sld = tid < RT * 16;
    if (sld) {
        sok = r0 + tid < cnt;
        const int pr = sok ? __ldg(A.plan.members + first + r0 + tid) : 0;
        const int row = A.slots ? pr / A.slots : pr;
        ssrc = A.as + (size_t)row * s_pitch;
    }
    const uint4* wblk = A.W + ((size_t)e * NB + (active ? cb : 0)) * (size_t)KG * (M * CELL);
    auto stage = [&](int g, int slot) {
        char* base = smem + slot * SM::STAGE;
#pragma unroll
        for (int i = 0; i < ACP; ++i)
            if (adst[i] >= 0) cp16z(base + adst[i], asrc[i] + g * 32, aok[i]);
        if (sld) cp4z(base + A_BYTES + tid * 4, ssrc + g * 4, sok);
        if (active) {
            const uint4* src = wblk + (size_t)g * (M * CELL);
            char* dst = base + A_BYTES + AS_BYTES + wp * (M * CELL * 16);
#pragma unroll
            for (int m = 0; m < M; ++m) {
                cp16(dst + (m * CELL + lane) * 16, src + m * CELL + lane);
                cp16(dst + (m * CELL + 32 + lane) * 16, src + m * CELL + 32 + lane);
                if (lane < 8) cp16(dst + (m * CELL + 64 + lane) * 16, src + m * CELL + 64 + lane);
            }
        }
    };
    float acc[M][RT][4][4];
#pragma unroll
    for (int m = 0; m < M; ++m)
#pragma unroll
        for (int r = 0; r < RT; ++r)
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[m][r][j][0] = acc[m][r][j][1] = acc[m][r][j][2] = acc[m][r][j][3] = 0.f;
#pragma unroll
    for (int s = 0; s < S - 1; ++s) {
        if (s < KG) stage(s, s);
        commit();
    }
    const int lq = lane >> 3, lr = (lane & 7) + (lq & 1) * 8;
    const int ldoff = lr * 32 + (((lq >> 1) ^ ((lr >> 2) & 1)) * 16);
    for (int g = 0; g < KG; ++g) {
        wait<S - 2>();
        __syncthreads();
        if (g + S - 1 < KG) stage(g + S - 1, (g + S - 1) % S);
        commit();
        if (active) {
            const char* base = smem + (g % S) * SM::STAGE;
            const uint4* wsm = reinterpret_cast<const uint4*>(base + A_BYTES + AS_BYTES + wp * (M * CELL * 16));
            uint4 wb0[M], wb1[M], wsb[M];
#pragma unroll
            for (int m = 0; m < M; ++m) {
                wb0[m] = live_codes(wsm + m * LIVE, lane);
                wb1[m] = live_codes(wsm + (M + m) * LIVE, lane);
                wsb[m] = live_scales(wsm + m * LIVE, wsm + (M + m) * LIVE, gq);
            }
            uint32_t comp0[M][4], comp1[M][4], comps[M][4];
#pragma unroll
            for (int m = 0; m < M; ++m)
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    comp0[m][j] = j == 0 ? wb0[m].x : j == 1 ? wb0[m].y : j == 2 ? wb0[m].z : wb0[m].w;
                    comp1[m][j] = j == 0 ? wb1[m].x : j == 1 ? wb1[m].y : j == 2 ? wb1[m].z : wb1[m].w;
                    comps[m][j] = j == 0 ? wsb[m].x : j == 1 ? wsb[m].y : j == 2 ? wsb[m].z : wsb[m].w;
                }
#pragma unroll
            for (int r = 0; r < RT; ++r) {
                uint32_t a[4];
                ldsm4(a, base + r * 512 + ldoff);
                const uint32_t sa =
                    *reinterpret_cast<const uint32_t*>(base + A_BYTES + (r * 16 + gq + 8 * (t & 1)) * 4);
#pragma unroll
                for (int m = 0; m < M; ++m)
#pragma unroll
                    for (int j = 0; j < 4; ++j)
                        omma(acc[m][r][j], a, comp0[m][j], comp1[m][j], sa, comps[m][j]);
            }
        }
    }
    wait<0>();
    __syncthreads();
    if (!active) return;
    const int col0 = cb * 32 + 2 * t;
    if constexpr (MODE == 0) {
        const float gs_g = __ldg(A.scale2 + e * 2), gs_u = __ldg(A.scale2 + e * 2 + 1);
        const float din = __ldg(A.in_scale + e);
        const float qg = (din > 0.f) ? __fdiv_rn(1.0f, din) : 0.f;
        const int np = A.N >> 1, ns = A.N >> 4;
#pragma unroll
        for (int r = 0; r < RT; ++r) {
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                const int rr = r0 + r * 16 + gq + 8 * hh;
                const bool ok = rr < cnt;
                const int pr = ok ? __ldg(A.plan.members + first + rr) : 0;
                const float xs2 = A.x_s2_row ? __ldg(A.x_s2_row + pr / A.slots) : A.x_s2;
                const float ag = gs_g * xs2, au = gs_u * xs2;
                float hv[4][2];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
#pragma unroll
                    for (int q = 0; q < 2; ++q) {
                        const float gv = acc[0][r][j][2 * hh + q] * ag;
                        const float uv = acc[1][r][j][2 * hh + q] * au;
                        hv[j][q] = ink_act<GATE_CAP_I, UP_CAP_I>(gv, uv);
                        if (A.gu_dbg && ok) {
                            A.gu_dbg[(size_t)pr * 2 * A.N + col0 + 8 * j + q] = gv;
                            A.gu_dbg[(size_t)pr * 2 * A.N + A.N + col0 + 8 * j + q] = uv;
                        }
                    }
                    if (A.h_dbg && ok)
                        *reinterpret_cast<__nv_bfloat162*>(A.h_dbg + (size_t)pr * A.N + col0 + 8 * j) =
                            __floats2bfloat162_rn(hv[j][0], hv[j][1]);
                }
                if (A.oq == nullptr) continue;
#pragma unroll
                for (int G = 0; G < 2; ++G) {
                    float am = fmaxf(fmaxf(fabsf(hv[2 * G][0]), fabsf(hv[2 * G][1])),
                                     fmaxf(fabsf(hv[2 * G + 1][0]), fabsf(hv[2 * G + 1][1])));
                    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 1));
                    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 2));
                    const BlockScale bs = compute_block_scale(am, qg);
                    const float mu = bs.mul;
                    const uint32_t blo = cvt_e2m1x2(hv[2 * G][0] * mu, hv[2 * G][1] * mu);
                    const uint32_t bhi = cvt_e2m1x2(hv[2 * G + 1][0] * mu, hv[2 * G + 1][1] * mu);
                    const uint32_t v16 = blo | (bhi << 8);
                    const uint32_t v1 = __shfl_down_sync(0xffffffffu, v16, 1);
                    const uint32_t v2 = __shfl_down_sync(0xffffffffu, v16, 2);
                    const uint32_t v3 = __shfl_down_sync(0xffffffffu, v16, 3);
                    if (ok && t == 0) {
                        const uint32_t lo = (v16 & 0xFF) | ((v1 & 0xFF) << 8) | ((v2 & 0xFF) << 16) | ((v3 & 0xFF) << 24);
                        const uint32_t hi = (v16 >> 8) | ((v1 >> 8) << 8) | ((v2 >> 8) << 16) | ((v3 >> 8) << 24);
                        *reinterpret_cast<uint2*>(A.oq + (size_t)pr * np + cb * 16 + G * 8) = make_uint2(lo, hi);
                        A.os[(size_t)pr * ns + cb * 2 + G] = bs.sf8;
                    }
                }
            }
        }
    } else {
        const float gs = __ldg(A.scale2 + e), ein = __ldg(A.in_scale + e);
#pragma unroll
        for (int r = 0; r < RT; ++r) {
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                const int rr = r0 + r * 16 + gq + 8 * hh;
                if (rr >= cnt) continue;
                const int pr = __ldg(A.plan.members + first + rr);
                const float alpha = gs * (A.a_s2_row ? __ldg(A.a_s2_row + pr) : ein);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float o0 = acc[0][r][j][2 * hh] * alpha, o1 = acc[0][r][j][2 * hh + 1] * alpha;
                    const size_t off = (size_t)pr * A.N + col0 + 8 * j;
                    if constexpr (MODE == 1)
                        *reinterpret_cast<float2*>(reinterpret_cast<float*>(A.out) + off) = make_float2(o0, o1);
                    else
                        *reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<__nv_bfloat16*>(A.out) + off) =
                            __floats2bfloat162_rn(o0, o1);
                }
            }
        }
    }
}

template <int M, int MODE, int WARPS, int S, int GATE_CAP_I, int UP_CAP_I>
__device__ __forceinline__ void tile_body(char* smem, const Args& A) {
    const int tid = threadIdx.x, lane = tid & 31, wp = tid >> 5;
    const int NB = A.N >> 5;
    const int groups = (NB + WARPS - 1) / WARPS;
    const int units = __ldg(A.plan.counts) * groups;
    for (int unit = blockIdx.x; unit < units; unit += gridDim.x) {
        const int it = unit / groups, cg = unit - it * groups;
        const int e = __ldg(A.plan.items + 3 * it), first = __ldg(A.plan.items + 3 * it + 1),
                  cnt = __ldg(A.plan.items + 3 * it + 2);
        if (e == A.plan.skip) continue;
        const int cb = cg * WARPS + wp;
        const bool active = cb < NB;
        for (int r0 = 0; r0 < cnt;) {
            const int rt = (cnt - r0 + 15) >> 4;
            if (rt >= 4) {
                chunk<M, MODE, WARPS, 4, S, GATE_CAP_I, UP_CAP_I>(smem, A, e, first, cnt, r0, cb, active, tid, lane, wp);
                r0 += 64;
            } else if (rt == 3) {
                chunk<M, MODE, WARPS, 3, S, GATE_CAP_I, UP_CAP_I>(smem, A, e, first, cnt, r0, cb, active, tid, lane, wp);
                r0 += 48;
            } else if (rt == 2) {
                chunk<M, MODE, WARPS, 2, S, GATE_CAP_I, UP_CAP_I>(smem, A, e, first, cnt, r0, cb, active, tid, lane, wp);
                r0 += 32;
            } else {
                chunk<M, MODE, WARPS, 1, S, GATE_CAP_I, UP_CAP_I>(smem, A, e, first, cnt, r0, cb, active, tid, lane, wp);
                r0 += 16;
            }
        }
    }
}

// ---- BF16 -> MXFP8 activation quantizer (e4m3 + per-32 ue8m0, kin order) ----
__device__ __forceinline__ uint16_t cvt_f32x2_to_e4m3x2(float f_lo, float f_hi) {
    uint16_t r;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(r) : "f"(f_hi), "f"(f_lo));
    return r;
}
__device__ __forceinline__ uint32_t cvt_f32x4_to_e4m3x4(float f0, float f1, float f2, float f3) {
    uint32_t lo = cvt_f32x2_to_e4m3x2(f0, f1);
    uint32_t hi = cvt_f32x2_to_e4m3x2(f2, f3);
    return lo | (hi << 16);
}
__device__ __forceinline__ void compute_mxfp8_scale(float amax, uint8_t& e_u8, float& inv_scale) {
    if (amax == 0.0f) {
        e_u8 = 127;
        inv_scale = 1.0f;
        return;
    }
    uint32_t u = __float_as_uint(amax);
    int exp_a = static_cast<int>((u >> 23) & 0xFFu);
    uint32_t mant_a = u & 0x7FFFFFu;
    int e = (mant_a <= 0x600000u) ? (exp_a - 8) : (exp_a - 7);
    if (e < 0) e = 0;
    if (e > 254) e = 254;
    e_u8 = static_cast<uint8_t>(e);
    uint32_t inv_bits = (e == 254) ? 0x00400000u : (static_cast<uint32_t>(254 - e) << 23);
    inv_scale = __uint_as_float(inv_bits);
}
__device__ __forceinline__ void quantize_bf16_to_mxfp8_kin_kernel(
    const __nv_bfloat16* __restrict__ x, int M, int K, int ldx,
    uint8_t* __restrict__ x_fp8, uint8_t* __restrict__ x_sf, int m_pad) {
    const int k_blocks = K >> 5;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int blk_id = tid >> 2;
    const int t = tid & 3;
    const int total_blks = m_pad * k_blocks;
    if (blk_id >= total_blks) return;
    const int row = blk_id / k_blocks;
    const int kb = blk_id - row * k_blocks;
    const size_t sf_idx = static_cast<size_t>(kb >> 1) * (static_cast<size_t>(m_pad) * 2)
                        + static_cast<size_t>(row) * 2 + static_cast<size_t>(kb & 1);
    if (row >= M) {
        if (t == 0) x_sf[sf_idx] = 0;
        return;
    }
    const __nv_bfloat16* row_ptr = x + static_cast<size_t>(row) * ldx + (kb << 5) + (t << 3);
    const uint4 raw = *reinterpret_cast<const uint4*>(row_ptr);
    const __nv_bfloat162* p2 = reinterpret_cast<const __nv_bfloat162*>(&raw);
    float f[8];
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float2 v = __bfloat1622float2(p2[i]);
        f[2 * i]     = v.x;
        f[2 * i + 1] = v.y;
        amax = fmaxf(amax, fmaxf(fabsf(v.x), fabsf(v.y)));
    }
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    uint8_t e_u8;
    float inv_scale;
    compute_mxfp8_scale(amax, e_u8, inv_scale);
    if (t == 0) x_sf[sf_idx] = e_u8;
    const uint32_t lo = cvt_f32x4_to_e4m3x4(f[0]*inv_scale, f[1]*inv_scale, f[2]*inv_scale, f[3]*inv_scale);
    const uint32_t hi = cvt_f32x4_to_e4m3x4(f[4]*inv_scale, f[5]*inv_scale, f[6]*inv_scale, f[7]*inv_scale);
    const uint32_t peer_lo = __shfl_xor_sync(0xffffffffu, lo, 1);
    const uint32_t peer_hi = __shfl_xor_sync(0xffffffffu, hi, 1);
    uint32_t out0, out1;
    if ((t & 1) == 0) {
        out0 = __byte_perm(lo, peer_lo, 0x5410u);
        out1 = __byte_perm(lo, peer_lo, 0x7632u);
    } else {
        out0 = __byte_perm(peer_hi, hi, 0x5410u);
        out1 = __byte_perm(peer_hi, hi, 0x7632u);
    }
    uint2* dst = reinterpret_cast<uint2*>(x_fp8 + static_cast<size_t>(row) * K + (kb << 5) + (t << 3));
    *dst = make_uint2(out0, out1);
}

// sm_121a QMMA.SF.16832: D += (A * sfa[BYTE_ID]) @ (B * sfb[BYTE_ID])^T (e4m3 x e4m3, ue8m0 block-32 scales).
template <int BYTE_ID>
__device__ __forceinline__ void qmma_sf_16832(
    float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, uint32_t sfa, uint32_t sfb) {
    static_assert(BYTE_ID == 0 || BYTE_ID == 1, "BYTE_ID must be 0 or 1");
    if constexpr (BYTE_ID == 0) {
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, %10, {0, 0}, %11, {0, 0};\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa), "r"(sfb));
    } else {
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, %10, {1, 0}, %11, {1, 0};\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa), "r"(sfb));
    }
}

template <int BM, int BN, int BK, int WM, int WN, int STAGES>
struct QmmaTile {
    static_assert(BK == 64 || BK == 128, "BK must be 64 or 128");
    static_assert(BN % 64 == 0, "BN must be a multiple of 64");
    static constexpr int THREADS = WM * WN * 32;
    static constexpr int MT = BM / WM / 16;
    static constexpr int NT = BN / WN / 8;
    static constexpr int K64_PER_STAGE = BK / 64;
    static constexpr int ROW = BK;
    static constexpr int CHUNKS = ROW / 16;
    static constexpr int X = BM * ROW;
    static constexpr int W = BN * BK;
    static constexpr int SX = K64_PER_STAGE * BM * 2;
    static constexpr int SW = K64_PER_STAGE * BN * 2;
    static constexpr int STAGE = (X + W + SX + SW + 127) / 128 * 128;
    static constexpr int SMEM = STAGES * STAGE;
};

template <int BM, int BN, int BK, int WM, int WN, int STAGES, bool F32>
__device__ __forceinline__ void mxfp8_qmma_kernel(
    const uint8_t* __restrict__ x_fp8, const uint8_t* __restrict__ x_sf, int m_pad,
    const uint8_t* __restrict__ w8, const uint8_t* __restrict__ bs, float scale,
    void* __restrict__ out, int M, int N, int K, int npad, int group) {
    using T = QmmaTile<BM, BN, BK, WM, WN, STAGES>;
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WN, wn = warp % WN;
    const int KG64 = K / 64;
    const int KSTEPS = K / BK;
    const int2 at = tile_of(blockIdx.x, M, N, BM, BN, group);
    const int m0 = at.x, n0 = at.y;
    auto stage = [&](int s) { return buf + s * T::STAGE; };
    auto load = [&](int s, int step) {
        unsigned char* p = stage(s);
        for (int c = tid; c < BM * T::CHUNKS; c += T::THREADS) {
            const int r = c / T::CHUNKS, ch = c % T::CHUNKS;
            const int row = m0 + r;
            cp16z(p + r * T::ROW + swz_x8<T::CHUNKS>(r, ch) * 16,
                  x_fp8 + static_cast<size_t>(min(row, M - 1)) * K + step * BK + ch * 16, row < M);
        }
        unsigned char* pw = p + T::X;
        constexpr int NT64 = BN / 64;
        constexpr int W_TILE_CHUNKS = 4096 / 16;
        constexpr int W_TOTAL_CHUNKS = T::K64_PER_STAGE * NT64 * W_TILE_CHUNKS;
        const int npad64 = npad / 64;
        for (int c = tid; c < W_TOTAL_CHUNKS; c += T::THREADS) {
            const int ksub = (T::K64_PER_STAGE == 1) ? 0 : (c / (NT64 * W_TILE_CHUNKS));
            const int rem  = (T::K64_PER_STAGE == 1) ? c : (c % (NT64 * W_TILE_CHUNKS));
            const int t    = rem / W_TILE_CHUNKS;
            const int off  = rem % W_TILE_CHUNKS;
            const int g64  = step * T::K64_PER_STAGE + ksub;
            const int tile_n = n0 / 64 + t;
            const bool valid = tile_n < npad64;
            cp16z(pw + c * 16,
                  w8 + (static_cast<size_t>(min(tile_n, npad64 - 1)) * KG64 + g64) * 4096 + off * 16, valid);
        }
        unsigned char* psx = pw + T::W;
        constexpr int SX_PER_K64 = (BM * 2) / 16;
        constexpr int SX_TOTAL = T::K64_PER_STAGE * SX_PER_K64;
        for (int c = tid; c < SX_TOTAL; c += T::THREADS) {
            const int ksub = (T::K64_PER_STAGE == 1) ? 0 : (c / SX_PER_K64);
            const int r8   = (T::K64_PER_STAGE == 1) ? c : (c % SX_PER_K64);
            const int g64  = step * T::K64_PER_STAGE + ksub;
            const int row0 = m0 + r8 * 8;
            cp16z(psx + c * 16,
                  x_sf + static_cast<size_t>(g64) * (static_cast<size_t>(m_pad) * 2) + static_cast<size_t>(min(row0, m_pad - 8)) * 2,
                  row0 < m_pad);
        }
        unsigned char* psw = psx + T::SX;
        constexpr int SW_TOTAL = T::K64_PER_STAGE * NT64 * 8;
        for (int c = tid; c < SW_TOTAL; c += T::THREADS) {
            const int ksub = (T::K64_PER_STAGE == 1) ? 0 : (c / (NT64 * 8));
            const int rem  = (T::K64_PER_STAGE == 1) ? c : (c % (NT64 * 8));
            const int t    = rem / 8;
            const int off  = rem % 8;
            const int g64  = step * T::K64_PER_STAGE + ksub;
            const int tile_n = n0 / 64 + t;
            const bool valid = tile_n < npad64;
            cp16z(psw + c * 16,
                  bs + (static_cast<size_t>(min(tile_n, npad64 - 1)) * KG64 + g64) * 128 + off * 16, valid);
        }
    };
    float acc[T::MT][T::NT][4];
#pragma unroll
    for (int i = 0; i < T::MT; ++i)
#pragma unroll
        for (int j = 0; j < T::NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < KSTEPS) load(s, s);
        commit();
    }
    for (int step = 0; step < KSTEPS; ++step) {
        wait<STAGES - 2>();
        __syncthreads();
        if (step + STAGES - 1 < KSTEPS) load((step + STAGES - 1) % STAGES, step + STAGES - 1);
        commit();
        const unsigned char* p   = stage(step % STAGES);
        const unsigned char* pw0 = p + T::X;
        const unsigned char* psx0 = pw0 + T::W;
        const unsigned char* psw0 = psx0 + T::SX;
#pragma unroll
        for (int ksub = 0; ksub < T::K64_PER_STAGE; ++ksub) {
            const unsigned char* pw  = pw0  + ksub * (BN * 64);
            const unsigned char* psx = psx0 + ksub * (BM * 2);
            const unsigned char* psw = psw0 + ksub * (BN * 2);
            uint4 wq[T::NT];
            uint32_t sfb[T::NT];
#pragma unroll
            for (int j = 0; j < T::NT; ++j) {
                const int jj  = wn * T::NT + j;
                const int col = (jj & 7) * 8 + (lane >> 2);
                wq[j]  = reinterpret_cast<const uint4*>(pw)[jj * 32 + lane];
                sfb[j] = reinterpret_cast<const uint16_t*>(psw + (jj >> 3) * 128)[col];
            }
            uint32_t sfa[T::MT];
#pragma unroll
            for (int i = 0; i < T::MT; ++i) {
                const int r_sf = wm * (BM / WM) + i * 16 + (lane >> 2) + (lane & 1) * 8;
                sfa[i] = reinterpret_cast<const uint16_t*>(psx)[r_sf];
            }
            {
                uint32_t a[T::MT][4];
#pragma unroll
                for (int i = 0; i < T::MT; ++i) {
                    const int r  = wm * (BM / WM) + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                    const int ch = ksub * 4 + (lane >> 4);
                    ldsm4(a[i], p + r * T::ROW + swz_x8<T::CHUNKS>(r, ch) * 16);
                }
#pragma unroll
                for (int j = 0; j < T::NT; ++j)
#pragma unroll
                    for (int i = 0; i < T::MT; ++i)
                        qmma_sf_16832<0>(acc[i][j], a[i], wq[j].x, wq[j].y, sfa[i], sfb[j]);
            }
            {
                uint32_t a[T::MT][4];
#pragma unroll
                for (int i = 0; i < T::MT; ++i) {
                    const int r  = wm * (BM / WM) + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                    const int ch = ksub * 4 + 2 + (lane >> 4);
                    ldsm4(a[i], p + r * T::ROW + swz_x8<T::CHUNKS>(r, ch) * 16);
                }
#pragma unroll
                for (int j = 0; j < T::NT; ++j)
#pragma unroll
                    for (int i = 0; i < T::MT; ++i)
                        qmma_sf_16832<1>(acc[i][j], a[i], wq[j].z, wq[j].w, sfa[i], sfb[j]);
            }
        }
    }
    wait<0>();
#pragma unroll
    for (int i = 0; i < T::MT; ++i)
#pragma unroll
        for (int j = 0; j < T::NT; ++j) {
            const int col = n0 + wn * (BN / WN) + j * 8 + (lane & 3) * 2;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + wm * (BM / WM) + i * 16 + (lane >> 2) + h * 8;
                if (row >= M) continue;
                const float v0 = acc[i][j][2 * h] * scale, v1 = acc[i][j][2 * h + 1] * scale;
                if constexpr (F32) {
                    float* dst = reinterpret_cast<float*>(out) + static_cast<size_t>(row) * N + col;
                    if (col < N) dst[0] = v0;
                    if (col + 1 < N) dst[1] = v1;
                } else {
                    __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(out) + static_cast<size_t>(row) * N + col;
                    if (col + 1 < N && (N & 1) == 0) {
                        *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(v0, v1);
                    } else {
                        if (col < N) dst[0] = __float2bfloat16_rn(v0);
                        if (col + 1 < N) dst[1] = __float2bfloat16_rn(v1);
                    }
                }
            }
        }
}

} // namespace ink2sm

// ---- extern "C" entries ----
#define INK2_PLAN_ARGS const int* items, const int* counts, const int* members, int skip
#define INK2_PLAN_SET a.plan.items = items; a.plan.counts = counts; a.plan.members = members; a.plan.skip = skip;

extern "C" __global__ void __launch_bounds__(64) tf_ink2_nvfp4_gu(
    const uint8_t* xq, const uint8_t* xs, const float* x_s2_row, float x_s2, int slots,
    const uint4* W, const float* scale2, const float* in_scale,
    INK2_PLAN_ARGS, int K, int N, uint8_t* oq, uint8_t* os, __nv_bfloat16* h, float* gu_dbg) {
    extern __shared__ __align__(128) char smem[];
    ink2sm::Args a{};
    a.aq = xq; a.as = xs; a.slots = slots; a.W = W; a.scale2 = scale2; a.in_scale = in_scale; a.x_s2 = x_s2;
    a.x_s2_row = x_s2_row; a.K = K; a.N = N; INK2_PLAN_SET a.oq = oq; a.os = os; a.h_dbg = h; a.gu_dbg = gu_dbg;
    ink2sm::tile_body<2, 0, 2, 4, 4, 25>(smem, a);
}
// The same gate|up with 2 stages (13824 B, 4 CTAs an SM instead of 3; 244 regs, no spill): the bits of
// tf_ink2_nvfp4_gu (stages only reorder loads), 10-24% faster at 1024-2048 rows, level below (M9.6 A4 sweep).
extern "C" __global__ void __launch_bounds__(64) tf_ink2_nvfp4_gu_s2(
    const uint8_t* xq, const uint8_t* xs, const float* x_s2_row, float x_s2, int slots,
    const uint4* W, const float* scale2, const float* in_scale,
    INK2_PLAN_ARGS, int K, int N, uint8_t* oq, uint8_t* os, __nv_bfloat16* h, float* gu_dbg) {
    extern __shared__ __align__(128) char smem[];
    ink2sm::Args a{};
    a.aq = xq; a.as = xs; a.slots = slots; a.W = W; a.scale2 = scale2; a.in_scale = in_scale; a.x_s2 = x_s2;
    a.x_s2_row = x_s2_row; a.K = K; a.N = N; INK2_PLAN_SET a.oq = oq; a.os = os; a.h_dbg = h; a.gu_dbg = gu_dbg;
    ink2sm::tile_body<2, 0, 2, 2, 4, 25>(smem, a);
}
#define INK2_OMMA_DOWN(NAME, MODE)                                                                     \
    extern "C" __global__ void __launch_bounds__(64) NAME(                                              \
        const uint8_t* aq, const uint8_t* as, const uint4* W, const float* scale2,                       \
        const float* in_scale, const float* a_s2_row, INK2_PLAN_ARGS, int K, int N, void* out) {        \
        extern __shared__ __align__(128) char smem[];                                                   \
        ink2sm::Args a{};                                                                               \
        a.aq = aq; a.as = as; a.slots = 0; a.W = W; a.scale2 = scale2; a.in_scale = in_scale;            \
        a.a_s2_row = a_s2_row; a.K = K; a.N = N; INK2_PLAN_SET a.out = out;                             \
        ink2sm::tile_body<1, MODE, 2, 4, 4, 25>(smem, a);                                         \
    }
INK2_OMMA_DOWN(tf_ink2_nvfp4_dn_f32, 1)
INK2_OMMA_DOWN(tf_ink2_nvfp4_dn_b16, 2)

extern "C" __global__ void __launch_bounds__(160) tf_ink2_nvfp4_quant_2304(
    const __nv_bfloat16* X, int ldx, int M, uint8_t* codes, uint8_t* scales, float* s2) {
    ink2sm::quant_dynamic_row_body<144>(X, ldx, M, 0.0f, codes, scales, s2);
}
extern "C" __global__ void __launch_bounds__(160) tf_ink2_nvfp4_quant_1152(
    const __nv_bfloat16* X, int ldx, int M, uint8_t* codes, uint8_t* scales, float* s2) {
    // 72 groups: threads 72..159 idle; grid is still one 160-thread block per row.
    ink2sm::quant_dynamic_row_body<72>(X, ldx, M, 0.0f, codes, scales, s2);
}
extern "C" __global__ void __launch_bounds__(256) tf_ink2_nvfp4_quant_static(
    const __nv_bfloat16* X, int ldx, int M, int K, float in_scale, uint8_t* codes, uint8_t* scales) {
    ink2sm::quant_static_body(X, ldx, M, K, in_scale, nullptr, codes, scales, nullptr);
}

extern "C" __global__ void __launch_bounds__(256) tf_ink2_mxfp8_quantize(
    const __nv_bfloat16* __restrict__ x, int M, int K, int ldx, uint8_t* __restrict__ x_fp8,
    uint8_t* __restrict__ x_sf, int m_pad) {
    ink2sm::quantize_bf16_to_mxfp8_kin_kernel(x, M, K, ldx, x_fp8, x_sf, m_pad);
}
#define INK2_MXFP8_GEMM(NAME, BM, BN, BK, WM, WN, ST, F32)                                              \
    extern "C" __global__ void __launch_bounds__(WM * WN * 32) NAME(                                   \
        const uint8_t* __restrict__ x_fp8, const uint8_t* __restrict__ x_sf, int m_pad,                 \
        const uint8_t* __restrict__ w8, const uint8_t* __restrict__ bs, float scale,                     \
        void* __restrict__ out, int M, int N, int K, int npad, int group) {                             \
        ink2sm::mxfp8_qmma_kernel<BM, BN, BK, WM, WN, ST, F32>(x_fp8, x_sf, m_pad, w8, bs, scale, out,   \
                                                              M, N, K, npad, group);                   \
    }
INK2_MXFP8_GEMM(tf_ink2_mxfp8_gemm_128x128x64_bf16, 128, 128, 64, 2, 4, 4, false)
INK2_MXFP8_GEMM(tf_ink2_mxfp8_gemm_128x128x64_f32, 128, 128, 64, 2, 4, 4, true)
INK2_MXFP8_GEMM(tf_ink2_mxfp8_gemm_128x128x128_bf16, 128, 128, 128, 2, 4, 2, false)
INK2_MXFP8_GEMM(tf_ink2_mxfp8_gemm_128x128x128_f32, 128, 128, 128, 2, 4, 2, true)
INK2_MXFP8_GEMM(tf_ink2_mxfp8_gemm_64x128x64_bf16, 64, 128, 64, 2, 2, 4, false)
INK2_MXFP8_GEMM(tf_ink2_mxfp8_gemm_64x128x64_f32, 64, 128, 64, 2, 2, 4, true)
INK2_MXFP8_GEMM(tf_ink2_mxfp8_gemm_64x64x64_bf16, 64, 64, 64, 2, 2, 4, false)
INK2_MXFP8_GEMM(tf_ink2_mxfp8_gemm_64x64x64_f32, 64, 64, 64, 2, 2, 4, true)

// ---- Grouped-expert SM121a MXFP8 QMMA for Mini layers 20-23 (E=128, H=2304, I=1152) ----
// Persistent-grid grouped expert QMMA reusing ink2sm's swizzled shared tiles, ldsm4 fragments and
// qmma_sf_16832: BM=16, BN=64, BK=64, 1x4 warps (MT=1, NT=2). Members scatter rows via plan members;
// gate+up share one activation tile (weights streamed gate-then-up through one W buffer).
namespace ink2mxq {
using ink2sm::cp16z;
using ink2sm::commit;
using ink2sm::wait;
using ink2sm::ldsm4;
using ink2sm::swz_x8;
using ink2sm::qmma_sf_16832;
using ink2sm::ink_act;

constexpr int BM = 16, BN = 64, BK = 64, THREADS = 128, NT = 2, STAGES = 2;  // 1x4 warps: wn = tid>>5
constexpr int ROW = 64, CHUNKS = 4;
constexpr int XX = BM * ROW;   // 1024: one 16x64 activation tile
constexpr int WW = BN * BK;   // 4096: one 64x64 weight tile
constexpr int SXB = BM * 2;   // 32: one u16 activation scale a row
constexpr int SWB = BN * 2;   // 128: one u16 weight scale a column
constexpr int STAGE_GU = (XX + 2 * WW + SXB + 2 * SWB + 127) / 128 * 128;  // 9600
constexpr int STAGE_DN = (XX + WW + SXB + SWB + 127) / 128 * 128;          // 5376

__device__ __forceinline__ void stage_x(unsigned char* p, const uint8_t* x_fp8, int K, int step,
                                         const int* members, int first, int r0, int cnt, int slots,
                                         int tid) {
    for (int c = tid; c < BM * CHUNKS; c += THREADS) {
        const int r = c / CHUNKS, ch = c % CHUNKS;
        const int lr = r0 + r;
        const bool ok = lr < cnt;
        const int src = ok ? __ldg(members + first + lr) / slots : 0;
        cp16z(p + r * ROW + swz_x8<CHUNKS>(r, ch) * 16,
              x_fp8 + static_cast<size_t>(src) * K + step * BK + ch * 16, ok);
    }
}

__device__ __forceinline__ void stage_x_pairs(unsigned char* p, const uint8_t* x_fp8, int K, int step,
                                               const int* members, int first, int r0, int cnt,
                                               int tid) {
    for (int c = tid; c < BM * CHUNKS; c += THREADS) {
        const int r = c / CHUNKS, ch = c % CHUNKS;
        const int lr = r0 + r;
        const bool ok = lr < cnt;
        const int src = ok ? __ldg(members + first + lr) : 0;
        cp16z(p + r * ROW + swz_x8<CHUNKS>(r, ch) * 16,
              x_fp8 + static_cast<size_t>(src) * K + step * BK + ch * 16, ok);
    }
}

__device__ __forceinline__ void load_sx(unsigned char* psx, const uint8_t* x_sf, int m_pad, int g64,
                                          const int* srows, int tid) {
    if (tid < BM)
        *reinterpret_cast<uint16_t*>(psx + tid * 2) =
            *reinterpret_cast<const uint16_t*>(x_sf + static_cast<size_t>(g64) * (m_pad * 2) + srows[tid] * 2);
}

__device__ __forceinline__ void load_w(unsigned char* pw, unsigned char* psw, const uint8_t* w8e,
                                         const uint8_t* bse, int tile_n, int g64, int KG64, int tid) {
    for (int c = tid; c < 256; c += THREADS)
        cp16z(pw + c * 16, w8e + (static_cast<size_t>(tile_n) * KG64 + g64) * 4096 + c * 16, true);
    for (int c = tid; c < 8; c += THREADS)
        cp16z(psw + c * 16, bse + (static_cast<size_t>(tile_n) * KG64 + g64) * 128 + c * 16, true);
}

__device__ __forceinline__ void mma_block(const unsigned char* p, const unsigned char* pw,
                                           const unsigned char* psx, const unsigned char* psw,
                                           float (&acc)[NT][4], int lane, int wn) {
    uint4 wq[NT];
    uint32_t sfb[NT];
    for (int j = 0; j < NT; ++j) {
        const int jj = wn * NT + j;
        const int col = (jj & 7) * 8 + (lane >> 2);
        wq[j] = reinterpret_cast<const uint4*>(pw)[jj * 32 + lane];
        sfb[j] = reinterpret_cast<const uint16_t*>(psw + (jj >> 3) * 128)[col];
    }
    const int r_sf = (lane >> 2) + (lane & 1) * 8;
    const uint32_t sfa = reinterpret_cast<const uint16_t*>(psx)[r_sf];
    uint32_t a[4];
    {
        const int r = (lane & 7) + ((lane >> 3) & 1) * 8, ch = (lane >> 4);
        ldsm4(a, p + r * ROW + swz_x8<CHUNKS>(r, ch) * 16);
        for (int j = 0; j < NT; ++j) qmma_sf_16832<0>(acc[j], a, wq[j].x, wq[j].y, sfa, sfb[j]);
    }
    {
        const int r = (lane & 7) + ((lane >> 3) & 1) * 8, ch = 2 + (lane >> 4);
        ldsm4(a, p + r * ROW + swz_x8<CHUNKS>(r, ch) * 16);
        for (int j = 0; j < NT; ++j) qmma_sf_16832<1>(acc[j], a, wq[j].z, wq[j].w, sfa, sfb[j]);
    }
}

// One (expert item, n-tile): K-loop over gate then up through one W buffer, fused ink_act to bf16 h_out.
__device__ __forceinline__ void gu_block(unsigned char* buf, const uint8_t* x_fp8, const uint8_t* x_sf,
                                          int m_pad, const uint8_t* w8_g, const uint8_t* bs_g,
                                          const uint8_t* w8_u, const uint8_t* bs_u, float scale,
                                          __nv_bfloat16* h_out, const int* members, int e, int first,
                                          int cnt, int slots, int K, int N, int tn, int tid, int lane,
                                          int wn) {
    const int KG64 = K >> 6;
    const size_t e_w = static_cast<size_t>(N >> 6) * KG64 * 4096;
    const size_t e_s = static_cast<size_t>(N >> 6) * KG64 * 128;
    const int n0 = tn * BN;
    const int KSTEPS = K >> 6;
    __shared__ int srows[BM];
    for (int r0 = 0; r0 < cnt; r0 += BM) {
        const int rr = min(BM, cnt - r0);
        __syncthreads();
        if (tid < BM) {
            const int lr = r0 + tid;
            srows[tid] = (lr < cnt) ? __ldg(members + first + lr) / slots : 0;
        }
        __syncthreads();
        float acc_g[NT][4] = {}, acc_u[NT][4] = {};
        auto stage = [&](int s) { return buf + s * STAGE_GU; };
        auto load_stage = [&](int s, int step) {
            unsigned char* p = stage(s);
            unsigned char* pw_g = p + XX;
            unsigned char* pw_u = pw_g + WW;
            unsigned char* psx = pw_u + WW;
            unsigned char* psw_g = psx + SXB;
            unsigned char* psw_u = psw_g + SWB;
            stage_x(p, x_fp8, K, step, members, first, r0, cnt, slots, tid);
            load_w(pw_g, psw_g, w8_g + e * e_w, bs_g + e * e_s, tn, step, KG64, tid);
            load_w(pw_u, psw_u, w8_u + e * e_w, bs_u + e * e_s, tn, step, KG64, tid);
            load_sx(psx, x_sf, m_pad, step, srows, tid);
        };
        load_stage(0, 0);
        commit();
        for (int step = 0; step < KSTEPS; ++step) {
            wait<0>();
            __syncthreads();
            if (step + 1 < KSTEPS) {
                load_stage((step + 1) & 1, step + 1);
                commit();
            }
            const unsigned char* p = stage(step & 1);
            const unsigned char* pw_g = p + XX;
            const unsigned char* pw_u = pw_g + WW;
            const unsigned char* psx = pw_u + WW;
            const unsigned char* psw_g = psx + SXB;
            const unsigned char* psw_u = psw_g + SWB;
            mma_block(p, pw_g, psx, psw_g, acc_g, lane, wn);
            mma_block(p, pw_u, psx, psw_u, acc_u, lane, wn);
        }
        wait<0>();
        __syncthreads();
        for (int j = 0; j < NT; ++j)
            for (int h = 0; h < 2; ++h) {
                const int lr = (lane >> 2) + h * 8;
                if (lr >= rr) continue;
                const int pr = __ldg(members + first + r0 + lr);
                const int col = n0 + wn * 16 + j * 8 + (lane & 3) * 2;
                const float gv0 = acc_g[j][2 * h] * scale, gv1 = acc_g[j][2 * h + 1] * scale;
                const float uv0 = acc_u[j][2 * h] * scale, uv1 = acc_u[j][2 * h + 1] * scale;
                __nv_bfloat16* dst = h_out + static_cast<size_t>(pr) * N + col;
                if (col + 1 < N)
                    *reinterpret_cast<__nv_bfloat162*>(dst) =
                        __floats2bfloat162_rn(ink_act<4, 25>(gv0, uv0), ink_act<4, 25>(gv1, uv1));
            }
    }
}

// One (expert item, n-tile) down projection to bf16 down_out.
__device__ __forceinline__ void dn_block(unsigned char* buf, const uint8_t* x_fp8, const uint8_t* x_sf,
                                          int m_pad, const uint8_t* w8_d, const uint8_t* bs_d, float scale,
                                          __nv_bfloat16* down_out, const int* members, int e, int first,
                                          int cnt, int K, int N, int tn, int tid, int lane, int wn) {
    const int KG64 = K >> 6;
    const size_t e_w = static_cast<size_t>(N >> 6) * KG64 * 4096;
    const size_t e_s = static_cast<size_t>(N >> 6) * KG64 * 128;
    const int n0 = tn * BN;
    const int KSTEPS = K >> 6;
    __shared__ int srows[BM];
    for (int r0 = 0; r0 < cnt; r0 += BM) {
        const int rr = min(BM, cnt - r0);
        __syncthreads();
        if (tid < BM) {
            const int lr = r0 + tid;
            srows[tid] = (lr < cnt) ? __ldg(members + first + lr) : 0;
        }
        __syncthreads();
        float acc[NT][4] = {};
        auto stage = [&](int s) { return buf + s * STAGE_DN; };
        auto load_stage = [&](int s, int step) {
            unsigned char* p = stage(s);
            unsigned char* pw = p + XX;
            unsigned char* psx = pw + WW;
            unsigned char* psw = psx + SXB;
            stage_x_pairs(p, x_fp8, K, step, members, first, r0, cnt, tid);
            load_w(pw, psw, w8_d + e * e_w, bs_d + e * e_s, tn, step, KG64, tid);
            load_sx(psx, x_sf, m_pad, step, srows, tid);
        };
        load_stage(0, 0);
        commit();
        for (int step = 0; step < KSTEPS; ++step) {
            wait<0>();
            __syncthreads();
            if (step + 1 < KSTEPS) {
                load_stage((step + 1) & 1, step + 1);
                commit();
            }
            const unsigned char* p = stage(step & 1);
            const unsigned char* pw = p + XX;
            const unsigned char* psx = pw + WW;
            const unsigned char* psw = psx + SXB;
            mma_block(p, pw, psx, psw, acc, lane, wn);
        }
        wait<0>();
        __syncthreads();
        for (int j = 0; j < NT; ++j)
            for (int h = 0; h < 2; ++h) {
                const int lr = (lane >> 2) + h * 8;
                if (lr >= rr) continue;
                const int pr = __ldg(members + first + r0 + lr);
                const int col = n0 + wn * 16 + j * 8 + (lane & 3) * 2;
                const float v0 = acc[j][2 * h] * scale, v1 = acc[j][2 * h + 1] * scale;
                __nv_bfloat16* dst = down_out + static_cast<size_t>(pr) * N + col;
                if (col + 1 < N)
                    *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(v0, v1);
            }
    }
}

} // namespace ink2mxq

// Persistent-grid grouped-expert MXFP8 gate|up with fused capped SwiGLU to bf16 h_out[pairs, N].
extern "C" __global__ void __launch_bounds__(128) tf_ink2_mxfp8_moe_gu(
    const uint8_t* x_fp8, const uint8_t* x_sf, int m_pad, const uint8_t* w8_g, const uint8_t* bs_g,
    const uint8_t* w8_u, const uint8_t* bs_u, float scale, __nv_bfloat16* h_out, int slots, int K, int N,
    const int* items, const int* counts, const int* members, int skip) {
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, wn = tid >> 5;
    const int ntiles = N >> 6;
    const int units = __ldg(counts) * ntiles;
    for (int unit = blockIdx.x; unit < units; unit += gridDim.x) {
        const int it = unit / ntiles, tn = unit - it * ntiles;
        const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1);
        const int cnt = __ldg(items + 3 * it + 2);
        if (e == skip) continue;
        ink2mxq::gu_block(buf, x_fp8, x_sf, m_pad, w8_g, bs_g, w8_u, bs_u, scale, h_out, members, e, first,
                           cnt, slots, K, N, tn, tid, lane, wn);
    }
}

// Persistent-grid grouped-expert MXFP8 down projection to bf16 down_out[pairs, N].
extern "C" __global__ void __launch_bounds__(128) tf_ink2_mxfp8_moe_dn_b16(
    const uint8_t* x_fp8, const uint8_t* x_sf, int m_pad, const uint8_t* w8_d, const uint8_t* bs_d,
    float scale, __nv_bfloat16* down_out, int K, int N, const int* items, const int* counts,
    const int* members, int skip) {
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, wn = tid >> 5;
    const int ntiles = N >> 6;
    const int units = __ldg(counts) * ntiles;
    for (int unit = blockIdx.x; unit < units; unit += gridDim.x) {
        const int it = unit / ntiles, tn = unit - it * ntiles;
        const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1);
        const int cnt = __ldg(items + 3 * it + 2);
        if (e == skip) continue;
        ink2mxq::dn_block(buf, x_fp8, x_sf, m_pad, w8_d, bs_d, scale, down_out, members, e, first, cnt,
                           K, N, tn, tid, lane, wn);
    }
}

// ---- M8e: grouped-expert SM121a MXFP8 QMMA with BM=64 (prefill rows) ----
// Same persistent-grid structure as ink2mxq (reused helpers), but BM=64 rows a block over 8 warps
// (wm = warp >> 1 in 0..3, wn = warp & 1 in 0..1; each warp does MT=1, NT=4 m16n8 tiles).
// Bit-identical numerics to the BM=16 path (same fragment order, same summation order per row).
namespace ink2mxq64 {
using ink2sm::cp16z;
using ink2sm::commit;
using ink2sm::wait;
using ink2sm::ldsm4;
using ink2sm::swz_x8;
using ink2sm::qmma_sf_16832;
using ink2sm::ink_act;

constexpr int BM = 64, BN = 64, BK = 64, THREADS = 256, NT = 4, STAGES = 2;
constexpr int ROW = 64, CHUNKS = 4;
constexpr int XX = BM * ROW;   // 4096: one 64x64 activation tile
constexpr int WW = BN * BK;   // 4096: one 64x64 weight tile
constexpr int SXB = BM * 2;   // 128: one u16 activation scale a row
constexpr int SWB = BN * 2;   // 128: one u16 weight scale a column
constexpr int STAGE_GU64 = (XX + 2 * WW + SXB + 2 * SWB + 127) / 128 * 128;  // 12672
constexpr int STAGE_DN64 = (XX + WW + SXB + SWB + 127) / 128 * 128;          // 8448

__device__ __forceinline__ void stage_x(unsigned char* p, const uint8_t* x_fp8, int K, int step,
                                        const int* members, int first, int r0, int cnt, int slots,
                                        int tid) {
    for (int c = tid; c < BM * CHUNKS; c += THREADS) {
        const int r = c / CHUNKS, ch = c % CHUNKS;
        const int lr = r0 + r;
        const bool ok = lr < cnt;
        const int src = ok ? __ldg(members + first + lr) / slots : 0;
        cp16z(p + r * ROW + swz_x8<CHUNKS>(r, ch) * 16,
              x_fp8 + static_cast<size_t>(src) * K + step * BK + ch * 16, ok);
    }
}

__device__ __forceinline__ void stage_x_pairs(unsigned char* p, const uint8_t* x_fp8, int K, int step,
                                              const int* members, int first, int r0, int cnt,
                                              int tid) {
    for (int c = tid; c < BM * CHUNKS; c += THREADS) {
        const int r = c / CHUNKS, ch = c % CHUNKS;
        const int lr = r0 + r;
        const bool ok = lr < cnt;
        const int src = ok ? __ldg(members + first + lr) : 0;
        cp16z(p + r * ROW + swz_x8<CHUNKS>(r, ch) * 16,
              x_fp8 + static_cast<size_t>(src) * K + step * BK + ch * 16, ok);
    }
}

__device__ __forceinline__ void load_sx(unsigned char* psx, const uint8_t* x_sf, int m_pad, int g64,
                                         const int* srows, int tid) {
    if (tid < BM)
        *reinterpret_cast<uint16_t*>(psx + tid * 2) =
            *reinterpret_cast<const uint16_t*>(x_sf + static_cast<size_t>(g64) * (m_pad * 2) + srows[tid] * 2);
}

__device__ __forceinline__ void load_w(unsigned char* pw, unsigned char* psw, const uint8_t* w8e,
                                        const uint8_t* bse, int tile_n, int g64, int KG64, int tid) {
    for (int c = tid; c < 256; c += THREADS)
        cp16z(pw + c * 16, w8e + (static_cast<size_t>(tile_n) * KG64 + g64) * 4096 + c * 16, true);
    for (int c = tid; c < 8; c += THREADS)
        cp16z(psw + c * 16, bse + (static_cast<size_t>(tile_n) * KG64 + g64) * 128 + c * 16, true);
}

__device__ __forceinline__ void mma_block(const unsigned char* p, const unsigned char* pw,
                                           const unsigned char* psx, const unsigned char* psw,
                                           float (&acc)[NT][4], int lane, int wn, int wm) {
    uint4 wq[NT];
    uint32_t sfb[NT];
    for (int j = 0; j < NT; ++j) {
        const int jj = wn * NT + j;
        const int col = (jj & 7) * 8 + (lane >> 2);
        wq[j] = reinterpret_cast<const uint4*>(pw)[jj * 32 + lane];
        sfb[j] = reinterpret_cast<const uint16_t*>(psw + (jj >> 3) * 128)[col];
    }
    const int r_sf = wm * 16 + (lane >> 2) + (lane & 1) * 8;
    const uint32_t sfa = reinterpret_cast<const uint16_t*>(psx)[r_sf];
    uint32_t a[4];
    {
        const int r = wm * 16 + (lane & 7) + ((lane >> 3) & 1) * 8, ch = (lane >> 4);
        ldsm4(a, p + r * ROW + swz_x8<CHUNKS>(r, ch) * 16);
        for (int j = 0; j < NT; ++j) qmma_sf_16832<0>(acc[j], a, wq[j].x, wq[j].y, sfa, sfb[j]);
    }
    {
        const int r = wm * 16 + (lane & 7) + ((lane >> 3) & 1) * 8, ch = 2 + (lane >> 4);
        ldsm4(a, p + r * ROW + swz_x8<CHUNKS>(r, ch) * 16);
        for (int j = 0; j < NT; ++j) qmma_sf_16832<1>(acc[j], a, wq[j].z, wq[j].w, sfa, sfb[j]);
    }
}

// One (expert item, n-tile): K-loop over gate then up through two W buffers, fused ink_act to bf16 h_out.
__device__ __forceinline__ void gu64_block(unsigned char* buf, const uint8_t* x_fp8, const uint8_t* x_sf,
                                            int m_pad, const uint8_t* w8_g, const uint8_t* bs_g,
                                            const uint8_t* w8_u, const uint8_t* bs_u, float scale,
                                            __nv_bfloat16* h_out, const int* members, int e, int first,
                                            int cnt, int slots, int K, int N, int tn, int tid, int lane,
                                            int wm, int wn) {
    const int KG64 = K >> 6;
    const size_t e_w = static_cast<size_t>(N >> 6) * KG64 * 4096;
    const size_t e_s = static_cast<size_t>(N >> 6) * KG64 * 128;
    const int n0 = tn * BN;
    const int KSTEPS = K >> 6;
    __shared__ int srows[BM];
    for (int r0 = 0; r0 < cnt; r0 += BM) {
        const int rr = min(BM, cnt - r0);
        __syncthreads();
        if (tid < BM) {
            const int lr = r0 + tid;
            srows[tid] = (lr < cnt) ? __ldg(members + first + lr) / slots : 0;
        }
        __syncthreads();
        float acc_g[NT][4] = {};
        float acc_u[NT][4] = {};
        auto stage = [&](int s) { return buf + s * STAGE_GU64; };
        auto load_stage = [&](int s, int step) {
            unsigned char* p = stage(s);
            unsigned char* pw_g = p + XX;
            unsigned char* pw_u = pw_g + WW;
            unsigned char* psx = pw_u + WW;
            unsigned char* psw_g = psx + SXB;
            unsigned char* psw_u = psw_g + SWB;
            stage_x(p, x_fp8, K, step, members, first, r0, cnt, slots, tid);
            load_w(pw_g, psw_g, w8_g + e * e_w, bs_g + e * e_s, tn, step, KG64, tid);
            load_w(pw_u, psw_u, w8_u + e * e_w, bs_u + e * e_s, tn, step, KG64, tid);
            load_sx(psx, x_sf, m_pad, step, srows, tid);
        };
        load_stage(0, 0);
        commit();
        for (int step = 0; step < KSTEPS; ++step) {
            wait<0>();
            __syncthreads();
            if (step + 1 < KSTEPS) {
                load_stage((step + 1) & 1, step + 1);
                commit();
            }
            const unsigned char* p = stage(step & 1);
            const unsigned char* pw_g = p + XX;
            const unsigned char* pw_u = pw_g + WW;
            const unsigned char* psx = pw_u + WW;
            const unsigned char* psw_g = psx + SXB;
            const unsigned char* psw_u = psw_g + SWB;
            mma_block(p, pw_g, psx, psw_g, acc_g, lane, wn, wm);
            mma_block(p, pw_u, psx, psw_u, acc_u, lane, wn, wm);
        }
        wait<0>();
        __syncthreads();
        for (int j = 0; j < NT; ++j)
            for (int h = 0; h < 2; ++h) {
                const int lr = wm * 16 + (lane >> 2) + h * 8;
                if (lr >= rr) continue;
                const int pr = __ldg(members + first + r0 + lr);
                const int col = n0 + wn * 32 + j * 8 + (lane & 3) * 2;
                const float gv0 = acc_g[j][2 * h] * scale, gv1 = acc_g[j][2 * h + 1] * scale;
                const float uv0 = acc_u[j][2 * h] * scale, uv1 = acc_u[j][2 * h + 1] * scale;
                __nv_bfloat16* dst = h_out + static_cast<size_t>(pr) * N + col;
                if (col + 1 < N)
                    *reinterpret_cast<__nv_bfloat162*>(dst) =
                        __floats2bfloat162_rn(ink_act<4, 25>(gv0, uv0), ink_act<4, 25>(gv1, uv1));
            }
    }
}

// One (expert item, n-tile) down projection to bf16 down_out.
__device__ __forceinline__ void dn64_block(unsigned char* buf, const uint8_t* x_fp8, const uint8_t* x_sf,
                                            int m_pad, const uint8_t* w8_d, const uint8_t* bs_d, float scale,
                                            __nv_bfloat16* down_out, const int* members, int e, int first,
                                            int cnt, int K, int N, int tn, int tid, int lane, int wm, int wn) {
    const int KG64 = K >> 6;
    const size_t e_w = static_cast<size_t>(N >> 6) * KG64 * 4096;
    const size_t e_s = static_cast<size_t>(N >> 6) * KG64 * 128;
    const int n0 = tn * BN;
    const int KSTEPS = K >> 6;
    __shared__ int srows[BM];
    for (int r0 = 0; r0 < cnt; r0 += BM) {
        const int rr = min(BM, cnt - r0);
        __syncthreads();
        if (tid < BM) {
            const int lr = r0 + tid;
            srows[tid] = (lr < cnt) ? __ldg(members + first + lr) : 0;
        }
        __syncthreads();
        float acc[NT][4] = {};
        auto stage = [&](int s) { return buf + s * STAGE_DN64; };
        auto load_stage = [&](int s, int step) {
            unsigned char* p = stage(s);
            unsigned char* pw = p + XX;
            unsigned char* psx = pw + WW;
            unsigned char* psw = psx + SXB;
            stage_x_pairs(p, x_fp8, K, step, members, first, r0, cnt, tid);
            load_w(pw, psw, w8_d + e * e_w, bs_d + e * e_s, tn, step, KG64, tid);
            load_sx(psx, x_sf, m_pad, step, srows, tid);
        };
        load_stage(0, 0);
        commit();
        for (int step = 0; step < KSTEPS; ++step) {
            wait<0>();
            __syncthreads();
            if (step + 1 < KSTEPS) {
                load_stage((step + 1) & 1, step + 1);
                commit();
            }
            const unsigned char* p = stage(step & 1);
            const unsigned char* pw = p + XX;
            const unsigned char* psx = pw + WW;
            const unsigned char* psw = psx + SXB;
            mma_block(p, pw, psx, psw, acc, lane, wn, wm);
        }
        wait<0>();
        __syncthreads();
        for (int j = 0; j < NT; ++j)
            for (int h = 0; h < 2; ++h) {
                const int lr = wm * 16 + (lane >> 2) + h * 8;
                if (lr >= rr) continue;
                const int pr = __ldg(members + first + r0 + lr);
                const int col = n0 + wn * 32 + j * 8 + (lane & 3) * 2;
                const float v0 = acc[j][2 * h] * scale, v1 = acc[j][2 * h + 1] * scale;
                __nv_bfloat16* dst = down_out + static_cast<size_t>(pr) * N + col;
                if (col + 1 < N)
                    *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(v0, v1);
            }
    }
}

}  // namespace ink2mxq64

// Persistent-grid grouped-expert MXFP8 gate|up, BM=64 rows a block, fused capped SwiGLU.
extern "C" __global__ void __launch_bounds__(256) tf_ink2_mxfp8_moe_gu_64(
    const uint8_t* x_fp8, const uint8_t* x_sf, int m_pad, const uint8_t* w8_g, const uint8_t* bs_g,
    const uint8_t* w8_u, const uint8_t* bs_u, float scale, __nv_bfloat16* h_out, int slots, int K, int N,
    const int* items, const int* counts, const int* members, int skip) {
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp >> 1, wn = warp & 1;
    const int ntiles = N >> 6;
    const int units = __ldg(counts) * ntiles;
    for (int unit = blockIdx.x; unit < units; unit += gridDim.x) {
        const int it = unit / ntiles, tn = unit - it * ntiles;
        const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1);
        const int cnt = __ldg(items + 3 * it + 2);
        if (e == skip) continue;
        ink2mxq64::gu64_block(buf, x_fp8, x_sf, m_pad, w8_g, bs_g, w8_u, bs_u, scale, h_out, members, e,
                               first, cnt, slots, K, N, tn, tid, lane, wm, wn);
    }
}

// Persistent-grid grouped-expert MXFP8 down projection, BM=64 rows a block.
extern "C" __global__ void __launch_bounds__(256) tf_ink2_mxfp8_moe_dn64(
    const uint8_t* x_fp8, const uint8_t* x_sf, int m_pad, const uint8_t* w8_d, const uint8_t* bs_d,
    float scale, __nv_bfloat16* down_out, int K, int N, const int* items, const int* counts,
    const int* members, int skip) {
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp >> 1, wn = warp & 1;
    const int ntiles = N >> 6;
    const int units = __ldg(counts) * ntiles;
    for (int unit = blockIdx.x; unit < units; unit += gridDim.x) {
        const int it = unit / ntiles, tn = unit - it * ntiles;
        const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1);
        const int cnt = __ldg(items + 3 * it + 2);
        if (e == skip) continue;
        ink2mxq64::dn64_block(buf, x_fp8, x_sf, m_pad, w8_d, bs_d, scale, down_out, members, e, first, cnt,
                               K, N, tn, tid, lane, wm, wn);
    }
}
