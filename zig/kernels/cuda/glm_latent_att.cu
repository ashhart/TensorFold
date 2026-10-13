// GLM-5.3-Flash latent.py part 2: 512-key chunk attention over the fp8/bf16 cache and the merge, Triton's orders.

#include <cstdint>
#include <cuda_bf16.h>

namespace tf_glm_latent {

using bf16 = __nv_bfloat16;

constexpr int H = 32;            // heads
constexpr int LW = 512;          // latent width
constexpr unsigned LOG2E_BITS = 0x3FB8AA3B;

__device__ __forceinline__ float neg_inf() { return __uint_as_float(0xFF800000u); }

__device__ __forceinline__ float fmax32(float a, float b) {
    float r;
    asm("max.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ float ex2(float x) {
    float r;
    asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ float div_full(float a, float b) {
    float r;
    asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ float log2e_mul(float x) {
    return __fmul_rn(x, __uint_as_float(LOG2E_BITS));
}

__device__ __forceinline__ unsigned pack_bf16x2(float lo, float hi) {
    unsigned r;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}

__device__ __forceinline__ unsigned cvt_e4m3x2_u16(unsigned short pair) {
    unsigned r;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(pair));
    return r;
}

__device__ __forceinline__ float f16_of(unsigned u, int half) {
    __half_raw hr;
    hr.x = half ? (unsigned short)(u >> 16) : (unsigned short)u;
    return __half2float(__half(hr));
}

__device__ __forceinline__ void mma_16n8k16(float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

struct ChunksArgs {
    const bf16* qa;    // bf16 [R, H, LW] (u32 view: 256 a row)
    const void* cache; // bf16 or fp8 [cap, LW]
    const float* sc;   // fp32 [cap] (fp8 only)
    const int* pos;    // int32 [1] (dense)
    const int* tok;    // int32 [R, W] (sparse)
    const int* cnt;    // int32 [R] (sparse)
    float* po;         // fp32 [nch, R, H, LW]
    float* pm;         // fp32 [nch, R, H]
    float* pl;         // fp32 [nch, R, H]
    int R;
    int tokw;          // W, the sparse token stride
    float scale;       // score scale (dense 0.0625, sparse 1/sqrt(128))
};

// One block per (row, head group, 512-key chunk): staged keys, 32 score mma steps, value mma, o seeded o*alpha.
template <int HBT, bool FP8, bool SPARSE>
__device__ __forceinline__ void attn_chunks(const ChunksArgs a) {
    const int r = blockIdx.x;
    const int t0 = blockIdx.y;
    const int c = blockIdx.z;
    const int t = threadIdx.x;
    const int w = t >> 5, l = t & 31, m = t & 3;
    const int cbase = c * 512;
    const bool run = SPARSE ? (cbase < a.cnt[r]) : (cbase <= a.pos[0] + r);
    __shared__ unsigned kv[32][256]; // 32 keys x 512 latents as bf16 pairs
    __shared__ unsigned psh[32][16];
    __shared__ float tmx[32][2][4]; // per row, a/b tile max over the warp's 8-key slice (w&3)
    __shared__ float lbuf[32][2][4]; // per row, a/b rowsum over the warp's 8-key slice
    const int rowA = (HBT == 32 ? 16 * (w >> 2) : 16 * t0) + (l >> 2);
    const int rowB = rowA + 8;
    const int kbase = (w & 3) * 8;
    const int pw = 4 * (w & 3);
    const int on_base = (HBT == 32 ? 128 * (w & 3) : 64 * w);
    const int NT = (HBT == 32) ? 16 : 8;
    float m_a = neg_inf(), m_b = neg_inf(), l_a = 0.0f, l_b = 0.0f;
    float oacc[16][4];
    for (int j = 0; j < NT; ++j) {
        oacc[j][0] = 0.0f;
        oacc[j][1] = 0.0f;
        oacc[j][2] = 0.0f;
        oacc[j][3] = 0.0f;
    }
    for (int tile = 0; run && tile < 16; ++tile) { // 16 tiles of 32 keys = the 512-key chunk
        // a tile past the last visible key leaves m, l and o as they are (alpha 1, p 0): stop at the first one
        if (SPARSE ? cbase + tile * 32 >= a.cnt[r] : cbase + tile * 32 > a.pos[0] + r) break;
        // stage 32 keys x 512 latents: thread t holds key (t>>3), u32 columns (t&7)*32..
        const int kk = t >> 3;
        int rowbase;
        float s = 1.0f;
        if (SPARSE) {
            const int idx = cbase + tile * 32 + kk;
            const bool ok = idx < a.cnt[r];
            rowbase = ok ? a.tok[(long long)r * a.tokw + idx] : 0;
            s = ok ? a.sc[rowbase] : 1.0f;
        } else {
            rowbase = cbase + tile * 32 + kk;
            if (FP8) s = a.sc[rowbase];
        }
        unsigned* dst = &kv[kk][(t & 7) * 32];
        if (FP8) {
            const unsigned char* src = reinterpret_cast<const unsigned char*>(a.cache) + (long long)rowbase * LW;
            for (int i = 0; i < 16; ++i) {
                const unsigned c4 = *reinterpret_cast<const unsigned*>(src + 4 * ((t & 7) * 16 + i));
                const unsigned p01 = cvt_e4m3x2_u16((unsigned short)c4);
                const unsigned p23 = cvt_e4m3x2_u16((unsigned short)(c4 >> 16));
                const float g0 = __fmul_rn(f16_of(p01, 0), s);
                const float g1 = __fmul_rn(f16_of(p01, 1), s);
                const float g2 = __fmul_rn(f16_of(p23, 0), s);
                const float g3 = __fmul_rn(f16_of(p23, 1), s);
                dst[2 * i] = pack_bf16x2(g0, g1);
                dst[2 * i + 1] = pack_bf16x2(g2, g3);
            }
        } else {
            const uint4* src = reinterpret_cast<const uint4*>(reinterpret_cast<const unsigned char*>(a.cache) +
                                                              (long long)rowbase * 2 * LW);
            for (int i = 0; i < 8; ++i) {
                const uint4 v = src[(t & 7) * 8 + i];
                unsigned* d = dst + 4 * i;
                d[0] = v.x;
                d[1] = v.y;
                d[2] = v.z;
                d[3] = v.w;
            }
        }
        __syncthreads();
        // scores: 32 ksteps of m16n8k16, A from the qa u32 view, B from shared
        const unsigned* qrowA = reinterpret_cast<const unsigned*>(a.qa) + ((long long)r * H + rowA) * 256;
        const unsigned* qrowB = qrowA + 2048;
        float sacc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int kstep = 0; kstep < 32; ++kstep) {
            // Triton's dot operand kWidth: bf16 2 (sequential 16-latent window), fp8 4 (even/odd pairs of a 32 window)
            const unsigned abase = FP8 ? 16 * (kstep >> 1) + 2 * m + (kstep & 1) : 8 * kstep + m;
            const unsigned boff = FP8 ? 8 : 4;
            const unsigned av[4] = {qrowA[abase], qrowB[abase],
                                    qrowA[abase + boff], qrowB[abase + boff]};
            const unsigned* krow = kv[kbase + (l >> 2)];
            const unsigned bv[2] = {krow[abase], krow[abase + boff]};
            mma_16n8k16(sacc, av, bv);
        }
        // scale then the validity mask (where(valid, scores, -inf))
        const int key0 = cbase + tile * 32 + kbase + 2 * m;
        bool valid[4];
        if (SPARSE) {
            valid[0] = key0 < a.cnt[r];
            valid[1] = key0 + 1 < a.cnt[r];
        } else {
            const int lim = a.pos[0] + r;
            valid[0] = key0 <= lim;
            valid[1] = key0 + 1 <= lim;
        }
        valid[2] = valid[0];
        valid[3] = valid[1];
        for (int i = 0; i < 4; ++i) {
            sacc[i] = __fmul_rn(sacc[i], a.scale);
            if (!valid[i]) sacc[i] = neg_inf();
        }
        // tile max: in-warp xor 2/1, then the cross-warp exchange (order-free)
        float ta = fmax32(sacc[0], sacc[1]);
        float tb = fmax32(sacc[2], sacc[3]);
        ta = fmax32(ta, __shfl_xor_sync(0xffffffffu, ta, 2));
        ta = fmax32(ta, __shfl_xor_sync(0xffffffffu, ta, 1));
        tb = fmax32(tb, __shfl_xor_sync(0xffffffffu, tb, 2));
        tb = fmax32(tb, __shfl_xor_sync(0xffffffffu, tb, 1));
        if ((l & 3) == 0) { // the row's first lane writes this warp's slice max into the row's slot
            tmx[rowA][0][w & 3] = ta;
            tmx[rowA][1][w & 3] = tb;
        }
        __syncthreads();
        const float tile_a = fmax32(fmax32(tmx[rowA][0][0], tmx[rowA][0][1]), fmax32(tmx[rowA][0][2], tmx[rowA][0][3]));
        const float tile_b = fmax32(fmax32(tmx[rowA][1][0], tmx[rowA][1][1]), fmax32(tmx[rowA][1][2], tmx[rowA][1][3]));
        const bool act_a = tile_a != neg_inf();
        const bool act_b = tile_b != neg_inf();
        const float nma = act_a ? fmax32(m_a, tile_a) : m_a;
        const float nmb = act_b ? fmax32(m_b, tile_b) : m_b;
        const float alpha_a = act_a ? (m_a == neg_inf() ? 0.0f : ex2(log2e_mul(__fsub_rn(m_a, nma)))) : 1.0f;
        const float alpha_b = act_b ? (m_b == neg_inf() ? 0.0f : ex2(log2e_mul(__fsub_rn(m_b, nmb)))) : 1.0f;
        const float p0 = (valid[0] && act_a) ? ex2(log2e_mul(__fsub_rn(sacc[0], nma))) : 0.0f;
        const float p1 = (valid[1] && act_a) ? ex2(log2e_mul(__fsub_rn(sacc[1], nma))) : 0.0f;
        const float p2 = (valid[2] && act_b) ? ex2(log2e_mul(__fsub_rn(sacc[2], nmb))) : 0.0f;
        const float p3 = (valid[3] && act_b) ? ex2(log2e_mul(__fsub_rn(sacc[3], nmb))) : 0.0f;
        for (int j = 0; j < NT; ++j) {
            oacc[j][0] = __fmul_rn(oacc[j][0], alpha_a);
            oacc[j][1] = __fmul_rn(oacc[j][1], alpha_a);
            oacc[j][2] = __fmul_rn(oacc[j][2], alpha_b);
            oacc[j][3] = __fmul_rn(oacc[j][3], alpha_b);
        }
        psh[rowA][pw + m] = pack_bf16x2(p0, p1);
        psh[rowB][pw + m] = pack_bf16x2(p2, p3);
        __syncthreads();
        // value mma: A from psh (key pairs pu0/pu1 as in the scores map), B the u16 latents of those keys
        for (int j = 0; j < NT; ++j) {
            for (int kstep = 0; kstep < 2; ++kstep) {
                const int pu0 = FP8 ? 2 * m + kstep : 8 * kstep + m;
                const int pu1 = pu0 + (FP8 ? 8 : 4);
                const int ncol = on_base + 8 * j + (l >> 2);
                const unsigned pav[4] = {psh[rowA][pu0], psh[rowB][pu0], psh[rowA][pu1], psh[rowB][pu1]};
                const unsigned cu = (unsigned)(ncol >> 1);
                const unsigned sh = (unsigned)(ncol & 1) << 4;
                const unsigned k0 = kv[2 * pu0][cu], k1 = kv[2 * pu0 + 1][cu];
                const unsigned k8 = kv[2 * pu1][cu], k9 = kv[2 * pu1 + 1][cu];
                const unsigned b0 = ((k0 >> sh) & 0xffffu) | (((k1 >> sh) & 0xffffu) << 16);
                const unsigned b1 = ((k8 >> sh) & 0xffffu) | (((k9 >> sh) & 0xffffu) << 16);
                const unsigned bv[2] = {b0, b1};
                mma_16n8k16(oacc[j], pav, bv);
            }
        }
        // l rowsum: per-lane pair sums, in-warp xor 2/1, cross-warp (P0+P2)+(P1+P3), l = l*alpha + rowsum
        const float t1a = __fadd_rn(p0, p1);
        const float t1b = __fadd_rn(p2, p3);
        float xa = __fadd_rn(t1a, __shfl_xor_sync(0xffffffffu, t1a, 2));
        xa = __fadd_rn(xa, __shfl_xor_sync(0xffffffffu, xa, 1));
        float xb = __fadd_rn(t1b, __shfl_xor_sync(0xffffffffu, t1b, 2));
        xb = __fadd_rn(xb, __shfl_xor_sync(0xffffffffu, xb, 1));
        if ((l & 3) == 0) { // per-row rowsum slots, same exchange discipline as tmx
            lbuf[rowA][0][w & 3] = xa;
            lbuf[rowA][1][w & 3] = xb;
        }
        __syncthreads();
        // SASS-confirmed tree: ((P0+P2)+(P1+P3)) over the warp key-slices (P_w = keys 8w..8w+7)
        const float rsa = __fadd_rn(__fadd_rn(lbuf[rowA][0][0], lbuf[rowA][0][2]),
                                    __fadd_rn(lbuf[rowA][0][1], lbuf[rowA][0][3]));
        const float rsb = __fadd_rn(__fadd_rn(lbuf[rowA][1][0], lbuf[rowA][1][2]),
                                    __fadd_rn(lbuf[rowA][1][1], lbuf[rowA][1][3]));
        l_a = __fmaf_rn(l_a, alpha_a, rsa);
        l_b = __fmaf_rn(l_b, alpha_b, rsb);
        m_a = nma;
        m_b = nmb;
    }
    const long long pbase = ((long long)c * a.R + r) * H;
    float* porow = a.po + (pbase + rowA) * LW;
    float* porowB = a.po + (pbase + rowB) * LW;
    for (int j = 0; j < NT; ++j) {
        porow[on_base + 8 * j + 2 * m] = oacc[j][0];
        porow[on_base + 8 * j + 2 * m + 1] = oacc[j][1];
        porowB[on_base + 8 * j + 2 * m] = oacc[j][2];
        porowB[on_base + 8 * j + 2 * m + 1] = oacc[j][3];
    }
    if (HBT == 32 || w < 4) {
        a.pm[pbase + rowA] = m_a;
        a.pl[pbase + rowA] = l_a;
        a.pm[pbase + rowB] = m_b;
        a.pl[pbase + rowB] = l_b;
    }
}

extern "C" __global__ void __launch_bounds__(256) tf_glm_attn_d16_bf16(ChunksArgs a) { attn_chunks<16, false, false>(a); }
extern "C" __global__ void __launch_bounds__(256) tf_glm_attn_d16_fp8(ChunksArgs a) { attn_chunks<16, true, false>(a); }
extern "C" __global__ void __launch_bounds__(256) tf_glm_attn_d32_bf16(ChunksArgs a) { attn_chunks<32, false, false>(a); }
extern "C" __global__ void __launch_bounds__(256) tf_glm_attn_d32_fp8(ChunksArgs a) { attn_chunks<32, true, false>(a); }
extern "C" __global__ void __launch_bounds__(256) tf_glm_attn_s16_fp8(ChunksArgs a) { attn_chunks<16, true, true>(a); }
extern "C" __global__ void __launch_bounds__(256) tf_glm_attn_s32_fp8(ChunksArgs a) { attn_chunks<32, true, true>(a); }

struct MergeArgs {
    const float* po; // fp32 [nch, R, H, LW]
    const float* pm; // fp32 [nch, R, H]
    const float* pl; // fp32 [nch, R, H]
    bf16* out;       // bf16 [R, H, LW]
    const int* cnt;  // int32 [R] (sparse)
    int R;
    int nch;
    int sparse;
};

// Merge one (row, head)'s chunk partials: running max and scale, fma the scaled partials, div.full to bf16.
extern "C" __global__ void tf_glm_merge(MergeArgs a) {
    const int r = blockIdx.x;
    const int h = blockIdx.y;
    const int t = threadIdx.x;
    if (a.sparse != 0 && a.cnt[r] == 0) return;
    float m = neg_inf(), lsum = 0.0f, o[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int c = 0; c < a.nch; ++c) {
        const long long base = ((long long)c * a.R + r) * H + h;
        const float cm = a.pm[base];
        const float cl = a.pl[base];
        const bool active = cl > 0.0f;
        const float nma = active ? fmax32(m, cm) : m;
        const float alpha = active ? (m == neg_inf() ? 0.0f : ex2(log2e_mul(__fsub_rn(m, nma)))) : 1.0f;
        const float beta = active ? ex2(log2e_mul(__fsub_rn(cm, nma))) : 0.0f;
        const float4 raw = *reinterpret_cast<const float4*>(a.po + base * LW + 4 * t);
        const float co[4] = {raw.x, raw.y, raw.z, raw.w};
        for (int i = 0; i < 4; ++i) o[i] = __fmaf_rn(o[i], alpha, __fmul_rn(co[i], beta));
        lsum = __fmaf_rn(lsum, alpha, __fmul_rn(cl, beta));
        m = nma;
    }
    unsigned ov[2];
    for (int i = 0; i < 2; ++i) ov[i] = pack_bf16x2(div_full(o[2 * i], lsum), div_full(o[2 * i + 1], lsum));
    *reinterpret_cast<uint2*>(reinterpret_cast<bf16*>(a.out) + ((long long)r * H + h) * LW + 4 * t) =
        make_uint2(ov[0], ov[1]);
}

} // namespace tf_glm_latent
