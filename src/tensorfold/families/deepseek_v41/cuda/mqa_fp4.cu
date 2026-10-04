// DeepSeek-V4.1 decode attention over the NVFP4 compressed KV cache (Fp4Rows) plus the bf16 window ring: the split
// pass of kernels.mqa for decode / verify rows and small prompt chunks; kernels._mqa_merge_wide combines the splits
// (sink, normalisation, inverse RoPE).
//
// SPDX-License-Identifier: Apache-2.0
// Adapted from FlashInfer's "Cake" DeepSeek-V4.1 mixed-cache decode, flashinfer-ai/flashinfer
// csrc/cake_dsv4/sm_120a/cake_sparse_mla_dsv41_mixed_h32.cu (decode_dual), commit
// 2c1c0525067452c411944fb1c40d8112040ea229 (PR #5983, generated from Cake 092b4b1f5fd8913d7c01390c7ceb6bc7e866d8e2),
// Apache License 2.0; Copyright 2025-2026 NVIDIA, Copyright 2023-2026 FlashInfer community (LICENSES/Apache-2.0.txt,
// NOTICE, THIRD_PARTY_NOTICES.md). Taken from it: 16-head tiles scored against one gathered candidate set, bf16
// mma.sync m16n8k16 with the FP4 rows widened exactly to bf16 (cvt.rn.bf16x2.e2m1x2 on CUDA 13.2+, else its two-prmt
// table; e4m3 scales widened and multiplied in bf16, exact), a lane owning whole 16-dim scale groups (Q and K permuted
// alike), V^T read with ldmatrix.trans.
// Modified for TensorFold dsv41-cuda: hand-written loops instead of the generated unrolled code; no IO warps or
// mbarrier ring: a CTA (one row x one split x one 16-head tile, 8 warps each owning 64 dims for QK and PV) gathers
// every compressed row of its split into registers up front and decodes each 32-row stage once into a bf16 stage in
// shared memory (double-buffered; window stages stream in with cp.async one stage ahead); separate Fp4Rows planes (q
// nibbles, s e4m3 bytes) instead of paged footers; the bf16 window ring read directly (rows from pos, the stream's
// ring base and the ring length) instead of FP8 528-byte main rows; QK split over dims across the 8 warps with the
// partial scores summed in a fixed tree; few long splits (a fixed partition that depends only on the layer's index
// count: row invariance) so that the partials stay small at 32 rows; TensorFold's natural-log online softmax, the
// sink applied in the merge; no lse_scale / out_lse, no main-only kernel, no PDL.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

#if !defined(TF_MQA4_LUT) && (__CUDACC_VER_MAJOR__ * 100 + __CUDACC_VER_MINOR__) >= 1302
#define TF_MQA4_CVT 1
#else
#define TF_MQA4_CVT 0
#endif

namespace tf_mqa4 {

constexpr int H = 32;           // local heads (TP=2 of 64)
constexpr int D = 512;          // latent: key and value
constexpr int W = 128;          // window
constexpr int ST = 32;          // candidates a stage
constexpr int NW = 8;           // warps a CTA: warp w owns dims w * 64 .. + 63
constexpr int NT = 32 * NW;
constexpr int T_BYTES = ST * D * 2;                 // bf16 stage, swizzled 16-byte granules
constexpr int PP = 40;                              // partial-score row pitch (floats): conflict-free float2
constexpr int PART_BYTES = NW * 16 * PP * 4;
constexpr int PBP = 20;                             // P row pitch (bf16 pairs): conflict-free A fragments
constexpr int FIN_BYTES = (16 * PBP + 32) * 4;
constexpr int SMEM = 2 * T_BYTES + PART_BYTES + FIN_BYTES + 3 * NT * 4;

// granule c of stage row r: low 3 bits xored with a bijection of r & 7 that puts rows 2m, 2m+1 four granules apart
// (QK: a quarter warp reads rows 2m, 2m+1 at four consecutive granules) and any 8 consecutive rows apart (ldmatrix)
__device__ __forceinline__ int phys(int c, int r) {
    const int x = r & 7;
    return c ^ (((x & 1) << 2) | (x >> 1));
}

// two E2M1 nibbles (low: the first element) -> bf16x2 (low half: the first), exact
__device__ __forceinline__ uint32_t e2m1x2_lut(uint32_t b) {
    const uint32_t hi = __byte_perm(0x3F3F3F00u, 0x40404040u, ((b & 7u) << 4) | ((b & 0x70u) << 8));
    const uint32_t lo = __byte_perm(0xC0800000u, 0xC0804000u, (b & 7u) | ((b & 0x70u) << 4));
    return hi | lo | ((b & 8u) << 12) | ((b & 0x80u) << 24);
}

// the four bytes of w -> four bf16x2
__device__ __forceinline__ void e2m1x8(uint32_t w, uint32_t (&o)[4]) {
#if TF_MQA4_CVT
    asm("{\n.reg .b8 b0, b1, b2, b3;\nmov.b32 {b0, b1, b2, b3}, %4;\n"
        "cvt.rn.bf16x2.e2m1x2 %0, b0;\ncvt.rn.bf16x2.e2m1x2 %1, b1;\n"
        "cvt.rn.bf16x2.e2m1x2 %2, b2;\ncvt.rn.bf16x2.e2m1x2 %3, b3;\n}"
        : "=r"(o[0]), "=r"(o[1]), "=r"(o[2]), "=r"(o[3]) : "r"(w));
#else
#pragma unroll
    for (int i = 0; i < 4; ++i) o[i] = e2m1x2_lut((w >> (8 * i)) & 0xFFu);
#endif
}

// e4m3 byte -> bf16x2 {s, s}, exact (e4m3 values all fit bf16)
__device__ __forceinline__ uint32_t e4m3_bf16x2(uint32_t sb) {
    const uint32_t two = sb | (sb << 8);
    uint32_t d;
#if TF_MQA4_CVT
    asm("{\n.reg .b16 h;\ncvt.u16.u32 h, %1;\ncvt.rn.bf16x2.e4m3x2 %0, h;\n}" : "=r"(d) : "r"(two));
#else
    uint32_t hh;
    asm("{\n.reg .b16 h;\ncvt.u16.u32 h, %1;\ncvt.rn.f16x2.e4m3x2 %0, h;\n}" : "=r"(hh) : "r"(two));
    const float f = __half2float(__ushort_as_half((unsigned short)(hh & 0xFFFFu)));
    const uint32_t b = __bfloat16_as_ushort(__float2bfloat16_rn(f));
    d = b | (b << 16);
#endif
    return d;
}

__device__ __forceinline__ uint32_t bmul(uint32_t a, uint32_t b) {
    uint32_t d;
    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
    return d;
}

__device__ __forceinline__ void mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&v);
}

__device__ __forceinline__ float ex(float x) {      // e^x
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(__fmul_rn(x, 1.4426950408889634f)));
    return y;
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool ok) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(ok ? 16 : 0));
}

// One stage of 32 candidates in T (bf16, swizzled) against the CTA's 16 heads, as a partial of its own: QK over warp
// w's 64 dims, the stage's max m, numerators p = e^(s - m) and their sum l (computed once, shared through fin), PV
// into the warp's 64 output dims (acc: zero on entry). vm: the stage's valid candidates (none: m = -inf, l = 0, acc
// stays zero).
__device__ __forceinline__ void attend(const uint8_t* T, float* part, float* fin, uint32_t vm, const uint4 (&qa)[2][2],
                                       float (&acc)[8][4], float& m0, float& m1, float& l0, float& l1,
                                       float scale) {
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, gr = lane >> 2, t = lane & 3;
    float sc[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        sc[j][0] = sc[j][1] = sc[j][2] = sc[j][3] = 0.f;
        const int row = 8 * j + gr;
        const uint8_t* base = T + row * 1024;
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            const uint4 kb = *reinterpret_cast<const uint4*>(base + phys(w * 8 + t + 4 * q, row) * 16);
            const uint32_t a0[4] = {qa[0][q].x, qa[1][q].x, qa[0][q].y, qa[1][q].y};
            const uint32_t a1[4] = {qa[0][q].z, qa[1][q].z, qa[0][q].w, qa[1][q].w};
            mma(sc[j], a0, kb.x, kb.y);
            mma(sc[j], a1, kb.z, kb.w);
        }
    }
    float* mine = part + w * 16 * PP;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        *reinterpret_cast<float2*>(mine + gr * PP + 8 * j + 2 * t) = make_float2(sc[j][0], sc[j][1]);
        *reinterpret_cast<float2*>(mine + (gr + 8) * PP + 8 * j + 2 * t) = make_float2(sc[j][2], sc[j][3]);
    }
    __syncthreads();
    // two scores a thread (row tid / 16): the 8 quarters summed in a fixed tree, scaled and masked; the row's max and
    // numerators e^(s - m) (natural log, as the Triton chunks), their sum over the row's 16 threads in a fixed tree;
    // P to bf16 for PV
    {
        const int hh = tid >> 4, c = (tid & 15) * 2;
        float2 v[NW];
#pragma unroll
        for (int u = 0; u < NW; ++u) v[u] = *reinterpret_cast<const float2*>(part + (u * 16 + hh) * PP + c);
        float sa = ((v[0].x + v[1].x) + (v[2].x + v[3].x)) + ((v[4].x + v[5].x) + (v[6].x + v[7].x));
        float sb = ((v[0].y + v[1].y) + (v[2].y + v[3].y)) + ((v[4].y + v[5].y) + (v[6].y + v[7].y));
        sa = (vm >> c) & 1u ? sa * scale : -INFINITY;
        sb = (vm >> (c + 1)) & 1u ? sb * scale : -INFINITY;
        float mx = fmaxf(sa, sb);
#pragma unroll
        for (int x = 1; x < 16; x <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, x));
        const float pa = sa == -INFINITY ? 0.f : ex(__fsub_rn(sa, mx));
        const float pb = sb == -INFINITY ? 0.f : ex(__fsub_rn(sb, mx));
        float sum = __fadd_rn(pa, pb);
#pragma unroll
        for (int x = 1; x < 16; x <<= 1) sum = __fadd_rn(sum, __shfl_xor_sync(0xFFFFFFFFu, sum, x));
        reinterpret_cast<uint32_t*>(fin)[hh * PBP + (tid & 15)] = pack_bf16(pa, pb);
        if ((tid & 15) == 0) {
            fin[16 * PBP + hh] = mx;
            fin[16 * PBP + 16 + hh] = sum;
        }
    }
    __syncthreads();
    const uint32_t* pw = reinterpret_cast<const uint32_t*>(fin);
    m0 = fin[16 * PBP + gr];
    m1 = fin[16 * PBP + gr + 8];
    l0 = fin[16 * PBP + 16 + gr];
    l1 = fin[16 * PBP + 16 + gr + 8];
    // PV: P (bf16, from the score fragments) x V^T (ldmatrix.trans) over the warp's 64 output dims
#pragma unroll
    for (int kk = 0; kk < 2; ++kk) {
        const uint32_t pa[4] = {pw[gr * PBP + 8 * kk + t], pw[(gr + 8) * PBP + 8 * kk + t],
                                pw[gr * PBP + 8 * kk + 4 + t], pw[(gr + 8) * PBP + 8 * kk + 4 + t]};
        const int mi = lane >> 3, cand = kk * 16 + (mi & 1) * 8 + (lane & 7);
        const uint8_t* rowp = T + cand * 1024;
#pragma unroll
        for (int np = 0; np < 4; ++np) {
            const int c = w * 8 + 2 * np + (mi >> 1);
            uint32_t b0, b1, b2, b3;
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                         : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3)
                         : "r"(smem_u32(rowp + phys(c, cand) * 16)));
            mma(acc[2 * np], pa, b0, b1);
            mma(acc[2 * np + 1], pa, b2, b3);
        }
    }
}

// A partial over some stages: unnormalised o, running max m, sum l (l == 0: nothing, m = -inf). fold() is the one
// way two partials combine, here and in merge_kernel alike (explicit roundings, no branches to diverge on), so a
// row's result is the same whichever kernel folds which part. An empty C (m = -inf, l = 0, o = 0) takes P exactly:
// a = 0, b = 1. An empty P leaves C as it is.
__device__ __forceinline__ bool fold_w(float& cm, float& cl, float pm, float pl, float& a, float& b) {
    const bool keep = !(pl > 0.f);
    const float mx = fmaxf(cm, pm);
    a = ex(__fsub_rn(cm, mx));
    b = ex(__fsub_rn(pm, mx));
    cl = keep ? cl : __fadd_rn(__fmul_rn(a, cl), __fmul_rn(b, pl));
    cm = keep ? cm : mx;
    return keep;
}

__device__ __forceinline__ float fold_o(float co, float po, float a, float b, bool keep) {
    return keep ? co : __fadd_rn(__fmul_rn(a, co), __fmul_rn(b, po));
}

// One CTA = one row x one part x one 16-head tile; grid (nparts, 2, R). The row's stages (32 candidates each):
// ncs = n_idx / 32 compressed (idx order), then W / 32 window (positions p - 127 + 32w ..); part k takes stages
// k * per .. (k + 1) * per - 1, each attended alone and folded in order into the part's partial. Thread (row tid / 8,
// seg tid % 8) of a compressed stage owns granules seg + 8u (8 dims each) of that row: 8 nibble words, plus scale
// word seg (shared with the row's other 7 threads by shuffle), loaded one stage ahead.
__global__ void __launch_bounds__(NT, 1)
split_kernel(const __nv_bfloat16* __restrict__ Q, const uint8_t* __restrict__ CQ, const uint8_t* __restrict__ CSC,
             int64_t qstride, int64_t sstride, const int* __restrict__ IDX, int64_t idx_stride, int n_idx,
             const __nv_bfloat16* __restrict__ SWA, const int64_t* __restrict__ POS,
             const int64_t* __restrict__ SBASE, int ring, float* __restrict__ PO, float* __restrict__ PM,
             float* __restrict__ PL, int per, int nparts, float scale) {
    extern __shared__ __align__(128) uint8_t smem[];
    float* part = reinterpret_cast<float*>(smem + 2 * T_BYTES);
    float* fin = reinterpret_cast<float*>(smem + 2 * T_BYTES + PART_BYTES);
    int* tab = reinterpret_cast<int*>(smem + 2 * T_BYTES + PART_BYTES + FIN_BYTES);
    const int k = blockIdx.x, tg = blockIdx.y, r = blockIdx.z, tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
    const int gr = lane >> 2, t = lane & 3;
    const int ncs = (n_idx + ST - 1) / ST, s0 = k * per;
    const int nst = min(per, ncs + W / ST - s0);             // this part's stages: s0 .. s0 + nst - 1
    const int nco = max(0, min(nst, ncs - s0));              // the compressed ones come first
    const int64_t p = POS[r];

    bool any = false;
    for (int e = tid; e < nst * ST; e += NT) {
        const int i = e >> 5, c = e & 31, s = s0 + i;
        int row = -1;
        if (s < ncs) {
            const int cc = s * ST + c;
            if (cc < n_idx) row = IDX[(int64_t)r * idx_stride + cc];
            row = row < 0 ? -1 : row;
        } else {
            const int64_t slot = p - (W - 1) + (int64_t)(s - ncs) * ST + c;
            if (slot >= 0) row = (int)((SBASE != nullptr ? SBASE[r] : 0) + (slot & (ring - 1)));
        }
        tab[e] = row;
        any |= row >= 0;
    }
    // Q fragments (in flight across the barrier): lane t owns dims w*64 + 8(t + 4q) .. +7 (q = 0, 1) of h0, h1
    const int h0 = tg * 16 + gr, h1 = h0 + 8;
    uint4 qa[2][2];
    {
        const uint4* q0 = reinterpret_cast<const uint4*>(Q + ((int64_t)r * H + h0) * D + w * 64);
        const uint4* q1 = reinterpret_cast<const uint4*>(Q + ((int64_t)r * H + h1) * D + w * 64);
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            qa[0][q] = __ldg(q0 + t + 4 * q);
            qa[1][q] = __ldg(q1 + t + 4 * q);
        }
    }
    if (!__syncthreads_or(any)) {                // nothing in this part: the merge skips l <= 0
        if (tid < 16) {
            const int64_t b = ((int64_t)r * nparts + k) * H + tg * 16 + tid;
            PM[b] = -INFINITY;
            PL[b] = 0.f;
        }
        return;
    }

    const int rrow = tid >> 3, seg = tid & 7;
    uint32_t rq[8], rs;
    auto load = [&](int i) {                     // compressed stage i's raw bytes into registers
        const int row = i < nco ? tab[i * ST + rrow] : -1;
        if (row >= 0) {
            const uint32_t* q = reinterpret_cast<const uint32_t*>(CQ + (int64_t)row * qstride);
#pragma unroll
            for (int u = 0; u < 8; ++u) rq[u] = __ldg(q + seg + 8 * u);
            rs = __ldg(reinterpret_cast<const uint32_t*>(CSC + (int64_t)row * sstride) + seg);
        } else {
#pragma unroll
            for (int u = 0; u < 8; ++u) rq[u] = 0u;
            rs = 0u;
        }
    };
    auto fill = [&](int i) {                     // stage i into T[i & 1]: decode, or the window rows' cp.async
        uint8_t* T = smem + (i & 1) * T_BYTES;
        if (i < nco) {
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                const int g = seg + 8 * u;                        // scale group g / 2 = 4u + seg / 2: word u
                const uint32_t sw = __shfl_sync(0xFFFFFFFFu, rs, (lane & ~7) | u);
                const uint32_t s2 = e4m3_bf16x2((sw >> (8 * (seg >> 1))) & 0xFFu);
                uint32_t v[4];
                e2m1x8(rq[u], v);
                uint4 o;
                o.x = bmul(v[0], s2); o.y = bmul(v[1], s2); o.z = bmul(v[2], s2); o.w = bmul(v[3], s2);
                *reinterpret_cast<uint4*>(T + rrow * 1024 + phys(g, rrow) * 16) = o;
            }
        } else {
            const int c = tid & 63;
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                const int rr = (tid >> 6) + 4 * u;
                const int row = tab[i * ST + rr];
                cp_async16(smem_u32(T + rr * 1024 + phys(c, rr) * 16),
                           SWA + (int64_t)(row < 0 ? 0 : row) * D + c * 8, row >= 0);
            }
            asm volatile("cp.async.commit_group;\n" ::);
        }
    };

    float co[8][4];
    float cm0 = -INFINITY, cm1 = -INFINITY, cl0 = 0.f, cl1 = 0.f;
#pragma unroll
    for (int n = 0; n < 8; ++n) co[n][0] = co[n][1] = co[n][2] = co[n][3] = 0.f;
    load(0);
    fill(0);
    load(1);
    for (int i = 0; i < nst; ++i) {
        if (i > 0) __syncthreads();                           // stage i - 1 is done with T[(i + 1) & 1]
        const bool ahead = i + 1 < nst;
        if (ahead) {
            fill(i + 1);
            load(i + 2);
        }
        if (i >= nco) {                                       // a window stage: its rows (one group may follow)
            if (ahead) asm volatile("cp.async.wait_group 1;\n" ::);
            else asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();
        const uint32_t vm = __ballot_sync(0xFFFFFFFFu, tab[i * ST + lane] >= 0);
        float po[8][4];
#pragma unroll
        for (int n = 0; n < 8; ++n) po[n][0] = po[n][1] = po[n][2] = po[n][3] = 0.f;
        float pm0 = -INFINITY, pm1 = -INFINITY, pl0 = 0.f, pl1 = 0.f;
        attend(smem + (i & 1) * T_BYTES, part, fin, vm, qa, po, pm0, pm1, pl0, pl1, scale);
        float a0, b0, a1, b1;
        const bool k0 = fold_w(cm0, cl0, pm0, pl0, a0, b0), k1 = fold_w(cm1, cl1, pm1, pl1, a1, b1);
#pragma unroll
        for (int n = 0; n < 8; ++n) {
            co[n][0] = fold_o(co[n][0], po[n][0], a0, b0, k0);
            co[n][1] = fold_o(co[n][1], po[n][1], a0, b0, k0);
            co[n][2] = fold_o(co[n][2], po[n][2], a1, b1, k1);
            co[n][3] = fold_o(co[n][3], po[n][3], a1, b1, k1);
        }
    }

    const int64_t b = ((int64_t)r * nparts + k) * H;
    float* o0 = PO + (b + h0) * D + w * 64 + 2 * t;
    float* o1 = PO + (b + h1) * D + w * 64 + 2 * t;
#pragma unroll
    for (int n = 0; n < 8; ++n) {
        *reinterpret_cast<float2*>(o0 + 8 * n) = make_float2(co[n][0], co[n][1]);
        *reinterpret_cast<float2*>(o1 + 8 * n) = make_float2(co[n][2], co[n][3]);
    }
    if (w == 0 && t == 0) {
        PM[b + h0] = cm0;
        PL[b + h0] = cl0;
        PM[b + h1] = cm1;
        PL[b + h1] = cl1;
    }
}

// Finish rows: per (row, head), the parts folded group by group (``ppg`` parts a group, in order: the group's
// stages folded left to right), then the groups folded onto the sink (a logit with a zero value vector); divide,
// inverse RoPE of the last 2 * half dims (cos / sin given: bf16 out) or fp32 out. 128 threads x 4 dims.
template <int MB>
__global__ void __launch_bounds__(128)
merge_kernel(const float* __restrict__ PO, const float* __restrict__ PM, const float* __restrict__ PL,
             const float* __restrict__ SINK, const int64_t* __restrict__ POS, const float* __restrict__ COS,
             const float* __restrict__ SIN, int half, void* __restrict__ OUT, int nparts, int ppg) {
    const int r = blockIdx.x, h = blockIdx.y, d = threadIdx.x * 4;
    float fm = SINK[h], fl = 1.f;
    float4 fo = make_float4(0.f, 0.f, 0.f, 0.f);
    // parts in batches of MB: every load of a batch in flight at once, then folded in order (a missing part is an
    // empty one; a group's end folds it onto the sink and starts the next group empty)
    float cm = -INFINITY, cl = 0.f;
    float4 cv = make_float4(0.f, 0.f, 0.f, 0.f);
    int left = ppg;                                       // parts until the current group ends
    for (int k0 = 0; k0 < nparts; k0 += MB) {
        float bm[MB], bl[MB];
        float4 bv[MB];
#pragma unroll
        for (int u = 0; u < MB; ++u) {
            bm[u] = -INFINITY;
            bl[u] = 0.f;
            bv[u] = make_float4(0.f, 0.f, 0.f, 0.f);
            if (k0 + u < nparts) {
                const int64_t b = ((int64_t)r * nparts + k0 + u) * H + h;
                bm[u] = PM[b];
                bl[u] = PL[b];
                bv[u] = *reinterpret_cast<const float4*>(PO + b * D + d);   // (unused when the part is empty)
            }
        }
#pragma unroll
        for (int u = 0; u < MB; ++u) {
            float a, bb;
            const bool keep = fold_w(cm, cl, bm[u], bl[u], a, bb);
            cv.x = fold_o(cv.x, bv[u].x, a, bb, keep);
            cv.y = fold_o(cv.y, bv[u].y, a, bb, keep);
            cv.z = fold_o(cv.z, bv[u].z, a, bb, keep);
            cv.w = fold_o(cv.w, bv[u].w, a, bb, keep);
            if (--left == 0 || k0 + u + 1 == nparts) {       // (uniform across the block: one row and head)
                const bool kg = fold_w(fm, fl, cm, cl, a, bb);   // onto the sink (never empty)
                fo.x = fold_o(fo.x, cv.x, a, bb, kg);
                fo.y = fold_o(fo.y, cv.y, a, bb, kg);
                fo.z = fold_o(fo.z, cv.z, a, bb, kg);
                fo.w = fold_o(fo.w, cv.w, a, bb, kg);
                cm = -INFINITY;
                cl = 0.f;
                cv = make_float4(0.f, 0.f, 0.f, 0.f);
                left = ppg;
            }
        }
    }
    float o[4] = {__fdiv_rn(fo.x, fl), __fdiv_rn(fo.y, fl), __fdiv_rn(fo.z, fl), __fdiv_rn(fo.w, fl)};
    const int64_t ob = ((int64_t)r * H + h) * D + d;
    if (COS != nullptr) {
        const int64_t p = POS[r];
#pragma unroll
        for (int e = 0; e < 4; e += 2) {                       // inverse RoPE: e' = e c + o s, o' = o c - e s
            if (d + e >= D - 2 * half) {
                const int i = (d + e - (D - 2 * half)) / 2;
                const float c = COS[p * half + i], sn = SIN[p * half + i];
                const float ev = o[e], od = o[e + 1];
                o[e] = __fadd_rn(__fmul_rn(ev, c), __fmul_rn(od, sn));
                o[e + 1] = __fsub_rn(__fmul_rn(od, c), __fmul_rn(ev, sn));
            }
        }
        __nv_bfloat16* out = reinterpret_cast<__nv_bfloat16*>(OUT) + ob;
        *reinterpret_cast<__nv_bfloat162*>(out) = __floats2bfloat162_rn(o[0], o[1]);
        *reinterpret_cast<__nv_bfloat162*>(out + 2) = __floats2bfloat162_rn(o[2], o[3]);
    } else {
        *reinterpret_cast<float4*>(reinterpret_cast<float*>(OUT) + ob) = make_float4(o[0], o[1], o[2], o[3]);
    }
}

__global__ void table_kernel(uint32_t* e2m1, uint16_t* e4m3) {
    const int b = threadIdx.x;
    uint32_t v[4];
    e2m1x8((uint32_t)b, v);
    e2m1[b] = v[0];
    e4m3[b] = (uint16_t)(e4m3_bf16x2((uint32_t)b) & 0xFFFFu);
}

}  // namespace tf_mqa4

int64_t stages_of(int64_t n_idx) { return (n_idx + tf_mqa4::ST - 1) / tf_mqa4::ST + tf_mqa4::W / tf_mqa4::ST; }

// The whole attention of rows <= 32: q [R, 32, 512] bf16; cq u8 [n, 256], cs u8 [n, 32] (Fp4Rows planes) and idx
// int32 [R, n_idx] (rows of cq, -1: none), or all three None (a window-only layer); swa bf16 [rows, 512] ring(s);
// pos int64 [R]; sbase int64 [R] (the row's ring's first row) or None (0); sink fp32 [32]; cos / sin fp32 [positions,
// half] (inverse RoPE, bf16 out) or None (fp32 out); out [R, 32, 512]; po / pm / pl partial scratch; ring: rows a
// ring (a power of two); group: stages a group (the fixed reduction tree); per: stages a CTA, 1 or ``group`` (the
// same result either way). Returns the parts written.
int64_t attend_rows(torch::Tensor q, c10::optional<torch::Tensor> cq, c10::optional<torch::Tensor> cs,
                    c10::optional<torch::Tensor> idx, torch::Tensor swa, torch::Tensor pos,
                    c10::optional<torch::Tensor> sbase, torch::Tensor sink, c10::optional<torch::Tensor> cosp,
                    c10::optional<torch::Tensor> sinp, torch::Tensor out, torch::Tensor po, torch::Tensor pm,
                    torch::Tensor pl, int64_t ring, int64_t group, int64_t per, double scale) {
    using namespace tf_mqa4;
    TORCH_CHECK(group >= 1 && (per == 1 || per == group), "per: 1 or group");
    TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kBFloat16 && q.dim() == 3 && q.size(1) == H && q.size(2) == D &&
                q.is_contiguous(), "q: contiguous bf16 [R, 32, 512]");
    const int64_t R = q.size(0);
    TORCH_CHECK(swa.scalar_type() == at::kBFloat16 && swa.dim() == 2 && swa.size(1) == D && swa.stride(1) == 1 &&
                swa.stride(0) == D && reinterpret_cast<uintptr_t>(swa.data_ptr()) % 16 == 0,
                "swa: bf16 [rows, 512] rows contiguous");
    TORCH_CHECK(ring > 0 && (ring & (ring - 1)) == 0, "ring: a power of two");
    TORCH_CHECK(pos.scalar_type() == at::kLong && pos.numel() == R && pos.is_contiguous(), "pos: int64 [R]");
    TORCH_CHECK(sink.scalar_type() == at::kFloat && sink.numel() == H && sink.is_contiguous(), "sink: fp32 [32]");
    const bool rope = cosp.has_value();
    int half = 1;
    if (rope) {
        TORCH_CHECK(sinp.has_value() && cosp->scalar_type() == at::kFloat && sinp->scalar_type() == at::kFloat &&
                    cosp->is_contiguous() && sinp->is_contiguous() && cosp->dim() == 2 && cosp->size(1) % 2 == 0,
                    "cos / sin: fp32 tables");
        half = (int)cosp->size(1);
    }
    TORCH_CHECK(out.is_contiguous() && out.numel() == R * H * D &&
                out.scalar_type() == (rope ? at::kBFloat16 : at::kFloat), "out: [R, 32, 512], bf16 with RoPE");
    const int64_t* sb = nullptr;
    if (sbase.has_value()) {
        TORCH_CHECK(sbase->scalar_type() == at::kLong && sbase->numel() == R && sbase->is_contiguous(),
                    "sbase: int64 [R]");
        sb = sbase->data_ptr<int64_t>();
    }
    int n_idx = 0;
    const uint8_t *qp = nullptr, *sp = nullptr;
    const int* ip = nullptr;
    int64_t qs = 0, ss = 0, is = 0;
    if (idx.has_value()) {
        TORCH_CHECK(cq.has_value() && cs.has_value(), "cq / cs with idx");
        TORCH_CHECK(idx->scalar_type() == at::kInt && idx->dim() == 2 && idx->size(0) == R && idx->stride(1) == 1,
                    "idx: int32 [R, n], unit column stride");
        TORCH_CHECK(cq->scalar_type() == at::kByte && cq->dim() == 2 && cq->size(1) == D / 2 && cq->stride(1) == 1 &&
                    cq->stride(0) % 16 == 0 && reinterpret_cast<uintptr_t>(cq->data_ptr()) % 16 == 0,
                    "cq: u8 [n, 256], 16-byte aligned rows");
        TORCH_CHECK(cs->scalar_type() == at::kByte && cs->dim() == 2 && cs->size(1) == D / 16 && cs->stride(1) == 1 &&
                    cs->stride(0) % 4 == 0 && reinterpret_cast<uintptr_t>(cs->data_ptr()) % 4 == 0,
                    "cs: u8 [n, 32], 4-byte aligned rows");
        n_idx = (int)idx->size(1);
        qp = cq->data_ptr<uint8_t>();
        sp = cs->data_ptr<uint8_t>();
        ip = idx->data_ptr<int>();
        qs = cq->stride(0);
        ss = cs->stride(0);
        is = idx->stride(0);
    }
    const int64_t stages = stages_of(n_idx);
    TORCH_CHECK(stages * ST <= NT * 3, "index table");
    const int nparts = (int)((stages + per - 1) / per);
    TORCH_CHECK(po.scalar_type() == at::kFloat && po.numel() >= R * nparts * H * D && pm.numel() >= R * nparts * H &&
                pl.numel() >= R * nparts * H, "partial buffers too small");
    if (R == 0) return nparts;
    const at::cuda::CUDAGuard guard(q.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream();
    static bool attr = false;
    if (!attr) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(split_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
        attr = true;
    }
    split_kernel<<<dim3(nparts, 2, (unsigned)R), NT, SMEM, st>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()), qp, sp, qs, ss, ip, is, n_idx,
        reinterpret_cast<const __nv_bfloat16*>(swa.data_ptr()), pos.data_ptr<int64_t>(), sb, (int)ring,
        po.data_ptr<float>(), pm.data_ptr<float>(), pl.data_ptr<float>(), (int)per, nparts, (float)scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto merge = nparts <= 4 ? merge_kernel<4> : merge_kernel<24>;    // (few parts: registers for occupancy)
    merge<<<dim3((unsigned)R, H), 128, 0, st>>>(
        po.data_ptr<float>(), pm.data_ptr<float>(), pl.data_ptr<float>(), sink.data_ptr<float>(),
        pos.data_ptr<int64_t>(), rope ? cosp->data_ptr<float>() : nullptr, rope ? sinp->data_ptr<float>() : nullptr,
        half, out.data_ptr(), nparts, (int)(group / per));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return nparts;
}

// the decode helpers over every byte: (bf16x2 of each nibble pair as int32 [256], bf16 bits of each e4m3 byte as
// int16 [256], whether cvt.rn.bf16x2.e2m1x2 / .e4m3x2 were compiled)
std::tuple<torch::Tensor, torch::Tensor, bool> decode_table(torch::Tensor like) {
    auto e2 = torch::empty({256}, like.options().dtype(at::kInt));
    auto e4 = torch::empty({256}, like.options().dtype(at::kShort));
    const at::cuda::CUDAGuard guard(like.device());
    tf_mqa4::table_kernel<<<1, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<uint32_t*>(e2.data_ptr()), reinterpret_cast<uint16_t*>(e4.data_ptr()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {e2, e4, TF_MQA4_CVT == 1};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attend_rows", &attend_rows);
    m.def("stages", &stages_of);
    m.def("decode_table", &decode_table);
}
