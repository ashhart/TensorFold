// DeepSeek-V4.1 decode attention over the NVFP4 compressed KV cache (Fp4Rows) plus the bf16 window ring: the chunk
// (split) pass of kernels.mqa for decode / verify rows and small prompt chunks. The Triton _mqa_merge combines the
// partials (sink, normalisation, inverse RoPE) exactly as for the Triton chunk kernel.
//
// SPDX-License-Identifier: Apache-2.0
// Adapted from FlashInfer's "Cake" DeepSeek-V4.1 mixed-cache decode, flashinfer-ai/flashinfer
// csrc/cake_dsv4/sm_120a/cake_sparse_mla_dsv41_mixed_h32.cu (decode_dual), commit
// 2c1c0525067452c411944fb1c40d8112040ea229 (PR #5983, generated from Cake 092b4b1f5fd8913d7c01390c7ceb6bc7e866d8e2),
// Apache License 2.0; Copyright 2025-2026 NVIDIA, Copyright 2023-2026 FlashInfer community (LICENSES/Apache-2.0.txt,
// NOTICE, THIRD_PARTY_NOTICES.md). Taken from it: one CTA scores every local head (two 16-head tiles) against one
// gathered candidate set, bf16 mma.sync m16n8k16 with the FP4 rows widened exactly to bf16 (cvt.rn.bf16x2.e2m1x2 on
// CUDA 13.2+, else its two-prmt table; e4m3 scales widened and multiplied in bf16, exact), a lane owning whole scale
// groups of dims (Q and K permuted alike), V^T read with ldmatrix.trans.
// Modified for TensorFold dsv41-cuda: hand-written loops instead of the generated unrolled code; no IO warps or
// mbarrier ring (each CTA gathers its whole split up front: decode rows are latency bound); the FP4 rows are decoded
// once into a bf16 stage shared by QK and PV; separate Fp4Rows planes (q nibbles, s e4m3 bytes) instead of paged
// footers; the bf16 window ring read directly (rows from pos, the stream's ring base and the ring length) instead of
// FP8 528-byte main rows; QK split over dims across a tile's 4 warps with the partial scores summed in a fixed order;
// a fixed split partition that depends only on the layer's index count (row invariance) instead of the planner;
// TensorFold's natural-log online softmax and partial layout (sinks applied in the merge); no lse_scale / out_lse, no
// main-only kernel, no PDL.
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
constexpr int WSPLIT = 32;      // window candidates a split (one stage)
constexpr int T_BYTES = ST * D * 2;                 // bf16 stage, swizzled 16-byte granules
constexpr int PP = 40;                              // partial-score row pitch (floats): conflict-free float2
constexpr int PART_TILE = 4 * 16 * PP * 4;
template <int TILES>
constexpr int smem_bytes() { return T_BYTES + TILES * PART_TILE + 64 * 4; }

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
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x * 1.4426950408889634f));
    return y;
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// raw FP4 bytes of one compressed stage row segment: thread (row tid / 8, seg tid % 8) owns granules seg + 8u
struct Raw {
    uint32_t n[8];          // nibbles of granule seg + 8u (8 dims each)
    uint32_t s[8];          // the row's 32 scale bytes
};

__device__ __forceinline__ void load_raw(Raw& x, const uint8_t* __restrict__ cq, const uint8_t* __restrict__ cs,
                                         int64_t qstride, int64_t sstride, int row, int seg) {
    if (row < 0) {
#pragma unroll
        for (int u = 0; u < 8; ++u) x.n[u] = x.s[u] = 0u;
        return;
    }
    const uint32_t* q = reinterpret_cast<const uint32_t*>(cq + (int64_t)row * qstride);
#pragma unroll
    for (int u = 0; u < 8; ++u) x.n[u] = __ldg(q + seg + 8 * u);
    const uint4* s = reinterpret_cast<const uint4*>(cs + (int64_t)row * sstride);
    const uint4 s0 = __ldg(s), s1 = __ldg(s + 1);
    x.s[0] = s0.x; x.s[1] = s0.y; x.s[2] = s0.z; x.s[3] = s0.w;
    x.s[4] = s1.x; x.s[5] = s1.y; x.s[6] = s1.z; x.s[7] = s1.w;
}

// decode the raw segment into stage row r: granule g = seg + 8u (dims 8g..8g+7), scale group g / 2
__device__ __forceinline__ void store_raw(uint8_t* T, const Raw& x, int r, int seg) {
#pragma unroll
    for (int u = 0; u < 8; ++u) {
        const int g = seg + 8 * u;
        const uint32_t sb = (x.s[u] >> (8 * (seg >> 1))) & 0xFFu;     // byte g / 2 = 4u + seg / 2
        const uint32_t s2 = e4m3_bf16x2(sb);
        uint32_t v[4];
        e2m1x8(x.n[u], v);
        uint4 o;
        o.x = bmul(v[0], s2); o.y = bmul(v[1], s2); o.z = bmul(v[2], s2); o.w = bmul(v[3], s2);
        *reinterpret_cast<uint4*>(T + r * 1024 + phys(g, r) * 16) = o;
    }
}

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool ok) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(ok ? 16 : 0));
}

// One CTA = one row x one split x TILES 16-head tiles (4 warps a tile: warp wt takes dims wt*128 .. +127 in QK and
// PV). Splits 0 .. ncomp-1: CS compressed candidates each (idx order), processed as CS / 32 stages; then W / 32
// window splits of 32 positions p - 127 + j. Grid (nsplit, 2 / TILES, R).
template <int CS, int TILES>
__global__ void __launch_bounds__(128 * TILES)
chunks_kernel(const __nv_bfloat16* __restrict__ Q, const uint8_t* __restrict__ CQ, const uint8_t* __restrict__ CSC,
              int64_t qstride, int64_t sstride, const int* __restrict__ IDX, int64_t idx_stride, int n_idx,
              int ncomp, const __nv_bfloat16* __restrict__ SWA, const int64_t* __restrict__ POS,
              const int64_t* __restrict__ SBASE, int ring, float* __restrict__ PO, float* __restrict__ PM,
              float* __restrict__ PL, int nsplit, float scale) {
    extern __shared__ __align__(128) uint8_t smem[];
    uint8_t* T = smem;
    float* part = reinterpret_cast<float*>(smem + T_BYTES);
    int* tab = reinterpret_cast<int*>(smem + T_BYTES + TILES * PART_TILE);
    constexpr int NT = 128 * TILES, RPT = 256 / NT;           // raw stage rows a thread
    const int k = blockIdx.x, r = blockIdx.z, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int lt = warp >> 2, tg = blockIdx.y * TILES + lt, wt = warp & 3, gr = lane >> 2, t = lane & 3;
    const bool comp = k < ncomp;
    const int nc = comp ? CS : WSPLIT;
    const int64_t p = POS[r];

    bool mine = false;
    if (tid < nc) {
        int row = -1;
        if (comp) {
            const int c = k * CS + tid;
            if (c < n_idx) row = IDX[(int64_t)r * idx_stride + c];
            row = row < 0 ? -1 : row;
        } else {
            const int64_t slot = p - (W - 1) + (int64_t)(k - ncomp) * WSPLIT + tid;
            if (slot >= 0) row = (int)((SBASE != nullptr ? SBASE[r] : 0) + (slot & (ring - 1)));
        }
        tab[tid] = row;
        mine = row >= 0;
    }
    const int h0 = tg * 16 + gr, h1 = h0 + 8;
    if (!__syncthreads_or(mine)) {               // nothing in this split: the merge skips l <= 0
        if (tid < 16 * TILES) {
            const int64_t b = ((int64_t)r * nsplit + k) * H + blockIdx.y * 16 * TILES + tid;
            PM[b] = -INFINITY;
            PL[b] = 0.f;
        }
        return;
    }

    // stage 0's gather first (window: cp.async straight into the stage; compressed: raw bytes into registers)
    Raw raw[RPT];
    const int rrow = tid >> 3, seg = tid & 7;
    if (comp) {
#pragma unroll
        for (int i = 0; i < RPT; ++i) load_raw(raw[i], CQ, CSC, qstride, sstride, tab[rrow + i * NT / 8], seg);
    } else {
        const int c = tid & 63;
#pragma unroll
        for (int i = 0; i < 2048 / NT; ++i) {
            const int rr = (tid >> 6) + (NT / 64) * i;
            const int row = tab[rr];
            cp_async16(smem_u32(T + rr * 1024 + phys(c, rr) * 16), SWA + (int64_t)(row < 0 ? 0 : row) * D + c * 8,
                       row >= 0);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    }
    // Q fragments: lane t of quarter wt owns dims wt*128 + 8(t + 4q) .. +7 (q = 0..3) of heads h0 and h1
    uint4 qa[2][4];
    {
        const uint4* q0 = reinterpret_cast<const uint4*>(Q + ((int64_t)r * H + h0) * D + wt * 128);
        const uint4* q1 = reinterpret_cast<const uint4*>(Q + ((int64_t)r * H + h1) * D + wt * 128);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            qa[0][q] = __ldg(q0 + t + 4 * q);
            qa[1][q] = __ldg(q1 + t + 4 * q);
        }
    }

    float acc[16][4];
#pragma unroll
    for (int n = 0; n < 16; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
    float* mypart = part + (lt * 4 + wt) * 16 * PP;
    const float* tpart = part + lt * 4 * 16 * PP;
    const int nstages = nc / ST;

    for (int s = 0; s < nstages; ++s) {
        if (comp) {
            if (s > 0) __syncthreads();                       // the previous stage's PV is done with T
#pragma unroll
            for (int i = 0; i < RPT; ++i) store_raw(T, raw[i], rrow + i * NT / 8, seg);
            if (s + 1 < nstages)
#pragma unroll
                for (int i = 0; i < RPT; ++i)
                    load_raw(raw[i], CQ, CSC, qstride, sstride, tab[(s + 1) * ST + rrow + i * NT / 8], seg);
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();
        const uint32_t vm = __ballot_sync(0xFFFFFFFFu, tab[s * ST + lane] >= 0);

        // QK over this warp's 128 dims: 4 candidate tiles x 8 k-steps
        float sc[4][4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            sc[j][0] = sc[j][1] = sc[j][2] = sc[j][3] = 0.f;
            const int row = 8 * j + gr;
            const uint8_t* base = T + row * 1024;
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const uint4 kb = *reinterpret_cast<const uint4*>(base + phys(wt * 16 + t + 4 * q, row) * 16);
                const uint32_t a0[4] = {qa[0][q].x, qa[1][q].x, qa[0][q].y, qa[1][q].y};
                const uint32_t a1[4] = {qa[0][q].z, qa[1][q].z, qa[0][q].w, qa[1][q].w};
                mma(sc[j], a0, kb.x, kb.y);
                mma(sc[j], a1, kb.z, kb.w);
            }
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            *reinterpret_cast<float2*>(mypart + gr * PP + 8 * j + 2 * t) = make_float2(sc[j][0], sc[j][1]);
            *reinterpret_cast<float2*>(mypart + (gr + 8) * PP + 8 * j + 2 * t) = make_float2(sc[j][2], sc[j][3]);
        }
        if (TILES == 2)
            asm volatile("bar.sync %0, 128;" ::"r"(1 + lt));
        else
            __syncthreads();
        // every warp of the tile sums the four quarters in the same order: the same scores, bit for bit
        float x[4][4];
        float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 v[4][2];
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                v[w][0] = *reinterpret_cast<const float2*>(tpart + w * 16 * PP + gr * PP + 8 * j + 2 * t);
                v[w][1] = *reinterpret_cast<const float2*>(tpart + w * 16 * PP + (gr + 8) * PP + 8 * j + 2 * t);
            }
            const int c = 8 * j + 2 * t;
            const bool ok0 = (vm >> c) & 1u, ok1 = (vm >> (c + 1)) & 1u;
            x[j][0] = ok0 ? ((v[0][0].x + v[1][0].x) + (v[2][0].x + v[3][0].x)) * scale : -INFINITY;
            x[j][1] = ok1 ? ((v[0][0].y + v[1][0].y) + (v[2][0].y + v[3][0].y)) * scale : -INFINITY;
            x[j][2] = ok0 ? ((v[0][1].x + v[1][1].x) + (v[2][1].x + v[3][1].x)) * scale : -INFINITY;
            x[j][3] = ok1 ? ((v[0][1].y + v[1][1].y) + (v[2][1].y + v[3][1].y)) * scale : -INFINITY;
            mx0 = fmaxf(mx0, fmaxf(x[j][0], x[j][1]));
            mx1 = fmaxf(mx1, fmaxf(x[j][2], x[j][3]));
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xFFFFFFFFu, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xFFFFFFFFu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xFFFFFFFFu, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xFFFFFFFFu, mx1, 2));
        const bool act0 = mx0 != -INFINITY, act1 = mx1 != -INFINITY;
        const float n0 = act0 ? fmaxf(m0, mx0) : m0, n1 = act1 ? fmaxf(m1, mx1) : m1;
        const float al0 = act0 ? (m0 == -INFINITY ? 0.f : ex(m0 - n0)) : 1.f;
        const float al1 = act1 ? (m1 == -INFINITY ? 0.f : ex(m1 - n1)) : 1.f;
        float s0 = 0.f, s1 = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            x[j][0] = x[j][0] == -INFINITY ? 0.f : ex(x[j][0] - n0);
            x[j][1] = x[j][1] == -INFINITY ? 0.f : ex(x[j][1] - n0);
            x[j][2] = x[j][2] == -INFINITY ? 0.f : ex(x[j][2] - n1);
            x[j][3] = x[j][3] == -INFINITY ? 0.f : ex(x[j][3] - n1);
            s0 += x[j][0] + x[j][1];
            s1 += x[j][2] + x[j][3];
        }
        s0 += __shfl_xor_sync(0xFFFFFFFFu, s0, 1);
        s0 += __shfl_xor_sync(0xFFFFFFFFu, s0, 2);
        s1 += __shfl_xor_sync(0xFFFFFFFFu, s1, 1);
        s1 += __shfl_xor_sync(0xFFFFFFFFu, s1, 2);
        l0 = l0 * al0 + s0;
        l1 = l1 * al1 + s1;
        m0 = n0;
        m1 = n1;
#pragma unroll
        for (int n = 0; n < 16; ++n) {
            acc[n][0] *= al0; acc[n][1] *= al0;
            acc[n][2] *= al1; acc[n][3] *= al1;
        }
        // PV: P (bf16, from the score fragments) x V^T (ldmatrix.trans) over this warp's 128 output dims
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            const uint32_t pa[4] = {pack_bf16(x[2 * kk][0], x[2 * kk][1]), pack_bf16(x[2 * kk][2], x[2 * kk][3]),
                                    pack_bf16(x[2 * kk + 1][0], x[2 * kk + 1][1]),
                                    pack_bf16(x[2 * kk + 1][2], x[2 * kk + 1][3])};
            const int mi = lane >> 3, cand = kk * 16 + (mi & 1) * 8 + (lane & 7);
            const uint8_t* rowp = T + cand * 1024;
#pragma unroll
            for (int np = 0; np < 8; ++np) {
                const int c = wt * 16 + 2 * np + (mi >> 1);
                uint32_t b0, b1, b2, b3;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3)
                             : "r"(smem_u32(rowp + phys(c, cand) * 16)));
                mma(acc[2 * np], pa, b0, b1);
                mma(acc[2 * np + 1], pa, b2, b3);
            }
        }
    }

    const int64_t b0 = ((int64_t)r * nsplit + k) * H;
    float* o0 = PO + (b0 + h0) * D + wt * 128 + 2 * t;
    float* o1 = PO + (b0 + h1) * D + wt * 128 + 2 * t;
#pragma unroll
    for (int n = 0; n < 16; ++n) {
        *reinterpret_cast<float2*>(o0 + 8 * n) = make_float2(acc[n][0], acc[n][1]);
        *reinterpret_cast<float2*>(o1 + 8 * n) = make_float2(acc[n][2], acc[n][3]);
    }
    if (wt == 0 && t == 0) {
        PM[b0 + h0] = m0;
        PL[b0 + h0] = l0;
        PM[b0 + h1] = m1;
        PL[b0 + h1] = l1;
    }
}

__global__ void table_kernel(uint32_t* e2m1, uint16_t* e4m3) {
    const int b = threadIdx.x;
    uint32_t v[4];
    e2m1x8((uint32_t)b, v);
    e2m1[b] = v[0];
    e4m3[b] = (uint16_t)(e4m3_bf16x2((uint32_t)b) & 0xFFFFu);
}

template <int CS, int TILES>
void launch(const int64_t R, int nsplit, cudaStream_t st, const __nv_bfloat16* q, const uint8_t* cq,
            const uint8_t* cs, int64_t qs, int64_t ss, const int* idx, int64_t istr, int n_idx, int ncomp,
            const __nv_bfloat16* swa, const int64_t* pos, const int64_t* sbase, int ring, float* po, float* pm,
            float* pl, float scale) {
    static bool attr = false;
    constexpr int SM = smem_bytes<TILES>();
    if (!attr) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(chunks_kernel<CS, TILES>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM));
        attr = true;
    }
    const dim3 grid(nsplit, 2 / TILES, (unsigned)R);
    chunks_kernel<CS, TILES><<<grid, 128 * TILES, SM, st>>>(q, cq, cs, qs, ss, idx, istr, n_idx, ncomp, swa, pos,
                                                           sbase, ring, po, pm, pl, nsplit, scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace tf_mqa4

int64_t nsplit_of(int64_t n_idx, int64_t split) {
    return (n_idx + split - 1) / split + tf_mqa4::W / tf_mqa4::WSPLIT;
}

// q [R, 32, 512] bf16; cq u8 [n, 256], cs u8 [n, 32] (Fp4Rows planes) and idx int32 [R, n_idx] (rows of cq, -1:
// none), or all three None (a window-only layer); swa bf16 [rows, 512] ring(s); pos int64 [R]; sbase int64 [R] (the
// row's ring's first row) or None (0); ring: rows a ring (a power of two). Writes partials for nsplit_of(n_idx) splits.
int64_t chunks(torch::Tensor q, c10::optional<torch::Tensor> cq, c10::optional<torch::Tensor> cs,
               c10::optional<torch::Tensor> idx, torch::Tensor swa, torch::Tensor pos,
               c10::optional<torch::Tensor> sbase, torch::Tensor po, torch::Tensor pm, torch::Tensor pl, int64_t ring,
               int64_t split, int64_t tiles, double scale) {
    using namespace tf_mqa4;
    TORCH_CHECK(split == 32 || split == 64, "split 32 or 64");
    TORCH_CHECK(tiles == 1 || tiles == 2, "tiles 1 or 2");
    TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kBFloat16 && q.dim() == 3 && q.size(1) == H && q.size(2) == D &&
                q.is_contiguous(), "q: contiguous bf16 [R, 32, 512]");
    const int64_t R = q.size(0);
    TORCH_CHECK(swa.scalar_type() == at::kBFloat16 && swa.dim() == 2 && swa.size(1) == D && swa.stride(1) == 1 &&
                swa.stride(0) == D && reinterpret_cast<uintptr_t>(swa.data_ptr()) % 16 == 0,
                "swa: bf16 [rows, 512] rows contiguous");
    TORCH_CHECK(ring > 0 && (ring & (ring - 1)) == 0, "ring: a power of two");
    TORCH_CHECK(pos.scalar_type() == at::kLong && pos.numel() == R && pos.is_contiguous(), "pos: int64 [R]");
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
                    cs->stride(0) % 16 == 0 && reinterpret_cast<uintptr_t>(cs->data_ptr()) % 16 == 0,
                    "cs: u8 [n, 32], 16-byte aligned rows");
        n_idx = (int)idx->size(1);
        qp = cq->data_ptr<uint8_t>();
        sp = cs->data_ptr<uint8_t>();
        ip = idx->data_ptr<int>();
        qs = cq->stride(0);
        ss = cs->stride(0);
        is = idx->stride(0);
    }
    const int ncomp = (int)((n_idx + split - 1) / split);
    const int nsplit = (int)nsplit_of(n_idx, split);
    TORCH_CHECK(po.scalar_type() == at::kFloat && po.numel() >= R * nsplit * H * D && pm.numel() >= R * nsplit * H &&
                pl.numel() >= R * nsplit * H, "partial buffers too small");
    if (R == 0) return nsplit;
    const at::cuda::CUDAGuard guard(q.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream();
    auto* qd = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr());
    auto* wd = reinterpret_cast<const __nv_bfloat16*>(swa.data_ptr());
    const int64_t* pd = pos.data_ptr<int64_t>();
    float *o = po.data_ptr<float>(), *m = pm.data_ptr<float>(), *l = pl.data_ptr<float>();
    const float sc = (float)scale;
#define TF_MQA4_GO(CS_, T_) launch<CS_, T_>(R, nsplit, st, qd, qp, sp, qs, ss, ip, is, n_idx, ncomp, wd, pd, sb, \
                                            (int)ring, o, m, l, sc)
    if (split == 64 && tiles == 2) TF_MQA4_GO(64, 2);
    else if (split == 64) TF_MQA4_GO(64, 1);
    else if (tiles == 2) TF_MQA4_GO(32, 2);
    else TF_MQA4_GO(32, 1);
#undef TF_MQA4_GO
    return nsplit;
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
    m.def("chunks", &chunks);
    m.def("nsplit", &nsplit_of);
    m.def("decode_table", &decode_table);
}
