// Copy of src/tensorfold/families/deepseek_v41/cuda/mhc_cuda.cu (blob caa24ba4bb72): sass_check.sh
#define DSV41_MHC_NO_TORCH



// TF_DSV41_MHC_CUDA=1: one mHC boundary (post + site + finish + next RMSNorm) as ONE launch for decode rows (R <= 16).

// Triton's arithmetic operation for operation (same bits); the last CTA's finish reads every partial, so order is moot.

#include <cstdint>
#include <cuda_runtime.h>

namespace dsv41_mhc {

constexpr int D = 5120;
constexpr int NB = 40;            // column blocks: part of the arithmetic
constexpr int CB = D / NB;        // 128 columns a block: one FMA chain
constexpr int G8 = CB / 8;        // 16-byte groups a block row
constexpr int NMIX = 24;
constexpr int MAXR = 16;
constexpr int NT = 128;           // threads a CTA
constexpr int PB = CB + 8;        // bf16 pitch of a staged row (272 B: conflict-free 16-byte reads)
constexpr int PF = CB + 4;        // fp32 pitch (fp32 fn)
constexpr int NE = 4 * NB;        // partial entries a row, (stream, block) order
constexpr int PW = 32;            // floats an entry
constexpr int NQ = 7;             // float4 groups of an entry the finish reads: [0, 28)
constexpr int STAGE4 = 640;       // float4 a finish stage (10 KiB; 2 stages)
constexpr int ROWB = 128;         // dynamic shared memory a row (the finish's row sums; nsys reads R from it)
static_assert(D % (NB * 8) == 0 && CB == 128, "the chains are 128 columns");

struct Args {
    const uint16_t* x; long long xs;            // streams [R, 4 D] bf16, row stride xs
    uint16_t* xout;                              // new streams [R, 4 D] (POST)
    const void* g; long long gr; int world; int gbf16;   // gathered partials [world, R, D], rank stride gr
    const float* post; const float* comb; const float* pre;   // [R, 4], [R, 16] (comb[i, j] at 4 i + j), [R, 4]
    const void* fn; const float* base; const float* scale;     // [24, 4 D] bf16 / fp32, [24], [3]
    float* part; uint16_t* c;                    // [R, 4, NB, 32] fp32, the collapsed row [R, D] bf16
    uint16_t* tap; long long ts;                 // [R, D] row stride ts (or null)
    const void* nw; int nwbf16;                  // norm weight [D] bf16 / fp32
    uint16_t* out;                               // the next sublayer's input [R, D] bf16
    float* opre; float* opost; float* ocomb;     // the next site's coefficients (MIX)
    int* cnt;                                    // [0] the ticket, [1] the tail's items, [2] exits (self-cleaning)
    int R;
    float eps, hc_eps, post_alpha; int iters;
    long long spin;                              // TAIL: SM cycles a CTA waits for the last arrival (0: it leaves)
    int defer;                                   // TAIL: the coefficient items run in coef_kernel (a side stream)
};

__device__ __forceinline__ uint4 ld_stream(const void* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
__device__ __forceinline__ uint4 ld_plain(const void* p) { return *reinterpret_cast<const uint4*>(p); }
__device__ __forceinline__ void widen8(const uint4 v, float (&f)[8]) {     // 8 bf16, column ascending -> fp32
    f[0] = __uint_as_float(v.x << 16); f[1] = __uint_as_float(v.x & 0xffff0000u);
    f[2] = __uint_as_float(v.y << 16); f[3] = __uint_as_float(v.y & 0xffff0000u);
    f[4] = __uint_as_float(v.z << 16); f[5] = __uint_as_float(v.z & 0xffff0000u);
    f[6] = __uint_as_float(v.w << 16); f[7] = __uint_as_float(v.w & 0xffff0000u);
}
__device__ __forceinline__ uint32_t bfbits(float f) {      // cvt.rn.bf16.f32: Triton's .to(tl.bfloat16)
    unsigned short h;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(h) : "f"(f));
    return h;
}
__device__ __forceinline__ float bfr(float f) { return __uint_as_float(bfbits(f) << 16); }
__device__ __forceinline__ uint4 pack8(const float (&f)[8]) {               // values already bf16: exact
    return make_uint4(bfbits(f[0]) | (bfbits(f[1]) << 16), bfbits(f[2]) | (bfbits(f[3]) << 16),
                      bfbits(f[4]) | (bfbits(f[5]) << 16), bfbits(f[6]) | (bfbits(f[7]) << 16));
}
__device__ __forceinline__ void load8f(const void* p, bool bf, float (&f)[8]) {   // 8 bf16 or 8 fp32, read-only
    if (bf) {
        widen8(ld_stream(p), f);
    } else {
        const float4 a = __ldg(reinterpret_cast<const float4*>(p)), b = __ldg(reinterpret_cast<const float4*>(p) + 1);
        f[0] = a.x; f[1] = a.y; f[2] = a.z; f[3] = a.w; f[4] = b.x; f[5] = b.y; f[6] = b.z; f[7] = b.w;
    }
}
__device__ __forceinline__ float ex2a(float x) {           // tl.exp's ex2.approx.f32 (not .ftz)
    float y;
    asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
__device__ __forceinline__ float divf(float a, float b) {  // Triton's fp32 '/': div.full.f32
    float y;
    asm("div.full.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
    return y;
}
__device__ __forceinline__ void cp16(void* s, const void* g) {
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(s));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(a), "l"(g) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory");
}

// X'_i of 8 columns: the post expression, one rounding an operation
__device__ __forceinline__ void post1(const Args& a, int r, int i, const float (&br)[8], const float (&x)[4][8],
                                      float (&y)[8]) {
    const float p = __ldg(a.post + r * 4 + i);
    const float c0 = __ldg(a.comb + r * 16 + i), c1 = __ldg(a.comb + r * 16 + 4 + i);
    const float c2 = __ldg(a.comb + r * 16 + 8 + i), c3 = __ldg(a.comb + r * 16 + 12 + i);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const float m = __fadd_rn(__fadd_rn(__fadd_rn(__fmul_rn(c0, x[0][k]), __fmul_rn(c1, x[1][k])),
                                            __fmul_rn(c2, x[2][k])), __fmul_rn(c3, x[3][k]));
        y[k] = bfr(__fadd_rn(__fmul_rn(p, br[k]), m));
    }
}

template <bool FN32> struct Smem {
    static constexpr int MAIN = 2 * MAXR * PB * 2 + (FN32 ? NMIX * PF * 4 : NMIX * PB * 2);
    static constexpr int BYTES = MAIN > 2 * STAGE4 * 16 ? MAIN : 2 * STAGE4 * 16;
};

// phase B: NS chains a thread at once (chain c = t + s NT -> row c / K, kind c % K; kind 25 = the collapsed row)
template <int NS, bool FN32, bool MIX>
__device__ __forceinline__ void chains(const Args& a, const uint16_t* xs, const uint16_t* cs, const void* fns, int K,
                                       int nch, int b, int jp, int t) {
    const uint16_t* src[NS];
    const void* wr[NS];
    int row[NS], kind[NS];
    float acc[NS];
#pragma unroll
    for (int s = 0; s < NS; ++s) {
        const int cc = min(t + s * NT, nch - 1);
        row[s] = cc / K;
        kind[s] = MIX ? cc % K : 25;
        src[s] = (kind[s] == 25 ? cs : xs) + row[s] * PB;
        wr[s] = kind[s] < NMIX ? (FN32 ? static_cast<const void*>(static_cast<const float*>(fns) + kind[s] * PF)
                                       : static_cast<const void*>(static_cast<const uint16_t*>(fns) + kind[s] * PB))
                               : static_cast<const void*>(src[s]);
        acc[s] = 0.0f;
    }
#pragma unroll
    for (int k8 = 0; k8 < G8; ++k8) {
#pragma unroll
        for (int s = 0; s < NS; ++s) {
            float v[8], w[8];
            widen8(*reinterpret_cast<const uint4*>(src[s] + k8 * 8), v);
            if (FN32 && kind[s] < NMIX) {
                const float4 w0 = reinterpret_cast<const float4*>(wr[s])[2 * k8];
                const float4 w1 = reinterpret_cast<const float4*>(wr[s])[2 * k8 + 1];
                w[0] = w0.x; w[1] = w0.y; w[2] = w0.z; w[3] = w0.w; w[4] = w1.x; w[5] = w1.y; w[6] = w1.z; w[7] = w1.w;
            } else {
                widen8(reinterpret_cast<const uint4*>(wr[s])[k8], w);
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[s] = __fmaf_rn(v[i], w[i], acc[s]);
        }
    }
#pragma unroll
    for (int s = 0; s < NS; ++s)
        if (t + s * NT < nch) a.part[((row[s] * 4 + jp) * NB + b) * PW + kind[s]] = acc[s];
}

// row r's coefficients on 16 lanes (comb (i, j) at lane bits 3..2, 1..0): _finish_k's ops and butterfly order
__device__ __forceinline__ void coefs_of(const Args& a, const float* mx, int r, bool store, int e) {
    const float rinv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(mx[24], 4.0f * D), a.eps)));
    const float LOG2E = __int_as_float(0x3FB8AA3B), NLOG2E = __int_as_float(0xBFB8AA3B);
    if (e < 4) {
        const float pl = __fadd_rn(__fmul_rn(__fmul_rn(mx[e], rinv), __ldg(a.scale + 0)), __ldg(a.base + e));
        const float ql = __fadd_rn(__fmul_rn(__fmul_rn(mx[4 + e], rinv), __ldg(a.scale + 1)), __ldg(a.base + 4 + e));
        const float pre = __fadd_rn(a.hc_eps, divf(1.0f, __fadd_rn(ex2a(__fmul_rn(pl, NLOG2E)), 1.0f)));
        const float post = __fmul_rn(a.post_alpha, divf(1.0f, __fadd_rn(ex2a(__fmul_rn(ql, NLOG2E)), 1.0f)));
        if (store) { a.opre[r * 4 + e] = pre; a.opost[r * 4 + e] = post; }
    }
    const float cl = __fadd_rn(__fmul_rn(__fmul_rn(mx[8 + e], rinv), __ldg(a.scale + 2)), __ldg(a.base + 8 + e));
    float mxv = fmaxf(cl, __shfl_xor_sync(0xffffffffu, cl, 2));
    mxv = fmaxf(mxv, __shfl_xor_sync(0xffffffffu, mxv, 1));
    const float ce = ex2a(__fmul_rn(__fsub_rn(cl, mxv), LOG2E));
    float s = __fadd_rn(ce, __shfl_xor_sync(0xffffffffu, ce, 2));
    s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, 1));
    float cm = __fadd_rn(a.hc_eps, divf(ce, s));
    s = __fadd_rn(cm, __shfl_xor_sync(0xffffffffu, cm, 8));
    s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, 4));
    cm = divf(cm, __fadd_rn(a.hc_eps, s));
    for (int it = 1; it < a.iters; ++it) {
        s = __fadd_rn(cm, __shfl_xor_sync(0xffffffffu, cm, 2));
        s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, 1));
        cm = divf(cm, __fadd_rn(a.hc_eps, s));
        s = __fadd_rn(cm, __shfl_xor_sync(0xffffffffu, cm, 8));
        s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, 4));
        cm = divf(cm, __fadd_rn(a.hc_eps, s));
    }
    if (store) a.ocomb[r * 16 + e] = cm;
}
__device__ __forceinline__ void coefs(const Args& a, const float* rowv, int r, bool store, int e) {
    coefs_of(a, rowv + r * (ROWB / 4), r, store, e);
}

__host__ __device__ constexpr int chunk_entries(int R) {      // partial entries a finish stage holds a row
    return R <= 1 ? 80 : R <= 2 ? 40 : R <= 4 ? 20 : R <= 9 ? 10 : 5;
}

// the last CTA: row sums (F1), then the coefficients (F3) beside the normed input (F2)
template <bool MIX>
__device__ void finish(const Args& a, float* rowv, float4* st, int t) {
    const int R = a.R;
    if (MIX) {          // F1: 7 float4 sums a row, a thread each; partials staged through 2 cp.async stages
        const int ch = chunk_entries(R), nck = NE / ch, per = R * ch * NQ;
        const int ri = t / NQ, qi = t % NQ;
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        float ssc = 0.0f;
        auto issue = [&](int c) {
            float4* dst = st + (c & 1) * STAGE4;
            for (int i = t; i < per; i += NT) {
                const int r = i / (ch * NQ), rem = i % (ch * NQ), e = rem / NQ, q = rem % NQ;
                cp16(dst + i, a.part + (static_cast<long long>(r) * NE + c * ch + e) * PW + q * 4);
            }
            cp_commit();
        };
        issue(0);
        if (nck > 1) issue(1);
        for (int c = 0; c < nck; ++c) {
            if (c + 1 < nck) cp_wait<1>(); else cp_wait<0>();
            __syncthreads();
            if (t < R * NQ) {
                const float4* src = st + (c & 1) * STAGE4 + ri * ch * NQ + qi;
#pragma unroll 5
                for (int e = 0; e < ch; ++e) {
                    const float4 v = src[e * NQ];
                    acc.x = __fadd_rn(acc.x, v.x);
                    if (qi < 6) {
                        acc.y = __fadd_rn(acc.y, v.y); acc.z = __fadd_rn(acc.z, v.z); acc.w = __fadd_rn(acc.w, v.w);
                    } else if (c * ch + e < NB) {
                        ssc = __fadd_rn(ssc, v.y);            // [25]: stream 0's blocks only
                    }
                }
            }
            __syncthreads();
            if (c + 2 < nck) issue(c + 2);
        }
        if (t < R * NQ) {
            float* rv = rowv + ri * (ROWB / 4);
            if (qi < 6) {
                rv[4 * qi] = acc.x; rv[4 * qi + 1] = acc.y; rv[4 * qi + 2] = acc.z; rv[4 * qi + 3] = acc.w;
            } else {
                rv[24] = acc.x;
                rv[25] = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(ssc, 1.0f * D), a.eps)));
            }
        }
    } else if (t < R) {  // final: the collapsed row's square sum alone
        const float* p = a.part + static_cast<long long>(t) * NE * PW + 25;
        float v[NB];
#pragma unroll
        for (int e = 0; e < NB; ++e) v[e] = __ldcg(p + e * PW);
        float ssc = 0.0f;
#pragma unroll
        for (int e = 0; e < NB; ++e) ssc = __fadd_rn(ssc, v[e]);
        rowv[t * (ROWB / 4) + 25] = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(ssc, 1.0f * D), a.eps)));
    }
    __syncthreads();
    const int warp = t >> 5, lane = t & 31;
    const int f3w = MIX ? min(4, (R + 1) / 2) : 0;          // warps with coefficient rows (2 rows a warp a pass)
    if (warp < f3w) {
#pragma unroll 1
        for (int p = 0; p < 2 && warp * 2 + 8 * p < R; ++p) {
            const int r = warp * 2 + (lane >> 4) + 8 * p;
            coefs(a, rowv, min(r, R - 1), r < R, lane & 15);
        }
    }
    const bool split = f3w < 4;                              // F2 beside F3 on the other warps, else after it
    if (split && warp < f3w) return;
    const int wid = split ? t - f3w * 32 : t, nwk = split ? NT - f3w * 32 : NT;
    const bool nwbf = a.nwbf16 != 0;
#pragma unroll 4
    for (int i = wid; i < R * (D / 8); i += nwk) {
        const int r = i / (D / 8), d0 = (i % (D / 8)) * 8;
        const float rn = rowv[r * (ROWB / 4) + 25];
        float c[8], w[8], o[8];
        widen8(__ldcg(reinterpret_cast<const uint4*>(a.c + static_cast<long long>(r) * D + d0)), c);
        load8f(static_cast<const char*>(a.nw) + d0 * (nwbf ? 2 : 4), nwbf, w);
#pragma unroll
        for (int k = 0; k < 8; ++k) o[k] = __fmul_rn(__fmul_rn(c[k], rn), w[k]);
        *reinterpret_cast<uint4*>(a.out + static_cast<long long>(r) * D + d0) = pack8(o);
    }
}

#include "mhc_tail.cuh"

// DEFER: a row's coefficient item on its own CTA, on a side stream that joins before the next boundary
__global__ void __launch_bounds__(NT) coef_kernel(const Args a) {
    __shared__ __align__(16) float4 st[NE * NQ];
    __shared__ float mx[32];
    coef_item(a, blockIdx.x, st, mx, threadIdx.x);
}

// POST: post from gathered partials; COLL 1 / 2: c = stream 0 / pre-mix; MIX: next site; ALL4: a CTA writes 4 streams
template <bool POST, int COLL, bool MIX, bool ALL4, bool FN32, bool TAIL = false>
__global__ void __launch_bounds__(NT, 4) boundary_kernel(const Args a) {
    extern __shared__ float4 dyn[];                          // R x ROWB: the finish's row sums
    __shared__ __align__(16) unsigned char sm[Smem<FN32>::BYTES];
    __shared__ int last;
    const int b = blockIdx.x, j = ALL4 ? 0 : blockIdx.y, t = threadIdx.x, R = a.R;
    const bool lead = j == 0;
    uint16_t* xs = reinterpret_cast<uint16_t*>(sm);          // stream j's new values: the chains' input
    uint16_t* cs = xs + MAXR * PB;                           // the collapsed row (lead)
    void* fns = cs + MAXR * PB;                              // this CTA's fn columns
    constexpr int FV = FN32 ? 6 : 3;                         // 16-byte fn vectors a thread
    uint4 fr[FV];
    if (MIX) {                                               // in flight through phase A
#pragma unroll
        for (int i = 0; i < FV; ++i) {
            const int v = t + i * NT, m = v / (FN32 ? 32 : 16), q = v % (FN32 ? 32 : 16);
            fr[i] = ld_stream(static_cast<const char*>(a.fn) +
                              (static_cast<long long>(m) * 4 * D + j * D + b * CB) * (FN32 ? 4 : 2) + q * 16);
        }
    }
    // ---- phase A: the post, the collapsed row, the taps; 8 columns a thread a step, every row ----
    for (int w = t; w < R * G8; w += NT) {
        const int r = w / G8, q = w % G8, col = b * CB + q * 8;
        const uint16_t* xr = a.x + static_cast<long long>(r) * a.xs + col;
        float x[4][8], vj[8];
        if (POST || lead) {
#pragma unroll
            for (int s = 0; s < 4; ++s) widen8(ld_plain(xr + s * D), x[s]);
        }
        if (POST) {
            float br[8], gv[8];
            const char* gp = static_cast<const char*>(a.g);
            const int gsz = a.gbf16 ? 2 : 4;
            load8f(gp + (static_cast<long long>(r) * D + col) * gsz, a.gbf16, br);
            for (int rk = 1; rk < a.world; ++rk) {
                load8f(gp + (rk * a.gr + static_cast<long long>(r) * D + col) * gsz, a.gbf16, gv);
#pragma unroll
                for (int k = 0; k < 8; ++k) br[k] = __fadd_rn(br[k], gv[k]);
            }
#pragma unroll
            for (int k = 0; k < 8; ++k) br[k] = bfr(br[k]);
            uint16_t* xo = a.xout + static_cast<long long>(r) * 4 * D + col;
            if (lead) {                                      // every stream's X' (the collapse, the taps)
                float y[4][8];
#pragma unroll
                for (int i = 0; i < 4; ++i) post1(a, r, i, br, x, y[i]);
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    if (ALL4 || i == 0) *reinterpret_cast<uint4*>(xo + i * D) = pack8(y[i]);
#pragma unroll
                for (int s = 0; s < 4; ++s)
#pragma unroll
                    for (int k = 0; k < 8; ++k) x[s][k] = y[s][k];
#pragma unroll
                for (int k = 0; k < 8; ++k) vj[k] = y[0][k];
            } else {
                post1(a, r, j, br, x, vj);
                *reinterpret_cast<uint4*>(xo + j * D) = pack8(vj);
            }
        } else if (lead) {
#pragma unroll
            for (int k = 0; k < 8; ++k) vj[k] = x[0][k];
        } else {
            widen8(ld_plain(xr + j * D), vj);
        }
        if (MIX) *reinterpret_cast<uint4*>(xs + r * PB + q * 8) = pack8(vj);
        if (lead) {
            float cv[8];
            if (COLL == 2) {
                const float q0 = __ldg(a.pre + r * 4), q1 = __ldg(a.pre + r * 4 + 1);
                const float q2 = __ldg(a.pre + r * 4 + 2), q3 = __ldg(a.pre + r * 4 + 3);
#pragma unroll
                for (int k = 0; k < 8; ++k)
                    cv[k] = bfr(__fadd_rn(__fadd_rn(__fadd_rn(__fmul_rn(q0, x[0][k]), __fmul_rn(q1, x[1][k])),
                                                    __fmul_rn(q2, x[2][k])), __fmul_rn(q3, x[3][k])));
            } else {
#pragma unroll
                for (int k = 0; k < 8; ++k) cv[k] = x[0][k];
            }
            const uint4 cp = pack8(cv);
            *reinterpret_cast<uint4*>(a.c + static_cast<long long>(r) * D + col) = cp;
            *reinterpret_cast<uint4*>(cs + r * PB + q * 8) = cp;
            if (POST && a.tap != nullptr) {
                float tp[8];
#pragma unroll
                for (int k = 0; k < 8; ++k)
                    tp[k] = bfr(__fmul_rn(__fadd_rn(__fadd_rn(__fadd_rn(x[0][k], x[1][k]), x[2][k]), x[3][k]), 0.25f));
                *reinterpret_cast<uint4*>(a.tap + static_cast<long long>(r) * a.ts + col) = pack8(tp);
            }
        }
    }
    if (MIX) {
#pragma unroll
        for (int i = 0; i < FV; ++i) {
            const int v = t + i * NT, m = v / (FN32 ? 32 : 16), q = v % (FN32 ? 32 : 16);
            reinterpret_cast<uint4*>(fns)[m * (FN32 ? PF / 4 : PB / 8) + q] = fr[i];
        }
    }
    __syncthreads();
    // ---- phase B: the FMA chains ----
    const int K = (MIX ? 25 : 0) + (lead ? 1 : 0);
    const int nch = R * K;
    switch ((nch + NT - 1) / NT) {
    case 1: chains<1, FN32, MIX>(a, xs, cs, fns, K, nch, b, j, t); break;
    case 2: chains<2, FN32, MIX>(a, xs, cs, fns, K, nch, b, j, t); break;
    case 3: chains<3, FN32, MIX>(a, xs, cs, fns, K, nch, b, j, t); break;
    case 4: chains<4, FN32, MIX>(a, xs, cs, fns, K, nch, b, j, t); break;
    default: break;
    }
    // ---- the ticket: the last CTA runs the finish ----
    __threadfence();
    __syncthreads();
    if (TAIL) {
        tail<MIX>(a, reinterpret_cast<float*>(dyn), reinterpret_cast<float4*>(sm), t);
        return;
    }
    if (t == 0) last = atomicAdd(a.cnt, 1) == static_cast<int>(gridDim.x * gridDim.y) - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    if (t == 0) atomicExch(a.cnt, 0);                        // self-cleaning: the next launch / replay starts at 0
    finish<MIX>(a, reinterpret_cast<float*>(dyn), reinterpret_cast<float4*>(sm), t);
}

template <bool POST, int COLL, bool MIX, bool ALL4, bool FN32>
cudaError_t launch_one(const Args& a, cudaStream_t stream, bool tail = false) {
    const dim3 grid(NB, ALL4 ? 1 : 4);
    if (tail) boundary_kernel<POST, COLL, MIX, ALL4, FN32, true><<<grid, NT, a.R * ROWB, stream>>>(a);
    else boundary_kernel<POST, COLL, MIX, ALL4, FN32><<<grid, NT, a.R * ROWB, stream>>>(a);
    return cudaGetLastError();
}

// mode: 0 boundary (POST, pre-mix collapse, MIX), 1 site from stream 0, 2 site with a carried pre-mix, 3 final
inline cudaError_t coef_launch(const Args& a, cudaStream_t stream) {
    if (a.R < 1 || a.R > MAXR) return cudaErrorInvalidValue;
    coef_kernel<<<a.R, NT, 0, stream>>>(a);
    return cudaGetLastError();
}

// tail: the parallel tail instead of the last CTA's finish (the same bits)
inline cudaError_t dispatch(const Args& a, int mode, bool fn32, cudaStream_t stream, bool tail = false) {
    if (a.R < 1 || a.R > MAXR || a.world < 1 || a.world > 8) return cudaErrorInvalidValue;
    switch (mode * 2 + (fn32 ? 1 : 0)) {
    case 0: return launch_one<true, 2, true, false, false>(a, stream, tail);
    case 1: return launch_one<true, 2, true, false, true>(a, stream, tail);
    case 2: return launch_one<false, 1, true, false, false>(a, stream, tail);
    case 3: return launch_one<false, 1, true, false, true>(a, stream, tail);
    case 4: return launch_one<false, 2, true, false, false>(a, stream, tail);
    case 5: return launch_one<false, 2, true, false, true>(a, stream, tail);
    case 6: case 7: return launch_one<true, 2, false, true, false>(a, stream, tail);
    default: return cudaErrorInvalidValue;
    }
}

}  // namespace dsv41_mhc

#ifndef DSV41_MHC_NO_TORCH       // the compile test builds the kernels alone (nvcc, no torch headers)
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

namespace {
template <typename T> T* ptr(const at::Tensor& t) { return t.numel() ? static_cast<T*>(t.data_ptr()) : nullptr; }
}

void dsv41_mhc_run_cuda(const at::Tensor& x, at::Tensor& xout, const at::Tensor& g, const at::Tensor& post,
                        const at::Tensor& comb, const at::Tensor& pre, const at::Tensor& fn, const at::Tensor& base,
                        const at::Tensor& scale, at::Tensor& part, at::Tensor& c, at::Tensor& tap,
                        const at::Tensor& nw, at::Tensor& out, at::Tensor& opre, at::Tensor& opost,
                        at::Tensor& ocomb, at::Tensor& cnt, int64_t R, int64_t mode, double eps, double hc_eps,
                        double post_alpha, int64_t iters, int64_t spin, int64_t defer) {
    using namespace dsv41_mhc;
    Args a{};
    a.x = ptr<const uint16_t>(x); a.xs = x.stride(0);
    a.xout = ptr<uint16_t>(xout);
    a.g = g.numel() ? g.data_ptr() : nullptr; a.gr = g.numel() ? g.stride(0) : 0;
    a.world = g.numel() ? static_cast<int>(g.size(0)) : 1;
    a.gbf16 = g.scalar_type() == at::kBFloat16 ? 1 : 0;
    a.post = ptr<const float>(post); a.comb = ptr<const float>(comb); a.pre = ptr<const float>(pre);
    a.fn = fn.numel() ? fn.data_ptr() : nullptr; a.base = ptr<const float>(base); a.scale = ptr<const float>(scale);
    a.part = ptr<float>(part); a.c = ptr<uint16_t>(c);
    a.tap = ptr<uint16_t>(tap); a.ts = tap.numel() ? tap.stride(0) : 0;
    a.nw = nw.data_ptr(); a.nwbf16 = nw.scalar_type() == at::kBFloat16 ? 1 : 0;
    a.out = ptr<uint16_t>(out);
    a.opre = ptr<float>(opre); a.opost = ptr<float>(opost); a.ocomb = ptr<float>(ocomb);
    a.cnt = ptr<int>(cnt);
    a.R = static_cast<int>(R);
    a.eps = static_cast<float>(eps); a.hc_eps = static_cast<float>(hc_eps);
    a.post_alpha = static_cast<float>(post_alpha); a.iters = static_cast<int>(iters);
    a.spin = spin;                                           // < 0: the last CTA's finish; >= 0: the parallel tail
    a.defer = spin >= 0 && defer ? 1 : 0;
    const bool fn32 = fn.numel() && fn.scalar_type() == at::kFloat;
    C10_CUDA_CHECK(dispatch(a, static_cast<int>(mode), fn32, at::cuda::getCurrentCUDAStream(), spin >= 0));
}

void dsv41_mhc_coef_cuda(const at::Tensor& part, const at::Tensor& base, const at::Tensor& scale, at::Tensor& opre,
                         at::Tensor& opost, at::Tensor& ocomb, int64_t R, double eps, double hc_eps, double post_alpha,
                         int64_t iters) {
    using namespace dsv41_mhc;
    Args a{};
    a.part = ptr<float>(part); a.base = ptr<const float>(base); a.scale = ptr<const float>(scale);
    a.opre = ptr<float>(opre); a.opost = ptr<float>(opost); a.ocomb = ptr<float>(ocomb);
    a.R = static_cast<int>(R);
    a.eps = static_cast<float>(eps); a.hc_eps = static_cast<float>(hc_eps);
    a.post_alpha = static_cast<float>(post_alpha); a.iters = static_cast<int>(iters);
    C10_CUDA_CHECK(coef_launch(a, at::cuda::getCurrentCUDAStream()));
}

std::vector<int64_t> dsv41_mhc_info_cuda(int64_t device) {   // what the GPU offers for cross-CTA reductions
    int cl = 0, sms = 0, optin = 0, maxcl = 0;
    cudaDeviceGetAttribute(&cl, cudaDevAttrClusterLaunch, static_cast<int>(device));
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, static_cast<int>(device));
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, static_cast<int>(device));
    if (cl) {
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = dim3(dsv41_mhc::NB, 4);
        cfg.blockDim = dim3(dsv41_mhc::NT);
        cfg.dynamicSmemBytes = dsv41_mhc::MAXR * dsv41_mhc::ROWB;
        cudaFuncSetAttribute(dsv41_mhc::boundary_kernel<true, 2, true, false, false>,
                             cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
        if (cudaOccupancyMaxPotentialClusterSize(&maxcl, dsv41_mhc::boundary_kernel<true, 2, true, false, false>,
                                                 &cfg) != cudaSuccess) {
            maxcl = -1;
            cudaGetLastError();
        }
    }
    int blocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, dsv41_mhc::boundary_kernel<true, 2, true, false, false>,
                                                  dsv41_mhc::NT, dsv41_mhc::MAXR * dsv41_mhc::ROWB);
    return {cl, maxcl, sms, optin, blocks};
}
#endif

// The instantiations dispatch() launches: modes 0-2 at bf16 / fp32 fn, mode 3 at bf16; tail off and on.
template __global__ void dsv41_mhc::boundary_kernel<true, 2, true, false, false, false>(const dsv41_mhc::Args);
template __global__ void dsv41_mhc::boundary_kernel<true, 2, true, false, false, true>(const dsv41_mhc::Args);
template __global__ void dsv41_mhc::boundary_kernel<true, 2, true, false, true, false>(const dsv41_mhc::Args);
template __global__ void dsv41_mhc::boundary_kernel<true, 2, true, false, true, true>(const dsv41_mhc::Args);
template __global__ void dsv41_mhc::boundary_kernel<false, 1, true, false, false, false>(const dsv41_mhc::Args);
template __global__ void dsv41_mhc::boundary_kernel<false, 1, true, false, false, true>(const dsv41_mhc::Args);
template __global__ void dsv41_mhc::boundary_kernel<false, 1, true, false, true, false>(const dsv41_mhc::Args);
template __global__ void dsv41_mhc::boundary_kernel<false, 1, true, false, true, true>(const dsv41_mhc::Args);
template __global__ void dsv41_mhc::boundary_kernel<false, 2, true, false, false, false>(const dsv41_mhc::Args);
template __global__ void dsv41_mhc::boundary_kernel<false, 2, true, false, false, true>(const dsv41_mhc::Args);
template __global__ void dsv41_mhc::boundary_kernel<false, 2, true, false, true, false>(const dsv41_mhc::Args);
template __global__ void dsv41_mhc::boundary_kernel<false, 2, true, false, true, true>(const dsv41_mhc::Args);
template __global__ void dsv41_mhc::boundary_kernel<true, 2, false, true, false, false>(const dsv41_mhc::Args);
template __global__ void dsv41_mhc::boundary_kernel<true, 2, false, true, false, true>(const dsv41_mhc::Args);
