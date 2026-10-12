// Copy of src/tensorfold/families/deepseek_v41/cuda/mhc_pf.cu (blob 8615dbba9caf): sass_check.sh
#define DSV41_MHC_PF_NO_TORCH



// TF_DSV41_MHC_PF: an mHC site of a prefill segment (R >= 256 rows) as one CUDA pass over the 4 streams.

// mhc_cuda.cu's decode arithmetic, operation for operation; rows never mix, no atomics, one writer an output.

#include <cstdint>
#include <cuda_runtime.h>

namespace dsv41_mhc_pf {

constexpr int D = 5120;
constexpr int NB = 40;            // column blocks: part of the arithmetic
constexpr int CB = D / NB;        // 128 columns a block: one FMA chain
constexpr int G8 = CB / 8;        // 16-byte groups a block row
constexpr int NMIX = 24;
constexpr int BM = 16;            // rows a CTA
constexpr int NT = 128;           // threads a CTA
constexpr int IPT = BM * G8 / NT; // phase A items a thread (2)
constexpr int PB = CB + 8;        // bf16 pitch of a staged row (272 B: conflict-free 16-byte reads across 8 rows)
constexpr int PF = CB + 4;        // fp32 pitch (fp32 fn)
constexpr int PW = 32;            // floats a PART entry
constexpr int NS = 13;            // chains a thread at most
constexpr int MAXW = 8;           // ranks
static_assert(D % (NB * 8) == 0 && CB == 128 && IPT * NT == BM * G8, "the chains are 128 columns");

template <bool FN32> struct Lay {
    static constexpr int XS = 4 * BM * PB * 2;                               // new streams bf16 [stream][row][PB]
    static constexpr int CS = BM * PB * 2;                                   // collapsed row bf16 [row][PB]
    static constexpr int FNB = FN32 ? 4 * NMIX * PF * 4 : 4 * NMIX * PB * 2;  // fn [stream][mix][pitch]
    static constexpr int BYTES = XS + CS + FNB;
};

struct Args {
    const uint16_t* x; long long xs;            // streams [R, 4 D] bf16, row stride xs
    uint16_t* xout;                              // new streams [R, 4 D] (POST; may be x)
    const void* g; long long gr; int world; int gbf16;   // gathered partials [world, R, D], rank stride gr
    const float* post; const float* comb; const float* pre;   // [R, 4], [R, 16] (comb[i, j] at 4 i + j), [R, 4]
    const void* fn;                              // [24, 4 D] bf16 / fp32
    float* part; uint16_t* c;                    // [R, 4, NB, 32] fp32, the collapsed row [R, D] bf16
    uint16_t* tap; long long ts;                 // [R, D] row stride ts (or null)
    int R;
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
__device__ __forceinline__ void cp16(void* s, const void* g) {
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(s));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(a), "l"(g) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
__device__ __forceinline__ void cp_wait_all() { asm volatile("cp.async.wait_group 0;" ::: "memory"); }

// X'_i of 8 columns of row r: mhc_cuda.cu post1's expression, one rounding an operation
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

// one k8 step of the fn row (stream j, mix m): 8 weights fp32
template <bool FN32>
__device__ __forceinline__ void fn8(const unsigned char* fns, int j, int m, int k8, float (&w)[8]) {
    if (FN32) {
        const float4* p = reinterpret_cast<const float4*>(fns + ((j * NMIX + m) * PF + k8 * 8) * 4);
        const float4 a = p[0], b = p[1];
        w[0] = a.x; w[1] = a.y; w[2] = a.z; w[3] = a.w; w[4] = b.x; w[5] = b.y; w[6] = b.z; w[7] = b.w;
    } else {
        widen8(*reinterpret_cast<const uint4*>(fns + ((j * NMIX + m) * PB + k8 * 8) * 2), w);
    }
}

// POST: the post from the gathered partials; COLL 1 = c is stream 0, 2 = the pre-mix; TAP: the DSpark taps (POST)
template <bool POST, int COLL, bool FN32, bool TAP>
__global__ void __launch_bounds__(NT, 2) site_kernel(const Args a) {
    extern __shared__ __align__(16) unsigned char sm[];
    uint16_t* xs = reinterpret_cast<uint16_t*>(sm);                          // [4][BM][PB]
    uint16_t* cs = reinterpret_cast<uint16_t*>(sm + Lay<FN32>::XS);           // [BM][PB]
    unsigned char* fns = sm + Lay<FN32>::XS + Lay<FN32>::CS;                 // [4][24][PB or PF]
    const int b = blockIdx.x, r0 = blockIdx.y * BM, t = threadIdx.x, R = a.R;

    // fn's slice of block b into shared memory, in flight through phase A
    {
        constexpr int QR = FN32 ? CB / 4 : CB / 8;                           // 16-byte pieces a (stream, mix) row
        const char* f = static_cast<const char*>(a.fn);
        for (int v = t; v < 4 * NMIX * QR; v += NT) {
            const int jm = v / QR, q = v % QR, j = jm / NMIX, m = jm % NMIX;
            const long long src = static_cast<long long>(m) * 4 * D + j * D + b * CB;
            if (FN32) cp16(fns + (jm * PF + q * 4) * 4, f + (src + q * 4) * 4);
            else      cp16(fns + (jm * PB + q * 8) * 2, f + (src + q * 8) * 2);
        }
        cp_commit();
    }

    // ---- phase A: every load of the thread's items first, then the post / collapse / taps and the stores ----
    uint4 xv[IPT][4];
    float br[IPT][8];
#pragma unroll
    for (int n = 0; n < IPT; ++n) {
        const int i = t + n * NT, r = r0 + i / G8, col = b * CB + (i % G8) * 8;
        const uint16_t* xr = a.x + static_cast<long long>(r < R ? r : 0) * a.xs + col;
#pragma unroll
        for (int s = 0; s < 4; ++s) xv[n][s] = r < R ? ld_plain(xr + s * D) : make_uint4(0u, 0u, 0u, 0u);
    }
    if (POST) {
        const char* gp = static_cast<const char*>(a.g);
        const int gsz = a.gbf16 ? 2 : 4;
#pragma unroll
        for (int n = 0; n < IPT; ++n) {
            const int i = t + n * NT, r = r0 + i / G8, col = b * CB + (i % G8) * 8;
            const long long off = static_cast<long long>(r < R ? r : 0) * D + col;
            load8f(gp + off * gsz, a.gbf16, br[n]);
            for (int rk = 1; rk < a.world; ++rk) {
                float gv[8];
                load8f(gp + (rk * a.gr + off) * gsz, a.gbf16, gv);
#pragma unroll
                for (int k = 0; k < 8; ++k) br[n][k] = __fadd_rn(br[n][k], gv[k]);
            }
#pragma unroll
            for (int k = 0; k < 8; ++k) br[n][k] = bfr(br[n][k]);
        }
    }
#pragma unroll
    for (int n = 0; n < IPT; ++n) {
        const int i = t + n * NT, rl = i / G8, q = i % G8, r = r0 + rl, col = b * CB + q * 8;
        const bool ok = r < R;
        float x[4][8];
#pragma unroll
        for (int s = 0; s < 4; ++s) widen8(xv[n][s], x[s]);
        if (POST && ok) {
            float y[4][8];
#pragma unroll
            for (int s = 0; s < 4; ++s) post1(a, r, s, br[n], x, y[s]);
            uint16_t* xo = a.xout + static_cast<long long>(r) * 4 * D + col;
#pragma unroll
            for (int s = 0; s < 4; ++s) *reinterpret_cast<uint4*>(xo + s * D) = pack8(y[s]);
#pragma unroll
            for (int s = 0; s < 4; ++s)
#pragma unroll
                for (int k = 0; k < 8; ++k) x[s][k] = y[s][k];
        }
        float cv[8];
        if (COLL == 2 && ok) {
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
        if (ok) {
            *reinterpret_cast<uint4*>(a.c + static_cast<long long>(r) * D + col) = cp;
            if (POST && TAP) {
                float tp[8];
#pragma unroll
                for (int k = 0; k < 8; ++k)
                    tp[k] = bfr(__fmul_rn(__fadd_rn(__fadd_rn(__fadd_rn(x[0][k], x[1][k]), x[2][k]), x[3][k]), 0.25f));
                *reinterpret_cast<uint4*>(a.tap + static_cast<long long>(r) * a.ts + col) = pack8(tp);
            }
        }
#pragma unroll
        for (int s = 0; s < 4; ++s) *reinterpret_cast<uint4*>(xs + (s * BM + rl) * PB + q * 8) = pack8(x[s]);
        *reinterpret_cast<uint4*>(cs + rl * PB + q * 8) = cp;
    }
    cp_wait_all();
    __syncthreads();

    // ---- phase B: chains of row rl, stream j: h 0 mixes 0-12; h 1 mixes 13-23 + square sums (24; 25 at stream 0)
    const int rl = t % BM, j = (t / BM) % 4, h = t / (4 * BM), r = r0 + rl;
    const bool coll = h == 1 && j == 0;
    const uint16_t* vrow = xs + (j * BM + rl) * PB;
    const uint16_t* crow = cs + rl * PB;
    float acc[NS];
#pragma unroll
    for (int s = 0; s < NS; ++s) acc[s] = 0.0f;
    if (h == 0) {
#pragma unroll 1
        for (int k8 = 0; k8 < G8; ++k8) {
            float v[8];
            widen8(*reinterpret_cast<const uint4*>(vrow + k8 * 8), v);
#pragma unroll
            for (int s = 0; s < 13; ++s) {
                float w[8];
                fn8<FN32>(fns, j, s, k8, w);
#pragma unroll
                for (int k = 0; k < 8; ++k) acc[s] = __fmaf_rn(v[k], w[k], acc[s]);
            }
        }
    } else {
#pragma unroll 1
        for (int k8 = 0; k8 < G8; ++k8) {
            float v[8];
            widen8(*reinterpret_cast<const uint4*>(vrow + k8 * 8), v);
#pragma unroll
            for (int s = 0; s < 11; ++s) {
                float w[8];
                fn8<FN32>(fns, j, 13 + s, k8, w);
#pragma unroll
                for (int k = 0; k < 8; ++k) acc[s] = __fmaf_rn(v[k], w[k], acc[s]);
            }
#pragma unroll
            for (int k = 0; k < 8; ++k) acc[11] = __fmaf_rn(v[k], v[k], acc[11]);
            if (coll) {
                float cvv[8];
                widen8(*reinterpret_cast<const uint4*>(crow + k8 * 8), cvv);
#pragma unroll
                for (int k = 0; k < 8; ++k) acc[12] = __fmaf_rn(cvv[k], cvv[k], acc[12]);
            }
        }
    }
    if (r < R) {
        float* pr = a.part + ((static_cast<long long>(r) * 4 + j) * NB + b) * PW;
        if (h == 0) {
#pragma unroll
            for (int s = 0; s < 13; ++s) pr[s] = acc[s];
        } else {
#pragma unroll
            for (int s = 0; s < 12; ++s) pr[13 + s] = acc[s];             // mixes 13-23, [24] the square sum
            if (coll) pr[25] = acc[12];
        }
    }
}

template <bool POST, int COLL, bool FN32, bool TAP>
cudaError_t launch_one(const Args& a, cudaStream_t stream) {
    static bool attr = false;                    // > 48 KiB of dynamic shared memory needs the opt-in, once
    if (!attr) {
        const cudaError_t e = cudaFuncSetAttribute(site_kernel<POST, COLL, FN32, TAP>,
                                                   cudaFuncAttributeMaxDynamicSharedMemorySize, Lay<FN32>::BYTES);
        if (e != cudaSuccess) return e;
        attr = true;
    }
    const dim3 grid(NB, (a.R + BM - 1) / BM);
    site_kernel<POST, COLL, FN32, TAP><<<grid, NT, Lay<FN32>::BYTES, stream>>>(a);
    return cudaGetLastError();
}

// mode: 0 boundary (POST, pre-mix collapse), 1 site from stream 0, 2 site with a carried pre-mix
inline cudaError_t dispatch(const Args& a, int mode, bool fn32, bool tap, cudaStream_t stream) {
    if (a.R < 1 || a.world < 1 || a.world > MAXW || (tap && mode != 0)) return cudaErrorInvalidValue;
    switch (mode * 4 + (fn32 ? 2 : 0) + (tap ? 1 : 0)) {
    case 0: return launch_one<true, 2, false, false>(a, stream);
    case 1: return launch_one<true, 2, false, true>(a, stream);
    case 2: return launch_one<true, 2, true, false>(a, stream);
    case 3: return launch_one<true, 2, true, true>(a, stream);
    case 4: return launch_one<false, 1, false, false>(a, stream);
    case 6: return launch_one<false, 1, true, false>(a, stream);
    case 8: return launch_one<false, 2, false, false>(a, stream);
    case 10: return launch_one<false, 2, true, false>(a, stream);
    default: return cudaErrorInvalidValue;
    }
}

}  // namespace dsv41_mhc_pf

#ifndef DSV41_MHC_PF_NO_TORCH    // the compile test builds the kernels alone (nvcc, no torch headers)
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

namespace {
template <typename T> T* ptr(const at::Tensor& t) { return t.numel() ? static_cast<T*>(t.data_ptr()) : nullptr; }
}

void dsv41_mhc_pf_run_cuda(const at::Tensor& x, at::Tensor& xout, const at::Tensor& g, const at::Tensor& post,
                           const at::Tensor& comb, const at::Tensor& pre, const at::Tensor& fn, at::Tensor& part,
                           at::Tensor& c, at::Tensor& tap, int64_t R, int64_t mode) {
    using namespace dsv41_mhc_pf;
    Args a{};
    a.x = ptr<const uint16_t>(x); a.xs = x.stride(0);
    a.xout = ptr<uint16_t>(xout);
    a.g = g.numel() ? g.data_ptr() : nullptr; a.gr = g.numel() ? g.stride(0) : 0;
    a.world = g.numel() ? static_cast<int>(g.size(0)) : 1;
    a.gbf16 = g.scalar_type() == at::kBFloat16 ? 1 : 0;
    a.post = ptr<const float>(post); a.comb = ptr<const float>(comb); a.pre = ptr<const float>(pre);
    a.fn = fn.data_ptr();
    a.part = ptr<float>(part); a.c = ptr<uint16_t>(c);
    a.tap = ptr<uint16_t>(tap); a.ts = tap.numel() ? tap.stride(0) : 0;
    a.R = static_cast<int>(R);
    const bool fn32 = fn.scalar_type() == at::kFloat;
    C10_CUDA_CHECK(dispatch(a, static_cast<int>(mode), fn32, tap.numel() > 0, at::cuda::getCurrentCUDAStream()));
}
#endif

// The instantiations dispatch() launches: (mode, fn32, tap) as its switch lists them.
template __global__ void dsv41_mhc_pf::site_kernel<true, 2, false, false>(const dsv41_mhc_pf::Args);
template __global__ void dsv41_mhc_pf::site_kernel<true, 2, false, true>(const dsv41_mhc_pf::Args);
template __global__ void dsv41_mhc_pf::site_kernel<true, 2, true, false>(const dsv41_mhc_pf::Args);
template __global__ void dsv41_mhc_pf::site_kernel<true, 2, true, true>(const dsv41_mhc_pf::Args);
template __global__ void dsv41_mhc_pf::site_kernel<false, 1, false, false>(const dsv41_mhc_pf::Args);
template __global__ void dsv41_mhc_pf::site_kernel<false, 1, true, false>(const dsv41_mhc_pf::Args);
template __global__ void dsv41_mhc_pf::site_kernel<false, 2, false, false>(const dsv41_mhc_pf::Args);
template __global__ void dsv41_mhc_pf::site_kernel<false, 2, true, false>(const dsv41_mhc_pf::Args);
