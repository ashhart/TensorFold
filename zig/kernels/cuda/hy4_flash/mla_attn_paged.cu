// Absorbed latent MLA tensor-core kernels (`mma.m16n8k16` bf16 -> fp32; see `hy4_mla_attn.cuh` and `hy4_mla_core.cuh`).
#include "mla_attn_paged.cuh"

#include <math.h>

#include "../common/cp_async.cuh"
#include "../common/glue_math.cuh"
#include "../common/qmm_frag.cuh"

#define TSTAMP(i)

namespace hy4_mla {
namespace detail {

typedef __nv_bfloat16 bf16;
constexpr unsigned FULL = 0xffffffffu;
constexpr float SCALE = 0.0625f;  // 256^-0.5, exact

// max that keeps NaN (float64's max over a set holding NaN is NaN); fmaxf would drop it
__device__ __forceinline__ float nmax(float a, float b) { return (a != a || a > b) ? a : b; }

__device__ __forceinline__ uint32_t saddr(const void* p) { return tf_common::smem(p); }

__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], uint32_t a) { qmm_frag::ldmatrix4(r, a); }
__device__ __forceinline__ void ldsm2(uint32_t (&r)[2], uint32_t a) { qmm_frag::ldmatrix2(r, a); }
__device__ __forceinline__ void ldsm4t(uint32_t (&r)[4], uint32_t a) { qmm_frag::ldmatrix4t(r, a); }
__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    qmm_frag::mma(d, a, b0, b1);
}

// element offset of 16-byte unit u of a 576-wide row (q and KV stages)
__device__ __forceinline__ int wide(int row, int u) { return row * KW + ((u ^ (row & 7)) << 3); }
// element offset of unit u of a KT-wide P row
template <int KT>
__device__ __forceinline__ int narrow(int row, int u) {
    if constexpr (KT == 32) return row * 32 + ((u ^ ((row >> 1) & 3)) << 3);
    else return row * 16 + ((u ^ ((row >> 2) & 1)) << 3);
}

__device__ __forceinline__ bool nonfinite8(uint4 v) {
    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
    bool bad = false;
#pragma unroll
    for (int i = 0; i < 4; ++i) bad |= ((w[i] & 0x7F800000u) == 0x7F800000u) || ((w[i] & 0x7F80u) == 0x7F80u);
    return bad;
}

__device__ __forceinline__ float fin(float a, float w, float d) {
    return d == 0.f ? 0.f : __fdiv_rn(__fmul_rn(a, w), d);  // d == 0: no visible key and a -inf sink (stated rule)
}

__device__ __forceinline__ void store2(void* out, size_t i, float a, float b, int f32) {
    if (f32 & 1) *reinterpret_cast<float2*>(static_cast<float*>(out) + i) = make_float2(a, b);
    else *reinterpret_cast<__nv_bfloat162*>(static_cast<bf16*>(out) + i) = __floats2bfloat162_rn(a, b);
}

template <int MT, int KT, bool QREG, bool EXACT = false>
struct Lay {
    static constexpr int NQ = MT * 16;             // queries a CTA
    static constexpr int NTILE = MT * KT / 8;      // (m16, n8) S tiles
    static constexpr int KSPLIT = 8 / NTILE;       // warps a tile, each a k-slice
    static constexpr int KSTEPS = 36 / KSPLIT;     // k16 steps a warp
    static constexpr int SPAD = KT + 8;            // S partial row stride (floats): conflict-free float2 stores
    static constexpr int TPR = NT / NQ;            // softmax threads a row
    static constexpr int CPT = KT / TPR;           // softmax columns a thread
    static constexpr int Q = 0;
    static constexpr int QLO = Q + (QREG ? 0 : NQ * KW * 2);
    static constexpr int KV = QLO + ((EXACT && !QREG) ? NQ * KW * 2 : 0);
    static constexpr int P = KV + 2 * KT * KW * 2;
    static constexpr int PLO = P + NQ * KT * 2;
    static constexpr int RED = PLO + (EXACT ? NQ * KT * 2 : 0);
    static constexpr int KP = RED + KSPLIT * NQ * SPAD * 4;
    static constexpr int ROW = KP + 2 * KT * 4;    // [6][NQ] fp32: alpha, m, l, w, den, spare
    static constexpr int RINFO = ROW + 6 * NQ * 4; // rows-in-M: [16] positions, [16] streams
    static constexpr int MISC = RINFO + 128;
    static constexpr int BYTES = MISC + 16;
    static constexpr int SPLIT_CAP = KSPLIT * SPAD;  // the merge's weights [nsplit][NQ] reuse the S partials
    static_assert(8 % NTILE == 0 && 36 % KSPLIT == 0 && KT % TPR == 0 && TPR <= 32, "tile shape");
    static_assert(CPT == 1 || CPT % 2 == 0, "P pairs");
    static_assert(!(EXACT && QREG && MT > 1), "exact qreg holds MT=1 q hi+lo in registers");
    static_assert(BYTES == smem_bytes(MT, KT, QREG, EXACT), "smem layout and the header's byte count agree");
};

struct Args {
    const bf16* QA;
    int Hl;
    const bf16* CACHE;
    int kind, ring, ctx;
    const int* ROW_STREAM;
    const int* ROW_POS;
    const int* LIST;
    const int* TOK;
    int ldtok, ntok;
    const float* SINKS;
    int ch;
    float* PM;
    float* PL;
    float* PO;
    int* TICKET;
    void* OUT;
    int out_f32, R;
    long long QLO = 0;  // EXACT: elements from QA to the q lo plane
    const unsigned long long* ROW_CACHE = nullptr;  // batched decode: per-row MLA cache pointer table
    const unsigned int* PAGE_TABLE = nullptr;  // paged indexed KV: [stream, max_pages] u32 (256 rows/page)
    int pt_stride = 0;
};

// Split fold for one `(row, head block)` in ascending split order, shared by ticket and `hy4_mla_fold`.
template <int NQ>
__device__ __forceinline__ void fold_rows(const Args& a, size_t pb0, int nspl, int h0, int r, int c0, int c1,
                                          float* wj, int wj_cap, float* d_s, float* top_s) {
    const int tid = threadIdx.x;
    if (tid < NQ) {
        const int h = h0 + tid;
        const float sk = h < a.Hl ? a.SINKS[h] : -INFINITY;
        float top = __ldcg(a.PM + pb0 + tid);
        for (int jj = 1; jj < nspl; ++jj) top = nmax(top, __ldcg(a.PM + pb0 + static_cast<size_t>(jj) * NQ + tid));
        top = nmax(top, sk);
        top_s[tid] = top;
        float den = sk == -INFINITY ? 0.f : tf_glue::tf_exp(__fsub_rn(sk, top));
        for (int jj = 0; jj < nspl; ++jj) {
            const float mj = __ldcg(a.PM + pb0 + static_cast<size_t>(jj) * NQ + tid);
            const float w = mj == -INFINITY ? 0.f : tf_glue::tf_exp(__fsub_rn(mj, top));
            if (jj < wj_cap) wj[jj * NQ + tid] = w;
            den = __fmaf_rn(__ldcg(a.PL + pb0 + static_cast<size_t>(jj) * NQ + tid), w, den);
        }
        d_s[tid] = den;
    }
    __syncthreads();
    const int cw = (c1 - c0) / 4;  // float4 units a row
    constexpr int CH = 8;          // splits in flight a unit
    const bool cached = nspl <= wj_cap;
    for (int u = tid; u < NQ * cw; u += blockDim.x) {
        const int row = u / cw, c4 = c0 + (u % cw) * 4;
        const int h = h0 + row;
        if (h >= a.Hl) continue;
        const float* src = a.PO + (pb0 + row) * KL + c4;
        const float top = top_s[row];
        float o0 = 0.f, o1 = 0.f, o2 = 0.f, o3 = 0.f;
        for (int j0 = 0; j0 < nspl; j0 += CH) {
            float4 v[CH];
#pragma unroll
            for (int jj = 0; jj < CH; ++jj)
                if (j0 + jj < nspl)
                    v[jj] = __ldcg(reinterpret_cast<const float4*>(src + static_cast<size_t>(j0 + jj) * NQ * KL));
            if (cached) {
#pragma unroll
                for (int jj = 0; jj < CH; ++jj)
                    if (j0 + jj < nspl) {
                        const float w = wj[(j0 + jj) * NQ + row];
                        o0 = __fmaf_rn(v[jj].x, w, o0);
                        o1 = __fmaf_rn(v[jj].y, w, o1);
                        o2 = __fmaf_rn(v[jj].z, w, o2);
                        o3 = __fmaf_rn(v[jj].w, w, o3);
                    }
            } else {
#pragma unroll
                for (int jj = 0; jj < CH; ++jj)
                    if (j0 + jj < nspl) {
                        const float mj = __ldcg(a.PM + pb0 + static_cast<size_t>(j0 + jj) * NQ + row);
                        const float w = mj == -INFINITY ? 0.f : tf_glue::tf_exp(__fsub_rn(mj, top));
                        o0 = __fmaf_rn(v[jj].x, w, o0);
                        o1 = __fmaf_rn(v[jj].y, w, o1);
                        o2 = __fmaf_rn(v[jj].z, w, o2);
                        o3 = __fmaf_rn(v[jj].w, w, o3);
                    }
            }
        }
        const float d = d_s[row];
        const size_t at = (static_cast<size_t>(r) * a.Hl + h) * KL + c4;
        store2(a.OUT, at, d == 0.f ? 0.f : __fdiv_rn(o0, d), d == 0.f ? 0.f : __fdiv_rn(o1, d), a.out_f32);
        store2(a.OUT, at + 2, d == 0.f ? 0.f : __fdiv_rn(o2, d), d == 0.f ? 0.f : __fdiv_rn(o3, d), a.out_f32);
    }
}

}  // namespace detail
}  // namespace hy4_mla

#include "mla_attn_core.cuh"
#include "mla_attn_x16_qreg.cuh"

using hy4_mla::detail::Args;
using hy4_mla::detail::attn_body;
using hy4_mla::detail::x16_qreg_body;

#define HY4_MLA_SPLIT_PACK                                                                                          \
    Args a{QA, Hl, CACHE, kind, ring, ctx, ROW_STREAM, ROW_POS, LIST, TOK, ldtok, ntok, SINKS, ch, PM, PL, PO, TICKET, \
           OUT, out_f32, R}

extern "C" __global__ void __launch_bounds__(256, 1) hy4_mla_dec_x16(HY4_MLA_SPLIT_ARGS, long long QLO) {
    HY4_MLA_SPLIT_PACK;
    a.QLO = QLO;
    attn_body<1, 16, false, false, true>(a);
}

extern "C" __global__ void __launch_bounds__(256, 1) hy4_mla_dec_x16_rows(HY4_MLA_SPLIT_ARGS, long long QLO) {
    HY4_MLA_SPLIT_PACK;
    a.QLO = QLO;
    a.ROW_CACHE = reinterpret_cast<const unsigned long long*>(CACHE);
    a.CACHE = nullptr;
    attn_body<1, 16, false, false, true>(a);
}

extern "C" __global__ void __launch_bounds__(256, 2) hy4_mla_dec_x16_qreg(HY4_MLA_SPLIT_ARGS, long long QLO) {
    HY4_MLA_SPLIT_PACK;
    a.QLO = QLO;
    x16_qreg_body(a);
}

extern "C" __global__ void __launch_bounds__(256, 2) hy4_mla_dec_x16_qreg_rows(HY4_MLA_SPLIT_ARGS, long long QLO) {
    HY4_MLA_SPLIT_PACK;
    a.QLO = QLO;
    a.ROW_CACHE = reinterpret_cast<const unsigned long long*>(CACHE);
    a.CACHE = nullptr;
    x16_qreg_body(a);
}

extern "C" __global__ void __launch_bounds__(256, 1)
    hy4_mla_dec_x16_paged(HY4_MLA_SPLIT_ARGS, long long QLO, const unsigned int* PAGE_TABLE, int pt_stride) {
    HY4_MLA_SPLIT_PACK;
    a.QLO = QLO;
    a.PAGE_TABLE = PAGE_TABLE;
    a.pt_stride = pt_stride;
    attn_body<1, 16, false, false, true>(a);
}

extern "C" __global__ void __launch_bounds__(256, 2)
    hy4_mla_dec_x16_qreg_paged(HY4_MLA_SPLIT_ARGS, long long QLO, const unsigned int* PAGE_TABLE, int pt_stride) {
    HY4_MLA_SPLIT_PACK;
    a.QLO = QLO;
    a.PAGE_TABLE = PAGE_TABLE;
    a.pt_stride = pt_stride;
    x16_qreg_body(a);
}

// Second pass of 2-launch split+fold (`TICKET = nullptr`): folds rows with `nspl > 1` (`grid (8, ceil(Hl / nq), R)`).
extern "C" __global__ void __launch_bounds__(256) hy4_mla_fold(int Hl, int nq, int kind, const int* __restrict__ ROW_POS,
                                                               int ntok, int ch, int NS, const float* __restrict__ PM,
                                                               const float* __restrict__ PL, const float* __restrict__ PO,
                                                               const float* __restrict__ SINKS, void* __restrict__ OUT,
                                                               int out_f32, int R) {
    using namespace hy4_mla;
    using namespace hy4_mla::detail;
    __shared__ float wj[128 * 32];
    __shared__ float d_s[32];
    __shared__ float top_s[32];
    qmm_frag::grid_wait();
    qmm_frag::grid_launch();
    const int s = blockIdx.x, hb = blockIdx.y, r = blockIdx.z;
    const int p = ROW_POS[r];
    const int n = kind == 0 ? min(p + 1, SWA_KEYS) : (kind == 1 ? (p >= DSA_KEYS ? DSA_KEYS : p + 1) : ntok);
    const int nspl = (n + ch - 1) / ch;
    if (nspl <= 1) return;           // finalised by the split kernel
    Args a{nullptr, Hl, nullptr, kind, 0, 0, nullptr, ROW_POS, nullptr, nullptr, 0, ntok, SINKS, ch, const_cast<float*>(PM),
           const_cast<float*>(PL), const_cast<float*>(PO), nullptr, OUT, out_f32, R};
    const size_t pb0 = (static_cast<size_t>(r) * gridDim.y + hb) * NS * nq;
    const int c0 = s * (hy4_mla::KL / gridDim.x), c1 = c0 + hy4_mla::KL / gridDim.x;
    if (nq == 32) fold_rows<32>(a, pb0, nspl, hb * nq, r, c0, c1, wj, 128, d_s, top_s);
    else fold_rows<16>(a, pb0, nspl, hb * nq, r, c0, c1, wj, 256, d_s, top_s);
}
