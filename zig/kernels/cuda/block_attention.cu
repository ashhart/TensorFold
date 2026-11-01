// DFlash-style block attention for every CUDA family (float64 reference: core/attention_ref.zig): context, then block.

// Block (stream, key head): a warp a query head, its 16 mma rows the block's rows; key tiles of 32 shared by the warps.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#include "attn_odd.cuh"
#include "common/qmm_frag.cuh"
#include "common/cp_async.cuh"

namespace tf_block {

using qmm_frag::commit;
using qmm_frag::cp16z;
using qmm_frag::ldmatrix4;
using qmm_frag::ldmatrix4t;
using qmm_frag::mma;
using qmm_frag::mma0;

constexpr int D = 128, C = D / 8, K16 = D / 16, NB = D / 8, KT = 32;
constexpr float LOG2E = 1.4426950408889634f;

using tf_common::cp_wait;

__device__ __forceinline__ float texp(float x) {
    float y;
    asm("ex2.approx.f32 %0, %1;\n" : "=f"(y) : "f"(__fmul_rn(x, LOG2E)));
    return y;
}

__device__ __forceinline__ uint32_t pack(float lo, float hi) {
    __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&v);
}

__device__ __forceinline__ int sw(int r, int c) { return r * C + (c ^ (r & 7)); }

// One warp's fold over a 32-key tile: scores masked by `vis(row, key)`, online softmax, P x V (KEEP: o's inf stays).
struct Fold {
    float o[NB][4];
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;

    __device__ __forceinline__ Fold() {
#pragma unroll
        for (int i = 0; i < NB; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    }

    template <bool KEEP = false, typename Vis>
    __device__ __forceinline__ void tile(const uint32_t (&qa)[K16][4], const uint4* ks, const uint4* vs, float scale,
                                         int lane, int k0, Vis vis) {
        const int g8 = lane / 4, qd = lane % 4, mt = lane / 8;
        float sc[4][4];
#pragma unroll
        for (int k = 0; k < K16; ++k) {
#pragma unroll
            for (int n = 0; n < 4; n += 2) {
                uint32_t b[4];
                ldmatrix4(b, ks + sw(8 * (n + mt / 2) + lane % 8, 2 * k + mt % 2));
                if (k == 0) {
                    mma0(sc[n], qa[k], b[0], b[1]);
                    mma0(sc[n + 1], qa[k], b[2], b[3]);
                } else {
                    mma(sc[n], qa[k], b[0], b[1]);
                    mma(sc[n + 1], qa[k], b[2], b[3]);
                }
            }
        }
        float tm0 = -INFINITY, tm1 = -INFINITY;
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            const int c = k0 + 8 * n + 2 * qd;
            sc[n][0] = vis(g8, c) ? __fmul_rn(sc[n][0], scale) : -INFINITY;
            sc[n][1] = vis(g8, c + 1) ? __fmul_rn(sc[n][1], scale) : -INFINITY;
            sc[n][2] = vis(g8 + 8, c) ? __fmul_rn(sc[n][2], scale) : -INFINITY;
            sc[n][3] = vis(g8 + 8, c + 1) ? __fmul_rn(sc[n][3], scale) : -INFINITY;
            tm0 = fmaxf(tm0, fmaxf(sc[n][0], sc[n][1]));
            tm1 = fmaxf(tm1, fmaxf(sc[n][2], sc[n][3]));
        }
        for (int x = 1; x <= 2; x *= 2) {
            tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, x));
            tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, x));
        }
        const float n0 = fmaxf(m0, tm0), n1 = fmaxf(m1, tm1);
        const float a0 = tm0 == -INFINITY ? 1.f : (m0 == -INFINITY ? 0.f : texp(__fsub_rn(m0, n0)));
        const float a1 = tm1 == -INFINITY ? 1.f : (m1 == -INFINITY ? 0.f : texp(__fsub_rn(m1, n1)));
        float c0 = 0.f, c1 = 0.f;
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            sc[n][0] = sc[n][0] != -INFINITY ? texp(__fsub_rn(sc[n][0], n0)) : 0.f;
            sc[n][1] = sc[n][1] != -INFINITY ? texp(__fsub_rn(sc[n][1], n0)) : 0.f;
            sc[n][2] = sc[n][2] != -INFINITY ? texp(__fsub_rn(sc[n][2], n1)) : 0.f;
            sc[n][3] = sc[n][3] != -INFINITY ? texp(__fsub_rn(sc[n][3], n1)) : 0.f;
            c0 = __fadd_rn(c0, __fadd_rn(sc[n][0], sc[n][1]));
            c1 = __fadd_rn(c1, __fadd_rn(sc[n][2], sc[n][3]));
        }
        for (int x = 1; x <= 2; x *= 2) {
            c0 = __fadd_rn(c0, __shfl_xor_sync(0xffffffffu, c0, x));
            c1 = __fadd_rn(c1, __shfl_xor_sync(0xffffffffu, c1, x));
        }
        l0 = __fmaf_rn(l0, a0, c0);
        l1 = __fmaf_rn(l1, a1, c1);
        m0 = n0;
        m1 = n1;
#pragma unroll
        for (int i = 0; i < NB; ++i) {
            o[i][0] = KEEP ? attn_odd::keep(o[i][0], a0) : __fmul_rn(o[i][0], a0);
            o[i][1] = KEEP ? attn_odd::keep(o[i][1], a0) : __fmul_rn(o[i][1], a0);
            o[i][2] = KEEP ? attn_odd::keep(o[i][2], a1) : __fmul_rn(o[i][2], a1);
            o[i][3] = KEEP ? attn_odd::keep(o[i][3], a1) : __fmul_rn(o[i][3], a1);
        }
        uint32_t pa[2][4];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            pa[j][0] = pack(sc[2 * j][0], sc[2 * j][1]);
            pa[j][1] = pack(sc[2 * j][2], sc[2 * j][3]);
            pa[j][2] = pack(sc[2 * j + 1][0], sc[2 * j + 1][1]);
            pa[j][3] = pack(sc[2 * j + 1][2], sc[2 * j + 1][3]);
        }
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int i = 0; i < NB; i += 2) {
                uint32_t b[4];
                ldmatrix4t(b, vs + sw(16 * j + 8 * (mt % 2) + lane % 8, i + mt / 2));
                mma(o[i], pa[j], b[0], b[1]);
                mma(o[i + 1], pa[j], b[2], b[3]);
            }
    }
};

}  // namespace tf_block

using namespace tf_block;

// Block rows r see context key k when lens[j] + r - k <= window, then block keys c < L (c <= r if causal).
extern "C" __global__ void __launch_bounds__(256) tf_block_attention(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kb, const __nv_bfloat16* __restrict__ vb,
    const long long* __restrict__ table, const int* __restrict__ lens, __nv_bfloat16* __restrict__ out, float scale,
    int window, int rows, int h, int hk, int L, int causal) {
    extern __shared__ uint4 smem[];
    uint4* ks = smem;
    uint4* vs = ks + KT * C;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g8 = lane / 4, qd = lane % 4;
    const int j = blockIdx.x, g = blockIdx.y, G = h / hk;
    const int head = g * G + warp;
    const int s = lens[j];
    const __nv_bfloat16* kc = reinterpret_cast<const __nv_bfloat16*>(table[2 * j]) + (int64_t)g * s * D;
    const __nv_bfloat16* vc = reinterpret_cast<const __nv_bfloat16*>(table[2 * j + 1]) + (int64_t)g * s * D;
    uint32_t qa[K16][4];
    {
        const int r0 = g8, r1 = g8 + 8;
        const uint32_t* q0 = reinterpret_cast<const uint32_t*>(q + ((int64_t)head * rows + j * L + min(r0, L - 1)) * D);
        const uint32_t* q1 = reinterpret_cast<const uint32_t*>(q + ((int64_t)head * rows + j * L + min(r1, L - 1)) * D);
#pragma unroll
        for (int k = 0; k < K16; ++k) {
            qa[k][0] = r0 < L ? q0[8 * k + qd] : 0u;
            qa[k][1] = r1 < L ? q1[8 * k + qd] : 0u;
            qa[k][2] = r0 < L ? q0[8 * k + qd + 4] : 0u;
            qa[k][3] = r1 < L ? q1[8 * k + qd + 4] : 0u;
        }
    }
    Fold f;
    const int lo = max(0, s - window); // the earliest context key any row sees (row 0: s - k <= window)
    const auto vis = [&](int r, int key) { return r < L && key < s && s + r - key <= window; };
    const int e0 = lo / KT * KT; // the first context tile
    auto context = [&](bool odd) { // odd: the tiles' NaN and infinite values leave the mma (added back below)
        for (int k0 = e0; k0 < s; k0 += KT) {
            for (int i = threadIdx.x; i < KT * C; i += blockDim.x) {
                const int key = k0 + i / C;
                const bool in = key < s;
                cp16z(ks + sw(i / C, i % C), kc + (int64_t)(in ? key : 0) * D + (i % C) * 8, in);
                cp16z(vs + sw(i / C, i % C), vc + (int64_t)(in ? key : 0) * D + (i % C) * 8, in);
            }
            commit();
            cp_wait<0>();
            if (odd) attn_odd::clean<D>(vs, blockDim.x);
            __syncthreads();
            f.tile(qa, ks, vs, scale, lane, k0, vis);
            __syncthreads();
        }
    };
    auto block_keys = [&] { // the block's own keys
        for (int i = threadIdx.x; i < KT * C; i += blockDim.x) {
            const int c = i / C;
            const bool in = c < L;
            const int64_t at = ((int64_t)g * rows + j * L + (in ? c : 0)) * D + (i % C) * 8;
            cp16z(ks + sw(c, i % C), kb + at, in);
            cp16z(vs + sw(c, i % C), vb + at, in);
        }
        commit();
        cp_wait<0>();
    };
    const auto own = [&](int r, int c) { return r < L && c < L && (!causal || c <= r); };
    context(false);
    block_keys();
    __syncthreads();
    f.tile(qa, ks, vs, scale, lane, 0, own);
    // a NaN or infinite output: again, NaN and infinite values out of the mma, added back (finite values: same bits)
    if (__syncthreads_or(attn_odd::nonfinite_out<NB>(f.o, g8 < L, g8 + 8 < L))) {
        f = Fold();
        context(true);
        const auto ck = [&](int k) { return kc + (int64_t)k * D; };
        const auto bk = [&](int c) { return kb + ((int64_t)g * rows + j * L + c) * D; };
        const auto qrow = [&](int r) { return q + ((int64_t)head * rows + j * L + r) * D; };
        const auto each = [&](int r, auto f) { // row r's keys: its window of the context, then the block's
            for (int k = lo; k < s; ++k)
                if (vis(r, k)) f(ck(k));
            for (int c = 0; c < L; ++c)
                if (own(r, c)) f(bk(c));
        };
        auto w = attn_odd::weights<D>(qrow, each, scale, lane);
        attn_odd::add_back<D>(f.o, [&](int k) { return vc + (int64_t)k * D; }, ck, e0, s, lane, vis, f.l0, f.l1, w);
        block_keys();
        attn_odd::clean<D>(vs, blockDim.x);
        __syncthreads();
        f.tile<true>(qa, ks, vs, scale, lane, 0, own); // its rescale (0 when it holds the max) keeps the infinities
        const auto row = [&](int c) { return vb + ((int64_t)g * rows + j * L + c) * D; };
        attn_odd::add_back<D>(f.o, row, bk, 0, L, lane, own, f.l0, f.l1, w);
    }
    if (g8 < L)
#pragma unroll
        for (int i = 0; i < NB; ++i)
            *reinterpret_cast<uint32_t*>(out + ((int64_t)(j * L + g8) * h + head) * D + 8 * i + 2 * qd) =
                pack(__fdiv_rn(f.o[i][0], f.l0), __fdiv_rn(f.o[i][1], f.l0));
    if (g8 + 8 < L)
#pragma unroll
        for (int i = 0; i < NB; ++i)
            *reinterpret_cast<uint32_t*>(out + ((int64_t)(j * L + g8 + 8) * h + head) * D + 8 * i + 2 * qd) =
                pack(__fdiv_rn(f.o[i][2], f.l1), __fdiv_rn(f.o[i][3], f.l1));
}
