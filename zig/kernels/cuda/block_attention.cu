// DFlash-style block attention for every CUDA family (float64 reference: core/attention_ref.zig): context, then block.

// Block (stream, key head): a warp a query head, its 16 mma rows the block's rows; key tiles of 32 shared by the warps.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#include "attn_fold.cuh"

using namespace attn_fold;

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
