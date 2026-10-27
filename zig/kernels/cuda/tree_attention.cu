// Tree attention (float64 references: core/attention_ref.zig): committed keys, then the row's path, groups in order.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#include "attn_odd.cuh"
#include "common/qmm_frag.cuh"

namespace tf_tree {

using qmm_frag::commit;
using qmm_frag::cp16z;
using qmm_frag::ldmatrix4;
using qmm_frag::ldmatrix4t;
using qmm_frag::mma;
using qmm_frag::mma0;

constexpr int SPAN = 512;      // keys a group: one partial (4 slices of its tiles)
constexpr int MAXD = 128;      // a path's most rows
constexpr float LOG2E = 1.4426950408889634f;

template <int N>
__device__ __forceinline__ void wait_group() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ float texp(float x) {
    float y;
    asm("ex2.approx.f32 %0, %1;\n" : "=f"(y) : "f"(__fmul_rn(x, LOG2E)));
    return y;
}

__device__ __forceinline__ uint32_t pack(float lo, float hi) {
    __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&v);
}

template <int D>
__device__ __forceinline__ int sw(int r, int c) { return r * (D / 8) + (c ^ (r & 7)); }

}  // namespace tf_tree

using namespace tf_tree;

// Each row's root-to-row path (global window rows, root first) and its depth, from parents (-1 at a root).
extern "C" __global__ void tf_tree_paths(const int* __restrict__ parents, int* __restrict__ paths,
                                         int* __restrict__ depths, int rows) {
    const int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= rows) return;
    int depth = 0;
    for (int cur = node; cur >= 0 && depth < MAXD; cur = parents[cur]) ++depth;
    depths[node] = depth;
    int cur = node;
    for (int slot = depth - 1; slot >= 0; --slot, cur = parents[cur]) paths[node * MAXD + slot] = cur;
}

// Group gi of key head kh for 16 (row, head) pairs from pf (m: row rq + m / G, head m % G); !ODD true: run tree_odd.
template <int D, int HB, int W, int ST, bool ODD>
__device__ __forceinline__ bool tree_group(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kn, const __nv_bfloat16* __restrict__ vn,
    const __nv_bfloat16* __restrict__ origin, const long long* __restrict__ offs, const int* __restrict__ row_stream,
    const int* __restrict__ streams, const int* __restrict__ paths, const int* __restrict__ depths,
    float* __restrict__ po, float* __restrict__ pm, float* __restrict__ pl, int h, int hk, float scale, int gi, int ng,
    int64_t rq, int pf, int npairs, int64_t row, int slice) {
    constexpr int C = D / 8, K16 = D / 16, NB = D / 8, NT8 = HB / 8, PK = HB / 16;
    constexpr bool QREG = D <= 128;
    extern __shared__ uint4 smem[];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g8 = lane / 4, qd = lane % 4, mt = lane / 8;
    const int wi = W == 4 ? warp : slice; // the group's slice: key tiles wi, wi + 4, ...
    const int64_t r = row; // the row whose stream and depth set the keys
    const int kh = blockIdx.y, G = h / hk;
    const int s = row_stream[r];
    const int p = streams[4 * s + 2], depth = depths[r];
    const int total = p + depth, k0 = gi * SPAN, k1 = min(total, k0 + SPAN);
    if (k0 >= total) return false; // the merge reads only the row's own groups
    const __nv_bfloat16* kc = origin + offs[2 * s];
    const __nv_bfloat16* vc = origin + offs[2 * s + 1];
    // ST tile stages: a one-warp block fetches ST - 1 tiles ahead while it computes one (stages never change bits)
    uint4* const kv0 = smem + warp * ST * 2 * HB * C;
    uint4* ks = kv0;
    uint4* vs = ks + HB * C;
    uint32_t qa[QREG ? K16 : 1][4];
    uint4* qs = smem + W * ST * 2 * HB * C; // !QREG: the row's heads [16][D], shared by the warps
    if constexpr (QREG) {
        const int a0 = pf + g8, a1 = pf + g8 + 8;
        const bool v0 = a0 < npairs, v1 = a1 < npairs;
        const uint32_t* q0 = reinterpret_cast<const uint32_t*>(q + ((rq + (v0 ? a0 : pf) / G) * h + kh * G + (v0 ? a0 : pf) % G) * D);
        const uint32_t* q1 = reinterpret_cast<const uint32_t*>(q + ((rq + (v1 ? a1 : pf) / G) * h + kh * G + (v1 ? a1 : pf) % G) * D);
#pragma unroll
        for (int k = 0; k < K16; ++k) {
            qa[k][0] = v0 ? q0[8 * k + qd] : 0u;
            qa[k][1] = v1 ? q1[8 * k + qd] : 0u;
            qa[k][2] = v0 ? q0[8 * k + qd + 4] : 0u;
            qa[k][3] = v1 ? q1[8 * k + qd + 4] : 0u;
        }
    } else {
        for (int i = threadIdx.x; i < 16 * C; i += blockDim.x) {
            const int a = pf + i / C;
            const bool in = a < npairs;
            const int64_t row = rq + (in ? a : pf) / G;
            cp16z(qs + sw<D>(i / C, i % C), q + (row * h + kh * G + (in ? a : pf) % G) * D + (i % C) * 8, in);
        }
        commit();
        wait_group<0>();
        __syncthreads();
    }
    float o[NB][4];
#pragma unroll
    for (int i = 0; i < NB; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
    // tile t's keys and values into stage st (committed keys, then the row's path), one cp.async group
    auto load = [&](int t, int st) {
        uint4* const kd = kv0 + st * 2 * HB * C;
        for (int i = lane; i < HB * C; i += 32) {
            const int key = t * HB + i / C;
            const bool in = key < k1;
            const __nv_bfloat16* ksrc;
            const __nv_bfloat16* vsrc;
            if (key < p) {
                ksrc = kc + ((int64_t)key * hk + kh) * D;
                vsrc = vc + ((int64_t)key * hk + kh) * D;
            } else { // the row's own path: key p + j is path row j
                const int node = in ? paths[r * MAXD + (key - p)] : 0;
                ksrc = kn + ((int64_t)node * hk + kh) * D;
                vsrc = vn + ((int64_t)node * hk + kh) * D;
            }
            if (!in) ksrc = kn, vsrc = vn;
            cp16z(kd + sw<D>(i / C, i % C), ksrc + (i % C) * 8, in);
            cp16z(kd + HB * C + sw<D>(i / C, i % C), vsrc + (i % C) * 8, in);
        }
        commit();
    };
    // ST - 1 tiles ahead: one cp.async group a tile, an empty one past the end, so the wait count never changes
    const int t0 = k0 / HB + wi;
#pragma unroll
    for (int a = 0; a < ST - 1; ++a) {
        if ((t0 + 4 * a) * HB < k1) load(t0 + 4 * a, a);
        else commit();
    }
    int st = 0;
    for (int t = t0; t * HB < k1; t += 4) {
        const int ahead = t + 4 * (ST - 1);
        if (ahead * HB < k1) load(ahead, (st + ST - 1) % ST);
        else commit();
        wait_group<ST - 1>(); // tile t landed; the next ST - 1 are in flight
        ks = kv0 + st * 2 * HB * C;
        vs = ks + HB * C;
        st = (st + 1) % ST;
        if constexpr (ODD) attn_odd::clean_rows<D, HB>(vs, lane, 32); // the chunks this lane copied
        __syncwarp();
        float sc[NT8][4];
#pragma unroll
        for (int k = 0; k < K16; ++k) {
            uint32_t qf[4];
            if constexpr (QREG) {
                qf[0] = qa[k][0], qf[1] = qa[k][1], qf[2] = qa[k][2], qf[3] = qa[k][3];
            } else {
                ldmatrix4(qf, qs + sw<D>(lane % 16, 2 * k + lane / 16));
            }
#pragma unroll
            for (int n = 0; n < NT8; n += 2) {
                uint32_t b[4];
                ldmatrix4(b, ks + sw<D>(8 * (n + mt / 2) + lane % 8, 2 * k + mt % 2));
                if (k == 0) {
                    mma0(sc[n], qf, b[0], b[1]);
                    mma0(sc[n + 1], qf, b[2], b[3]);
                } else {
                    mma(sc[n], qf, b[0], b[1]);
                    mma(sc[n + 1], qf, b[2], b[3]);
                }
            }
        }
        const int kb = t * HB + 2 * qd;
        float tm0 = -INFINITY, tm1 = -INFINITY;
#pragma unroll
        for (int n = 0; n < NT8; ++n) {
            const int c = kb + 8 * n;
            const bool i0 = c < k1, i1 = c + 1 < k1;
            sc[n][0] = i0 ? __fmul_rn(sc[n][0], scale) : -INFINITY;
            sc[n][1] = i1 ? __fmul_rn(sc[n][1], scale) : -INFINITY;
            sc[n][2] = i0 ? __fmul_rn(sc[n][2], scale) : -INFINITY;
            sc[n][3] = i1 ? __fmul_rn(sc[n][3], scale) : -INFINITY;
            tm0 = fmaxf(tm0, fmaxf(sc[n][0], sc[n][1]));
            tm1 = fmaxf(tm1, fmaxf(sc[n][2], sc[n][3]));
        }
        for (int x = 1; x <= 2; x *= 2) {
            tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, x));
            tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, x));
        }
        const bool act0 = tm0 != -INFINITY, act1 = tm1 != -INFINITY; // each row's own: a NaN head moves only its row
        const float n0 = act0 ? fmaxf(m0, tm0) : m0, n1 = act1 ? fmaxf(m1, tm1) : m1;
        const float a0 = act0 ? (m0 == -INFINITY ? 0.0f : texp(__fsub_rn(m0, n0))) : 1.0f;
        const float a1 = act1 ? (m1 == -INFINITY ? 0.0f : texp(__fsub_rn(m1, n1))) : 1.0f;
        float c0 = 0.0f, c1 = 0.0f;
#pragma unroll
        for (int n = 0; n < NT8; ++n) {
            sc[n][0] = sc[n][0] != -INFINITY ? texp(__fsub_rn(sc[n][0], n0)) : 0.0f;
            sc[n][1] = sc[n][1] != -INFINITY ? texp(__fsub_rn(sc[n][1], n0)) : 0.0f;
            sc[n][2] = sc[n][2] != -INFINITY ? texp(__fsub_rn(sc[n][2], n1)) : 0.0f;
            sc[n][3] = sc[n][3] != -INFINITY ? texp(__fsub_rn(sc[n][3], n1)) : 0.0f;
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
            o[i][0] = __fmul_rn(o[i][0], a0);
            o[i][1] = __fmul_rn(o[i][1], a0);
            o[i][2] = __fmul_rn(o[i][2], a1);
            o[i][3] = __fmul_rn(o[i][3], a1);
        }
        uint32_t pa[PK][4];
#pragma unroll
        for (int j = 0; j < PK; ++j) {
            pa[j][0] = pack(sc[2 * j][0], sc[2 * j][1]);
            pa[j][1] = pack(sc[2 * j][2], sc[2 * j][3]);
            pa[j][2] = pack(sc[2 * j + 1][0], sc[2 * j + 1][1]);
            pa[j][3] = pack(sc[2 * j + 1][2], sc[2 * j + 1][3]);
        }
#pragma unroll
        for (int j = 0; j < PK; ++j)
#pragma unroll
            for (int i = 0; i < NB; i += 2) {
                uint32_t b[4];
                ldmatrix4t(b, vs + sw<D>(16 * j + 8 * (mt % 2) + lane % 8, i + mt / 2));
                mma(o[i], pa[j], b[0], b[1]);
                mma(o[i + 1], pa[j], b[2], b[3]);
            }
        __syncwarp();
    }
    if constexpr (!ODD) { // a NaN or infinite partial: the block runs again as ODD (finite values: the same bits)
        if (__syncthreads_or(attn_odd::nonfinite_out<NB>(o, pf + g8 < npairs, pf + g8 + 8 < npairs))) return true;
    } else {
        // key k of the group's row: committed, then the row's path; pair m: row rq + (pf + m) / G, head (pf + m) % G
        const auto at = [&](const __nv_bfloat16* c, const __nv_bfloat16* n, int k) {
            return k < p ? c + ((int64_t)k * hk + kh) * D : n + ((int64_t)paths[r * MAXD + (k - p)] * hk + kh) * D;
        };
        const auto qrow = [&](int m) { return q + ((rq + (pf + m) / G) * h + kh * G + (pf + m) % G) * D; };
        const auto each = [&](int m, auto f) { // pair m's keys: its stream's committed ones, then its own row's path
            const int64_t pr = rq + (pf + m) / G;
            for (int k = 0; k < p; ++k) f(kc + ((int64_t)k * hk + kh) * D);
            for (int j = 0; j < depths[pr]; ++j) f(kn + ((int64_t)paths[pr * MAXD + j] * hk + kh) * D);
        };
        auto w = attn_odd::weights<D>(qrow, each, scale, lane);
        const auto vis = [&](int m, int) { return pf + m < npairs; }; // a pair sees every key of its group
        for (int t = t0; t * HB < k1; t += 4) // after the last tile: no rescale turns an infinity into NaN
            attn_odd::add_back<D>(o, [&](int k) { return at(vc, vn, k); }, [&](int k) { return at(kc, kn, k); },
                                  t * HB, min(k1, t * HB + HB), lane, vis, l0, l1, w);
    }
    if constexpr (W == 1) {
        // one warp a block: each slice's partial (o, max, sum) of its 16 pairs; the merge combines the 4 in order
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int hd = g8 + 8 * half;
            if (pf + hd >= npairs) continue;
            const int64_t part = (((rq + (pf + hd) / G) * hk + kh) * ng + gi) * 4 + wi;
            const int head = (pf + hd) % G;
#pragma unroll
            for (int i = 0; i < NB; ++i) {
                const int d = 8 * i + 2 * qd;
                po[(part * G + head) * D + d] = o[i][2 * half];
                po[(part * G + head) * D + d + 1] = o[i][2 * half + 1];
            }
            if (qd == 0) pm[part * G + head] = half ? m1 : m0, pl[part * G + head] = half ? l1 : l0;
        }
        return false;
    } else {
        __syncthreads();
        float* os = reinterpret_cast<float*>(smem); // [4 warps][16 heads][D], then max and sum [4][16]
        float* ms = os + 4 * 16 * D;
        float* ls = ms + 4 * 16;
#pragma unroll
        for (int i = 0; i < NB; ++i) {
            const int d = 8 * i + 2 * qd;
            os[(warp * 16 + g8) * D + d] = o[i][0];
            os[(warp * 16 + g8) * D + d + 1] = o[i][1];
            os[(warp * 16 + g8 + 8) * D + d] = o[i][2];
            os[(warp * 16 + g8 + 8) * D + d + 1] = o[i][3];
        }
        if (qd == 0) {
            ms[warp * 16 + g8] = m0, ls[warp * 16 + g8] = l0;
            ms[warp * 16 + g8 + 8] = m1, ls[warp * 16 + g8 + 8] = l1;
        }
        __syncthreads();
        for (int idx = threadIdx.x; idx < 16 * D; idx += 128) {
            const int hd = idx / D, e = idx % D;
            if (pf + hd >= npairs) continue;
            const int64_t part = ((rq + (pf + hd) / G) * hk + kh) * ng + gi;
            const int head = (pf + hd) % G;
            float mm = -INFINITY;
            for (int w = 0; w < 4; ++w) mm = fmaxf(mm, ms[w * 16 + hd]);
            float acc = 0.0f, l = 0.0f;
            for (int w = 0; w < 4; ++w) {
                const float mw = ms[w * 16 + hd];
                if (mw == -INFINITY) continue;
                const float a = texp(__fsub_rn(mw, mm));
                acc = __fmaf_rn(a, os[(w * 16 + hd) * D + e], acc);
                l = __fmaf_rn(a, ls[w * 16 + hd], l);
            }
            if (acc != acc) { // NaN: a replayed infinity (weighed in float64) may have met a weight of 0; again, kept
                acc = 0.0f;
                for (int w = 0; w < 4; ++w)
                    if (ms[w * 16 + hd] != -INFINITY)
                        acc = attn_odd::fold(os[(w * 16 + hd) * D + e], texp(__fsub_rn(ms[w * 16 + hd], mm)), acc);
            }
            po[(part * G + head) * D + e] = acc;
            if (e == 0) pm[part * G + head] = mm, pl[part * G + head] = l;
        }
        return false;
    }
}

// A block with a NaN or infinite partial, again as ODD: those values out of the mma, added back at float64's weights.
template <int D, int HB, int W, int ST>
__device__ __noinline__ void tree_odd(
    const __nv_bfloat16* q, const __nv_bfloat16* kn, const __nv_bfloat16* vn, const __nv_bfloat16* origin,
    const long long* offs, const int* row_stream, const int* streams, const int* paths, const int* depths, float* po,
    float* pm, float* pl, int h, int hk, float scale, int gi, int ng, int64_t rq, int pf, int npairs, int64_t row,
    int slice) {
    tree_group<D, HB, W, ST, true>(q, kn, vn, origin, offs, row_stream, streams, paths, depths, po, pm, pl, h, hk, scale,
                                   gi, ng, rq, pf, npairs, row, slice);
}

// A stream's full committed groups once for its rows, 16 (row, head) pairs a block: each partial a row block's bits.
template <int D, int HB, int W, int ST>
__device__ __forceinline__ void tree_shared(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ origin, const long long* __restrict__ offs,
    const int* __restrict__ row_stream, const int* __restrict__ streams, const int* __restrict__ paths,
    const int* __restrict__ depths, float* __restrict__ po, float* __restrict__ pm, float* __restrict__ pl, int h, int hk,
    float scale, int ng, int n, const __nv_bfloat16* __restrict__ kn) {
    const int G = h / hk;
    const int per = W == 4 ? 1 : 4; // slices an item takes a block
    for (int item = blockIdx.x; item < n * ng * per; item += gridDim.x) { // (row, group[, slice]): a fixed grid
        const int64_t r = item / per / ng;
        const int gi = item / per % ng;
        const int st = row_stream[r];
        const int64_t r0 = streams[4 * st];
        const int pairs = streams[4 * st + 1] * G, p = streams[4 * st + 2];
        const int tile = static_cast<int>(r - r0); // a stream's j-th row hosts its j-th pair tile (tiles <= rows)
        if (tile * 16 >= pairs || (gi + 1) * SPAN > p) continue;
        // vn: never read, a full committed group has no path keys (the replay weighs them, from kn)
        if (tree_group<D, HB, W, ST, false>(q, kn, q, origin, offs, row_stream, streams, paths, depths, po, pm, pl, h, hk,
                                            scale, gi, ng, r0, tile * 16, pairs, r, item % per))
            tree_odd<D, HB, W, ST>(q, kn, q, origin, offs, row_stream, streams, paths, depths, po, pm, pl, h, hk, scale, gi,
                                   ng, r0, tile * 16, pairs, r, item % per);
        __syncthreads();
    }
}

#define TF_TREE_SHARED(NAME, D, HB, W, ST)                                                                              \
    extern "C" __global__ void __launch_bounds__(32 * W) NAME(                                                           \
        const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ origin,                                \
        const long long* __restrict__ offs, const int* __restrict__ row_stream, const int* __restrict__ streams,      \
        const int* __restrict__ paths, const int* __restrict__ depths, float* __restrict__ po,                        \
        float* __restrict__ pm, float* __restrict__ pl, int h, int hk, float scale, int ng, int n,                    \
        const __nv_bfloat16* __restrict__ kn) {                                                                       \
        tree_shared<D, HB, W, ST>(q, origin, offs, row_stream, streams, paths, depths, po, pm, pl, h, hk, scale, ng, n, \
                                  kn);                                                                                \
    }

TF_TREE_SHARED(tf_tree_shared, 128, 32, 4, 1)
TF_TREE_SHARED(tf_tree_shared_256, 256, 16, 4, 1)
TF_TREE_SHARED(tf_tree_shared_w1, 128, 32, 1, 2)
TF_TREE_SHARED(tf_tree_shared_256_w1, 256, 16, 1, 2)
TF_TREE_SHARED(tf_tree_shared_w1s3, 128, 32, 1, 3)
TF_TREE_SHARED(tf_tree_shared_256_w1s3, 256, 16, 1, 3)

// Block (row, key head, group slot): the row's groups past its stream's full committed ones, slot, slot + slots, ...
template <int D, int HB, int W, int ST>
__device__ __forceinline__ void tree_part(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kn, const __nv_bfloat16* __restrict__ vn,
    const __nv_bfloat16* __restrict__ origin, const long long* __restrict__ offs, const int* __restrict__ row_stream,
    const int* __restrict__ streams, const int* __restrict__ paths, const int* __restrict__ depths,
    float* __restrict__ po, float* __restrict__ pm, float* __restrict__ pl, int h, int hk, float scale, int ng) {
    const int64_t r = blockIdx.x;
    const int p = streams[4 * row_stream[r] + 2], total = p + depths[r];
    const int per = W == 4 ? 1 : 4, slot = blockIdx.z / per, slots = gridDim.z / per;
    for (int gi = slot + p / SPAN; gi < ng && gi * SPAN < total; gi += slots) { // full committed: shared
        if (tree_group<D, HB, W, ST, false>(q, kn, vn, origin, offs, row_stream, streams, paths, depths, po, pm, pl, h, hk,
                                            scale, gi, ng, r, 0, h / hk, r, blockIdx.z % per))
            tree_odd<D, HB, W, ST>(q, kn, vn, origin, offs, row_stream, streams, paths, depths, po, pm, pl, h, hk, scale, gi,
                                   ng, r, 0, h / hk, r, blockIdx.z % per);
        __syncthreads(); // the next group's tiles reuse the partials' shared memory
    }
}

#define TF_TREE_PART(NAME, D, HB, W, ST)                                                                              \
    extern "C" __global__ void __launch_bounds__(32 * W) NAME(                                                           \
        const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kn,                                    \
        const __nv_bfloat16* __restrict__ vn, const __nv_bfloat16* __restrict__ origin,                               \
        const long long* __restrict__ offs, const int* __restrict__ row_stream, const int* __restrict__ streams,      \
        const int* __restrict__ paths, const int* __restrict__ depths, float* __restrict__ po,                        \
        float* __restrict__ pm, float* __restrict__ pl, int h, int hk, float scale, int ng) {                         \
        tree_part<D, HB, W, ST>(q, kn, vn, origin, offs, row_stream, streams, paths, depths, po, pm, pl, h, hk, scale, \
                            ng);                                                                                    \
    }

TF_TREE_PART(tf_tree_part, 128, 32, 4, 1)
TF_TREE_PART(tf_tree_part_256, 256, 16, 4, 1)
TF_TREE_PART(tf_tree_part_w1, 128, 32, 1, 2)
TF_TREE_PART(tf_tree_part_256_w1, 256, 16, 1, 2)
TF_TREE_PART(tf_tree_part_w1s3, 128, 32, 1, 3)
TF_TREE_PART(tf_tree_part_256_w1s3, 256, 16, 1, 3)

// out [W, H, D] bf16: block (row, key head, head) folds the row's groups in order (sliced: a group's 4 slices first).
constexpr int MERGE_CHUNK = 16; // partials per chunk: 16 groups, or 4 groups of 4 slices

extern "C" __global__ void __launch_bounds__(256) tf_tree_merge(
    const float* __restrict__ po, const float* __restrict__ pm, const float* __restrict__ pl,
    const int* __restrict__ row_stream, const int* __restrict__ streams, const int* __restrict__ depths,
    __nv_bfloat16* __restrict__ out, int h, int hk, int ng, int d, int sliced) {
    const int64_t r = blockIdx.x;
    const int kh = blockIdx.y, G = h / hk, hd = blockIdx.z; // block (row, key head, head): each element alone
    const int total = streams[4 * row_stream[r] + 2] + depths[r];
    const int groups = (total + SPAN - 1) / SPAN;
    const int per = sliced ? 4 : 1, slots = groups * per;
    const int64_t base = (r * hk + kh) * ng * per;
    for (int e = threadIdx.x; e < d; e += 256) {
        float m = -INFINITY, l = 0.0f, acc = 0.0f;
        for (int s0 = 0; s0 < slots; s0 += MERGE_CHUNK) { // a chunk's loads all issued before any is used
            float cm_[MERGE_CHUNK], co_[MERGE_CHUNK], cl_[MERGE_CHUNK];
#pragma unroll
            for (int i = 0; i < MERGE_CHUNK; ++i) {
                const int64_t q = (base + s0 + i) * G + hd;
                const bool in = s0 + i < slots;
                cm_[i] = in ? pm[q] : -INFINITY, cl_[i] = in ? pl[q] : 0.0f, co_[i] = in ? po[q * d + e] : 0.0f;
            }
#pragma unroll
            for (int i = 0; i < MERGE_CHUNK; ++i) {
                if (s0 + i >= slots) break;
                if (sliced && (i & 3)) continue;
                float cm = -INFINITY, co = 0.0f, cl = 0.0f;
                if (!sliced) { // four-warp blocks combined the group's slices themselves
                    cm = cm_[i], co = co_[i], cl = cl_[i];
                } else { // one-warp blocks left the 4 slices: their combine, in the same order
                    for (int w = 0; w < 4; ++w) cm = fmaxf(cm, cm_[i + w]);
                    for (int w = 0; w < 4; ++w) {
                        const float mw = cm_[i + w];
                        if (mw == -INFINITY) continue;
                        const float aw = texp(__fsub_rn(mw, cm));
                        co = __fmaf_rn(aw, co_[i + w], co);
                        cl = __fmaf_rn(aw, cl_[i + w], cl);
                    }
                }
                const bool active = cl > 0.0f;
                const float next = active ? fmaxf(m, cm) : m;
                const float a = active ? (m == -INFINITY ? 0.0f : texp(__fsub_rn(m, next))) : 1.0f;
                const float b = active ? texp(__fsub_rn(cm, next)) : 0.0f;
                acc = __fmaf_rn(acc, a, __fmul_rn(co, b));
                l = __fmaf_rn(l, a, __fmul_rn(cl, b));
                m = next;
            }
        }
        if (acc != acc) { // NaN: a replayed infinity (weighed in float64) may have met a weight of 0 here; again, kept
            float mm = -INFINITY;
            acc = 0.0f;
            for (int s0 = 0; s0 < slots; s0 += per) {
                float cm = -INFINITY, co = 0.0f, cl = 0.0f;
                for (int w = 0; w < per; ++w) cm = fmaxf(cm, pm[(base + s0 + w) * G + hd]);
                for (int w = 0; w < per; ++w) {
                    const int64_t q = (base + s0 + w) * G + hd;
                    if (pm[q] == -INFINITY) continue;
                    const float aw = texp(__fsub_rn(pm[q], cm));
                    co = attn_odd::fold(po[q * d + e], aw, co);
                    cl = __fmaf_rn(aw, pl[q], cl);
                }
                if (!(cl > 0.0f)) continue;
                const float next = fmaxf(mm, cm);
                acc = attn_odd::fold(acc, mm == -INFINITY ? 0.0f : texp(__fsub_rn(mm, next)),
                                     attn_odd::keep(co, texp(__fsub_rn(cm, next))));
                mm = next;
            }
        }
        out[(r * h + kh * G + hd) * d + e] = __float2bfloat16_rn(__fdiv_rn(acc, l));
    }
}
