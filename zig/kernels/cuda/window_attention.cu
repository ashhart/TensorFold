// Sliding-window attention for every CUDA family (float64 references: core/attention_ref.zig): decode and prompt rows.

// Decode: a block a (row, key head); keys walk 64-key tiles by absolute position, so a row's bits are its own.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#include "qmm_frag.cuh"

namespace tf_window {

constexpr int NT = 256;
constexpr int BN = 64;     // keys a tile
constexpr int GMAX = 16;   // query heads a key head
constexpr int KT = BN + 2;  // kt's row stride: a bank further each dim
constexpr int DMAX = 128;  // head dim (static shared memory: q, a K and a V tile, scores within 48 KiB)

}  // namespace tf_window

using namespace tf_window;

// Decode split by key tile (block: row, key head, tile); tf_window_merge folds the partials in tile order.
extern "C" __global__ void __launch_bounds__(NT) tf_window_part(const __nv_bfloat16* __restrict__ q,
                                                                const __nv_bfloat16* __restrict__ kc,
                                                                const __nv_bfloat16* __restrict__ vc,
                                                                const int* __restrict__ pos,
                                                                const int* __restrict__ slot, float* __restrict__ po,
                                                                float* __restrict__ pm, float* __restrict__ pl, int h,
                                                                int hk, int d, int win, long long ring, float scale) {
    __shared__ float qs[GMAX * DMAX];
    __shared__ __nv_bfloat16 kt[DMAX * KT]; // K transposed, [d][key], padded: a warp reads 32 keys of one dim
    __shared__ __nv_bfloat16 vs[BN * DMAX];
    __shared__ float sc[GMAX * BN];
    __shared__ float mx[GMAX], ls[GMAX];
    const int64_t r = blockIdx.x;
    const int kh = blockIdx.y, tiles = gridDim.z, j = blockIdx.z;
    const int g = h / hk;
    const int p = pos[r];
    const int64_t base = (slot != nullptr ? static_cast<int64_t>(slot[r]) : 0) * ring;
    const int lo = win > 0 ? max(p - win + 1, 0) : 0;
    const int t = lo / BN + j;
    const int64_t part = (r * hk + kh) * tiles + j; // this block's partial, g heads of d values
    if (t > p / BN) { // past the row's last tile: an empty partial
        for (int i = threadIdx.x; i < g; i += NT) pm[part * g + i] = -INFINITY, pl[part * g + i] = 0.f;
        return;
    }
    for (int i = threadIdx.x; i < g * d; i += NT) qs[i] = __bfloat162float(q[(r * h + kh * g + i / d) * d + i % d]);
    for (int i = threadIdx.x; i < BN * d; i += NT) {
        const int key = t * BN + i / d;
        const bool in = key >= lo && key <= p;
        const int64_t at = ((base + (in ? key : lo) % ring) * hk + kh) * d + i % d;
        kt[(i % d) * KT + i / d] = kc[at];
        vs[i] = vc[at];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < g * BN; i += NT) { // score (head i / BN, key i % BN)
        const int hd = i / BN, jj = i % BN, key = t * BN + jj;
        float s = 0.f;
#pragma unroll 8
        for (int e = 0; e < d; ++e) s = __fmaf_rn(qs[hd * d + e], __bfloat162float(kt[e * KT + jj]), s);
        sc[i] = (key >= lo && key <= p) ? __fmul_rn(s, scale) : -INFINITY;
    }
    __syncthreads();
    { // a warp a head: the tile's max, the weights exp(s - max) and their sum, lanes in a fixed tree
        const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
        for (int hd = warp; hd < g; hd += NT / 32) {
            float a = sc[hd * BN + lane], b = sc[hd * BN + 32 + lane];
            float m = fmaxf(a, b);
            for (int k = 16; k > 0; k >>= 1) m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, k));
            a = a == -INFINITY ? 0.f : expf(__fsub_rn(a, m));
            b = b == -INFINITY ? 0.f : expf(__fsub_rn(b, m));
            sc[hd * BN + lane] = a;
            sc[hd * BN + 32 + lane] = b;
            float l = __fadd_rn(a, b);
            for (int k = 16; k > 0; k >>= 1) l = __fadd_rn(l, __shfl_xor_sync(0xFFFFFFFFu, l, k));
            if (lane == 0) mx[hd] = m, ls[hd] = l;
        }
    }
    __syncthreads();
    for (int o = threadIdx.x; o < g * d; o += NT) {
        const int hd = o / d, e = o % d;
        float a = 0.f;
        for (int jj = 0; jj < BN; ++jj) a = __fmaf_rn(sc[hd * BN + jj], __bfloat162float(vs[jj * d + e]), a);
        po[part * g * d + o] = a;
    }
    if (threadIdx.x < g) pm[part * g + threadIdx.x] = mx[threadIdx.x], pl[part * g + threadIdx.x] = ls[threadIdx.x];
}

// out [W, H, D] bf16: a row's partials folded in tile order, o = sum_j exp(m_j - M) o_j / sum_j exp(m_j - M) l_j.
extern "C" __global__ void __launch_bounds__(NT) tf_window_merge(const float* __restrict__ po,
                                                                 const float* __restrict__ pm,
                                                                 const float* __restrict__ pl,
                                                                 __nv_bfloat16* __restrict__ out, int h, int hk, int d,
                                                                 int tiles) {
    const int64_t r = blockIdx.x;
    const int kh = blockIdx.y, g = h / hk;
    const int64_t part0 = (r * hk + kh) * tiles;
    for (int o = threadIdx.x; o < g * d; o += NT) {
        const int hd = o / d;
        float m = -INFINITY;
        for (int j = 0; j < tiles; ++j) m = fmaxf(m, pm[(part0 + j) * g + hd]);
        float acc = 0.f, l = 0.f;
        for (int j = 0; j < tiles; ++j) {
            const float mj = pm[(part0 + j) * g + hd];
            if (mj == -INFINITY) continue;
            const float a = expf(__fsub_rn(mj, m));
            acc = __fmaf_rn(a, po[(part0 + j) * g * d + o], acc);
            l = __fmaf_rn(a, pl[(part0 + j) * g + hd], l);
        }
        out[(r * h + kh * g + hd) * d + o % d] = __float2bfloat16_rn(__fdiv_rn(acc, l));
    }
}

// Prompt rows: prefill_attention.cu's tensor-core kernel with a window (keys (pos - win, pos]) and a ring of keys.
namespace tf_window_prompt {

using qmm_frag::commit;
using qmm_frag::cp16z;
using qmm_frag::ldmatrix4;
using qmm_frag::mma;
using qmm_frag::mma0;

constexpr int BN = 64, HALF = 32;             // keys a tile (the fold's unit), keys a staging slot
constexpr float LOG2E = 1.4426950408889634f;

template <int N>
__device__ __forceinline__ void wait_group() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldmatrix4t(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(qmm_frag::smem(p)));
}

__device__ __forceinline__ float texp(float x) {
    float y;
    asm("ex2.approx.f32 %0, %1;\n" : "=f"(y) : "f"(__fmul_rn(x, LOG2E)));
    return y;
}

__device__ __forceinline__ float tdiv(float a, float b) {
    float y;
    asm("div.full.f32 %0, %1, %2;\n" : "=f"(y) : "f"(a), "f"(b));
    return y;
}

__device__ __forceinline__ uint32_t pack(float lo, float hi) {
    __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&v);
}

template <int D>
__device__ __forceinline__ int sw(int r, int c) { return r * (D / 8) + (c ^ (r & 7)); }

__device__ __forceinline__ float rowsum(const float (&c)[8]) {
    float w[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float x = __fadd_rn(c[j], __shfl_xor_sync(0xffffffffu, c[j], 2));
        w[j] = __fadd_rn(x, __shfl_xor_sync(0xffffffffu, x, 1));
    }
    return __fadd_rn(__fadd_rn(__fadd_rn(w[0], w[4]), __fadd_rn(w[2], w[6])),
                     __fadd_rn(__fadd_rn(w[1], w[5]), __fadd_rn(w[3], w[7])));
}

template <int D, int WARPS, int HPC, int NS>
__global__ void __launch_bounds__(32 * WARPS, 1)
wattn_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kc,
             const __nv_bfloat16* __restrict__ vc, __nv_bfloat16* __restrict__ out, int p0, int W, int H, int HK,
             int G, float scale, int win, int ring) {
    constexpr int C = D / 8, K16 = D / 16, NB = D / 8, RB = WARPS / HPC, THREADS = 32 * WARPS;
    constexpr bool QREG = D <= 128;               // the queries' fragments stay in registers, else in shared memory
    extern __shared__ uint4 smem[];
    uint4* slots = smem;
    uint4* qs = smem + NS * HALF * C;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, qd = lane % 4, mt = lane / 8;
    const int groups = G / HPC, hk = blockIdx.y / groups, hg = blockIdx.y % groups;
    const int row0 = blockIdx.x * 16 * RB, last = min(row0 + 16 * RB, W) - 1;
    const int head = hk * G + hg * HPC + warp % HPC, wrow = row0 + (warp / HPC) * 16;
    // the first tile any row of the block sees: its first row's window start (win 0: key 0)
    const int t0 = win > 0 ? max(p0 + row0 - win + 1, 0) / BN : 0;
    const int tiles = (p0 + last) / BN + 1 - t0, items = 4 * tiles, limit = p0 + W;
    auto stage = [&](int item) {
        const __nv_bfloat16* src = item % 4 < 2 ? kc : vc;
        const int key0 = (t0 + item / 4) * BN + (item % 2) * HALF;
        uint4* dst = slots + (item % NS) * HALF * C;
        for (int i = threadIdx.x; i < HALF * C; i += THREADS) {
            const int key = key0 + i / C;
            const bool in = key < limit;             // past the cache's filled keys: zeros, never read
            const int row = in ? (ring > 0 ? key % ring : key) : 0; // a ring keeps key p at row p % ring
            cp16z(dst + sw<D>(i / C, i % C), src + ((int64_t)row * HK + hk) * D + (i % C) * 8, in);
        }
    };
    uint32_t qa[QREG ? K16 : 1][4];
    if constexpr (QREG) {
        const int r0 = wrow + g, r1 = r0 + 8;
        const uint32_t* q0 = reinterpret_cast<const uint32_t*>(q + ((int64_t)min(r0, W - 1) * H + head) * D);
        const uint32_t* q1 = reinterpret_cast<const uint32_t*>(q + ((int64_t)min(r1, W - 1) * H + head) * D);
#pragma unroll
        for (int k = 0; k < K16; ++k) {
            qa[k][0] = r0 < W ? q0[8 * k + qd] : 0u;
            qa[k][1] = r1 < W ? q1[8 * k + qd] : 0u;
            qa[k][2] = r0 < W ? q0[8 * k + qd + 4] : 0u;
            qa[k][3] = r1 < W ? q1[8 * k + qd + 4] : 0u;
        }
    } else {
        for (int i = threadIdx.x; i < WARPS * 16 * C; i += THREADS) {
            const int w = i / (16 * C), r = (i / C) % 16, row = row0 + (w / HPC) * 16 + r;
            const int hd = hk * G + hg * HPC + w % HPC;
            cp16z(qs + sw<D>(i / C, i % C), q + ((int64_t)min(row, W - 1) * H + hd) * D + (i % C) * 8, row < W);
        }
        commit();
    }
#pragma unroll
    for (int i = 0; i < NS; ++i) {
        if (i < items) stage(i);
        commit();
    }
    float o[NB][4];
#pragma unroll
    for (int i = 0; i < NB; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float sc[8][4];
    uint32_t pa[4][4];
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
    const int pos0 = p0 + wrow + g, pos1 = pos0 + 8;
    auto take = [&](int it) -> const uint4* {           // item it's slot, once every thread's copy has landed
        wait_group<NS - 1>();
        __syncthreads();
        return slots + (it % NS) * HALF * C;
    };
    auto done = [&](int it) {                            // the slot is free: stage the item NS ahead into it
        __syncthreads();
        if (it + NS < items) stage(it + NS);
        commit();
    };
    auto scores = [&](float (&s)[4][4], const uint4* sl) {
#pragma unroll
        for (int k = 0; k < K16; ++k) {
            uint32_t a[4];
            if constexpr (QREG) {
                a[0] = qa[k][0]; a[1] = qa[k][1]; a[2] = qa[k][2]; a[3] = qa[k][3];
            } else {
                ldmatrix4(a, qs + sw<D>(warp * 16 + lane % 16, 2 * k + lane / 16));
            }
#pragma unroll
            for (int n = 0; n < 4; n += 2) {
                uint32_t b[4];
                ldmatrix4(b, sl + sw<D>(8 * (n + mt / 2) + lane % 8, 2 * k + mt % 2));
                if (k == 0) {
                    mma0(s[n], a, b[0], b[1]);
                    mma0(s[n + 1], a, b[2], b[3]);
                } else {
                    mma(s[n], a, b[0], b[1]);
                    mma(s[n + 1], a, b[2], b[3]);
                }
            }
        }
    };
    auto values = [&](const uint32_t (&pa0)[4], const uint32_t (&pa1)[4], const uint4* sl) {
#pragma unroll
        for (int i = 0; i < NB; i += 2) {
            uint32_t b[4];
            ldmatrix4t(b, sl + sw<D>(8 * (mt % 2) + lane % 8, i + mt / 2));
            mma(o[i], pa0, b[0], b[1]);
            mma(o[i + 1], pa0, b[2], b[3]);
        }
#pragma unroll
        for (int i = 0; i < NB; i += 2) {
            uint32_t b[4];
            ldmatrix4t(b, sl + sw<D>(16 + 8 * (mt % 2) + lane % 8, i + mt / 2));
            mma(o[i], pa1, b[0], b[1]);
            mma(o[i + 1], pa1, b[2], b[3]);
        }
    };
    for (int t = 0; t < tiles; ++t) {
        const int it = 4 * t;
        scores(*reinterpret_cast<float(*)[4][4]>(&sc[0]), take(it));
        done(it);
        scores(*reinterpret_cast<float(*)[4][4]>(&sc[4]), take(it + 1));
        done(it + 1);
        {                                                // _tile's fold of the tile's 64 scores
            const int kb = (t0 + t) * BN + 2 * qd;
            const int lo0 = win > 0 ? pos0 - win + 1 : 0, lo1 = win > 0 ? pos1 - win + 1 : 0;
            float tm0 = -INFINITY, tm1 = -INFINITY;
#pragma unroll
            for (int n = 0; n < 8; ++n) {
                const int c = kb + 8 * n;
                sc[n][0] = c >= lo0 && c <= pos0 ? __fmul_rn(sc[n][0], scale) : -INFINITY;
                sc[n][1] = c + 1 >= lo0 && c + 1 <= pos0 ? __fmul_rn(sc[n][1], scale) : -INFINITY;
                sc[n][2] = c >= lo1 && c <= pos1 ? __fmul_rn(sc[n][2], scale) : -INFINITY;
                sc[n][3] = c + 1 >= lo1 && c + 1 <= pos1 ? __fmul_rn(sc[n][3], scale) : -INFINITY;
                tm0 = fmaxf(tm0, fmaxf(sc[n][0], sc[n][1]));
                tm1 = fmaxf(tm1, fmaxf(sc[n][2], sc[n][3]));
            }
            for (int x = 1; x <= 2; x *= 2) {
                tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, x));
                tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, x));
            }
            const bool act0 = tm0 != -INFINITY, act1 = tm1 != -INFINITY;
            const float n0 = act0 ? fmaxf(m0, tm0) : m0, n1 = act1 ? fmaxf(m1, tm1) : m1;
            const float a0 = act0 ? (m0 == -INFINITY ? 0.0f : texp(__fsub_rn(m0, n0))) : 1.0f;
            const float a1 = act1 ? (m1 == -INFINITY ? 0.0f : texp(__fsub_rn(m1, n1))) : 1.0f;
            float c0[8], c1[8];
#pragma unroll
            for (int n = 0; n < 8; ++n) {
                const int c = kb + 8 * n;
                sc[n][0] = act0 && c >= lo0 && c <= pos0 ? texp(__fsub_rn(sc[n][0], n0)) : 0.0f;
                sc[n][1] = act0 && c + 1 >= lo0 && c + 1 <= pos0 ? texp(__fsub_rn(sc[n][1], n0)) : 0.0f;
                sc[n][2] = act1 && c >= lo1 && c <= pos1 ? texp(__fsub_rn(sc[n][2], n1)) : 0.0f;
                sc[n][3] = act1 && c + 1 >= lo1 && c + 1 <= pos1 ? texp(__fsub_rn(sc[n][3], n1)) : 0.0f;
                c0[n] = __fadd_rn(sc[n][0], sc[n][1]);
                c1[n] = __fadd_rn(sc[n][2], sc[n][3]);
            }
            l0 = __fmaf_rn(l0, a0, rowsum(c0));
            l1 = __fmaf_rn(l1, a1, rowsum(c1));
            m0 = n0;
            m1 = n1;
#pragma unroll
            for (int i = 0; i < NB; ++i) {
                o[i][0] = __fmul_rn(o[i][0], a0);
                o[i][1] = __fmul_rn(o[i][1], a0);
                o[i][2] = __fmul_rn(o[i][2], a1);
                o[i][3] = __fmul_rn(o[i][3], a1);
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                pa[j][0] = pack(sc[2 * j][0], sc[2 * j][1]);
                pa[j][1] = pack(sc[2 * j][2], sc[2 * j][3]);
                pa[j][2] = pack(sc[2 * j + 1][0], sc[2 * j + 1][1]);
                pa[j][3] = pack(sc[2 * j + 1][2], sc[2 * j + 1][3]);
            }
        }
        values(pa[0], pa[1], take(it + 2));
        done(it + 2);
        values(pa[2], pa[3], take(it + 3));
        done(it + 3);
    }
    wait_group<0>();
    const int r0 = wrow + g, r1 = r0 + 8;
#pragma unroll
    for (int i = 0; i < NB; ++i) {
        const int d = 8 * i + 2 * qd;
        if (r0 < W)
            *reinterpret_cast<uint32_t*>(out + ((int64_t)r0 * H + head) * D + d) =
                pack(tdiv(o[i][0], l0), tdiv(o[i][1], l0));
        if (r1 < W)
            *reinterpret_cast<uint32_t*>(out + ((int64_t)r1 * H + head) * D + d) =
                pack(tdiv(o[i][2], l1), tdiv(o[i][3], l1));
    }
}


// Decode on tensor cores (d 128): a row's heads one mma tile; warps take every 4th 32-key tile, fold in warp order.
extern "C" __global__ void __launch_bounds__(128) tf_window_decode_tc(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kc, const __nv_bfloat16* __restrict__ vc,
    __nv_bfloat16* __restrict__ out, const int* __restrict__ pos, const int* __restrict__ slot, int h, int hk, int win,
    long long ring, float scale) {
    constexpr int D = 128, C = D / 8, K16 = D / 16, NB = D / 8, HB = 32;
    extern __shared__ uint4 smem[];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g8 = lane / 4, qd = lane % 4, mt = lane / 8;
    const int64_t r = blockIdx.x;
    const int kh = blockIdx.y, G = h / hk;
    uint4* ks = smem + warp * 2 * HB * C;
    uint4* vs = ks + HB * C;
    const int p = pos[r];
    const int64_t base = (slot != nullptr ? static_cast<int64_t>(slot[r]) : 0) * ring;
    const int lo = win > 0 ? max(p - win + 1, 0) : 0;
    const int lor = static_cast<int>(lo % ring);
    uint32_t qa[K16][4];
    {
        const int h0 = g8, h1 = g8 + 8;
        const uint32_t* q0 = reinterpret_cast<const uint32_t*>(q + ((int64_t)r * h + kh * G + min(h0, G - 1)) * D);
        const uint32_t* q1 = reinterpret_cast<const uint32_t*>(q + ((int64_t)r * h + kh * G + min(h1, G - 1)) * D);
#pragma unroll
        for (int k = 0; k < K16; ++k) {
            qa[k][0] = h0 < G ? q0[8 * k + qd] : 0u;
            qa[k][1] = h1 < G ? q1[8 * k + qd] : 0u;
            qa[k][2] = h0 < G ? q0[8 * k + qd + 4] : 0u;
            qa[k][3] = h1 < G ? q1[8 * k + qd + 4] : 0u;
        }
    }
    float o[NB][4];
#pragma unroll
    for (int i = 0; i < NB; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
    // K of tile t loads while tile t - 4's values fold in, V of tile t while its scores run: one commit group each
    auto stage = [&](uint4* dst, const __nv_bfloat16* src, int t) {
        if (t <= p / HB) {
            const int r0 = static_cast<int>((t * HB) % ring);
#pragma unroll 4
            for (int i = lane; i < HB * C; i += 32) {
                const int j = i / C, key = t * HB + j;
                const bool in = key >= lo && key <= p;
                int row = r0 + j;
                if (row >= ring) row -= static_cast<int>(ring);
                cp16z(dst + sw<D>(j, i % C), src + ((base + (in ? row : lor)) * hk + kh) * D + (i % C) * 8, in);
            }
        }
        commit();
    };
    stage(ks, kc, lo / HB + warp);
    stage(vs, vc, lo / HB + warp);
    for (int t = lo / HB + warp; t <= p / HB; t += 4) {
        wait_group<1>(); // tile t's K (its V may still be in flight)
        __syncwarp();
        float sc[4][4];
#pragma unroll
        for (int k = 0; k < K16; ++k) {
#pragma unroll
            for (int n = 0; n < 4; n += 2) {
                uint32_t b[4];
                ldmatrix4(b, ks + sw<D>(8 * (n + mt / 2) + lane % 8, 2 * k + mt % 2));
                if (k == 0) {
                    mma0(sc[n], qa[k], b[0], b[1]);
                    mma0(sc[n + 1], qa[k], b[2], b[3]);
                } else {
                    mma(sc[n], qa[k], b[0], b[1]);
                    mma(sc[n + 1], qa[k], b[2], b[3]);
                }
            }
        }
        __syncwarp();
        stage(ks, kc, t + 4);
        const int kb = t * HB + 2 * qd;
        float tm0 = -INFINITY, tm1 = -INFINITY;
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            const int c = kb + 8 * n;
            const bool i0 = c >= lo && c <= p, i1 = c + 1 >= lo && c + 1 <= p;
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
        const bool act = tm0 != -INFINITY; // the keys' mask is the same for every head
        const float n0 = act ? fmaxf(m0, tm0) : m0, n1 = act ? fmaxf(m1, tm1) : m1;
        const float a0 = act ? (m0 == -INFINITY ? 0.0f : texp(__fsub_rn(m0, n0))) : 1.0f;
        const float a1 = act ? (m1 == -INFINITY ? 0.0f : texp(__fsub_rn(m1, n1))) : 1.0f;
        float c0 = 0.0f, c1 = 0.0f;
#pragma unroll
        for (int n = 0; n < 4; ++n) {
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
        wait_group<1>(); // tile t's V (the next K may still be in flight)
        __syncwarp();
        uint32_t pa[2][4];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            pa[j][0] = pack(sc[2 * j][0], sc[2 * j][1]);
            pa[j][1] = pack(sc[2 * j][2], sc[2 * j][3]);
            pa[j][2] = pack(sc[2 * j + 1][0], sc[2 * j + 1][1]);
            pa[j][3] = pack(sc[2 * j + 1][2], sc[2 * j + 1][3]);
        }
#pragma unroll
        for (int i = 0; i < NB; i += 2) {
            uint32_t b[4];
            ldmatrix4t(b, vs + sw<D>(8 * (mt % 2) + lane % 8, i + mt / 2));
            mma(o[i], pa[0], b[0], b[1]);
            mma(o[i + 1], pa[0], b[2], b[3]);
        }
#pragma unroll
        for (int i = 0; i < NB; i += 2) {
            uint32_t b[4];
            ldmatrix4t(b, vs + sw<D>(16 + 8 * (mt % 2) + lane % 8, i + mt / 2));
            mma(o[i], pa[1], b[0], b[1]);
            mma(o[i + 1], pa[1], b[2], b[3]);
        }
        __syncwarp();
        stage(vs, vc, t + 4);
    }
    wait_group<0>();
    const int na = min(4, p / HB - lo / HB + 1); // warps that saw keys
    if (na == 1) { // one warp's fold is the result (the merge below would scale it by exp(0) = 1)
        if (warp == 0) {
#pragma unroll
            for (int i = 0; i < NB; ++i) {
                const int d = 8 * i + 2 * qd;
                if (g8 < G)
                    *reinterpret_cast<uint32_t*>(out + ((int64_t)r * h + kh * G + g8) * D + d) =
                        pack(tdiv(o[i][0], l0), tdiv(o[i][1], l0));
                if (g8 + 8 < G)
                    *reinterpret_cast<uint32_t*>(out + ((int64_t)r * h + kh * G + g8 + 8) * D + d) =
                        pack(tdiv(o[i][2], l1), tdiv(o[i][3], l1));
            }
        }
        return;
    }
    __syncthreads();
    float* os = reinterpret_cast<float*>(smem);   // [4 warps][16 heads][D]
    float* ms = os + 4 * 16 * D;                  // [4][16]
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
    float* as = ls + 4 * 16; // [4][16]: warp w's factor for head hd
    float* il = as + 4 * 16;  // [16]: the folded l
    if (threadIdx.x < 16) {
        const int hd = threadIdx.x;
        float mm = -INFINITY;
        for (int w = 0; w < 4; ++w) mm = fmaxf(mm, ms[w * 16 + hd]);
        float l = 0.0f;
        for (int w = 0; w < 4; ++w) {
            const float mw = ms[w * 16 + hd];
            const float a = mw == -INFINITY ? 0.0f : texp(__fsub_rn(mw, mm));
            as[w * 16 + hd] = a;
            if (mw != -INFINITY) l = __fmaf_rn(a, ls[w * 16 + hd], l);
        }
        il[hd] = l;
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < G * D / 2; idx += 128) {
        const int hd = idx / (D / 2), e = 2 * (idx % (D / 2));
        float acc0 = 0.0f, acc1 = 0.0f;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            if (ms[w * 16 + hd] == -INFINITY) continue;
            const float a = as[w * 16 + hd];
            const float2 v = *reinterpret_cast<const float2*>(os + (w * 16 + hd) * D + e);
            acc0 = __fmaf_rn(a, v.x, acc0);
            acc1 = __fmaf_rn(a, v.y, acc1);
        }
        const float l = il[hd];
        *reinterpret_cast<uint32_t*>(out + ((int64_t)r * h + kh * G + hd) * D + e) = pack(tdiv(acc0, l), tdiv(acc1, l));
    }
}

} // namespace tf_window_prompt

// Head dim 128, eight warps, four or eight query heads a block, eight staging slots.
template __global__ void tf_window_prompt::wattn_kernel<128, 8, 4, 8>(const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, __nv_bfloat16*, int, int, int, int, int, float, int, int);
template __global__ void tf_window_prompt::wattn_kernel<128, 8, 8, 8>(const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, __nv_bfloat16*, int, int, int, int, int, float, int, int);
