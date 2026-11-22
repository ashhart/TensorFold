// Inkling 2 global-layer prompt attention: 128 (row, head) pairs share each K/V tile, with the split path's bits.
#include <cstdint>
#include <cuda_bf16.h>

#include "common/qmm_frag.cuh"

namespace ink2p {

constexpr int kDim = 128;
constexpr int kExtent = 1024;
constexpr int kKeys = 64;
constexpr int kSub = 1024;
constexpr int kWarps = 8;
constexpr int kPairs = kWarps * 16;
constexpr int kThreads = kWarps * 32;
constexpr int kStages = 3;
constexpr int kTile = kKeys * kDim;  // halves in a K or V tile: [key][128], 16-byte pieces swizzled by key
constexpr int kStage = 2 * kTile;    // a stage: its K tile, then its V tile
constexpr float kLog2e = 1.4426950408889634f;

struct Seg {
    int row0;
    int n;
    int p0;
    int pad;
    long long off;
};

__device__ __forceinline__ float bf(uint16_t u) { return __uint_as_float(static_cast<uint32_t>(u) << 16); }

__device__ __forceinline__ uint16_t tobf(float f) { return __bfloat16_as_ushort(__float2bfloat16_rn(f)); }

__device__ __forceinline__ uint32_t pack(uint16_t lo, uint16_t hi) {
    return static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
}

__device__ __forceinline__ void fold(float& mm, float& ll, float m, float l, float& a, float& b) {
    const float mn = fmaxf(mm, m);
    a = exp2f(mm - mn);
    b = exp2f(m - mn);
    ll = __fadd_rn(__fmul_rn(ll, a), __fmul_rn(l, b));
    mm = mn;
}

__device__ __forceinline__ float scaled(float o, float a, float p, float b) {
    return __fadd_rn(__fmul_rn(o, a), __fmul_rn(p, b));
}

// A (row, head)'s prepared rel vector, kept packed: rp[j / 2] holds rel[j] (low half) and rel[j + 1] (high).
__device__ __forceinline__ void loadRel(uint32_t (&rp)[8], const uint16_t* __restrict__ rel, size_t rh, bool ok) {
    const uint4* src = reinterpret_cast<const uint4*>(rel + rh * 16);
    const uint4 a = ok ? src[0] : make_uint4(0u, 0u, 0u, 0u), b = ok ? src[1] : make_uint4(0u, 0u, 0u, 0u);
    rp[0] = a.x, rp[1] = a.y, rp[2] = a.z, rp[3] = a.w, rp[4] = b.x, rp[5] = b.y, rp[6] = b.z, rp[7] = b.w;
}

// The bias at key distance d, formed as tf_ink2_rel_table forms its table entry (attention.cu, -O3: one fma per j
// from 0.0f, j ascending, rounded to bf16), from the projection transposed to [extent][16]: two 16-byte loads.
__device__ __forceinline__ float relBias(const uint32_t (&rp)[8], const uint16_t* __restrict__ proj_t, int d) {
    const uint4* src = reinterpret_cast<const uint4*>(proj_t + static_cast<size_t>(d) * 16);
    const uint4 a = src[0], b = src[1];
    const uint32_t w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        acc = __fmaf_rn(__uint_as_float(rp[j] << 16), __uint_as_float(w[j] << 16), acc);
        acc = __fmaf_rn(__uint_as_float(rp[j] & 0xffff0000u), __uint_as_float(w[j] & 0xffff0000u), acc);
    }
    return bf(tobf(acc));
}

// The half offset of a tile's 16-byte piece `c` (dims 8c..8c+7) of key `key`: pieces XOR-swizzled by key & 7.
__device__ __forceinline__ int piece(int key, int c) { return key * kDim + ((c ^ (key & 7)) << 3); }

// The K/V tile load: keys kt..kt+63 of a key head into the stage's bf16 tiles, zeros outside [k_lo, k_hi] (async).
__device__ __forceinline__ void loadTile(uint16_t* ks, uint16_t* vs, const uint16_t* __restrict__ kc,
                                         const uint16_t* __restrict__ vc, size_t head_base, int ring, int kt,
                                         int k_lo, int k_hi) {
    const size_t slot0 = static_cast<size_t>(kt % ring);
    for (int e = threadIdx.x; e < kKeys * (kDim / 8); e += kThreads) {
        const int key = e >> 4, col = (e & 15) * 8;
        const int kp = kt + key;
        const bool in = kp >= k_lo && kp <= k_hi;
        const size_t at = in ? head_base + (slot0 + key) * kDim + col : head_base;
        tf_common::cp16z(ks + piece(key, e & 15), kc + at, in);
        tf_common::cp16z(vs + piece(key, e & 15), vc + at, in);
    }
    tf_common::cp_commit();
}

// A block's prompt pairs (tile of kPairs pairs, key head, chunk).
__device__ __forceinline__ void promptTile(const uint16_t* __restrict__ q, const uint16_t* __restrict__ kc,
                                           const uint16_t* __restrict__ vc, const uint16_t* __restrict__ rel,
                                           const uint16_t* __restrict__ proj_t, const Seg* __restrict__ segs,
                                           float* __restrict__ part_o,
                                           float2* __restrict__ part_ml, int heads, int kv_heads, int ring,
                                           int chunk, int c0, int chunks, float scale, uint16_t* smem) {
    const Seg s = segs[0];
    const int group = heads / kv_heads;
    const int total = s.n * group;
    const int m0 = static_cast<int>(blockIdx.x) * kPairs;
    if (m0 >= total) return;
    const int tile_hi = s.p0 + min(m0 + kPairs - 1, total - 1) / group;
    const int kvh = blockIdx.y;
    const int c = c0 + static_cast<int>(blockIdx.z);
    const int b_lo = c * chunk, b_hi = b_lo + chunk - 1;
    if (b_lo > tile_hi) return;
    const int k_hi = min(b_hi, tile_hi);
    const int n_tiles = (k_hi - b_lo) / kKeys + 1;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int w0 = m0 + warp * 16;
    const bool busy = w0 < total;
    const bool full = w0 + 15 < total;
    const int w_first = s.p0 + w0 / group, w_last = s.p0 + min(w0 + 15, total - 1) / group;
    int pos[2];
    bool ok[2];
    size_t prow[2];
    uint32_t qa[8][4];
    for (int i = 0; i < 2; ++i) {
        const int mr = w0 + g + i * 8;
        ok[i] = mr < total;
        const int lr = ok[i] ? mr / group : 0;
        pos[i] = ok[i] ? s.p0 + lr : -1;
        prow[i] = static_cast<size_t>(s.row0 + lr) * heads + kvh * group + (ok[i] ? mr % group : 0);
    }
    uint32_t rp[2][8];
    for (int i = 0; i < 2; ++i) loadRel(rp[i], rel, prow[i], ok[i]);
    for (int k = 0; k < 8; ++k) {
        for (int i = 0; i < 2; ++i) {
            const uint32_t* src = reinterpret_cast<const uint32_t*>(q + prow[i] * kDim + 16 * k + 2 * t);
            qa[k][i] = ok[i] ? src[0] : 0u;
            qa[k][i + 2] = ok[i] ? src[4] : 0u;
        }
    }
    float acc[16][4];
    float m[2] = {-INFINITY, -INFINITY};
    float l[2] = {0.0f, 0.0f};
    float O[16][4];
    float M[2] = {-INFINITY, -INFINITY}, L[2] = {0.0f, 0.0f};
    for (int n = 0; n < 16; ++n) O[n][0] = O[n][1] = O[n][2] = O[n][3] = 0.0f;
    const size_t head_base = s.off + static_cast<size_t>(kvh) * ring * kDim;
    // ldmatrix lanes: K rows 8j + (lane & 7), pieces 2k + lane / 8; V rows 16u + lane % 16, pieces n + lane / 16
    const int k_row = lane & 7, k_pc = lane >> 3;
    const int v_row = lane & 15, v_pc = lane >> 4;
    const uint16_t* k16 = kc;
    const uint16_t* v16 = vc;
    for (int st = 0; st < kStages - 1; ++st) {
        if (st < n_tiles) loadTile(smem + st * kStage, smem + st * kStage + kTile, k16, v16, head_base, ring,
                                   b_lo + st * kKeys, b_lo, k_hi);
        else tf_common::cp_commit();
    }
    for (int it = 0; it < n_tiles; ++it) {
        const int kt = b_lo + it * kKeys;
        const int s_lo = b_lo + (it / (kSub / kKeys)) * kSub, s_hi = s_lo + kSub - 1;
        if (it % (kSub / kKeys) == 0) {
            for (int n = 0; n < 16; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.0f;
            m[0] = m[1] = -INFINITY;
            l[0] = l[1] = 0.0f;
        }
        tf_common::cp_wait<kStages - 2>();
        __syncthreads();  // tile `it` is in; every warp is past tile it-1, whose stage the next load takes
        if (it + kStages - 1 < n_tiles) {
            uint16_t* nx = smem + ((it + kStages - 1) % kStages) * kStage;
            loadTile(nx, nx + kTile, k16, v16, head_base, ring, kt + (kStages - 1) * kKeys, b_lo, k_hi);
        } else {
            tf_common::cp_commit();
        }
        const uint16_t* ks = smem + (it % kStages) * kStage;
        const uint16_t* vs = ks + kTile;
        if (busy && kt <= w_last) {  // a tile past all of a warp's rows is an exact no-op
            float sc[8][4];
            for (int j = 0; j < 8; ++j) {
                sc[j][0] = sc[j][1] = sc[j][2] = sc[j][3] = 0.0f;
                for (int k = 0; k < 8; k += 2) {
                    uint32_t b[4];
                    qmm_frag::ldmatrix4(b, tf_common::smem(ks + piece(8 * j + k_row, 2 * k + k_pc)));
                    qmm_frag::mma(sc[j], qa[k][0], qa[k][1], qa[k][2], qa[k][3], b[0], b[1]);
                    qmm_frag::mma(sc[j], qa[k + 1][0], qa[k + 1][1], qa[k + 1][2], qa[k + 1][3], b[2], b[3]);
                }
            }
            float mx[2] = {-INFINITY, -INFINITY};
            if (full && kt + kKeys - 1 + kExtent <= w_first) {  // every key live, every bias 0
                for (int j = 0; j < 8; ++j) {
                    for (int e = 0; e < 4; ++e) {
                        const int i = e >> 1;
                        const float bias = 0.0f;
                        sc[j][e] = (sc[j][e] * scale + bias) * kLog2e;
                        mx[i] = fmaxf(mx[i], sc[j][e]);
                    }
                }
            } else {
#pragma unroll
                for (int j = 0; j < 8; ++j) {
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const int i = e >> 1;
                        const int kp = kt + 8 * j + 2 * t + (e & 1);
                        const bool live = ok[i] && kp <= pos[i] && kp >= s_lo && kp <= s_hi;
                        const int d = pos[i] - kp;
                        const float bias = live && d < kExtent ? relBias(rp[i], proj_t, d) : 0.0f;
                        sc[j][e] = live ? (sc[j][e] * scale + bias) * kLog2e : -INFINITY;
                        mx[i] = fmaxf(mx[i], sc[j][e]);
                    }
                }
            }
            float alpha[2], mu[2];
            for (int i = 0; i < 2; ++i) {
                mx[i] = fmaxf(mx[i], __shfl_xor_sync(0xffffffffu, mx[i], 1));
                mx[i] = fmaxf(mx[i], __shfl_xor_sync(0xffffffffu, mx[i], 2));
                const float mn = fmaxf(m[i], mx[i]);
                mu[i] = mn == -INFINITY ? 0.0f : mn;
                alpha[i] = exp2f(m[i] - mu[i]);
                m[i] = mn;
                l[i] *= alpha[i];
            }
            uint32_t pk[8][2];  // P as the A fragments: pk[j][i] packs keys 8j+2t, +1 of row g + 8i
            for (int j = 0; j < 8; ++j) {
                uint16_t p[4];
                for (int e = 0; e < 4; ++e) {
                    const int i = e >> 1;
                    p[e] = tobf(exp2f(sc[j][e] - mu[i]));
                    l[i] += bf(p[e]);
                }
                pk[j][0] = pack(p[0], p[1]);
                pk[j][1] = pack(p[2], p[3]);
            }
            for (int n = 0; n < 16; ++n) {
                acc[n][0] *= alpha[0];
                acc[n][1] *= alpha[0];
                acc[n][2] *= alpha[1];
                acc[n][3] *= alpha[1];
            }
            for (int u = 0; u < 4; ++u) {
                const uint32_t a0 = pk[2 * u][0], a1 = pk[2 * u][1], a2 = pk[2 * u + 1][0], a3 = pk[2 * u + 1][1];
                for (int n = 0; n < 16; n += 2) {
                    uint32_t b[4];
                    qmm_frag::ldmatrix4t(b, tf_common::smem(vs + piece(16 * u + v_row, n + v_pc)));
                    qmm_frag::mma(acc[n], a0, a1, a2, a3, b[0], b[1]);
                    qmm_frag::mma(acc[n + 1], a0, a1, a2, a3, b[2], b[3]);
                }
            }
        }
        if (busy && (it % (kSub / kKeys) == kSub / kKeys - 1 || it == n_tiles - 1)) {
            for (int i = 0; i < 2; ++i) {
                l[i] += __shfl_xor_sync(0xffffffffu, l[i], 1);
                l[i] += __shfl_xor_sync(0xffffffffu, l[i], 2);
            }
            for (int i = 0; i < 2; ++i) {
                if (!ok[i] || s_lo > pos[i]) continue;  // the merge folds a row's sub-chunks up to its own
                float a, b;
                fold(M[i], L[i], m[i], l[i], a, b);
                for (int n = 0; n < 16; ++n) {
                    O[n][2 * i] = scaled(O[n][2 * i], a, acc[n][2 * i], b);
                    O[n][2 * i + 1] = scaled(O[n][2 * i + 1], a, acc[n][2 * i + 1], b);
                }
            }
        }
    }
    for (int i = 0; i < 2; ++i) {
        if (!ok[i] || b_lo > pos[i]) continue;  // the merge reads chunks up to the row's own
        const size_t at = prow[i] * chunks + (c - c0);
        if (t == 0) part_ml[at] = make_float2(M[i], L[i]);
        float* o = part_o + at * kDim + 2 * t;
        for (int n = 0; n < 16; ++n) *reinterpret_cast<float2*>(o + 8 * n) = make_float2(O[n][2 * i], O[n][2 * i + 1]);
    }
}

}  // namespace ink2p

#define TF_INK2_PROMPT_ARGS                                                                                       \
    const uint16_t *__restrict__ q, const uint16_t *__restrict__ kc, const uint16_t *__restrict__ vc,             \
        const uint16_t *__restrict__ rel, const uint16_t *__restrict__ proj_t, const ink2p::Seg *__restrict__ segs, \
        float *__restrict__ part_o, float2 *__restrict__ part_ml, int heads, int kv_heads, int ring, int chunk, int c0, int chunks, float scale

namespace ink2p {

// One stream's prompt rows on a sliding window layer with bf16 caches. Same 8-warp kPairs=128,
// 3-stage cp16z, qmm_frag MMA skeleton as tf_ink2_attn_prompt, but a single online softmax over the
// window [b_lo, k_hi] with per-element masking (d = pos - kp, live iff 0 <= d < window) and a direct
// bf16 epilogue (sink+gate merge folded in, no part_o/part_ml, no separate merge kernel).
// b_lo is 64-aligned so kt % ring never wraps a tile mid-load.
__device__ __forceinline__ void promptTileSliding(const uint16_t* __restrict__ q, const uint16_t* __restrict__ kc,
                                                  const uint16_t* __restrict__ vc, const uint16_t* __restrict__ rel,
                                                  const uint16_t* __restrict__ proj_t, const Seg* __restrict__ segs,
                                                  const uint16_t* __restrict__ sink,
                                                  const uint16_t* __restrict__ gate, uint16_t* __restrict__ out,
                                                  int heads, int kv_heads, int ring, int window, float scale,
                                                  int gate_ld, uint16_t* smem) {
    const Seg s = segs[0];
    const int group = heads / kv_heads;
    const int total = s.n * group;
    const int m0 = static_cast<int>(blockIdx.x) * kPairs;
    if (m0 >= total) return;
    const int tile_first_pos = s.p0 + m0 / group;
    const int tile_last_pos = s.p0 + min(m0 + kPairs - 1, total - 1) / group;
    const int kvh = blockIdx.y;
    const int lo_key = tile_first_pos + 1 > window ? tile_first_pos + 1 - window : 0;
    const int b_lo = lo_key / kKeys * kKeys;
    const int k_hi = tile_last_pos;
    const int n_tiles = (k_hi - b_lo) / kKeys + 1;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int w0 = m0 + warp * 16;
    const bool busy = w0 < total;
    const int w_first = s.p0 + w0 / group, w_last = s.p0 + min(w0 + 15, total - 1) / group;
    const int w_lo = w_first + 1 > window ? w_first + 1 - window : 0;
    int pos[2];
    bool ok[2];
    size_t prow[2];
    uint32_t qa[8][4];
    for (int i = 0; i < 2; ++i) {
        const int mr = w0 + g + i * 8;
        ok[i] = mr < total;
        const int lr = ok[i] ? mr / group : 0;
        pos[i] = ok[i] ? s.p0 + lr : -1;
        prow[i] = static_cast<size_t>(s.row0 + lr) * heads + kvh * group + (ok[i] ? mr % group : 0);
    }
    uint32_t rp[2][8];
    for (int i = 0; i < 2; ++i) loadRel(rp[i], rel, prow[i], ok[i]);
    for (int k = 0; k < 8; ++k) {
        for (int i = 0; i < 2; ++i) {
            const uint32_t* src = reinterpret_cast<const uint32_t*>(q + prow[i] * kDim + 16 * k + 2 * t);
            qa[k][i] = ok[i] ? src[0] : 0u;
            qa[k][i + 2] = ok[i] ? src[4] : 0u;
        }
    }
    float acc[16][4];
    float m[2] = {-INFINITY, -INFINITY};
    float l[2] = {0.0f, 0.0f};
    for (int n = 0; n < 16; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.0f;
    const size_t head_base = s.off + static_cast<size_t>(kvh) * ring * kDim;
    const int k_row = lane & 7, k_pc = lane >> 3;
    const int v_row = lane & 15, v_pc = lane >> 4;
    const uint16_t* k16 = kc;
    const uint16_t* v16 = vc;
    for (int st = 0; st < kStages - 1; ++st) {
        if (st < n_tiles) loadTile(smem + st * kStage, smem + st * kStage + kTile, k16, v16, head_base, ring,
                                   b_lo + st * kKeys, b_lo, k_hi);
        else tf_common::cp_commit();
    }
    for (int it = 0; it < n_tiles; ++it) {
        const int kt = b_lo + it * kKeys;
        // Warps whose rows see no live keys in this tile still join every barrier; they skip only math.
        const bool live_tile = !busy || (kt <= w_last && kt + kKeys - 1 >= w_lo);
        tf_common::cp_wait<kStages - 2>();
        __syncthreads();
        if (it + kStages - 1 < n_tiles) {
            uint16_t* nx = smem + ((it + kStages - 1) % kStages) * kStage;
            loadTile(nx, nx + kTile, k16, v16, head_base, ring, kt + (kStages - 1) * kKeys, b_lo, k_hi);
        } else {
            tf_common::cp_commit();
        }
        const uint16_t* ks = smem + (it % kStages) * kStage;
        const uint16_t* vs = ks + kTile;
        if (!busy || !live_tile) continue;
        float sc[8][4];
        for (int j = 0; j < 8; ++j) {
            sc[j][0] = sc[j][1] = sc[j][2] = sc[j][3] = 0.0f;
            for (int k = 0; k < 8; k += 2) {
                uint32_t b[4];
                qmm_frag::ldmatrix4(b, tf_common::smem(ks + piece(8 * j + k_row, 2 * k + k_pc)));
                qmm_frag::mma(sc[j], qa[k][0], qa[k][1], qa[k][2], qa[k][3], b[0], b[1]);
                qmm_frag::mma(sc[j], qa[k + 1][0], qa[k + 1][1], qa[k + 1][2], qa[k + 1][3], b[2], b[3]);
            }
        }
        float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int j = 0; j < 8; ++j) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int i = e >> 1;
                const int kp = kt + 8 * j + 2 * t + (e & 1);
                const int d = pos[i] - kp;
                const bool live = ok[i] && d >= 0 && d < window;
                const float bias = live && d < kExtent ? relBias(rp[i], proj_t, d) : 0.0f;
                sc[j][e] = live ? (sc[j][e] * scale + bias) * kLog2e : -INFINITY;
                mx[i] = fmaxf(mx[i], sc[j][e]);
            }
        }
        float alpha[2], mu[2];
        for (int i = 0; i < 2; ++i) {
            mx[i] = fmaxf(mx[i], __shfl_xor_sync(0xffffffffu, mx[i], 1));
            mx[i] = fmaxf(mx[i], __shfl_xor_sync(0xffffffffu, mx[i], 2));
            const float mn = fmaxf(m[i], mx[i]);
            mu[i] = mn == -INFINITY ? 0.0f : mn;
            alpha[i] = exp2f(m[i] - mu[i]);
            m[i] = mn;
            l[i] *= alpha[i];
        }
        uint32_t pk[8][2];
        for (int j = 0; j < 8; ++j) {
            uint16_t p[4];
            for (int e = 0; e < 4; ++e) {
                const int i = e >> 1;
                p[e] = tobf(exp2f(sc[j][e] - mu[i]));
                l[i] += bf(p[e]);
            }
            pk[j][0] = pack(p[0], p[1]);
            pk[j][1] = pack(p[2], p[3]);
        }
        for (int n = 0; n < 16; ++n) {
            acc[n][0] *= alpha[0];
            acc[n][1] *= alpha[0];
            acc[n][2] *= alpha[1];
            acc[n][3] *= alpha[1];
        }
        for (int u = 0; u < 4; ++u) {
            const uint32_t a0 = pk[2 * u][0], a1 = pk[2 * u][1], a2 = pk[2 * u + 1][0], a3 = pk[2 * u + 1][1];
            for (int n = 0; n < 16; n += 2) {
                uint32_t b[4];
                qmm_frag::ldmatrix4t(b, tf_common::smem(vs + piece(16 * u + v_row, n + v_pc)));
                qmm_frag::mma(acc[n], a0, a1, a2, a3, b[0], b[1]);
                qmm_frag::mma(acc[n + 1], a0, a1, a2, a3, b[2], b[3]);
            }
        }
    }
    for (int i = 0; i < 2; ++i) {
        l[i] += __shfl_xor_sync(0xffffffffu, l[i], 1);
        l[i] += __shfl_xor_sync(0xffffffffu, l[i], 2);
    }
    // Direct fused epilogue (attention.cu merge formula): sink-adjusted softmax, gated, bf16 out.
    for (int i = 0; i < 2; ++i) {
        if (!ok[i]) continue;
        const float sk = bf(sink[prow[i]]) * kLog2e;
        const float mf = fmaxf(m[i], sk);
        const float a = exp2f(m[i] - mf);
        const float inv_denom = a / (l[i] * a + exp2f(sk - mf));
        const uint16_t* grow = gate + (prow[i] / heads) * gate_ld + (prow[i] % heads) * kDim;  // gate_ld apart a row
        for (int n = 0; n < 16; ++n) {
            const int col = 8 * n + 2 * t;
            const float y0 = acc[n][2 * i] * inv_denom;
            const float y1 = acc[n][2 * i + 1] * inv_denom;
            const float g0 = bf(grow[col]);
            const float g1 = bf(grow[col + 1]);
            out[prow[i] * kDim + col] = tobf(y0 * (1.0f / (1.0f + expf(-g0))));
            out[prow[i] * kDim + col + 1] = tobf(y1 * (1.0f / (1.0f + expf(-g1))));
        }
    }
}


}  // namespace ink2p

// One stream's prompt rows on a sliding window: direct bf16 out, no merge pass.
extern "C" __global__ void __launch_bounds__(ink2p::kThreads, 1) tf_ink2_attn_prompt_sliding(
    const uint16_t* __restrict__ q, const uint16_t* __restrict__ kc, const uint16_t* __restrict__ vc,
    const uint16_t* __restrict__ rel, const uint16_t* __restrict__ proj_t, const ink2p::Seg* __restrict__ segs,
    const uint16_t* __restrict__ sink, const uint16_t* __restrict__ gate, uint16_t* __restrict__ out, int heads, int kv_heads, int ring,
    int window, float scale, int gate_ld) {
    extern __shared__ __align__(16) uint16_t smem[];
    ink2p::promptTileSliding(q, kc, vc, rel, proj_t, segs, sink, gate, out, heads, kv_heads, ring, window, scale, gate_ld, smem);
}

// One stream's prompt rows on a global layer with bf16 caches.
extern "C" __global__ void __launch_bounds__(ink2p::kThreads, 1) tf_ink2_attn_prompt(TF_INK2_PROMPT_ARGS) {
    extern __shared__ __align__(16) uint16_t smem[];
    ink2p::promptTile(q, kc, vc, rel, proj_t, segs, part_o, part_ml, heads, kv_heads, ring, chunk, c0, chunks, scale, smem);
}

namespace ink2p {

__device__ __forceinline__ uint4 dequant8(uint2 p, float sc) {
    const uint32_t lo = p.x, hi = p.y;
    const uint16_t b0 = tobf(static_cast<float>(static_cast<int8_t>(lo & 0xff)) * sc);
    const uint16_t b1 = tobf(static_cast<float>(static_cast<int8_t>((lo >> 8) & 0xff)) * sc);
    const uint16_t b2 = tobf(static_cast<float>(static_cast<int8_t>((lo >> 16) & 0xff)) * sc);
    const uint16_t b3 = tobf(static_cast<float>(static_cast<int8_t>((lo >> 24) & 0xff)) * sc);
    const uint16_t b4 = tobf(static_cast<float>(static_cast<int8_t>(hi & 0xff)) * sc);
    const uint16_t b5 = tobf(static_cast<float>(static_cast<int8_t>((hi >> 8) & 0xff)) * sc);
    const uint16_t b6 = tobf(static_cast<float>(static_cast<int8_t>((hi >> 16) & 0xff)) * sc);
    const uint16_t b7 = tobf(static_cast<float>(static_cast<int8_t>((hi >> 24) & 0xff)) * sc);
    return make_uint4(pack(b0, b1), pack(b2, b3), pack(b4, b5), pack(b6, b7));
}

__device__ __forceinline__ void promptTileI8(const uint16_t* __restrict__ q, const int8_t* __restrict__ kc,
                                             const int8_t* __restrict__ vc, const float* __restrict__ k_scales,
                                             const float* __restrict__ v_scales, const uint16_t* __restrict__ rel,
                                             const uint16_t* __restrict__ proj_t, const Seg* __restrict__ segs,
                                             float* __restrict__ part_o, float2* __restrict__ part_ml, int heads,
                                             int kv_heads, int ring, int chunk, int c0, int chunks, float scale,
                                             uint16_t* smem) {
    const Seg s = segs[0];
    const int group = heads / kv_heads;
    const int total = s.n * group;
    const int m0 = static_cast<int>(blockIdx.x) * kPairs;
    if (m0 >= total) return;
    const int tile_hi = s.p0 + min(m0 + kPairs - 1, total - 1) / group;
    const int kvh = blockIdx.y;
    const int c = c0 + static_cast<int>(blockIdx.z);
    const int b_lo = c * chunk, b_hi = b_lo + chunk - 1;
    if (b_lo > tile_hi) return;
    const int k_hi = min(b_hi, tile_hi);
    const int n_tiles = (k_hi - b_lo) / kKeys + 1;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int w0 = m0 + warp * 16;
    const bool busy = w0 < total;
    const bool full = w0 + 15 < total;
    const int w_first = s.p0 + w0 / group, w_last = s.p0 + min(w0 + 15, total - 1) / group;
    int pos[2];
    bool ok[2];
    size_t prow[2];
    uint32_t qa[8][4];
    for (int i = 0; i < 2; ++i) {
        const int mr = w0 + g + i * 8;
        ok[i] = mr < total;
        const int lr = ok[i] ? mr / group : 0;
        pos[i] = ok[i] ? s.p0 + lr : -1;
        prow[i] = static_cast<size_t>(s.row0 + lr) * heads + kvh * group + (ok[i] ? mr % group : 0);
    }
    uint32_t rp[2][8];
    for (int i = 0; i < 2; ++i) loadRel(rp[i], rel, prow[i], ok[i]);
    for (int k = 0; k < 8; ++k) {
        for (int i = 0; i < 2; ++i) {
            const uint32_t* src = reinterpret_cast<const uint32_t*>(q + prow[i] * kDim + 16 * k + 2 * t);
            qa[k][i] = ok[i] ? src[0] : 0u;
            qa[k][i + 2] = ok[i] ? src[4] : 0u;
        }
    }
    float acc[16][4];
    float m[2] = {-INFINITY, -INFINITY};
    float l[2] = {0.0f, 0.0f};
    float O[16][4];
    float M[2] = {-INFINITY, -INFINITY}, L[2] = {0.0f, 0.0f};
    for (int n = 0; n < 16; ++n) O[n][0] = O[n][1] = O[n][2] = O[n][3] = 0.0f;
    const size_t byte_off = static_cast<size_t>(s.off) * 2;
    const int8_t* kc_s = kc + byte_off;
    const int8_t* vc_s = vc + byte_off;
    const float* ks_s = reinterpret_cast<const float*>(reinterpret_cast<const uint8_t*>(k_scales) + byte_off);
    const float* vs_s = reinterpret_cast<const float*>(reinterpret_cast<const uint8_t*>(v_scales) + byte_off);
    const size_t head_slot_base = static_cast<size_t>(kvh) * ring;
    const int k_row = lane & 7, k_pc = lane >> 3;
    const int v_row = lane & 15, v_pc = lane >> 4;
    uint16_t* ks = smem;
    uint16_t* vs = smem + kTile;
    for (int it = 0; it < n_tiles; ++it) {
        const int kt = b_lo + it * kKeys;
        const int s_lo = b_lo + (it / (kSub / kKeys)) * kSub, s_hi = s_lo + kSub - 1;
        if (it % (kSub / kKeys) == 0) {
            for (int n = 0; n < 16; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.0f;
            m[0] = m[1] = -INFINITY;
            l[0] = l[1] = 0.0f;
        }
        __syncthreads();
        const size_t slot0 = static_cast<size_t>(kt % ring);
        for (int e = threadIdx.x; e < kKeys * (kDim / 8); e += kThreads) {
            const int key = e >> 4, col = (e & 15) * 8;
            const int kp = kt + key;
            uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
            if (kp >= b_lo && kp <= k_hi) {
                const size_t slot = head_slot_base + slot0 + key;
                const size_t at = slot * kDim + col;
                kv = dequant8(*reinterpret_cast<const uint2*>(kc_s + at), ks_s[slot]);
                vv = dequant8(*reinterpret_cast<const uint2*>(vc_s + at), vs_s[slot]);
            }
            const int pc = piece(key, e & 15);
            *reinterpret_cast<uint4*>(ks + pc) = kv;
            *reinterpret_cast<uint4*>(vs + pc) = vv;
        }
        __syncthreads();
        if (busy && kt <= w_last) {
            float sc[8][4];
            for (int j = 0; j < 8; ++j) {
                sc[j][0] = sc[j][1] = sc[j][2] = sc[j][3] = 0.0f;
                for (int k = 0; k < 8; k += 2) {
                    uint32_t b[4];
                    qmm_frag::ldmatrix4(b, tf_common::smem(ks + piece(8 * j + k_row, 2 * k + k_pc)));
                    qmm_frag::mma(sc[j], qa[k][0], qa[k][1], qa[k][2], qa[k][3], b[0], b[1]);
                    qmm_frag::mma(sc[j], qa[k + 1][0], qa[k + 1][1], qa[k + 1][2], qa[k + 1][3], b[2], b[3]);
                }
            }
            float mx[2] = {-INFINITY, -INFINITY};
            if (full && kt + kKeys - 1 + kExtent <= w_first) {
                for (int j = 0; j < 8; ++j) {
                    for (int e = 0; e < 4; ++e) {
                        const int i = e >> 1;
                        sc[j][e] = (sc[j][e] * scale) * kLog2e;
                        mx[i] = fmaxf(mx[i], sc[j][e]);
                    }
                }
            } else {
#pragma unroll
                for (int j = 0; j < 8; ++j) {
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const int i = e >> 1;
                        const int kp = kt + 8 * j + 2 * t + (e & 1);
                        const bool live = ok[i] && kp <= pos[i] && kp >= s_lo && kp <= s_hi;
                        const int d = pos[i] - kp;
                        const float bias = live && d < kExtent ? relBias(rp[i], proj_t, d) : 0.0f;
                        sc[j][e] = live ? (sc[j][e] * scale + bias) * kLog2e : -INFINITY;
                        mx[i] = fmaxf(mx[i], sc[j][e]);
                    }
                }
            }
            float alpha[2], mu[2];
            for (int i = 0; i < 2; ++i) {
                mx[i] = fmaxf(mx[i], __shfl_xor_sync(0xffffffffu, mx[i], 1));
                mx[i] = fmaxf(mx[i], __shfl_xor_sync(0xffffffffu, mx[i], 2));
                const float mn = fmaxf(m[i], mx[i]);
                mu[i] = mn == -INFINITY ? 0.0f : mn;
                alpha[i] = exp2f(m[i] - mu[i]);
                m[i] = mn;
                l[i] *= alpha[i];
            }
            uint32_t pk[8][2];
            for (int j = 0; j < 8; ++j) {
                uint16_t p[4];
                for (int e = 0; e < 4; ++e) {
                    const int i = e >> 1;
                    p[e] = tobf(exp2f(sc[j][e] - mu[i]));
                    l[i] += bf(p[e]);
                }
                pk[j][0] = pack(p[0], p[1]);
                pk[j][1] = pack(p[2], p[3]);
            }
            for (int n = 0; n < 16; ++n) {
                acc[n][0] *= alpha[0];
                acc[n][1] *= alpha[0];
                acc[n][2] *= alpha[1];
                acc[n][3] *= alpha[1];
            }
            for (int u = 0; u < 4; ++u) {
                const uint32_t a0 = pk[2 * u][0], a1 = pk[2 * u][1], a2 = pk[2 * u + 1][0], a3 = pk[2 * u + 1][1];
                for (int n = 0; n < 16; n += 2) {
                    uint32_t b[4];
                    qmm_frag::ldmatrix4t(b, tf_common::smem(vs + piece(16 * u + v_row, n + v_pc)));
                    qmm_frag::mma(acc[n], a0, a1, a2, a3, b[0], b[1]);
                    qmm_frag::mma(acc[n + 1], a0, a1, a2, a3, b[2], b[3]);
                }
            }
        }
        if (busy && (it % (kSub / kKeys) == kSub / kKeys - 1 || it == n_tiles - 1)) {
            for (int i = 0; i < 2; ++i) {
                l[i] += __shfl_xor_sync(0xffffffffu, l[i], 1);
                l[i] += __shfl_xor_sync(0xffffffffu, l[i], 2);
            }
            for (int i = 0; i < 2; ++i) {
                if (!ok[i] || s_lo > pos[i]) continue;
                float a, b;
                fold(M[i], L[i], m[i], l[i], a, b);
                for (int n = 0; n < 16; ++n) {
                    O[n][2 * i] = scaled(O[n][2 * i], a, acc[n][2 * i], b);
                    O[n][2 * i + 1] = scaled(O[n][2 * i + 1], a, acc[n][2 * i + 1], b);
                }
            }
        }
    }
    for (int i = 0; i < 2; ++i) {
        if (!ok[i] || b_lo > pos[i]) continue;
        const size_t at = prow[i] * chunks + (c - c0);
        if (t == 0) part_ml[at] = make_float2(M[i], L[i]);
        float* o = part_o + at * kDim + 2 * t;
        for (int n = 0; n < 16; ++n) *reinterpret_cast<float2*>(o + 8 * n) = make_float2(O[n][2 * i], O[n][2 * i + 1]);
    }
}

}  // namespace ink2p

extern "C" __global__ void __launch_bounds__(ink2p::kThreads, 1) tf_ink2_attn_prompt_i8(
    const uint16_t* __restrict__ q, const int8_t* __restrict__ kc, const int8_t* __restrict__ vc,
    const float* __restrict__ k_scales, const float* __restrict__ v_scales, const uint16_t* __restrict__ rel,
    const uint16_t* __restrict__ proj_t, const ink2p::Seg* __restrict__ segs, float* __restrict__ part_o,
    float2* __restrict__ part_ml, int heads, int kv_heads, int ring, int chunk, int c0, int chunks, float scale) {
    extern __shared__ __align__(16) uint16_t smem[];
    ink2p::promptTileI8(q, kc, vc, k_scales, v_scales, rel, proj_t, segs, part_o, part_ml, heads, kv_heads, ring,
                        chunk, c0, chunks, scale, smem);
}
