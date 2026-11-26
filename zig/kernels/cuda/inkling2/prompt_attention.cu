#include "../common/softmax_identity.cuh"
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
    const bool same = TF_CUDA_SOFTMAX_IDENTITY_GUARD && isfinite(mn) && mn == mm; a = same ? 1.0f : exp2f(mm - mn);
    b = exp2f(m - mn);
    ll = (same && isfinite(__fadd_rn(ll, __fmul_rn(l, b)))) ? __fadd_rn(ll, __fmul_rn(l, b)) : __fadd_rn(__fmul_rn(ll, a), __fmul_rn(l, b));
    mm = mn;
}

__device__ __forceinline__ float scaled(float o, float a, float p, float b) {
    return __fadd_rn(__fmul_rn(o, a), __fmul_rn(p, b));
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
                                           const uint16_t* __restrict__ vc, const uint16_t* __restrict__ tab,
                                           const Seg* __restrict__ segs, float* __restrict__ part_o,
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
    for (int k = 0; k < 8; ++k) {
        for (int i = 0; i < 2; ++i) {
            const uint32_t* src = reinterpret_cast<const uint32_t*>(q + prow[i] * kDim + 16 * k + 2 * t);
            qa[k][i] = ok[i] ? src[0] : 0u;
            qa[k][i + 2] = ok[i] ? src[4] : 0u;
        }
    }
    float acc[16][4];
    float m[2], l[2];
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
                for (int j = 0; j < 8; ++j) {
                    for (int e = 0; e < 4; ++e) {
                        const int i = e >> 1;
                        const int kp = kt + 8 * j + 2 * t + (e & 1);
                        const bool live = ok[i] && kp <= pos[i] && kp >= s_lo && kp <= s_hi;
                        const int d = pos[i] - kp;
                        const float bias = live && d < kExtent ? bf(tab[prow[i] * kExtent + d]) : 0.0f;
                        sc[j][e] = live ? (sc[j][e] * scale + bias) * kLog2e : -INFINITY;
                        mx[i] = fmaxf(mx[i], sc[j][e]);
                    }
                }
            }
            float alpha[2], mu[2]; bool same[2];
            for (int i = 0; i < 2; ++i) {
                mx[i] = fmaxf(mx[i], __shfl_xor_sync(0xffffffffu, mx[i], 1));
                mx[i] = fmaxf(mx[i], __shfl_xor_sync(0xffffffffu, mx[i], 2));
                const float mn = fmaxf(m[i], mx[i]);
                mu[i] = mn == -INFINITY ? 0.0f : mn;
                same[i] = TF_CUDA_SOFTMAX_IDENTITY_GUARD && isfinite(mu[i]) && mu[i] == m[i]; alpha[i] = same[i] ? 1.0f : exp2f(m[i] - mu[i]);
                m[i] = mn;
                if (!same[i] || !isfinite(l[i])) l[i] *= alpha[i];
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
                if (!same[0] || !isfinite(acc[n][0])) acc[n][0] *= alpha[0];
                if (!same[0] || !isfinite(acc[n][1])) acc[n][1] *= alpha[0];
                if (!same[1] || !isfinite(acc[n][2])) acc[n][2] *= alpha[1];
                if (!same[1] || !isfinite(acc[n][3])) acc[n][3] *= alpha[1];
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
                const bool same = TF_CUDA_SOFTMAX_IDENTITY_GUARD && isfinite(fmaxf(M[i], m[i])) && fmaxf(M[i], m[i]) == M[i]; fold(M[i], L[i], m[i], l[i], a, b);
                for (int n = 0; n < 16; ++n) {
                    O[n][2 * i] = (same && isfinite(__fadd_rn(O[n][2 * i], __fmul_rn(acc[n][2 * i], b)))) ? __fadd_rn(O[n][2 * i], __fmul_rn(acc[n][2 * i], b)) : scaled(O[n][2 * i], a, acc[n][2 * i], b);
                    O[n][2 * i + 1] = (same && isfinite(__fadd_rn(O[n][2 * i + 1], __fmul_rn(acc[n][2 * i + 1], b)))) ? __fadd_rn(O[n][2 * i + 1], __fmul_rn(acc[n][2 * i + 1], b)) : scaled(O[n][2 * i + 1], a, acc[n][2 * i + 1], b);
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
        const uint16_t *__restrict__ tab, const ink2p::Seg *__restrict__ segs, float *__restrict__ part_o,      \
        float2 *__restrict__ part_ml, int heads, int kv_heads, int ring, int chunk, int c0, int chunks, float scale

// One stream's prompt rows on a global layer with bf16 caches.
extern "C" __global__ void __launch_bounds__(ink2p::kThreads, 1) tf_ink2_attn_prompt(TF_INK2_PROMPT_ARGS) {
    extern __shared__ __align__(16) uint16_t smem[];
    ink2p::promptTile(q, kc, vc, tab, segs, part_o, part_ml, heads, kv_heads, ring, chunk, c0, chunks, scale, smem);
}
