#pragma once

// Inkling 2's attention tile on HIP: a wave a (row, head) pair, shared K/V tiles, the same steps on every path.

#include <hip/hip_runtime.h>
#include <cstdint>
#include <math.h>

namespace ink2 {

constexpr int kDim = 128;
constexpr int kRel = 16;
constexpr int kExtent = 1024;
constexpr int kKeys = 64;
constexpr int kSub = 1024;
constexpr int kWaves = 4;        // pairs a block on the decode paths (tiles of 4 pairs)
constexpr int kPromptWaves = 8;  // pairs a block on one stream's prompt rows
constexpr int kKRow = kDim / 2 + 1;  // dwords a K row in LDS: the pad dword puts lane i's row in bank i
constexpr int kVRow = kDim / 2;      // dwords a V row: a wave reads one row's consecutive dwords
constexpr float kLog2e = 1.4426950408889634f;

// One stream's rows in a call: rows row0..row0+n-1 sit at positions p0..p0+n-1; `off` is its cache's element offset.
struct Seg {
    int row0;
    int n;
    int p0;
    int pad;
    long long off;
};

// A block's pairs: segment `seg`, pairs m0.. of it (pair m is row m / group, head m % group of the key head).
struct Tile {
    int seg;
    int m0;
};

__device__ __forceinline__ float bf(uint16_t u) { return __uint_as_float(static_cast<uint32_t>(u) << 16); }
__device__ __forceinline__ float lo16(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float hi16(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }

// Round to nearest even; NaN to 0x7fc0.
__device__ __forceinline__ uint16_t tobf(float f) {
    if (f != f) return 0x7fc0;
    const uint32_t u = __float_as_uint(f);
    return static_cast<uint16_t>((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

// An xor butterfly over the wave: every lane ends with the same bits (each add is commutative).
__device__ __forceinline__ float waveSum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o);
    return v;
}

__device__ __forceinline__ float waveMax(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o));
    return v;
}

// A part (m, l) into a running (mm, ll); a, b scale old and new sums. Each product and sum rounds once.
__device__ __forceinline__ void fold(float& mm, float& ll, float m, float l, float& a, float& b) {
    const float mn = fmaxf(mm, m);
    a = exp2f(mm - mn);
    b = exp2f(m - mn);
    ll = ll * a + l * b;
    mm = mn;
}

// The call's buffers and geometry, as the launch passes them.
struct Ptrs {
    const uint16_t* q;
    const uint16_t* kc;
    const uint16_t* vc;
    const uint16_t* tab;
    float* part_o;
    float2* part_ml;
};

struct Geo {
    int heads, kv_heads, ring, window, chunk, c0, chunks, split;
};

// A K/V tile's 16-byte pieces a thread holds: keys kt..kt+63 of a key head, zeros outside [k_lo, k_hi].
template <int THREADS>
struct Stage {
    static constexpr int kPer = kKeys * (kDim / 8) / THREADS;
    uint4 k[kPer], v[kPer];

    __device__ __forceinline__ void load(const uint16_t* kc, const uint16_t* vc, size_t head_base, int ring, int kt,
                                         int k_lo, int k_hi) {
        const size_t slot0 = static_cast<size_t>(kt % ring);
        for (int i = 0; i < kPer; ++i) {
            const int e = static_cast<int>(threadIdx.x) + i * THREADS;
            const int key = e >> 4, col = (e & 15) * 8;
            const int kp = kt + key;
            k[i] = v[i] = make_uint4(0u, 0u, 0u, 0u);
            if (kp >= k_lo && kp <= k_hi) {
                const size_t at = head_base + (slot0 + key) * kDim + col;
                k[i] = *reinterpret_cast<const uint4*>(kc + at);
                v[i] = *reinterpret_cast<const uint4*>(vc + at);
            }
        }
    }

    __device__ __forceinline__ void store(uint32_t* ks, uint32_t* vs) const {
        for (int i = 0; i < kPer; ++i) {
            const int e = static_cast<int>(threadIdx.x) + i * THREADS;
            const int key = e >> 4, c4 = (e & 15) * 4;
            uint32_t* kr = ks + key * kKRow + c4;
            kr[0] = k[i].x;
            kr[1] = k[i].y;
            kr[2] = k[i].z;
            kr[3] = k[i].w;
            *reinterpret_cast<uint4*>(vs + key * kVRow + c4) = v[i];
        }
    }
};

// A key's score in log2 units: (q.k / sqrt(128) + bias at its distance) log2e when live, else -inf.
__device__ __forceinline__ float score(float dot, int kp, int live_lo, int live_hi, int pos, const uint16_t* trow,
                                       float scale) {
    if (kp < live_lo || kp > live_hi) return -INFINITY;
    const int d = pos - kp;
    const float bias = d < kExtent ? bf(trow[d]) : 0.0f;
    return (dot * scale + bias) * kLog2e;
}

// One wave's pair over the tile in LDS: its live keys live_lo..live_hi, an online max, P rounded to bf16 for l and P.V.
__device__ __forceinline__ void tileStep(const uint32_t* ks, const uint32_t* vs, const float* qw, int kt, int live_lo,
                                         int live_hi, int pos, const uint16_t* trow, float scale, float& m, float& l,
                                         float* acc) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const uint32_t* ra = ks + lane * kKRow;
    const uint32_t* rb = ks + (lane + 32) * kKRow;
    const float2* q2 = reinterpret_cast<const float2*>(qw);
    float da = 0.0f, db = 0.0f;
    for (int j = 0; j < kDim / 2; ++j) {
        const float2 x = q2[j];
        const uint32_t a = ra[j], b = rb[j];
        da += x.x * lo16(a);
        da += x.y * hi16(a);
        db += x.x * lo16(b);
        db += x.y * hi16(b);
    }
    const float sa = score(da, kt + lane, live_lo, live_hi, pos, trow, scale);
    const float sb = score(db, kt + lane + 32, live_lo, live_hi, pos, trow, scale);
    const float mn = fmaxf(m, waveMax(fmaxf(sa, sb)));
    const float mu = mn == -INFINITY ? 0.0f : mn;
    const float alpha = exp2f(m - mu);
    m = mn;
    const float pa = bf(tobf(exp2f(sa - mu))), pb = bf(tobf(exp2f(sb - mu)));
    l = l * alpha + waveSum(pa + pb);
    for (int i = 0; i < 4; ++i) acc[i] *= alpha;
    const uint32_t* vcol = vs + 2 * lane;
    for (int j = live_lo - kt; j <= live_hi - kt; ++j) {
        const int from = __float_as_int(j < 32 ? pa : pb);
        const float p = __int_as_float(__builtin_amdgcn_readlane(from, j & 31));
        const uint32_t v0 = vcol[j * kVRow], v1 = vcol[j * kVRow + 1];
        acc[0] += p * lo16(v0);
        acc[1] += p * hi16(v0);
        acc[2] += p * lo16(v1);
        acc[3] += p * hi16(v1);
    }
}

// WAVES pairs m0.. of segment s, key head kvh, block z (FOLD: a global chunk); PIPE loads the next tile meanwhile.
template <int WAVES, bool FOLD, bool PIPE>
__device__ __forceinline__ void attnTile(Ptrs p, const Seg s, int m0, int kvh, int z, Geo g, float scale,
                                         uint32_t* ks, uint32_t* vs, float* qs) {
    const int group = g.heads / g.kv_heads;
    const int total = s.n * group;
    const int tile_hi = s.p0 + min(m0 + WAVES - 1, total - 1) / group;
    const int tile_lo = g.window > 0 ? max(0, s.p0 + m0 / group - g.window + 1) : 0;
    const int subs = g.chunk / kSub;
    int c, sub = 0;
    if (g.split) {
        c = g.c0 + z / subs;
        sub = z % subs;
    } else {
        c = (g.window > 0 ? tile_lo / g.chunk : g.c0) + z;
    }
    const int b_lo = c * g.chunk + sub * kSub, b_hi = b_lo + (g.split ? kSub : g.chunk) - 1;
    if (b_lo > tile_hi || b_hi < tile_lo) return;
    // a wave's pair: uniform values, so the P.V loop's bounds and readlane's lane sit in scalar registers
    const int w = __builtin_amdgcn_readfirstlane(static_cast<int>(threadIdx.x) >> 5);
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int mr = m0 + w;
    const bool ok = mr < total;
    const int lr = ok ? mr / group : 0;
    const int pos = ok ? s.p0 + lr : -1;
    const int lo = g.window > 0 ? max(0, pos - g.window + 1) : 0;
    const size_t prow = static_cast<size_t>(s.row0 + lr) * g.heads + kvh * group + (ok ? mr % group : 0);
    float* qw = qs + w * kDim;
    for (int i = 0; i < 4; ++i) qw[lane * 4 + i] = ok ? bf(p.q[prow * kDim + lane * 4 + i]) : 0.0f;
    const uint16_t* trow = p.tab + prow * kExtent;
    const size_t head_base = s.off + static_cast<size_t>(kvh) * g.ring * kDim;
    const int part = FOLD ? kSub : b_hi - b_lo + 1;
    const int k_lo = max(b_lo, tile_lo), k_hi = min(b_hi, tile_hi);
    const int first = k_lo & ~(kKeys - 1);
    const int n_tiles = (k_hi - first) / kKeys + 1;
    float m = -INFINITY, l = 0.0f, acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float M = -INFINITY, L = 0.0f, O[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    Stage<WAVES * 32> st;
    if (PIPE) st.load(p.kc, p.vc, head_base, g.ring, first, k_lo, k_hi);
    for (int it = 0; it < n_tiles; ++it) {
        const int kt = first + it * kKeys;
        const int s_lo = b_lo + (kt - b_lo) / part * part, s_hi = s_lo + part - 1;
        if (FOLD && kt == s_lo) {
            m = -INFINITY;
            l = 0.0f;
            for (int i = 0; i < 4; ++i) acc[i] = 0.0f;
        }
        __syncthreads();  // every wave is past the previous tile
        if (!PIPE) st.load(p.kc, p.vc, head_base, g.ring, kt, k_lo, k_hi);
        st.store(ks, vs);
        __syncthreads();
        if (PIPE && it + 1 < n_tiles) st.load(p.kc, p.vc, head_base, g.ring, kt + kKeys, k_lo, k_hi);
        const int live_lo = max(max(lo, s_lo), kt), live_hi = min(min(pos, s_hi), kt + kKeys - 1);
        if (ok) tileStep(ks, vs, qw, kt, live_lo, live_hi, pos, trow, scale, m, l, acc);
        if (FOLD && ok && s_lo <= pos && (kt + kKeys > s_hi || it == n_tiles - 1)) {
            float a, b;
            fold(M, L, m, l, a, b);
            for (int i = 0; i < 4; ++i) O[i] = O[i] * a + acc[i] * b;
        }
    }
    // a chunk outside the row's keys stores nothing: its slot is a neighbour (row, head)'s (hip_attention.partialSlot)
    if (!ok || c < lo / g.chunk || c > pos / g.chunk || b_lo > pos) return;
    const size_t at = g.split ? prow * g.chunks + (c - g.c0) * subs + sub
                              : prow * g.chunks + (c - (g.window > 0 ? lo / g.chunk : g.c0));
    if (lane == 0) p.part_ml[at] = FOLD ? make_float2(M, L) : make_float2(m, l);
    const float4 o4 = FOLD ? make_float4(O[0], O[1], O[2], O[3]) : make_float4(acc[0], acc[1], acc[2], acc[3]);
    *reinterpret_cast<float4*>(p.part_o + at * kDim + lane * 4) = o4;
}

}  // namespace ink2
