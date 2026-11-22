// Inkling 2 attention: q/k/v/rel preparation, the relative-bias table, chunked tensor-core attention and the merge.
#include <cstdint>
#include <cuda_bf16.h>

#include "common/qmm_frag.cuh"

namespace ink2 {

constexpr int kDim = 128;
constexpr int kRel = 16;
constexpr int kExtent = 1024;
constexpr int kKeys = 64;
constexpr int kPad = kDim + 8;
constexpr float kLog2e = 1.4426950408889634f;

// One stream's rows in a call: rows row0..row0+n-1 sit at positions p0..p0+n-1; `off` is its cache's element offset.
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

__device__ __forceinline__ float warpSum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// L2 clip of one head vector held 4 values a lane: scale down to `thr` when longer, as Inkling2L2Clip.
__device__ __forceinline__ void clip(float* x, float thr) {
    const float norm = sqrtf(warpSum(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]));
    const float f = fminf(thr / (norm + 1e-6f), 1.0f);
    for (int i = 0; i < 4; ++i) x[i] *= f;
}

// Per row and head: q = clip(q * q_scale) (* tau), rel = clip(rel * rel_scale) (* tau), k = clip(k), v = rms(v).
__device__ __forceinline__ void prepRow(const uint16_t* __restrict__ q_raw, const uint16_t* __restrict__ k_raw,
                                        const uint16_t* __restrict__ v_raw, const uint16_t* __restrict__ rel_raw,
                                        const uint16_t* __restrict__ q_scale, const uint16_t* __restrict__ rel_scale,
                                        const int* __restrict__ rpos, const int* __restrict__ rseg,
                                        const Seg* __restrict__ segs, uint16_t* __restrict__ q_out,
                                        uint16_t* __restrict__ rel_out, uint16_t* __restrict__ kc,
                                        uint16_t* __restrict__ vc, int heads, int kv_heads, int ring, int global,
                                        float qk_clip, float rel_clip, float floor_pos, float alpha, float eps,
                                        int q_ld, int rel_ld) {

    const int row = blockIdx.x, y = blockIdx.y, lane = threadIdx.x;
    const int pos = rpos[row];
    const float tau = global ? 1.0f + alpha * logf(fmaxf(static_cast<float>(pos + 1) / floor_pos, 1.0f)) : 1.0f;
    float x[4];
    if (y < heads) {
        const uint16_t* src = q_raw + static_cast<size_t>(row) * q_ld + y * kDim;
        for (int i = 0; i < 4; ++i) x[i] = bf(src[lane * 4 + i]) * bf(q_scale[y * kDim + lane * 4 + i]);
        clip(x, qk_clip);
        uint16_t* dst = q_out + (static_cast<size_t>(row) * heads + y) * kDim;
        for (int i = 0; i < 4; ++i) dst[lane * 4 + i] = global ? tobf(bf(tobf(x[i])) * tau) : tobf(x[i]);
        float r = lane < kRel ? bf(rel_raw[static_cast<size_t>(row) * rel_ld + y * kRel + lane]) *
                                    bf(rel_scale[y * kRel + lane])
                              : 0.0f;
        const float norm = sqrtf(warpSum(r * r));
        r *= fminf(rel_clip / (norm + 1e-6f), 1.0f);
        if (lane < kRel) {
            const uint16_t v = global ? tobf(bf(tobf(r)) * tau) : tobf(r);
            rel_out[(static_cast<size_t>(row) * heads + y) * kRel + lane] = v;
        }
        return;
    }
    const bool key = y < heads + kv_heads;
    const int h = key ? y - heads : y - heads - kv_heads;
    const uint16_t* src = (key ? k_raw : v_raw) + (static_cast<size_t>(row) * kv_heads + h) * kDim;
    for (int i = 0; i < 4; ++i) x[i] = bf(src[lane * 4 + i]);
    if (key) {
        clip(x, qk_clip);
    } else {
        const float r = rsqrtf(warpSum(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]) / kDim + eps);
        for (int i = 0; i < 4; ++i) x[i] *= r;
    }
    const Seg s = segs[rseg[row]];
    uint16_t* dst = (key ? kc : vc) + s.off + (static_cast<size_t>(h) * ring + pos % ring) * kDim;
    for (int i = 0; i < 4; ++i) dst[lane * 4 + i] = tobf(x[i]);
}

}  // namespace ink2

using ink2::Seg;

#define TF_INK2_PREP_ARGS                                                                                          \
    const uint16_t *__restrict__ q_raw, const uint16_t *__restrict__ k_raw, const uint16_t *__restrict__ v_raw,    \
        const uint16_t *__restrict__ rel_raw, const uint16_t *__restrict__ q_scale,                                \
        const uint16_t *__restrict__ rel_scale, const int *__restrict__ rpos, const int *__restrict__ rseg,        \
        const Seg *__restrict__ segs, uint16_t *__restrict__ q_out, uint16_t *__restrict__ rel_out,                \
        uint16_t *__restrict__ kc, uint16_t *__restrict__ vc, int heads, int kv_heads, int ring, int global,       \
        float qk_clip, float rel_clip, float floor_pos, float alpha, float eps, int q_ld, int rel_ld
#define TF_INK2_PREP_PASS                                                                                          \
    q_raw, k_raw, v_raw, rel_raw, q_scale, rel_scale, rpos, rseg, segs, q_out, rel_out, kc, vc, heads, kv_heads,   \
        ring, global, qk_clip, rel_clip, floor_pos, alpha, eps, q_ld, rel_ld

extern "C" __global__ void tf_ink2_prep(TF_INK2_PREP_ARGS) {
    ink2::prepRow(TF_INK2_PREP_PASS);
}

extern "C" __global__ void tf_ink2_prep_i8(
    const uint16_t* __restrict__ q_raw, const uint16_t* __restrict__ k_raw, const uint16_t* __restrict__ v_raw,
    const uint16_t* __restrict__ rel_raw, const uint16_t* __restrict__ q_scale,
    const uint16_t* __restrict__ rel_scale, const int* __restrict__ rpos, const int* __restrict__ rseg,
    const Seg* __restrict__ segs, uint16_t* __restrict__ q_out, uint16_t* __restrict__ rel_out,
    int8_t* __restrict__ kc, int8_t* __restrict__ vc, float* __restrict__ k_scales, float* __restrict__ v_scales,
    int heads, int kv_heads, int ring, int global, float qk_clip, float rel_clip, float floor_pos, float alpha,
    float eps, int q_ld, int rel_ld) {
    using namespace ink2;
    const int row = blockIdx.x, y = blockIdx.y, lane = threadIdx.x;
    const int pos = rpos[row];
    const float tau = global ? 1.0f + alpha * logf(fmaxf(static_cast<float>(pos + 1) / floor_pos, 1.0f)) : 1.0f;
    float x[4];
    if (y < heads) {
        const uint16_t* src = q_raw + static_cast<size_t>(row) * q_ld + y * kDim;
        for (int i = 0; i < 4; ++i) x[i] = bf(src[lane * 4 + i]) * bf(q_scale[y * kDim + lane * 4 + i]);
        clip(x, qk_clip);
        uint16_t* dst = q_out + (static_cast<size_t>(row) * heads + y) * kDim;
        for (int i = 0; i < 4; ++i) dst[lane * 4 + i] = global ? tobf(bf(tobf(x[i])) * tau) : tobf(x[i]);
        float r = lane < kRel ? bf(rel_raw[static_cast<size_t>(row) * rel_ld + y * kRel + lane]) *
                                    bf(rel_scale[y * kRel + lane])
                              : 0.0f;
        const float norm = sqrtf(warpSum(r * r));
        r *= fminf(rel_clip / (norm + 1e-6f), 1.0f);
        if (lane < kRel) {
            const uint16_t v = global ? tobf(bf(tobf(r)) * tau) : tobf(r);
            rel_out[(static_cast<size_t>(row) * heads + y) * kRel + lane] = v;
        }
        return;
    }
    const bool key = y < heads + kv_heads;
    const int h = key ? y - heads : y - heads - kv_heads;
    const uint16_t* src = (key ? k_raw : v_raw) + (static_cast<size_t>(row) * kv_heads + h) * kDim;
    for (int i = 0; i < 4; ++i) x[i] = bf(src[lane * 4 + i]);
    if (key) {
        clip(x, qk_clip);
    } else {
        const float r = rsqrtf(warpSum(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]) / kDim + eps);
        for (int i = 0; i < 4; ++i) x[i] *= r;
    }
    float amax = fmaxf(fmaxf(fabsf(x[0]), fabsf(x[1])), fmaxf(fabsf(x[2]), fabsf(x[3])));
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float sc = amax > 0.0f ? amax / 127.0f : 0.0f;
    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    const Seg s = segs[rseg[row]];
    const size_t byte_off = static_cast<size_t>(s.off) * 2;
    const size_t slot = static_cast<size_t>(h) * ring + pos % ring;
    int8_t* dst = (key ? kc : vc) + byte_off + slot * kDim + lane * 4;
    for (int i = 0; i < 4; ++i) {
        int qv = __float2int_rn(x[i] * inv);
        if (qv > 127) qv = 127;
        if (qv < -127) qv = -127;
        dst[i] = static_cast<int8_t>(qv);
    }
    if (lane == 0) {
        float* sdst = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(key ? k_scales : v_scales) + byte_off);
        sdst[slot] = sc;
    }
}

// tab[row, head, d] = bf16(rel[row, head, :] . proj[:, d]) for d < 1024, fp32 sums in tap order (rel_logits_proj).
extern "C" __global__ void tf_ink2_rel_table(const uint16_t* __restrict__ rel, const uint16_t* __restrict__ proj,
                                             uint16_t* __restrict__ tab) {
    using namespace ink2;
    const size_t rh = blockIdx.x;
    float r[kRel];
    for (int j = 0; j < kRel; ++j) r[j] = bf(rel[rh * kRel + j]);
    for (int d = threadIdx.x; d < kExtent; d += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < kRel; ++j) acc += r[j] * bf(proj[j * kExtent + d]);
        tab[rh * kExtent + d] = tobf(acc);
    }
}

namespace ink2 {

// The bias at key distance d as tf_ink2_rel_table forms its entry (one fma per j from 0.0f, j ascending, rounded to
// bf16; prompt_attention.cu's relBias), from a (row, head)'s prepared rel vector and the projection transposed to
// [extent][16]: the table entry's bits, no table.
__device__ __forceinline__ uint16_t relBiasBits(const uint4 (&rv)[2], const uint16_t* __restrict__ proj_t, int d) {
    const uint4* src = reinterpret_cast<const uint4*>(proj_t + static_cast<size_t>(d) * kRel);
    const uint4 a = src[0], b = src[1];
    const uint32_t w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    const uint32_t r[8] = {rv[0].x, rv[0].y, rv[0].z, rv[0].w, rv[1].x, rv[1].y, rv[1].z, rv[1].w};
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        acc = __fmaf_rn(__uint_as_float(r[j] << 16), __uint_as_float(w[j] << 16), acc);
        acc = __fmaf_rn(__uint_as_float(r[j] & 0xffff0000u), __uint_as_float(w[j] & 0xffff0000u), acc);
    }
    return tobf(acc);
}

// A block's bias scratch: each of its 64 pairs' rel vector and position, and one key tile's biases [pair][key].
struct BiasSmem {
    uint4 rel[64][2];
    int pos[64];
    uint16_t bias[64 * kKeys];
};

// A global chunk's keys fold in sub-chunks of kSub, in a block each (split) or in order in the chunk's: same bits.
constexpr int kSub = 1024;

// A part (m, l) into a running (mm, ll); a, b scale old and new sums (explicit roundings: both kernels fold alike).
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

// 64 (row, head) pairs of a key head over a block's keys (FOLD: a global chunk, split: a sub-chunk, else a chunk).
template <bool FOLD>
__device__ __forceinline__ void attnTile(const uint16_t* __restrict__ q, const uint16_t* __restrict__ kc,
                                         const uint16_t* __restrict__ vc, const uint16_t* __restrict__ rel,
                                         const uint16_t* __restrict__ proj_t, const Seg* __restrict__ segs, const int2* __restrict__ tiles,
                                         float* __restrict__ part_o, float2* __restrict__ part_ml, int heads,
                                         int kv_heads, int ring, int window, int chunk, int c0, int chunks,
                                         float scale, int split, uint16_t* ks, uint16_t* vs, BiasSmem& bs) {
    const int2 tile = tiles[blockIdx.x];
    const Seg s = segs[tile.x];
    const int group = heads / kv_heads;
    const int total = s.n * group;
    const int m0 = tile.y;
    const int m_last = min(m0 + 63, total - 1);
    const int tile_hi = s.p0 + m_last / group;
    const int tile_lo = window > 0 ? max(0, s.p0 + m0 / group - window + 1) : 0;
    const int subs = chunk / kSub;
    const int kvh = blockIdx.y;
    // sliding layers count chunks from the tile's first key and store them from each row's first chunk
    int c, sub = 0;
    if (split) {
        c = c0 + static_cast<int>(blockIdx.z) / subs;
        sub = static_cast<int>(blockIdx.z) % subs;
    } else {
        c = (window > 0 ? tile_lo / chunk : c0) + static_cast<int>(blockIdx.z);
    }
    const int b_lo = c * chunk + sub * kSub, b_hi = b_lo + (split ? kSub : chunk) - 1;
    if (b_lo > tile_hi || b_hi < tile_lo) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const bool busy = m0 + warp * 16 < total;  // a warp past the tile's pairs only loads keys
    int mr[2], pos[2], lo[2];
    bool ok[2];
    size_t prow[2];
    uint32_t qa[8][4];
    for (int i = 0; i < 2; ++i) {
        mr[i] = m0 + warp * 16 + g + i * 8;
        ok[i] = mr[i] < total;
        const int lr = ok[i] ? mr[i] / group : 0;
        pos[i] = ok[i] ? s.p0 + lr : -1;
        lo[i] = window > 0 ? max(0, pos[i] - window + 1) : 0;
        prow[i] = static_cast<size_t>(s.row0 + lr) * heads + kvh * group + (ok[i] ? mr[i] % group : 0);
    }
    // the tile's pairs' rel vectors and positions, once (pair p is m0 + p: warp p / 16, lane group p % 8, half p / 8 % 2)
    const int n_pairs = min(64, total - m0);
    for (int p = threadIdx.x; p < n_pairs; p += 128) {
        const int m = m0 + p;
        const size_t rh = static_cast<size_t>(s.row0 + m / group) * heads + kvh * group + m % group;
        const uint4* src = reinterpret_cast<const uint4*>(rel + rh * kRel);
        bs.rel[p][0] = src[0];
        bs.rel[p][1] = src[1];
        bs.pos[p] = s.p0 + m / group;
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
    float O[FOLD ? 16 : 1][4];
    float M[2] = {-INFINITY, -INFINITY}, L[2] = {0.0f, 0.0f};
    if (FOLD)
        for (int n = 0; n < 16; ++n) O[n][0] = O[n][1] = O[n][2] = O[n][3] = 0.0f;
    const size_t head_base = s.off + static_cast<size_t>(kvh) * ring * kDim;
    const int step = FOLD ? kSub : b_hi - b_lo + 1;
    for (int s_lo = b_lo; s_lo <= b_hi && s_lo <= tile_hi; s_lo += step) {
        const int s_hi = s_lo + step - 1;
        for (int n = 0; n < 16; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.0f;
        m[0] = m[1] = -INFINITY;
        l[0] = l[1] = 0.0f;
        const int k_lo = max(s_lo, tile_lo), k_hi = min(s_hi, tile_hi);
        for (int kt = k_lo & ~(kKeys - 1); kt <= k_hi; kt += kKeys) {
            __syncthreads();
            const size_t slot0 = static_cast<size_t>(kt % ring);
            for (int e = threadIdx.x; e < kKeys * (kDim / 8); e += 128) {
                const int key = e >> 4, col = (e & 15) * 8;
                const int kp = kt + key;
                uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
                if (kp >= k_lo && kp <= k_hi) {
                    const size_t at = head_base + (slot0 + key) * kDim + col;
                    kv = *reinterpret_cast<const uint4*>(kc + at);
                    vv = *reinterpret_cast<const uint4*>(vc + at);
                }
                *reinterpret_cast<uint4*>(ks + key * kPad + col) = kv;
                *reinterpret_cast<uint4*>(vs + key * kPad + col) = vv;
            }
            // the tile's biases, every warp a share (busy or not): a pair's keys within kExtent of it and inside its keys
            // here (else 0, as the table's reader took), as bf16 bits
            for (int e = threadIdx.x; e < n_pairs * kKeys; e += 128) {
                const int p = e / kKeys, key = e % kKeys;
                const int kp = kt + key, pp = bs.pos[p], d = pp - kp;
                const int plo = window > 0 ? max(0, pp - window + 1) : 0;
                const bool need = kp >= plo && kp <= pp && kp >= s_lo && kp <= s_hi && d < kExtent;
                bs.bias[e] = need ? relBiasBits(bs.rel[p], proj_t, d) : 0;
            }
            __syncthreads();
            if (!busy) continue;
            float sc[8][4];
            for (int j = 0; j < 8; ++j) {
                sc[j][0] = sc[j][1] = sc[j][2] = sc[j][3] = 0.0f;
                const uint16_t* krow = ks + (8 * j + g) * kPad + 2 * t;
                for (int k = 0; k < 8; ++k) {
                    const uint32_t b0 = *reinterpret_cast<const uint32_t*>(krow + 16 * k);
                    const uint32_t b1 = *reinterpret_cast<const uint32_t*>(krow + 16 * k + 8);
                    qmm_frag::mma(sc[j], qa[k][0], qa[k][1], qa[k][2], qa[k][3], b0, b1);
                }
            }
            float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
            for (int j = 0; j < 8; ++j) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int i = e >> 1;
                    const int kp = kt + 8 * j + 2 * t + (e & 1);
                    const bool live = ok[i] && kp >= lo[i] && kp <= pos[i] && kp >= s_lo && kp <= s_hi;
                    const int d = pos[i] - kp;
                    const float bias = live && d < kExtent ? bf(bs.bias[(warp * 16 + g + i * 8) * kKeys + (kp - kt)]) : 0.0f;
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
            uint16_t p[8][4];
            for (int j = 0; j < 8; ++j) {
                for (int e = 0; e < 4; ++e) {
                    const int i = e >> 1;
                    p[j][e] = tobf(exp2f(sc[j][e] - mu[i]));
                    l[i] += bf(p[j][e]);
                }
            }
            for (int n = 0; n < 16; ++n) {
                acc[n][0] *= alpha[0];
                acc[n][1] *= alpha[0];
                acc[n][2] *= alpha[1];
                acc[n][3] *= alpha[1];
            }
            for (int u = 0; u < 4; ++u) {
                const uint32_t a0 = pack(p[2 * u][0], p[2 * u][1]), a1 = pack(p[2 * u][2], p[2 * u][3]);
                const uint32_t a2 = pack(p[2 * u + 1][0], p[2 * u + 1][1]);
                const uint32_t a3 = pack(p[2 * u + 1][2], p[2 * u + 1][3]);
                const uint16_t* v0 = vs + (16 * u + 2 * t) * kPad + g;
                for (int n = 0; n < 16; ++n) {
                    const uint16_t* vp = v0 + 8 * n;
                    const uint32_t b0 = pack(vp[0], vp[kPad]);
                    const uint32_t b1 = pack(vp[8 * kPad], vp[9 * kPad]);
                    qmm_frag::mma(acc[n], a0, a1, a2, a3, b0, b1);
                }
            }
        }
        for (int i = 0; i < 2; ++i) {
            l[i] += __shfl_xor_sync(0xffffffffu, l[i], 1);
            l[i] += __shfl_xor_sync(0xffffffffu, l[i], 2);
        }
        if (FOLD) {
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
        // a chunk outside the row's keys stores nothing: its slot is a neighbour (row, head)'s (kernels.partialSlot)
        if (!ok[i] || c < lo[i] / chunk || c > pos[i] / chunk || b_lo > pos[i]) continue;
        const size_t at = split ? prow[i] * chunks + (c - c0) * subs + sub
                                : prow[i] * chunks + (c - (window > 0 ? lo[i] / chunk : c0));
        if (t == 0) part_ml[at] = FOLD ? make_float2(M[i], L[i]) : make_float2(m[i], l[i]);
        float* o = part_o + at * kDim + 2 * t;
        for (int n = 0; n < 16; ++n) {
            const float x = FOLD ? O[FOLD ? n : 0][2 * i] : acc[n][2 * i];
            const float y = FOLD ? O[FOLD ? n : 0][2 * i + 1] : acc[n][2 * i + 1];
            *reinterpret_cast<float2*>(o + 8 * n) = make_float2(x, y);
        }
    }
}

}  // namespace ink2

// Sliding layers (a chunk a block) and split global calls (a sub-chunk a block, `chunks` partials a row and head).
// Biases are formed inline from `rel` (prepared, [rows*heads][16]) and `proj_t` ([extent][16]), as the prompt kernels.
extern "C" __global__ void __launch_bounds__(128)
    tf_ink2_attn(const uint16_t* __restrict__ q, const uint16_t* __restrict__ kc, const uint16_t* __restrict__ vc,
                 const uint16_t* __restrict__ rel, const uint16_t* __restrict__ proj_t, const Seg* __restrict__ segs,
                 const int2* __restrict__ tiles,
                 float* __restrict__ part_o, float2* __restrict__ part_ml, int heads, int kv_heads, int ring,
                 int window, int chunk, int c0, int chunks, float scale, int split) {
    using namespace ink2;
    __shared__ __align__(16) uint16_t ks[kKeys * kPad];
    __shared__ __align__(16) uint16_t vs[kKeys * kPad];
    __shared__ __align__(16) BiasSmem bs;
    attnTile<false>(q, kc, vc, rel, proj_t, segs, tiles, part_o, part_ml, heads, kv_heads, ring, window, chunk, c0, chunks,
                    scale, split, ks, vs, bs);
}

// Global layers a chunk a block, its sub-chunks folded in order (prompt calls: no partial a sub-chunk).
extern "C" __global__ void __launch_bounds__(128)
    tf_ink2_attn_fold(const uint16_t* __restrict__ q, const uint16_t* __restrict__ kc,
                      const uint16_t* __restrict__ vc, const uint16_t* __restrict__ rel,
                      const uint16_t* __restrict__ proj_t, const Seg* __restrict__ segs,
                      const int2* __restrict__ tiles, float* __restrict__ part_o, float2* __restrict__ part_ml,
                      int heads, int kv_heads, int ring, int chunk, int c0, int chunks, float scale) {
    using namespace ink2;
    __shared__ __align__(16) uint16_t ks[kKeys * kPad];
    __shared__ __align__(16) uint16_t vs[kKeys * kPad];
    __shared__ __align__(16) BiasSmem bs;
    attnTile<true>(q, kc, vc, rel, proj_t, segs, tiles, part_o, part_ml, heads, kv_heads, ring, 0, chunk, c0, chunks, scale,
                   0, ks, vs, bs);
}

// Per (row, head): its chunks folded in order (subs > 1: each from its sub-chunks first), the sink, then the gate
// (a row's heads `gate_ld` elements after the last row's: the fused projection's rows are read in place).
extern "C" __global__ void tf_ink2_merge(const float* __restrict__ part_o, const float2* __restrict__ part_ml,
                                         const uint16_t* __restrict__ sink, const uint16_t* __restrict__ gate,
                                         const int* __restrict__ rpos, uint16_t* __restrict__ out, int heads,
                                         int window, int chunk, int c0, int chunks, int subs, int gate_ld) {
    using namespace ink2;
    const size_t rh = blockIdx.x;
    const int row = static_cast<int>(rh / heads), d = threadIdx.x;
    const int pos = rpos[row];
    const int lo = window > 0 ? max(0, pos - window + 1) : 0;
    float mm = -INFINITY, ll = 0.0f, o = 0.0f;
    for (int c = lo / chunk; c <= pos / chunk; ++c) {
        float2 ml;
        float po;
        if (subs > 1) {
            float M = -INFINITY, L = 0.0f, O = 0.0f;
            const int last = min(subs - 1, (pos - c * chunk) / kSub);
            for (int u = 0; u <= last; ++u) {
                const size_t at = rh * chunks + (c - c0) * subs + u;
                const float2 sm = part_ml[at];
                float a, b;
                fold(M, L, sm.x, sm.y, a, b);
                O = scaled(O, a, part_o[at * kDim + d], b);
            }
            ml = make_float2(M, L);
            po = O;
        } else {
            const size_t at = rh * chunks + (c - (window > 0 ? lo / chunk : c0));
            ml = part_ml[at];
            po = part_o[at * kDim + d];
        }
        const float mn = fmaxf(mm, ml.x);
        const float a = exp2f(mm - mn), b = exp2f(ml.x - mn);
        ll = ll * a + ml.y * b;
        o = o * a + po * b;
        mm = mn;
    }
    const float sk = bf(sink[rh]) * kLog2e;
    const float mf = fmaxf(mm, sk);
    const float a = exp2f(mm - mf);
    const float y = o * a / (ll * a + exp2f(sk - mf));
    const float gv = bf(gate[static_cast<size_t>(row) * gate_ld + (rh - static_cast<size_t>(row) * heads) * kDim + d]);
    out[rh * kDim + d] = tobf(y * (1.0f / (1.0f + expf(-gv))));
}

namespace ink2 {

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

template <bool FOLD>
__device__ __forceinline__ void attnTileI8(const uint16_t* __restrict__ q, const int8_t* __restrict__ kc,
                                           const int8_t* __restrict__ vc, const float* __restrict__ k_scales,
                                           const float* __restrict__ v_scales, const uint16_t* __restrict__ rel,
                                           const uint16_t* __restrict__ proj_t, const Seg* __restrict__ segs,
                                           const int2* __restrict__ tiles, float* __restrict__ part_o,
                                           float2* __restrict__ part_ml, int heads, int kv_heads, int ring,
                                           int window, int chunk, int c0, int chunks, float scale, int split,
                                           uint16_t* ks, uint16_t* vs, BiasSmem& bs) {
    const int2 tile = tiles[blockIdx.x];
    const Seg s = segs[tile.x];
    const int group = heads / kv_heads;
    const int total = s.n * group;
    const int m0 = tile.y;
    const int m_last = min(m0 + 63, total - 1);
    const int tile_hi = s.p0 + m_last / group;
    const int tile_lo = window > 0 ? max(0, s.p0 + m0 / group - window + 1) : 0;
    const int subs = chunk / kSub;
    const int kvh = blockIdx.y;
    int c, sub = 0;
    if (split) {
        c = c0 + static_cast<int>(blockIdx.z) / subs;
        sub = static_cast<int>(blockIdx.z) % subs;
    } else {
        c = (window > 0 ? tile_lo / chunk : c0) + static_cast<int>(blockIdx.z);
    }
    const int b_lo = c * chunk + sub * kSub, b_hi = b_lo + (split ? kSub : chunk) - 1;
    if (b_lo > tile_hi || b_hi < tile_lo) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const bool busy = m0 + warp * 16 < total;
    int mr[2], pos[2], lo[2];
    bool ok[2];
    size_t prow[2];
    uint32_t qa[8][4];
    for (int i = 0; i < 2; ++i) {
        mr[i] = m0 + warp * 16 + g + i * 8;
        ok[i] = mr[i] < total;
        const int lr = ok[i] ? mr[i] / group : 0;
        pos[i] = ok[i] ? s.p0 + lr : -1;
        lo[i] = window > 0 ? max(0, pos[i] - window + 1) : 0;
        prow[i] = static_cast<size_t>(s.row0 + lr) * heads + kvh * group + (ok[i] ? mr[i] % group : 0);
    }
    const int n_pairs = min(64, total - m0);
    for (int p = threadIdx.x; p < n_pairs; p += 128) {
        const int m = m0 + p;
        const size_t rh = static_cast<size_t>(s.row0 + m / group) * heads + kvh * group + m % group;
        const uint4* src = reinterpret_cast<const uint4*>(rel + rh * kRel);
        bs.rel[p][0] = src[0];
        bs.rel[p][1] = src[1];
        bs.pos[p] = s.p0 + m / group;
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
    float O[FOLD ? 16 : 1][4];
    float M[2] = {-INFINITY, -INFINITY}, L[2] = {0.0f, 0.0f};
    if (FOLD)
        for (int n = 0; n < 16; ++n) O[n][0] = O[n][1] = O[n][2] = O[n][3] = 0.0f;
    const size_t byte_off = static_cast<size_t>(s.off) * 2;
    const int8_t* kc_s = kc + byte_off;
    const int8_t* vc_s = vc + byte_off;
    const float* ks_s = reinterpret_cast<const float*>(reinterpret_cast<const uint8_t*>(k_scales) + byte_off);
    const float* vs_s = reinterpret_cast<const float*>(reinterpret_cast<const uint8_t*>(v_scales) + byte_off);
    const size_t head_slot_base = static_cast<size_t>(kvh) * ring;
    const int step = FOLD ? kSub : b_hi - b_lo + 1;
    for (int s_lo = b_lo; s_lo <= b_hi && s_lo <= tile_hi; s_lo += step) {
        const int s_hi = s_lo + step - 1;
        for (int n = 0; n < 16; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.0f;
        m[0] = m[1] = -INFINITY;
        l[0] = l[1] = 0.0f;
        const int k_lo = max(s_lo, tile_lo), k_hi = min(s_hi, tile_hi);
        for (int kt = k_lo & ~(kKeys - 1); kt <= k_hi; kt += kKeys) {
            __syncthreads();
            const size_t slot0 = static_cast<size_t>(kt % ring);
            for (int e = threadIdx.x; e < kKeys * (kDim / 8); e += 128) {
                const int key = e >> 4, col = (e & 15) * 8;
                const int kp = kt + key;
                uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
                if (kp >= k_lo && kp <= k_hi) {
                    const size_t slot = head_slot_base + slot0 + key;
                    const size_t at = slot * kDim + col;
                    kv = dequant8(*reinterpret_cast<const uint2*>(kc_s + at), ks_s[slot]);
                    vv = dequant8(*reinterpret_cast<const uint2*>(vc_s + at), vs_s[slot]);
                }
                *reinterpret_cast<uint4*>(ks + key * kPad + col) = kv;
                *reinterpret_cast<uint4*>(vs + key * kPad + col) = vv;
            }
            for (int e = threadIdx.x; e < n_pairs * kKeys; e += 128) {
                const int p = e / kKeys, key = e % kKeys;
                const int kp = kt + key, pp = bs.pos[p], d = pp - kp;
                const int plo = window > 0 ? max(0, pp - window + 1) : 0;
                const bool need = kp >= plo && kp <= pp && kp >= s_lo && kp <= s_hi && d < kExtent;
                bs.bias[e] = need ? relBiasBits(bs.rel[p], proj_t, d) : 0;
            }
            __syncthreads();
            if (!busy) continue;
            float sc[8][4];
            for (int j = 0; j < 8; ++j) {
                sc[j][0] = sc[j][1] = sc[j][2] = sc[j][3] = 0.0f;
                const uint16_t* krow = ks + (8 * j + g) * kPad + 2 * t;
                for (int k = 0; k < 8; ++k) {
                    const uint32_t b0 = *reinterpret_cast<const uint32_t*>(krow + 16 * k);
                    const uint32_t b1 = *reinterpret_cast<const uint32_t*>(krow + 16 * k + 8);
                    qmm_frag::mma(sc[j], qa[k][0], qa[k][1], qa[k][2], qa[k][3], b0, b1);
                }
            }
            float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
            for (int j = 0; j < 8; ++j) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int i = e >> 1;
                    const int kp = kt + 8 * j + 2 * t + (e & 1);
                    const bool live = ok[i] && kp >= lo[i] && kp <= pos[i] && kp >= s_lo && kp <= s_hi;
                    const int d = pos[i] - kp;
                    const float bias = live && d < kExtent ? bf(bs.bias[(warp * 16 + g + i * 8) * kKeys + (kp - kt)]) : 0.0f;
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
            uint16_t p[8][4];
            for (int j = 0; j < 8; ++j) {
                for (int e = 0; e < 4; ++e) {
                    const int i = e >> 1;
                    p[j][e] = tobf(exp2f(sc[j][e] - mu[i]));
                    l[i] += bf(p[j][e]);
                }
            }
            for (int n = 0; n < 16; ++n) {
                acc[n][0] *= alpha[0];
                acc[n][1] *= alpha[0];
                acc[n][2] *= alpha[1];
                acc[n][3] *= alpha[1];
            }
            for (int u = 0; u < 4; ++u) {
                const uint32_t a0 = pack(p[2 * u][0], p[2 * u][1]), a1 = pack(p[2 * u][2], p[2 * u][3]);
                const uint32_t a2 = pack(p[2 * u + 1][0], p[2 * u + 1][1]);
                const uint32_t a3 = pack(p[2 * u + 1][2], p[2 * u + 1][3]);
                const uint16_t* v0 = vs + (16 * u + 2 * t) * kPad + g;
                for (int n = 0; n < 16; ++n) {
                    const uint16_t* vp = v0 + 8 * n;
                    const uint32_t b0 = pack(vp[0], vp[kPad]);
                    const uint32_t b1 = pack(vp[8 * kPad], vp[9 * kPad]);
                    qmm_frag::mma(acc[n], a0, a1, a2, a3, b0, b1);
                }
            }
        }
        for (int i = 0; i < 2; ++i) {
            l[i] += __shfl_xor_sync(0xffffffffu, l[i], 1);
            l[i] += __shfl_xor_sync(0xffffffffu, l[i], 2);
        }
        if (FOLD) {
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
        if (!ok[i] || c < lo[i] / chunk || c > pos[i] / chunk || b_lo > pos[i]) continue;
        const size_t at = split ? prow[i] * chunks + (c - c0) * subs + sub
                                : prow[i] * chunks + (c - (window > 0 ? lo[i] / chunk : c0));
        if (t == 0) part_ml[at] = FOLD ? make_float2(M[i], L[i]) : make_float2(m[i], l[i]);
        float* o = part_o + at * kDim + 2 * t;
        for (int n = 0; n < 16; ++n) {
            const float x = FOLD ? O[FOLD ? n : 0][2 * i] : acc[n][2 * i];
            const float y = FOLD ? O[FOLD ? n : 0][2 * i + 1] : acc[n][2 * i + 1];
            *reinterpret_cast<float2*>(o + 8 * n) = make_float2(x, y);
        }
    }
}

}  // namespace ink2

extern "C" __global__ void __launch_bounds__(128)
    tf_ink2_attn_i8(const uint16_t* __restrict__ q, const int8_t* __restrict__ kc, const int8_t* __restrict__ vc,
                    const float* __restrict__ k_scales, const float* __restrict__ v_scales,
                    const uint16_t* __restrict__ rel, const uint16_t* __restrict__ proj_t,
                    const Seg* __restrict__ segs, const int2* __restrict__ tiles, float* __restrict__ part_o,
                    float2* __restrict__ part_ml, int heads, int kv_heads, int ring, int window, int chunk, int c0,
                    int chunks, float scale, int split) {
    using namespace ink2;
    __shared__ __align__(16) uint16_t ks[kKeys * kPad];
    __shared__ __align__(16) uint16_t vs[kKeys * kPad];
    __shared__ __align__(16) BiasSmem bs;
    attnTileI8<false>(q, kc, vc, k_scales, v_scales, rel, proj_t, segs, tiles, part_o, part_ml, heads, kv_heads, ring,
                      window, chunk, c0, chunks, scale, split, ks, vs, bs);
}

extern "C" __global__ void __launch_bounds__(128)
    tf_ink2_attn_fold_i8(const uint16_t* __restrict__ q, const int8_t* __restrict__ kc, const int8_t* __restrict__ vc,
                         const float* __restrict__ k_scales, const float* __restrict__ v_scales,
                         const uint16_t* __restrict__ rel, const uint16_t* __restrict__ proj_t,
                         const Seg* __restrict__ segs, const int2* __restrict__ tiles, float* __restrict__ part_o,
                         float2* __restrict__ part_ml, int heads, int kv_heads, int ring, int chunk, int c0, int chunks,
                         float scale) {
    using namespace ink2;
    __shared__ __align__(16) uint16_t ks[kKeys * kPad];
    __shared__ __align__(16) uint16_t vs[kKeys * kPad];
    __shared__ __align__(16) BiasSmem bs;
    attnTileI8<true>(q, kc, vc, k_scales, v_scales, rel, proj_t, segs, tiles, part_o, part_ml, heads, kv_heads, ring,
                     0, chunk, c0, chunks, scale, 0, ks, vs, bs);
}
