// Transcribe's audio encoder glue, each sum in one fixed order; the controls build each step with CTL set (see CTL:).
#pragma once

#include <cuda_bf16.h>
#include <stdint.h>

#include "common/glue_math.cuh"
#include "common/reduce.cuh"

namespace tf_nemo_audio {

typedef __nv_bfloat16 bf16;

__device__ __forceinline__ float ld(const bf16* p) { return __bfloat162float(*p); }
__device__ __forceinline__ float rb(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

using tf_common::warp_max;
using tf_common::warp_sum;

constexpr int FFT = 512, BINS = FFT / 2 + 1, WIN = 400, HOP = 160, MELS = 128;
constexpr int WIN_AT = (FFT - WIN) / 2;  // the window's place in its 512-sample frame
constexpr int PAD = FFT / 2;             // zeros before the first sample (centred frames)

// A frame a block: pre-emphasis, window, |DFT|^2, mel filters, log(mel + guard) to bf16. CTL: the last bin left out.
template <bool CTL>
__device__ __forceinline__ void mel(const float* __restrict__ X, long lo, long n, const bf16* __restrict__ win,
                                    const bf16* __restrict__ fb, const float* __restrict__ cs,
                                    const float* __restrict__ sn, bf16* __restrict__ out, int first, int valid,
                                    float preemph, float guard) {
    __shared__ float xw[WIN];
    __shared__ float pw[BINS];
    const int f = first + blockIdx.x;
    bf16* o = out + (size_t)f * MELS;
    if (f >= valid) {
        if (threadIdx.x < MELS) o[threadIdx.x] = __float2bfloat16_rn(0.0f);
        return;
    }
    for (int t = threadIdx.x; t < WIN; t += blockDim.x) {
        const long i = (long)f * HOP - PAD + WIN_AT + t;
        float y = 0.0f;
        if (i >= 0 && i < n) y = i == 0 ? X[-lo] : __fsub_rn(X[i - lo], __fmul_rn(preemph, X[i - 1 - lo]));
        xw[t] = __fmul_rn(y, ld(win + t));
    }
    __syncthreads();
    for (int k = threadIdx.x; k < BINS; k += blockDim.x) {
        float re = 0.0f, im = 0.0f;
        for (int t = 0; t < WIN; ++t) {
            const int j = (k * (t + WIN_AT)) & (FFT - 1);
            re = __fmaf_rn(xw[t], cs[j], re);
            im = __fmaf_rn(xw[t], sn[j], im);
        }
        pw[k] = __fadd_rn(__fmul_rn(re, re), __fmul_rn(im, im));
    }
    __syncthreads();
    for (int m = threadIdx.x; m < MELS; m += blockDim.x) {
        const bf16* row = fb + (size_t)m * BINS;
        float a = 0.0f;
        for (int k = 0; k < (CTL ? BINS - 1 : BINS); ++k) a = __fmaf_rn(ld(row + k), pw[k], a);
        o[m] = __float2bfloat16_rn(logf(__fadd_rn(a, guard)));
    }
}

// Each feature over `valid` frames: (x - mean) / (sd + eps), sd with n - 1, 8 warps' sums in order. CTL: sd with n.
template <bool CTL>
__device__ __forceinline__ void norm(const bf16* __restrict__ x, bf16* __restrict__ y, int frames, int valid,
                                     float eps) {
    __shared__ float part[8][32];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, c = blockIdx.x * 32 + lane;
    float s = 0.0f;
    for (int f = w; f < valid; f += 8) s = __fadd_rn(s, ld(x + (size_t)f * MELS + c));
    part[w][lane] = s;
    __syncthreads();
    float total = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) total = __fadd_rn(total, part[i][lane]);
    const float mean = __fdiv_rn(total, (float)valid);
    __syncthreads();
    s = 0.0f;
    for (int f = w; f < valid; f += 8) {
        const float d = __fsub_rn(ld(x + (size_t)f * MELS + c), mean);
        s = __fmaf_rn(d, d, s);
    }
    part[w][lane] = s;
    __syncthreads();
    total = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) total = __fadd_rn(total, part[i][lane]);
    const float sd = __fadd_rn(__fsqrt_rn(__fdiv_rn(total, (float)(CTL ? valid : valid - 1))), eps);
    for (int f = w; f < frames; f += 8) {
        const float v = f < valid ? __fdiv_rn(__fsub_rn(ld(x + (size_t)f * MELS + c), mean), sd) : 0.0f;
        y[(size_t)f * MELS + c] = __float2bfloat16_rn(v);
    }
}

// q, k heads LayerNormed (qw set), dims (i, i + 32) turned with bf16 products, a warp a head. CTL: a c not rounded.
template <bool CTL>
__device__ __forceinline__ void qk(bf16* __restrict__ qkv, const bf16* __restrict__ qw, const bf16* __restrict__ qb,
                                   const bf16* __restrict__ kw, const bf16* __restrict__ kb,
                                   const bf16* __restrict__ cs, const bf16* __restrict__ sn, int rows, int heads,
                                   int dim, float eps) {
    const int r = blockIdx.x, slot = blockIdx.y * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (r >= rows || slot >= 2 * heads) return;
    bf16* h = qkv + (size_t)r * 3 * heads * dim + (size_t)slot * dim;
    const int per = dim / 32;  // 2 or 4: dims lane + 32 e
    float v[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) v[e] = e < per ? ld(h + lane + 32 * e) : 0.0f;
    if (qw != nullptr) {
        const bool q = slot < heads;
        const bf16* nw = q ? qw : kw;
        const bf16* nb = q ? qb : kb;
        float s = 0.0f;
#pragma unroll
        for (int e = 0; e < 4; ++e) s = __fadd_rn(s, v[e]);
        const float mean = __fdiv_rn(warp_sum(s), (float)dim);
        s = 0.0f;
#pragma unroll
        for (int e = 0; e < 4; ++e)
            if (e < per) s = __fmaf_rn(__fsub_rn(v[e], mean), __fsub_rn(v[e], mean), s);
        const float rstd = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(warp_sum(s), (float)dim), eps)));
#pragma unroll
        for (int e = 0; e < 4; ++e)
            if (e < per) {
                const int d = lane + 32 * e;
                v[e] = rb(__fadd_rn(__fmul_rn(__fmul_rn(__fsub_rn(v[e], mean), rstd), ld(nw + d)), ld(nb + d)));
            }
    }
    const float c = ld(cs + (size_t)r * 32 + lane), sv = ld(sn + (size_t)r * 32 + lane);
    const float a = v[0], b = v[1];
    const float ac = CTL ? __fmul_rn(a, c) : rb(__fmul_rn(a, c));
    v[0] = rb(__fsub_rn(ac, rb(__fmul_rn(b, sv))));
    v[1] = rb(__fadd_rn(rb(__fmul_rn(b, c)), rb(__fmul_rn(a, sv))));
#pragma unroll
    for (int e = 0; e < 4; ++e)
        if (e < per) h[lane + 32 * e] = __float2bfloat16_rn(v[e]);
}

constexpr int AT_WARPS = 8, AT_TILE = 64, AT_KEYS = 1024;

// Every row over every key, a warp a query row: f32 scores, sums in key order, rows up to AT_KEYS. CTL: drops a key.
template <int D, bool CTL>
__device__ __forceinline__ void attend(const bf16* __restrict__ qkv, bf16* __restrict__ out, int rows, int heads,
                                       float scale) {
    constexpr int RS = D + 2, PER = D / 32;  // padded rows: a lane a key reads its own bank
    __shared__ float sc[AT_WARPS][AT_KEYS];
    __shared__ float qs[AT_WARPS][D];
    __shared__ bf16 tile[AT_TILE * RS];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, head = blockIdx.y;
    const int r = blockIdx.x * AT_WARPS + w, keys = CTL ? rows - 1 : rows;
    const bool live = r < rows;
    const size_t stride = (size_t)3 * heads * D;
    const bf16* q = qkv + (size_t)(live ? r : 0) * stride + (size_t)head * D;
    for (int d = lane; d < D; d += 32) qs[w][d] = ld(q + d);
    const int tiles = (keys + AT_TILE - 1) / AT_TILE;
    for (int t = 0; t < tiles; ++t) {
        __syncthreads();
        for (int i = threadIdx.x; i < AT_TILE * D; i += blockDim.x) {
            const int key = t * AT_TILE + i / D, d = i % D;
            tile[(i / D) * RS + d] = key < keys ? qkv[(size_t)key * stride + (size_t)(heads + head) * D + d] : bf16();
        }
        __syncthreads();
        for (int j = lane; j < AT_TILE; j += 32) {
            const int key = t * AT_TILE + j;
            if (key >= keys) break;
            float s = 0.0f;
#pragma unroll 8
            for (int d = 0; d < D; ++d) s = __fmaf_rn(qs[w][d], __bfloat162float(tile[j * RS + d]), s);
            sc[w][key] = __fmul_rn(s, scale);
        }
    }
    __syncwarp();
    float m = -INFINITY;
    for (int key = lane; key < keys; key += 32) m = fmaxf(m, sc[w][key]);
    m = warp_max(m);
    float l = 0.0f;
    for (int key = lane; key < keys; key += 32) {
        const float p = tf_glue::tf_exp(__fsub_rn(sc[w][key], m));
        sc[w][key] = p;
        l = __fadd_rn(l, p);
    }
    l = warp_sum(l);
    __syncwarp();
    float acc[PER];
#pragma unroll
    for (int e = 0; e < PER; ++e) acc[e] = 0.0f;
    for (int t = 0; t < tiles; ++t) {
        __syncthreads();
        for (int i = threadIdx.x; i < AT_TILE * D; i += blockDim.x) {
            const int key = t * AT_TILE + i / D, d = i % D;
            tile[(i / D) * RS + d] = key < keys ? qkv[(size_t)key * stride + (size_t)(2 * heads + head) * D + d] : bf16();
        }
        __syncthreads();
        const int n = min(AT_TILE, keys - t * AT_TILE);
        for (int j = 0; j < n; ++j) {
            const float p = sc[w][t * AT_TILE + j];
#pragma unroll
            for (int e = 0; e < PER; ++e) acc[e] = __fmaf_rn(p, __bfloat162float(tile[j * RS + lane + 32 * e]), acc[e]);
        }
    }
    if (!live) return;
    bf16* o = out + ((size_t)r * heads + head) * D;
#pragma unroll
    for (int e = 0; e < PER; ++e) o[lane + 32 * e] = __float2bfloat16_rn(__fdiv_rn(acc[e], l));
}

// The kernel-3 conv's taps: y [r][c 3 + k] = x[r + k - 1][c], zeros past either end. CTL: the taps one row late.
template <bool CTL>
__device__ __forceinline__ void taps(const bf16* __restrict__ x, bf16* __restrict__ y, int rows, int width) {
    const int r = blockIdx.x;
    for (int i = threadIdx.x; i < 3 * width; i += blockDim.x) {
        const int c = i / 3, k = i % 3, from = r + k - (CTL ? 0 : 1);
        y[(size_t)r * 3 * width + i] = from >= 0 && from < rows ? x[(size_t)from * width + c] : bf16();
    }
}

// x = max(x, 0) in place, NaN kept. CTL: fmaxf, which drops a NaN.
template <bool CTL>
__device__ __forceinline__ void relu(bf16* __restrict__ x, long count) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    if (CTL)
        x[i] = __float2bfloat16_rn(fmaxf(ld(x + i), 0.0f));
    else if (ld(x + i) < 0.0f)
        x[i] = __float2bfloat16_rn(0.0f);
}

// preds [r][s] = bf16(mean over `sub` rows of bf16(sigmoid(logits))), as f32. CTL: the sigmoids not rounded.
template <bool CTL>
__device__ __forceinline__ void preds(const bf16* __restrict__ logits, float* __restrict__ out, int rows, int spk,
                                      int sub) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * spk) return;
    const int r = i / spk, s = i % spk;
    float a = 0.0f;
    for (int j = 0; j < sub; ++j) {
        const float p = tf_glue::tf_sigmoid(ld(logits + ((size_t)r * sub + j) * spk + s));
        a = __fadd_rn(a, CTL ? p : rb(p));
    }
    out[i] = rb(__fdiv_rn(a, (float)sub));
}

// out = bf16(an + bf16(LN8(preds > thr) . K)), sn the normed activity when set; a block a row. CTL: product unrounded.
template <bool CTL>
__device__ __forceinline__ void fuse(const bf16* __restrict__ an, const float* __restrict__ preds,
                                     const bf16* __restrict__ nw, const bf16* __restrict__ nb,
                                     const bf16* __restrict__ K, bf16* __restrict__ out, bf16* __restrict__ sn,
                                     int width, float thr, float eps) {
    __shared__ float a[8];
    const int r = blockIdx.x;
    if (threadIdx.x == 0) {
        float v[8], s = 0.0f;
        for (int i = 0; i < 8; ++i) {
            v[i] = preds[(size_t)r * 8 + i] > thr ? 1.0f : 0.0f;
            s = __fadd_rn(s, v[i]);
        }
        const float mean = __fdiv_rn(s, 8.0f);
        s = 0.0f;
        for (int i = 0; i < 8; ++i) s = __fmaf_rn(__fsub_rn(v[i], mean), __fsub_rn(v[i], mean), s);
        const float rstd = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(s, 8.0f), eps)));
        for (int i = 0; i < 8; ++i) {
            a[i] = rb(__fadd_rn(__fmul_rn(__fmul_rn(__fsub_rn(v[i], mean), rstd), ld(nw + i)), ld(nb + i)));
            if (sn != nullptr) sn[(size_t)r * 8 + i] = __float2bfloat16_rn(a[i]);
        }
    }
    __syncthreads();
    for (int c = threadIdx.x; c < width; c += blockDim.x) {
        float p = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) p = __fmaf_rn(a[i], ld(K + (size_t)i * width + c), p);
        out[(size_t)r * width + c] = __float2bfloat16_rn(__fadd_rn(ld(an + (size_t)r * width + c), CTL ? p : rb(p)));
    }
}

}  // namespace tf_nemo_audio
