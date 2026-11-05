// Qwen3-ASR's ops no shared Gen12LP kernel covers: bf16 rows and weights, fp32 sums, bf16 where torch rounds.
#pragma OPENCL FP_CONTRACT OFF

inline float bf(ushort v) { return as_float((uint)v << 16); }
inline ushort to_bf(float f) {
    uint u = as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (ushort)((u >> 16) | 0x40u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (ushort)(u >> 16);
}
inline float bfr(float f) { return bf(to_bf(f)); }
inline float gelu(float x) { return 0.5f * x * (1.0f + erf(x * 0.70710678118654752f)); }

// Sum over a work-group of 256 items (16 sub-groups of 16) in a fixed order.
inline float wg_sum256(float v, __local float *part) {
    v = sub_group_reduce_add(v);
    if (get_sub_group_local_id() == 0) part[get_sub_group_id()] = v;
    barrier(CLK_LOCAL_MEM_FENCE);
    float t = 0.0f;
    for (uint i = 0; i < 16; i++) t += part[i];
    barrier(CLK_LOCAL_MEM_FENCE);
    return t;
}

// The epilogue of a Linear: bf16(act(bf16(s + bias)) + res), each optional (act 1: GELU).
inline ushort epilogue(float s, __global const ushort *bias, uint has_bias, uint o, uint act, __global const ushort *res, uint has_res, ulong at) {
    float t = bfr(has_bias ? s + bf(bias[o]) : s);
    if (act == 1) t = bfr(gelu(t));
    if (has_res) t = bfr(t + bf(res[at]));
    return to_bf(t);
}

// A 64 x 64 tile of y in 16 x 16 items of 4 x 4, 16 k a step through local memory (vit_gemm's shape, bf16 rows).
#define TILE_BODY(LOADX)                                                                                                       \
    __local float xs[16][64];                                                                                                \
    __local float ws[16][64];                                                                                                \
    const uint lx = get_local_id(0), ly = get_local_id(1), t = ly * 16 + lx;                                                \
    const ulong m0 = (ulong)get_group_id(1) * 64, n0 = (ulong)get_group_id(0) * 64;                                         \
    const uint lr = t / 4, lk = (t % 4) * 4;                                                                                 \
    float acc[4][4];                                                                                                         \
    for (uint i = 0; i < 4; i++)                                                                                             \
        for (uint j = 0; j < 4; j++) acc[i][j] = 0.0f;                                                                       \
    for (uint k0 = 0; k0 < k; k0 += 16) {                                                                                    \
        const ulong xr = m0 + lr, wr = n0 + lr;                                                                              \
        for (uint j = 0; j < 4; j++) {                                                                                       \
            const uint kk = k0 + lk + j;                                                                                     \
            xs[lk + j][lr] = (xr < m && kk < k) ? LOADX : 0.0f;                                                              \
            ws[lk + j][lr] = (wr < n && kk < k) ? bf(w[wr * k + kk]) : 0.0f;                                                 \
        }                                                                                                                    \
        barrier(CLK_LOCAL_MEM_FENCE);                                                                                        \
        float part[4][4];                                                                                                    \
        for (uint i = 0; i < 4; i++)                                                                                         \
            for (uint j = 0; j < 4; j++) part[i][j] = 0.0f;                                                                  \
        _Pragma("unroll") for (uint q = 0; q < 16; q++) {                                                                    \
            const float4 a = vload4(ly, &xs[q][0]);                                                                          \
            const float4 b4 = vload4(lx, &ws[q][0]);                                                                         \
            const float av[4] = {a.s0, a.s1, a.s2, a.s3}, bv[4] = {b4.s0, b4.s1, b4.s2, b4.s3};                              \
            _Pragma("unroll") for (uint i = 0; i < 4; i++) _Pragma("unroll") for (uint j = 0; j < 4; j++) part[i][j] =       \
                fma(av[i], bv[j], part[i][j]);                                                                               \
        }                                                                                                                    \
        for (uint i = 0; i < 4; i++)                                                                                         \
            for (uint j = 0; j < 4; j++) acc[i][j] += part[i][j];                                                            \
        barrier(CLK_LOCAL_MEM_FENCE);                                                                                        \
    }

// y[r][o] (rows ldy apart) = epilogue(x[r] . w[o]): x [m, k] and w [n, k] bf16; grid ((n + 63) / 64, (m + 63) / 64).
__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void asr_gemm(__global const ushort *x, __global const ushort *w, __global const ushort *bias, uint has_bias, __global const ushort *res,
                       uint has_res, __global ushort *y, uint m, uint n, uint k, uint ldy, uint act) {
    TILE_BODY(bf(x[xr * k + kk]))
    for (uint i = 0; i < 4; i++) {
        const ulong r = m0 + ly * 4 + i;
        if (r >= m) continue;
        for (uint j = 0; j < 4; j++) {
            const uint o = (uint)(n0 + lx * 4 + j);
            if (o >= n) continue;
            const ulong at = r * ldy + o;
            y[at] = epilogue(acc[i][j], bias, has_bias, o, act, res, has_res, at);
        }
    }
}

// x value (ci, kh, kw) of output position p for a 3 x 3 stride-2 conv, zero padding 1, channels last [n][fi][ti][c].
inline float conv_tap(__global const ushort *x, ulong p, uint kk, uint fi, uint ti, uint c, uint fo, uint to) {
    const uint ci = kk / 9, kh = (kk % 9) / 3, kw = kk % 3;
    const ulong nn = p / ((ulong)fo * to);
    const uint rem = (uint)(p % ((ulong)fo * to)), f = rem / to, tt = rem % to;
    const int fx = (int)(2 * f + kh) - 1, tx = (int)(2 * tt + kw) - 1;
    if (fx < 0 || tx < 0 || fx >= (int)fi || tx >= (int)ti) return 0.0f;
    return bf(x[((nn * fi + (uint)fx) * ti + (uint)tx) * c + ci]);
}

// y [n][fo][to][co] = bf16(taps(x) . w[co][ci * 9 + kh * 3 + kw] + b); m = n * fo * to, k = 9 c; grid as asr_gemm.
__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void asr_conv(__global const ushort *x, __global const ushort *w, __global const ushort *bias, __global ushort *y, uint m, uint n, uint k,
                       uint fi, uint ti, uint c) {
    const uint fo = (fi - 1) / 2 + 1, to = (ti - 1) / 2 + 1;
    TILE_BODY(conv_tap(x, xr, kk, fi, ti, c, fo, to))
    for (uint i = 0; i < 4; i++) {
        const ulong r = m0 + ly * 4 + i;
        if (r >= m) continue;
        for (uint j = 0; j < 4; j++) {
            const uint o = (uint)(n0 + lx * 4 + j);
            if (o < n) y[r * n + o] = to_bf(acc[i][j] + bf(bias[o]));
        }
    }
}

// [n][f][t][c] channels last to rows [n * t][c * f] (torch's permute(0, 3, 1, 2) of [n][c][f][t]); a value an item.
__kernel void asr_flatten(__global const ushort *x, __global ushort *y, uint f, uint t, uint c, uint total) {
    const ulong i = get_global_id(0);
    if (i >= total) return;
    const uint cf = c * f;
    const ulong row = i / cf;
    const uint col = (uint)(i % cf), ci = col / f, ff = col % f;
    const ulong nn = row / t, tt = row % t;
    y[i] = x[((nn * f + ff) * t + tt) * c + ci];
}

// LayerNorm of row get_group_id(0): bf16((x - mean) * rsqrt(var + eps) * w + b), fp32 inside; 256 items.
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void asr_layernorm(__global const ushort *x, __global const ushort *w, __global const ushort *b, __global ushort *y, uint n, float eps) {
    __local float part[16];
    const ulong row = get_group_id(0);
    const uint id = get_local_id(0);
    __global const ushort *xr = x + row * n;
    float s = 0.0f;
    for (uint i = id; i < n; i += 256) s += bf(xr[i]);
    const float mean = wg_sum256(s, part) / (float)n;
    float v = 0.0f;
    for (uint i = id; i < n; i += 256) {
        const float d = bf(xr[i]) - mean;
        v += d * d;
    }
    const float rs = rsqrt(wg_sum256(v, part) / (float)n + eps);
    for (uint i = id; i < n; i += 256) y[row * n + i] = to_bf((bf(xr[i]) - mean) * rs * bf(w[i]) + bf(b[i]));
}

// y = bf16(gelu(x)) with erf, as torch's GELU; a value an item.
__kernel void asr_gelu(__global const ushort *x, __global ushort *y, uint total) {
    const ulong i = get_global_id(0);
    if (i < total) y[i] = to_bf(gelu(bf(x[i])));
}

// Bidirectional attention over len rows of [q | k | v] (64-dim heads), online softmax as vit_attn; a sub-group a row.
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void asr_attn_full(__global const ushort *qkv, __global ushort *out, uint len, uint heads, float scale) {
    const uint row = get_group_id(0), h = get_group_id(1), lane = get_sub_group_local_id();
    const uint d = heads * 64, ld = 3 * d;
    float q[64];
    for (uint i = 0; i < 64; i++) q[i] = bf(qkv[(ulong)row * ld + h * 64 + i]);
    float m = -INFINITY, l = 0.0f, o[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint k0 = 0; k0 < len; k0 += 16) {
        const uint key = k0 + lane;
        float s = -INFINITY;
        if (key < len) {
            __global const ushort *kr = qkv + (ulong)key * ld + d + h * 64;
            float a = 0.0f;
            for (uint i = 0; i < 64; i++) a = fma(q[i], bf(kr[i]), a);
            s = a * scale;
        }
        const float mn = fmax(m, sub_group_reduce_max(s));
        const float p = key < len ? exp(s - mn) : 0.0f;
        const float c = exp(m - mn);
        l = l * c + sub_group_reduce_add(p);
        m = mn;
        for (uint i = 0; i < 4; i++) o[i] *= c;
        const uint kn = min(16u, len - k0);
        for (uint j = 0; j < kn; j++) {
            const float pj = sub_group_shuffle(p, j);
            __global const ushort *vr = qkv + (ulong)(k0 + j) * ld + 2 * d + h * 64 + lane * 4;
            for (uint i = 0; i < 4; i++) o[i] = fma(pj, bf(vr[i]), o[i]);
        }
    }
    const float il = 1.0f / l;
    for (uint i = 0; i < 4; i++) out[(ulong)row * d + h * 64 + lane * 4 + i] = to_bf(o[i] * il);
}

// A head of 128: bf16(w * bf16(x * rs)), then RoPE (d, d + 64), each product and sum rounded as Qwen3 (lfm3: once).
inline void norm_rope(__global const ushort *src, __global ushort *dst, __global const ushort *w, __global const float *cs,
                      __global const float *sn, uint p, float eps) {
    const uint lane = get_sub_group_local_id();
    float v[8];
    float s = 0.0f;
    for (uint i = 0; i < 8; i++) {
        v[i] = bf(src[lane * 8 + i]);
        s += v[i] * v[i];
    }
    const float rs = rsqrt(sub_group_reduce_add(s) / 128.0f + eps);
    for (uint i = 0; i < 8; i++) v[i] = bfr(bf(w[lane * 8 + i]) * bfr(v[i] * rs));
    // the partner of dim d is d + 64 (or d - 64): lane l's dims pair with lane l ^ 8's
    for (uint i = 0; i < 8; i++) {
        const uint dd = lane * 8 + i;
        const float other = sub_group_shuffle_xor(v[i], 8u);
        const float rot = dd < 64 ? -other : other;
        dst[dd] = to_bf(bfr(v[i] * cs[(ulong)p * 128 + dd]) + bfr(rot * sn[(ulong)p * 128 + dd]));
    }
}

// Row r at pos0 + r of [q | k | v]: q heads to q, k heads to k and kc, v to v and vc; grid (heads + 2 kv_heads, rows).
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void asr_qk_rope(__global const ushort *qkv, __global const ushort *qn, __global const ushort *kn, __global const float *cs,
                          __global const float *sn, __global ushort *q, __global ushort *k, __global ushort *v, __global ushort *kc,
                          __global ushort *vc, uint heads, uint kv_heads, uint pos0, float eps) {
    const uint hh = get_group_id(0), r = get_group_id(1), p = pos0 + r, lane = get_sub_group_local_id();
    const uint qd = heads * 128, kvd = kv_heads * 128;
    __global const ushort *row = qkv + (ulong)r * (qd + 2 * kvd);
    if (hh < heads) {
        norm_rope(row + hh * 128, q + (ulong)r * qd + hh * 128, qn, cs, sn, p, eps);
    } else if (hh < heads + kv_heads) {
        const uint kh = hh - heads;
        norm_rope(row + qd + kh * 128, k + (ulong)r * kvd + kh * 128, kn, cs, sn, p, eps);
        for (uint i = 0; i < 8; i++) kc[(ulong)p * kvd + kh * 128 + lane * 8 + i] = k[(ulong)r * kvd + kh * 128 + lane * 8 + i];
    } else {
        const uint vh = hh - heads - kv_heads;
        for (uint i = 0; i < 8; i++) {
            const ushort x = row[qd + kvd + vh * 128 + lane * 8 + i];
            v[(ulong)r * kvd + vh * 128 + lane * 8 + i] = x;
            vc[(ulong)p * kvd + vh * 128 + lane * 8 + i] = x;
        }
    }
}
