// LFM3 ops in plain OpenCL C (no DPAS): bf16 rows, fp32 math, rounded to bf16 where modeling_lfm3.py rounds.
#pragma OPENCL FP_CONTRACT OFF
#define HD 128 // head dim
#ifndef GQ
#define GQ 4 // query heads per KV head (16 / 4)
#endif
#define NPI 16 // values an item of 256 holds in the row ops: n <= 4096

inline float bf(ushort v) { return as_float((uint)v << 16); }
inline ushort to_bf(float f) {
    uint u = as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (ushort)(u >> 16);
}
inline float bfr(float f) { return bf(to_bf(f)); }

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

// RMSNorm of row get_group_id(0): y = bf16(w * bf16(x * rsqrt(mean(x^2) + eps))); 256 items, n % 256 == 0, n <= 4096.
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void lfm3_rmsnorm(__global const ushort *x, __global const ushort *w, __global ushort *y, uint n, float eps) {
    __local float part[16];
    const uint row = get_group_id(0);
    const uint lid = get_local_id(0);
    const uint per = n / 256;
    float v[NPI];
    float ss = 0.0f;
    for (uint k = 0; k < per; k++) {
        v[k] = bf(x[row * n + lid + 256 * k]);
        ss += v[k] * v[k];
    }
    const float rs = rsqrt(wg_sum256(ss, part) / (float)n + eps);
    for (uint k = 0; k < per; k++) y[row * n + lid + 256 * k] = to_bf(bf(w[lid + 256 * k]) * bfr(v[k] * rs));
}

// Sandwich norm step: x = bf16(x + rmsnorm(d, wp)) in place, then y = rmsnorm(x, wn); 256 items a row.
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void lfm3_post_add_norm(__global const ushort *d, __global ushort *x, __global const ushort *wp, __global const ushort *wn, __global ushort *y,
                                 uint n, float eps) {
    __local float part[16];
    const uint row = get_group_id(0);
    const uint lid = get_local_id(0);
    const uint per = n / 256;
    float v[NPI];
    float ss = 0.0f;
    for (uint k = 0; k < per; k++) {
        v[k] = bf(d[row * n + lid + 256 * k]);
        ss += v[k] * v[k];
    }
    const float rd = rsqrt(wg_sum256(ss, part) / (float)n + eps);
    ss = 0.0f;
    for (uint k = 0; k < per; k++) {
        const uint i = row * n + lid + 256 * k;
        const float dn = bfr(bf(wp[lid + 256 * k]) * bfr(v[k] * rd));
        v[k] = bfr(bf(x[i]) + dn);
        x[i] = to_bf(v[k]);
        ss += v[k] * v[k];
    }
    const float rx = rsqrt(wg_sum256(ss, part) / (float)n + eps);
    for (uint k = 0; k < per; k++) y[row * n + lid + 256 * k] = to_bf(bf(wn[lid + 256 * k]) * bfr(v[k] * rx));
}

// Gated short conv: rows [B | C | x], y = bf16(C * bf16(conv7(bf16(B * x)))); state: last six B * x, oldest first.
__kernel void lfm3_conv_rows(__global const ushort *proj, uint pd, __global const ushort *sin, __global ushort *sout, __global const ushort *cw,
                             __global ushort *y, uint n, uint D) {
    const uint c = get_global_id(0);
    if (c >= D) return;
    float h[6], w[7];
    for (uint k = 0; k < 6; k++) h[k] = bf(sin[k * D + c]);
    for (uint k = 0; k < 7; k++) w[k] = bf(cw[c * 7 + k]);
    for (uint r = 0; r < n; r++) {
        __global const ushort *p = proj + r * pd;
        const float bx = bfr(bf(p[c]) * bf(p[2 * D + c]));
        float acc = 0.0f;
        for (uint k = 0; k < 6; k++) acc += h[k] * w[k];
        acc += bx * w[6];
        y[r * D + c] = to_bf(bf(p[D + c]) * bfr(acc));
        for (uint k = 0; k < 5; k++) h[k] = h[k + 1];
        h[5] = bx;
    }
    for (uint k = 0; k < 6; k++) sout[k * D + c] = to_bf(h[k]);
}

// lfm3_conv_rows for a verified window, also ext rows 0..5 = the state before it, 6 + r = row r's B * x (rollback).
__kernel void lfm3_conv_hist(__global const ushort *proj, uint pd, __global const ushort *sin, __global ushort *sout, __global const ushort *cw,
                             __global ushort *y, uint n, uint D, __global ushort *ext) {
    const uint c = get_global_id(0);
    if (c >= D) return;
    float h[6], w[7];
    for (uint k = 0; k < 6; k++) {
        h[k] = bf(sin[k * D + c]);
        ext[k * D + c] = to_bf(h[k]);
    }
    for (uint k = 0; k < 7; k++) w[k] = bf(cw[c * 7 + k]);
    for (uint r = 0; r < n; r++) {
        __global const ushort *p = proj + r * pd;
        const float bx = bfr(bf(p[c]) * bf(p[2 * D + c]));
        float acc = 0.0f;
        for (uint k = 0; k < 6; k++) acc += h[k] * w[k];
        acc += bx * w[6];
        y[r * D + c] = to_bf(bf(p[D + c]) * bfr(acc));
        ext[(6 + r) * D + c] = to_bf(bx);
        for (uint k = 0; k < 5; k++) h[k] = h[k + 1];
        h[5] = bx;
    }
    for (uint k = 0; k < 6; k++) sout[k * D + c] = to_bf(h[k]);
}

// One head of row r at position pos: RMSNorm over 128 dims, RoPE on pairs (d, d + 64) with bf16 cos/sin; 64 items.
inline void norm_rope_head(__global const ushort *s, __global ushort *o, __global const ushort *w, __global const float *inv_freq, uint pos,
                           float eps, __local float *part) {
    const uint i = get_local_id(0); // 64 items in 4 sub-groups of 16 (the kernels require 16): part[4]
    const float a = bf(s[i]), b = bf(s[i + 64]);
    float ss = sub_group_reduce_add(a * a + b * b);
    if (get_sub_group_local_id() == 0) part[get_sub_group_id()] = ss;
    barrier(CLK_LOCAL_MEM_FENCE);
    const uint nsg = get_num_sub_groups();
    float t = 0.0f;
    for (uint k = 0; k < nsg; k++) t += part[k];
    const float rs = rsqrt(t / (float)HD + eps);
    const float na = bfr(bf(w[i]) * bfr(a * rs));
    const float nb = bfr(bf(w[i + 64]) * bfr(b * rs));
    const float f = (float)pos * inv_freq[i];
    const float co = bfr(cos(f)), si = bfr(sin(f));
    o[i] = to_bf(na * co - nb * si);
    o[i + 64] = to_bf(nb * co + na * si);
}

// Per-head RMSNorm, RoPE on pairs (d, d + 64), bf16 cos/sin; grid (heads, rows); to row r or slot (pos0 + r) % ring.
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void lfm3_qk_norm_rope(__global const ushort *src, uint src_stride, uint src_off, __global ushort *dst, uint dst_stride,
                                __global const ushort *w, __global const float *inv_freq, uint pos0, uint ring, float eps) {
    __local float part[4];
    const uint head = get_group_id(0);
    const uint r = get_group_id(1);
    const uint row = ring == 0 ? r : (pos0 + r) % ring;
    norm_rope_head(src + r * src_stride + src_off + head * HD, dst + row * dst_stride + head * HD, w, inv_freq, pos0 + r, eps, part);
}

// q heads, then k heads and v rows into slot (pos0 + r) % ring, in one launch; grid (nq + nkv + nkv * 2, rows).
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void lfm3_rope_store(__global const ushort *qkv, uint stride, __global ushort *q, __global ushort *kc, __global ushort *vc,
                              __global const ushort *qw, __global const ushort *kw, __global const float *inv_freq, uint nq, uint nkv,
                              uint pos0, uint ring, float eps) {
    __local float part[4];
    const uint g = get_group_id(0);
    const uint r = get_group_id(1);
    const uint slot = (pos0 + r) % ring;
    __global const ushort *src = qkv + r * stride;
    if (g < nq) {
        norm_rope_head(src + g * HD, q + (r * nq + g) * HD, qw, inv_freq, pos0 + r, eps, part);
    } else if (g < nq + nkv) {
        const uint h = g - nq;
        norm_rope_head(src + (nq + h) * HD, kc + (slot * nkv + h) * HD, kw, inv_freq, pos0 + r, eps, part);
    } else {
        const uint e = (g - nq - nkv) * 64 + get_local_id(0);
        vc[slot * nkv * HD + e] = src[(nq + nkv) * HD + e];
    }
}

// Copies n_elem bf16 of row r to KV slot (pos0 + r) % ring; grid (ceil(n_elem / 64), rows).
__kernel void lfm3_v_store(__global const ushort *src, uint src_stride, uint src_off, __global ushort *dst, uint n_elem, uint pos0, uint ring) {
    const uint e = get_global_id(0);
    const uint r = get_group_id(1);
    if (e < n_elem) dst[((pos0 + r) % ring) * n_elem + e] = src[r * src_stride + src_off + e];
}

// First key of row r's attention: window == 0 sees every earlier position, else the last `window` positions.
inline uint key_lo(uint p, uint window) { return (window == 0 || p + 1 <= window) ? 0 : p + 1 - window; }

// Split-K GQA pass 1: group (kv head, chunk, row), a sub-group per q head, keys at slot k % ring; unnormalised o, m, l.
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void lfm3_attn_partial(__global const ushort *q, __global const ushort *kc, __global const ushort *vc, __global float *po, __global float *pm,
                                __global float *pl, uint pos0, uint chunk, uint ring, uint window, uint nkv, uint maxc, float scale) {
    const uint kvh = get_group_id(0);
    const uint c = get_group_id(1);
    const uint z = get_group_id(2);
    const uint g = get_sub_group_id();
    const uint lane = get_sub_group_local_id();
    const uint nq = nkv * GQ;
    const uint h = kvh * GQ + g;
    const uint p = pos0 + z;
    const uint lo = key_lo(p, window) + c * chunk;
    const uint hi = min(p + 1, lo + chunk);
    const float8 qv = as_float8(convert_uint8(vload8(0, q + (z * nq + h) * HD + lane * 8)) << 16);
    float m = -INFINITY, l = 0.0f;
    float8 o = (float8)(0.0f);
    for (uint k = lo; k < hi; k++) {
        const uint base = ((k % ring) * nkv + kvh) * HD + lane * 8;
        const float8 kv = as_float8(convert_uint8(vload8(0, kc + base)) << 16);
        const float8 pr = qv * kv;
        const float s = sub_group_reduce_add(pr.s0 + pr.s1 + pr.s2 + pr.s3 + pr.s4 + pr.s5 + pr.s6 + pr.s7) * scale;
        const float mn = fmax(m, s);
        const float corr = exp(m - mn);
        const float e = exp(s - mn);
        const float8 vv = as_float8(convert_uint8(vload8(0, vc + base)) << 16);
        l = l * corr + e;
        o = o * corr + e * vv;
        m = mn;
    }
    const uint slot = (z * nq + h) * maxc + c;
    vstore8(o, 0, po + slot * HD + lane * 8);
    if (lane == 0) {
        pm[slot] = m;
        pl[slot] = l;
    }
}

// Pass 2: row z's chunks of head h merged in order, bf16(O / L) times bf16(sigmoid(gate)); grid (nq, rows), 16 items.
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void lfm3_attn_merge_gate(__global const float *po, __global const float *pm, __global const float *pl, __global const ushort *g,
                                   __global ushort *out, uint pos0, uint chunk, uint window, uint nq, uint maxc) {
    const uint h = get_group_id(0);
    const uint z = get_group_id(1);
    const uint lane = get_sub_group_local_id();
    const uint p = pos0 + z;
    const uint len = p + 1 - key_lo(p, window);
    const uint nch = (len + chunk - 1) / chunk;
    const uint base = (z * nq + h) * maxc;
    float mx = -INFINITY;
    for (uint c = 0; c < nch; c++) mx = fmax(mx, pm[base + c]);
    float L = 0.0f;
    float8 O = (float8)(0.0f);
    for (uint c = 0; c < nch; c++) {
        const float e = exp(pm[base + c] - mx);
        L += pl[base + c] * e;
        O += vload8(0, po + (base + c) * HD + lane * 8) * e;
    }
    const float gate = bfr(1.0f / (1.0f + exp(-bf(g[z * nq + h]))));
    const float8 r = O / L;
    __global ushort *dst = out + (z * nq + h) * HD + lane * 8;
    dst[0] = to_bf(bfr(r.s0) * gate);
    dst[1] = to_bf(bfr(r.s1) * gate);
    dst[2] = to_bf(bfr(r.s2) * gate);
    dst[3] = to_bf(bfr(r.s3) * gate);
    dst[4] = to_bf(bfr(r.s4) * gate);
    dst[5] = to_bf(bfr(r.s5) * gate);
    dst[6] = to_bf(bfr(r.s6) * gate);
    dst[7] = to_bf(bfr(r.s7) * gate);
}

// SwiGLU over n bf16 values: out = bf16(bf16(silu(a)) * b), a = w1 x (gate), b = w3 x (up).
__kernel void lfm3_swiglu(__global const ushort *a, __global const ushort *b, __global ushort *out, uint n) {
    const uint i = get_global_id(0);
    if (i >= n) return;
    const float x = bf(a[i]);
    out[i] = to_bf(bfr(x / (1.0f + exp(-x))) * bf(b[i]));
}
