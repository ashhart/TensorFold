// LFM3-VL's vision tower and projector without DPAS: bf16 storage at transformers' rounding points, fp32 inside.
#define TM 64 // GEMM tile: patch rows
#define TN 64 // GEMM tile: output columns
#define TK 16 // GEMM tile: inputs a step
#define VHD 64 // vision head dim
#define QR 8 // attention: query rows a sub-group holds
inline float bf(ushort v) { return as_float((uint)v << 16); }
inline ushort to_bf(float f) {
    uint u = as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (ushort)(u >> 16);
}
inline float bfr(float f) { return bf(to_bf(f)); }

// y[r][o] = bf16(x[r] . w[o] + bias[o]) for r < m (x rows ldx apart, y rows ldy apart); n % 64 == 0, k % 16 == 0.
__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void vit_gemm(__global const ushort *x, uint ldx, __global const ushort *w, __global const ushort *bias, uint has_bias,
                       __global ushort *y, uint ldy, uint m, uint k) {
    __local float xs[TK][TM];
    __local float ws[TK][TN];
    const uint lx = get_local_id(0), ly = get_local_id(1);
    const uint t = ly * 16 + lx;
    const ulong m0 = (ulong)get_group_id(1) * TM, n0 = (ulong)get_group_id(0) * TN;
    const uint lr = t / 4, lk = (t % 4) * 4;
    float acc[4][4];
    for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 4; j++) acc[i][j] = 0.0f;
    for (uint k0 = 0; k0 < k; k0 += TK) {
        const ulong xr = m0 + lr;
        ushort4 xv = (ushort4)(0);
        if (xr < m) xv = vload4(0, x + xr * ldx + k0 + lk);
        const ushort4 wv = vload4(0, w + (n0 + lr) * k + k0 + lk);
        xs[lk][lr] = bf(xv.s0);
        xs[lk + 1][lr] = bf(xv.s1);
        xs[lk + 2][lr] = bf(xv.s2);
        xs[lk + 3][lr] = bf(xv.s3);
        ws[lk][lr] = bf(wv.s0);
        ws[lk + 1][lr] = bf(wv.s1);
        ws[lk + 2][lr] = bf(wv.s2);
        ws[lk + 3][lr] = bf(wv.s3);
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint kk = 0; kk < TK; kk++) {
            const float4 a = vload4(ly, &xs[kk][0]);
            const float4 b = vload4(lx, &ws[kk][0]);
            acc[0][0] = fma(a.s0, b.s0, acc[0][0]);
            acc[0][1] = fma(a.s0, b.s1, acc[0][1]);
            acc[0][2] = fma(a.s0, b.s2, acc[0][2]);
            acc[0][3] = fma(a.s0, b.s3, acc[0][3]);
            acc[1][0] = fma(a.s1, b.s0, acc[1][0]);
            acc[1][1] = fma(a.s1, b.s1, acc[1][1]);
            acc[1][2] = fma(a.s1, b.s2, acc[1][2]);
            acc[1][3] = fma(a.s1, b.s3, acc[1][3]);
            acc[2][0] = fma(a.s2, b.s0, acc[2][0]);
            acc[2][1] = fma(a.s2, b.s1, acc[2][1]);
            acc[2][2] = fma(a.s2, b.s2, acc[2][2]);
            acc[2][3] = fma(a.s2, b.s3, acc[2][3]);
            acc[3][0] = fma(a.s3, b.s0, acc[3][0]);
            acc[3][1] = fma(a.s3, b.s1, acc[3][1]);
            acc[3][2] = fma(a.s3, b.s2, acc[3][2]);
            acc[3][3] = fma(a.s3, b.s3, acc[3][3]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (uint i = 0; i < 4; i++) {
        const ulong r = m0 + ly * 4 + i;
        if (r >= m) continue;
        for (uint j = 0; j < 4; j++) {
            const ulong o = n0 + lx * 4 + j;
            const float b = has_bias ? bf(bias[o]) : 0.0f;
            y[r * ldy + o] = to_bf(acc[i][j] + b);
        }
    }
}

// Sum over a work-group of 64 items (4 sub-groups of 16) in a fixed order.
inline float wg_sum64(float v, __local float *part) {
    const float s = sub_group_reduce_add(v);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (get_sub_group_local_id() == 0) part[get_sub_group_id()] = s;
    barrier(CLK_LOCAL_MEM_FENCE);
    return ((part[0] + part[1]) + part[2]) + part[3];
}

// The tower's RMSNorm as transformers rounds it: y = bf16(w * bf16(x * rsqrt(mean(x^2) + eps))); a row a work-group.
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void vit_rms(__global const ushort *x, __global const ushort *w, __global ushort *y, uint n, float eps) {
    __local float part[4];
    const ulong row = get_group_id(0);
    float ss = 0.0f;
    for (uint i = get_local_id(0); i < n; i += 64) {
        const float v = bf(x[row * n + i]);
        ss = fma(v, v, ss);
    }
    const float inv = 1.0f / sqrt(wg_sum64(ss, part) / (float)n + eps);
    for (uint i = get_local_id(0); i < n; i += 64) y[row * n + i] = to_bf(bf(w[i]) * bfr(bf(x[row * n + i]) * inv));
}

// The residual step: h = bf16(h + d) in place, then y = the RMSNorm of h (vit_rms); a row a work-group.
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void vit_add_rms(__global ushort *h, __global const ushort *d, __global const ushort *w, __global ushort *y, uint n, float eps) {
    __local float part[4];
    const ulong row = get_group_id(0);
    float ss = 0.0f;
    for (uint i = get_local_id(0); i < n; i += 64) {
        const float v = bfr(bf(h[row * n + i]) + bf(d[row * n + i]));
        h[row * n + i] = to_bf(v);
        ss = fma(v, v, ss);
    }
    const float inv = 1.0f / sqrt(wg_sum64(ss, part) / (float)n + eps);
    for (uint i = get_local_id(0); i < n; i += 64) y[row * n + i] = to_bf(bf(w[i]) * bfr(bf(h[row * n + i]) * inv));
}

// The projector's LayerNorm: y = bf16((x - mean) * rsqrt(var + eps) * w + b) over n values; a row a work-group.
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void vit_layernorm(__global const ushort *x, __global const ushort *w, __global const ushort *b, __global ushort *y, uint n, float eps) {
    __local float part[4];
    const ulong row = get_group_id(0);
    float s = 0.0f;
    for (uint i = get_local_id(0); i < n; i += 64) s += bf(x[row * n + i]);
    const float mean = wg_sum64(s, part) / (float)n;
    float ss = 0.0f;
    for (uint i = get_local_id(0); i < n; i += 64) {
        const float c = bf(x[row * n + i]) - mean;
        ss = fma(c, c, ss);
    }
    const float inv = 1.0f / sqrt(wg_sum64(ss, part) / (float)n + eps);
    for (uint i = get_local_id(0); i < n; i += 64) y[row * n + i] = to_bf((bf(x[row * n + i]) - mean) * inv * bf(w[i]) + bf(b[i]));
}

// GELU with erf, as transformers' "gelu": y = bf16(0.5 x (1 + erf(x / sqrt 2))).
__kernel void vit_gelu(__global const ushort *x, __global ushort *y, uint n) {
    const uint i = get_global_id(0);
    if (i >= n) return;
    const float v = bf(x[i]);
    y[i] = to_bf(0.5f * v * (1.0f + erf(v * M_SQRT1_2_F)));
}

// Grid row and column of patch p in merge order (2 x 2 blocks of a grid gw wide).
inline int2 patch_hw(uint p, uint gw) {
    const uint b = p / 4, a = (p % 4) / 2, c = p % 2;
    return (int2)((int)((b / (gw / 2)) * 2 + a), (int)((b % (gw / 2)) * 2 + c));
}

// The conv layers' mixer: out = bf16(C * bf16(dwconv3x3(bf16(B * x)))) over the merge-order rows; grid (1024, patches).
__kernel void vit_dwconv(__global const ushort *proj, __global const ushort *cw, __global ushort *out, uint gh, uint gw) {
    const uint c = get_global_id(0), p = get_global_id(1);
    const uint D = get_global_size(0);
    const int2 hw = patch_hw(p, gw);
    float acc = 0.0f;
    for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++) {
            const int y = hw.x + dy, x = hw.y + dx;
            if (y < 0 || x < 0 || y >= (int)gh || x >= (int)gw) continue;
            const ulong q = ((ulong)(y / 2) * (gw / 2) + x / 2) * 4 + (y % 2) * 2 + x % 2;
            const float g = bfr(bf(proj[q * 3 * D + c]) * bf(proj[q * 3 * D + 2 * D + c]));
            acc = fma(g, bf(cw[c * 9 + (dy + 1) * 3 + dx + 1]), acc);
        }
    out[(ulong)p * D + c] = to_bf(bfr(acc) * bf(proj[(ulong)p * 3 * D + D + c]));
}

// q or k heads: RMSNorm over the head's 64 values, then axial RoPE (h on angles 0-15 and 32-47, w on 16-31 and 48-63).
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void vit_qk_norm_rope(__global const ushort *src, uint ld, __global ushort *dst, __global const ushort *w,
                               __global const float *inv_freq, uint heads, uint gw, float eps) {
    const uint lane = get_sub_group_local_id();
    const ulong g = (ulong)get_group_id(0) * 4 + get_sub_group_id();
    const ulong p = g / heads, hd = g % heads;
    const ushort4 raw = vload4(0, src + p * ld + hd * VHD + lane * 4);
    float4 v = (float4)(bf(raw.s0), bf(raw.s1), bf(raw.s2), bf(raw.s3));
    const float inv = 1.0f / sqrt(sub_group_reduce_add(v.s0 * v.s0 + v.s1 * v.s1 + v.s2 * v.s2 + v.s3 * v.s3) / (float)VHD + eps);
    const ushort4 wr = vload4(0, w + lane * 4);
    float4 nv;
    nv.s0 = bfr(bf(wr.s0) * bfr(v.s0 * inv));
    nv.s1 = bfr(bf(wr.s1) * bfr(v.s1 * inv));
    nv.s2 = bfr(bf(wr.s2) * bfr(v.s2 * inv));
    nv.s3 = bfr(bf(wr.s3) * bfr(v.s3 * inv));
    float4 other;
    other.s0 = sub_group_shuffle_xor(nv.s0, 8);
    other.s1 = sub_group_shuffle_xor(nv.s1, 8);
    other.s2 = sub_group_shuffle_xor(nv.s2, 8);
    other.s3 = sub_group_shuffle_xor(nv.s3, 8);
    const float4 rot = lane < 8 ? -other : other;
    const int2 hw = patch_hw((uint)p, gw);
    const uint j0 = lane * 4;
    const float pos = (float)(((j0 / 16) % 2 == 0) ? hw.x : hw.y);
    float o[4];
    for (uint e = 0; e < 4; e++) {
        const float ang = pos * inv_freq[(j0 + e) % 16];
        const float cv = cos(ang), sv = sin(ang);
        const float a = e == 0 ? nv.s0 : e == 1 ? nv.s1 : e == 2 ? nv.s2 : nv.s3;
        const float r = e == 0 ? rot.s0 : e == 1 ? rot.s1 : e == 2 ? rot.s2 : rot.s3;
        o[e] = a * cv + r * sv;
    }
    vstore4((ushort4)(to_bf(o[0]), to_bf(o[1]), to_bf(o[2]), to_bf(o[3])), 0, dst + p * ld + hd * VHD + lane * 4);
}

// Unmasked attention over n patches: a sub-group takes QR query rows of a head, lanes keys (scores) or dims (sums).
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void vit_attn(__global const ushort *q, __global const ushort *k, __global const ushort *v, __global ushort *out,
                       uint n, uint hq, uint hkv, float scale) {
    __local float qs[4][QR][VHD];
    const uint lane = get_sub_group_local_id(), sg = get_sub_group_id();
    const uint head = get_group_id(1), kvh = head / (hq / hkv);
    const ulong r0 = ((ulong)get_group_id(0) * 4 + sg) * QR;
    for (uint i = lane; i < QR * VHD; i += 16) {
        const ulong r = r0 + i / VHD;
        qs[sg][i / VHD][i % VHD] = r < n ? bf(q[(r * hq + head) * VHD + i % VHD]) : 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float m[QR], l[QR], o[QR][4];
    for (uint r = 0; r < QR; r++) {
        m[r] = -INFINITY;
        l[r] = 0.0f;
        o[r][0] = o[r][1] = o[r][2] = o[r][3] = 0.0f;
    }
    for (uint k0 = 0; k0 < n; k0 += 16) {
        const uint key = k0 + lane;
        float s[QR];
        for (uint r = 0; r < QR; r++) s[r] = 0.0f;
        if (key < n) {
            __global const ushort *kr = k + ((ulong)key * hkv + kvh) * VHD;
            for (uint d = 0; d < VHD; d += 8) {
                const ushort8 kw = vload8(0, kr + d);
                const float8 kv = (float8)(bf(kw.s0), bf(kw.s1), bf(kw.s2), bf(kw.s3), bf(kw.s4), bf(kw.s5), bf(kw.s6), bf(kw.s7));
                for (uint r = 0; r < QR; r++) {
                    const float8 qv = vload8(0, &qs[sg][r][d]);
                    s[r] += dot(qv.lo, kv.lo) + dot(qv.hi, kv.hi);
                }
            }
        }
        for (uint r = 0; r < QR; r++) {
            const float sc = key < n ? s[r] * scale : -INFINITY;
            const float mn = fmax(m[r], sub_group_reduce_max(sc));
            const float pr = key < n ? exp(sc - mn) : 0.0f;
            const float a = exp(m[r] - mn);
            l[r] = l[r] * a + sub_group_reduce_add(pr);
            m[r] = mn;
            o[r][0] *= a;
            o[r][1] *= a;
            o[r][2] *= a;
            o[r][3] *= a;
            s[r] = pr;
        }
        const uint kn = min(16u, n - k0);
        for (uint j = 0; j < kn; j++) {
            const ushort4 vw = vload4(0, v + ((ulong)(k0 + j) * hkv + kvh) * VHD + lane * 4);
            const float4 vv = (float4)(bf(vw.s0), bf(vw.s1), bf(vw.s2), bf(vw.s3));
            for (uint r = 0; r < QR; r++) {
                const float pj = sub_group_shuffle(s[r], j);
                o[r][0] = fma(pj, vv.s0, o[r][0]);
                o[r][1] = fma(pj, vv.s1, o[r][1]);
                o[r][2] = fma(pj, vv.s2, o[r][2]);
                o[r][3] = fma(pj, vv.s3, o[r][3]);
            }
        }
    }
    for (uint r = 0; r < QR; r++) {
        const ulong row = r0 + r;
        if (row >= n) continue;
        const float il = 1.0f / l[r];
        vstore4((ushort4)(to_bf(o[r][0] * il), to_bf(o[r][1] * il), to_bf(o[r][2] * il), to_bf(o[r][3] * il)), 0,
                out + (row * hq + head) * VHD + lane * 4);
    }
}

// bf16(silu(bf16 a) rounded, times b), as lfm3_swiglu and transformers' bf16 MLP round it.
__kernel void vit_swiglu(__global const ushort *a, __global const ushort *b, __global ushort *out, uint n) {
    const uint i = get_global_id(0);
    if (i >= n) return;
    const float x = bf(a[i]);
    out[i] = to_bf(bfr(x / (1.0f + exp(-x))) * bf(b[i]));
}
