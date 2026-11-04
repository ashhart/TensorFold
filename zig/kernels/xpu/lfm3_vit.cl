// LFM3-VL's vision tower and projector without DPAS: fp32 activations, bf16 weights as stored, fp32 sums.
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
// `rnd` rounds a kernel's output (and its inner points) to bf16 values where transformers' bf16 tower does: C11's form.
inline float rd(float f, uint rnd) { return rnd ? bfr(f) : f; }

// y[r][o] = x[r] . w[o] + bias[o], r < m (fp32, rows ldx / ldy apart; out_bf: y bf16); n % 64 == 0, k % 16 == 0.
__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void vit_gemm(__global const float *x, uint ldx, __global const ushort *w, __global const ushort *bias, uint has_bias,
                       __global void *y, uint ldy, uint m, uint k, uint rnd, uint out_bf) {
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
        float4 xv = (float4)(0.0f);
        if (xr < m) xv = vload4(0, x + xr * ldx + k0 + lk);
        const ushort4 wv = vload4(0, w + (n0 + lr) * k + k0 + lk);
        xs[lk][lr] = xv.s0;
        xs[lk + 1][lr] = xv.s1;
        xs[lk + 2][lr] = xv.s2;
        xs[lk + 3][lr] = xv.s3;
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
            const float v = acc[i][j] + (has_bias ? bf(bias[o]) : 0.0f);
            if (out_bf) {
                ((__global ushort *)y)[r * ldy + o] = to_bf(v);
            } else ((__global float *)y)[r * ldy + o] = rd(v, rnd);
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

// The tower's RMSNorm: y = w * (x * rsqrt(mean(x^2) + eps)), rnd: bf16(w * bf16(x * rsqrt)); a row a work-group.
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void vit_rms(__global const float *x, __global const ushort *w, __global float *y, uint n, float eps, uint rnd) {
    __local float part[4];
    const ulong row = get_group_id(0);
    float ss = 0.0f;
    for (uint i = get_local_id(0); i < n; i += 64) {
        const float v = x[row * n + i];
        ss = fma(v, v, ss);
    }
    const float inv = 1.0f / sqrt(wg_sum64(ss, part) / (float)n + eps);
    for (uint i = get_local_id(0); i < n; i += 64) y[row * n + i] = rd(bf(w[i]) * rd(x[row * n + i] * inv, rnd), rnd);
}

// The residual step: h = h + d in place (rres), then y = the RMSNorm of h (vit_rms, rnorm); a row a work-group.
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void vit_add_rms(__global float *h, __global const float *d, __global const ushort *w, __global float *y, uint n, float eps,
                          uint rres, uint rnorm) {
    __local float part[4];
    const ulong row = get_group_id(0);
    float ss = 0.0f;
    for (uint i = get_local_id(0); i < n; i += 64) {
        const float v = rd(h[row * n + i] + d[row * n + i], rres);
        h[row * n + i] = v;
        ss = fma(v, v, ss);
    }
    const float inv = 1.0f / sqrt(wg_sum64(ss, part) / (float)n + eps);
    for (uint i = get_local_id(0); i < n; i += 64) y[row * n + i] = rd(bf(w[i]) * rd(h[row * n + i] * inv, rnorm), rnorm);
}

// The projector's LayerNorm: y = (x - mean) * rsqrt(var + eps) * w + b (rnd: rounded once); a row a work-group.
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void vit_layernorm(__global const float *x, __global const ushort *w, __global const ushort *b, __global float *y, uint n, float eps,
                            uint rnd) {
    __local float part[4];
    const ulong row = get_group_id(0);
    float s = 0.0f;
    for (uint i = get_local_id(0); i < n; i += 64) s += x[row * n + i];
    const float mean = wg_sum64(s, part) / (float)n;
    float ss = 0.0f;
    for (uint i = get_local_id(0); i < n; i += 64) {
        const float c = x[row * n + i] - mean;
        ss = fma(c, c, ss);
    }
    const float inv = 1.0f / sqrt(wg_sum64(ss, part) / (float)n + eps);
    for (uint i = get_local_id(0); i < n; i += 64) y[row * n + i] = rd((x[row * n + i] - mean) * inv * bf(w[i]) + bf(b[i]), rnd);
}

// GELU with erf, as transformers' "gelu": y = 0.5 x (1 + erf(x / sqrt 2)).
__kernel void vit_gelu(__global const float *x, __global float *y, uint n, uint rnd) {
    const uint i = get_global_id(0);
    if (i >= n) return;
    const float v = x[i];
    y[i] = rd(0.5f * v * (1.0f + erf(v * M_SQRT1_2_F)), rnd);
}

// Grid row and column of patch p in merge order (2 x 2 blocks of a grid gw wide).
inline int2 patch_hw(uint p, uint gw) {
    const uint b = p / 4, a = (p % 4) / 2, c = p % 2;
    return (int2)((int)((b / (gw / 2)) * 2 + a), (int)((b % (gw / 2)) * 2 + c));
}

// The conv mixer: out = C * dwconv3x3(B * x) over merge-order rows (rnd: B * x, the sum, the gate); grid (1024, P).
__kernel void vit_dwconv(__global const float *proj, __global const ushort *cw, __global float *out, uint gh, uint gw, uint rnd) {
    const uint c = get_global_id(0), p = get_global_id(1);
    const uint D = get_global_size(0);
    const int2 hw = patch_hw(p, gw);
    float acc = 0.0f;
    for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++) {
            const int y = hw.x + dy, x = hw.y + dx;
            if (y < 0 || x < 0 || y >= (int)gh || x >= (int)gw) continue;
            const ulong q = ((ulong)(y / 2) * (gw / 2) + x / 2) * 4 + (y % 2) * 2 + x % 2;
            const float g = rd(proj[q * 3 * D + c] * proj[q * 3 * D + 2 * D + c], rnd);
            acc = fma(g, bf(cw[c * 9 + (dy + 1) * 3 + dx + 1]), acc);
        }
    out[(ulong)p * D + c] = rd(rd(acc, rnd) * proj[(ulong)p * 3 * D + D + c], rnd);
}

// q or k heads: RMSNorm over the head's 64 values, then axial RoPE (h on angles 0-15 and 32-47, w on 16-31 and 48-63).
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void vit_qk_norm_rope(__global const float *src, uint ld, __global float *dst, __global const ushort *w,
                               __global const float *inv_freq, uint heads, uint gw, float eps, uint rnd) {
    const uint lane = get_sub_group_local_id();
    const ulong g = (ulong)get_group_id(0) * 4 + get_sub_group_id();
    const ulong p = g / heads, hd = g % heads;
    const float4 v = vload4(0, src + p * ld + hd * VHD + lane * 4);
    const float inv = 1.0f / sqrt(sub_group_reduce_add(v.s0 * v.s0 + v.s1 * v.s1 + v.s2 * v.s2 + v.s3 * v.s3) / (float)VHD + eps);
    const ushort4 wr = vload4(0, w + lane * 4);
    float4 nv;
    nv.s0 = rd(bf(wr.s0) * rd(v.s0 * inv, rnd), rnd);
    nv.s1 = rd(bf(wr.s1) * rd(v.s1 * inv, rnd), rnd);
    nv.s2 = rd(bf(wr.s2) * rd(v.s2 * inv, rnd), rnd);
    nv.s3 = rd(bf(wr.s3) * rd(v.s3 * inv, rnd), rnd);
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
        o[e] = rd(a * cv + r * sv, rnd);
    }
    vstore4((float4)(o[0], o[1], o[2], o[3]), 0, dst + p * ld + hd * VHD + lane * 4);
}

// Unmasked attention over n patches: a sub-group takes QR query rows of a head, lanes keys (scores) or dims (sums).
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void vit_attn(__global const float *q, __global const float *k, __global const float *v, __global float *out,
                       uint n, uint hq, uint hkv, float scale, uint rnd) {
    __local float qs[4][QR][VHD];
    const uint lane = get_sub_group_local_id(), sg = get_sub_group_id();
    const uint head = get_group_id(1), kvh = head / (hq / hkv);
    const ulong r0 = ((ulong)get_group_id(0) * 4 + sg) * QR;
    for (uint i = lane; i < QR * VHD; i += 16) {
        const ulong r = r0 + i / VHD;
        qs[sg][i / VHD][i % VHD] = r < n ? q[(r * hq + head) * VHD + i % VHD] : 0.0f;
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
            __global const float *kr = k + ((ulong)key * hkv + kvh) * VHD;
            for (uint d = 0; d < VHD; d += 8) {
                const float8 kv = vload8(0, kr + d);
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
            const float4 vv = vload4(0, v + ((ulong)(k0 + j) * hkv + kvh) * VHD + lane * 4);
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
        vstore4((float4)(rd(o[r][0] * il, rnd), rd(o[r][1] * il, rnd), rd(o[r][2] * il, rnd), rd(o[r][3] * il, rnd)), 0,
                out + (row * hq + head) * VHD + lane * 4);
    }
}

// vit_attn at 4 query rows a sub-group, 8 sub-groups (32 rows a work-group, as vit_attn's): each row's ops unchanged.
#define QR4 4
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(128, 1, 1)))
__kernel void vit_attn_q4(__global const float *q, __global const float *k, __global const float *v, __global float *out,
                          uint n, uint hq, uint hkv, float scale, uint rnd) {
    __local float qs[8][QR4][VHD];
    const uint lane = get_sub_group_local_id(), sg = get_sub_group_id();
    const uint head = get_group_id(1), kvh = head / (hq / hkv);
    const ulong r0 = ((ulong)get_group_id(0) * 8 + sg) * QR4;
    for (uint i = lane; i < QR4 * VHD; i += 16) {
        const ulong r = r0 + i / VHD;
        qs[sg][i / VHD][i % VHD] = r < n ? q[(r * hq + head) * VHD + i % VHD] : 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float m[QR4], l[QR4], o[QR4][4];
    for (uint r = 0; r < QR4; r++) {
        m[r] = -INFINITY;
        l[r] = 0.0f;
        o[r][0] = o[r][1] = o[r][2] = o[r][3] = 0.0f;
    }
    for (uint k0 = 0; k0 < n; k0 += 16) {
        const uint key = k0 + lane;
        float s[QR4];
        for (uint r = 0; r < QR4; r++) s[r] = 0.0f;
        if (key < n) {
            __global const float *kr = k + ((ulong)key * hkv + kvh) * VHD;
            for (uint d = 0; d < VHD; d += 8) {
                const float8 kv = vload8(0, kr + d);
                for (uint r = 0; r < QR4; r++) {
                    const float8 qv = vload8(0, &qs[sg][r][d]);
                    s[r] += dot(qv.lo, kv.lo) + dot(qv.hi, kv.hi);
                }
            }
        }
        for (uint r = 0; r < QR4; r++) {
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
            const float4 vv = vload4(0, v + ((ulong)(k0 + j) * hkv + kvh) * VHD + lane * 4);
            for (uint r = 0; r < QR4; r++) {
                const float pj = sub_group_shuffle(s[r], j);
                o[r][0] = fma(pj, vv.s0, o[r][0]);
                o[r][1] = fma(pj, vv.s1, o[r][1]);
                o[r][2] = fma(pj, vv.s2, o[r][2]);
                o[r][3] = fma(pj, vv.s3, o[r][3]);
            }
        }
    }
    for (uint r = 0; r < QR4; r++) {
        const ulong row = r0 + r;
        if (row >= n) continue;
        const float il = 1.0f / l[r];
        vstore4((float4)(rd(o[r][0] * il, rnd), rd(o[r][1] * il, rnd), rd(o[r][2] * il, rnd), rd(o[r][3] * il, rnd)), 0,
                out + (row * hq + head) * VHD + lane * 4);
    }
}

// vit_attn_q4 with q in registers (lane l holds dims 4l to 4l + 3, shuffled to every lane), 32-bit offsets: no SLM.
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(128, 1, 1)))
__kernel void vit_attn_qr(__global const float *q, __global const float *k, __global const float *v, __global float *out,
                          uint n, uint hq, uint hkv, float scale, uint rnd) {
    const uint lane = get_sub_group_local_id(), sg = get_sub_group_id();
    const uint head = get_group_id(1), kvh = head / (hq / hkv);
    const uint r0 = (get_group_id(0) * 8 + sg) * QR4;
    const uint ks = hkv * VHD; // floats from one key's row to the next
    float4 qd[QR4];
    float m[QR4], l[QR4], o[QR4][4];
    for (uint r = 0; r < QR4; r++) {
        const float4 qa = vload4(0, q + (min(r0 + r, n - 1) * hq + head) * VHD + lane * 4); // past n: a real row, never stored
        qd[r] = r0 + r < n ? qa : (float4)(0.0f);
        m[r] = -INFINITY;
        l[r] = 0.0f;
        o[r][0] = o[r][1] = o[r][2] = o[r][3] = 0.0f;
    }
    for (uint k0 = 0; k0 < n; k0 += 16) {
        const uint key = k0 + lane, kr = min(key, n - 1) * ks + kvh * VHD; // past n: a real row, s is never read
        float s[QR4];
        for (uint r = 0; r < QR4; r++) s[r] = 0.0f;
        for (uint d = 0; d < VHD; d += 8) {
            const float8 kv = vload8(0, k + kr + d);
            for (uint r = 0; r < QR4; r++) {
                const float4 qa = qd[r];
                const float8 qv = (float8)(sub_group_shuffle(qa.s0, d / 4), sub_group_shuffle(qa.s1, d / 4), sub_group_shuffle(qa.s2, d / 4),
                                           sub_group_shuffle(qa.s3, d / 4), sub_group_shuffle(qa.s0, d / 4 + 1), sub_group_shuffle(qa.s1, d / 4 + 1),
                                           sub_group_shuffle(qa.s2, d / 4 + 1), sub_group_shuffle(qa.s3, d / 4 + 1));
                s[r] += dot(qv.lo, kv.lo) + dot(qv.hi, kv.hi);
            }
        }
        for (uint r = 0; r < QR4; r++) {
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
        if (k0 + 16 <= n) { // a full tile: j constant once unrolled, so each shuffle is a register region
            __attribute__((opencl_unroll_hint(16))) for (uint j = 0; j < 16; j++) {
                const float4 vv = vload4(0, v + (k0 + j) * ks + kvh * VHD + lane * 4);
                for (uint r = 0; r < QR4; r++) {
                    const float pj = sub_group_shuffle(s[r], j);
                    o[r][0] = fma(pj, vv.s0, o[r][0]);
                    o[r][1] = fma(pj, vv.s1, o[r][1]);
                    o[r][2] = fma(pj, vv.s2, o[r][2]);
                    o[r][3] = fma(pj, vv.s3, o[r][3]);
                }
            }
        } else
            for (uint j = 0; j < n - k0; j++) {
                const float4 vv = vload4(0, v + (k0 + j) * ks + kvh * VHD + lane * 4);
                for (uint r = 0; r < QR4; r++) {
                    const float pj = sub_group_shuffle(s[r], j);
                    o[r][0] = fma(pj, vv.s0, o[r][0]);
                    o[r][1] = fma(pj, vv.s1, o[r][1]);
                    o[r][2] = fma(pj, vv.s2, o[r][2]);
                    o[r][3] = fma(pj, vv.s3, o[r][3]);
                }
            }
    }
    for (uint r = 0; r < QR4; r++) {
        const ulong row = r0 + r;
        if (row >= n) continue;
        const float il = 1.0f / l[r];
        vstore4((float4)(rd(o[r][0] * il, rnd), rd(o[r][1] * il, rnd), rd(o[r][2] * il, rnd), rd(o[r][3] * il, rnd)), 0,
                out + (row * hq + head) * VHD + lane * 4);
    }
}

// SwiGLU: silu(a) * b (rnd: transformers' bf16(bf16(silu(a)) * b)).
__kernel void vit_swiglu(__global const float *a, __global const float *b, __global float *out, uint n, uint rnd) {
    const uint i = get_global_id(0);
    if (i >= n) return;
    const float x = a[i];
    out[i] = rd(rd(x / (1.0f + exp(-x)), rnd) * b[i], rnd);
}
