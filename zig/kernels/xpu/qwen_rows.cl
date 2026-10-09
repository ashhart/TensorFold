// Qwen3.8 multi-row (window) building blocks: R tokens per forward with per-row arithmetic identical to the single-row kernels
// (qwen_basic.cl / qwen_gdn.cl / qwen_attn.cl), so a row's bits do not depend on how many rows share the call.
// GDN: conv and delta-rule state advance row by row inside the kernel (each (head, value row) recurrence is independent), the new
// state goes to a separate buffer (or in place) so a rejected window can be rolled back; attention: causal inside the window.
#define DK 128
#define DV 128
#define KH 16
#define VH 48
#define HD 256
#define NH 24
#define NKV 4
#define GQ 6
#define CH 64
#define RD 32
inline float bf(ushort v) { return as_float((uint)v << 16); }
inline ushort to_bf(float f) {
    uint u = as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (ushort)(u >> 16);
}
inline float bfr(float f) { return bf(to_bf(f)); }
inline float silu(float x) { return x / (1.0f + exp(-x)); }

#define SUMSQ_STEP(ss, f) ss = fma(f, f, ss)
// Sum of squares in the order of the original 64-item rmsnorm (see qwen_basic.cl sumsq_ordered).
inline float sumsq_ordered(__local float *vals, uint n, __local float *part) {
    const uint lid = get_local_id(0);
    if (lid < 64) {
        float ss = 0.0f;
        for (uint i = lid; i < n; i += 64) {
            const float f = vals[i];
            SUMSQ_STEP(ss, f);
        }
        ss = sub_group_reduce_add(ss);
        if (get_sub_group_local_id() == 0) part[get_sub_group_id()] = ss;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    return part[0] + part[1] + part[2] + part[3];
}

// Row `get_group_id(0)` of x [rows][n] (n <= 5120): x = bf16(x + d); y = bf16(x * rsqrt(mean(x^2) + eps) * w). Same arithmetic as add_rmsnorm.
#define ADD_RMS_ROWS(LOADW) \
    __local float part[4]; \
    __local float vals[5120]; \
    const uint lid = get_local_id(0); \
    const ulong base = (ulong)get_group_id(0) * n; \
    for (uint i = lid; i < n; i += 512) { \
        const ushort v = to_bf(bf(x[base + i]) + bf(d[base + i])); \
        x[base + i] = v; \
        vals[i] = bf(v); \
    } \
    barrier(CLK_LOCAL_MEM_FENCE); \
    const float scale = rsqrt(sumsq_ordered(vals, n, part) / (float)n + eps); \
    for (uint i = lid; i < n; i += 512) y[base + i] = to_bf(vals[i] * scale * LOADW(w[i]));
#define W_BF(v) bf(v)
#define W_F32(v) (v)

__attribute__((reqd_work_group_size(512, 1, 1))) __attribute__((intel_reqd_sub_group_size(16)))
__kernel void add_rmsnorm_r(__global ushort *x, __global const ushort *d, __global const ushort *w, __global ushort *y, uint n, float eps) {
    ADD_RMS_ROWS(W_BF)
}

__attribute__((reqd_work_group_size(512, 1, 1))) __attribute__((intel_reqd_sub_group_size(16)))
__kernel void add_rmsnorm_r_f32w(__global ushort *x, __global const ushort *d, __global const float *w, __global ushort *y, uint n, float eps) {
    ADD_RMS_ROWS(W_F32)
}

#define RMS_ROWS(LOADW) \
    __local float part[4]; \
    __local float vals[5120]; \
    const uint lid = get_local_id(0); \
    const ulong base = (ulong)get_group_id(0) * n; \
    for (uint i = lid; i < n; i += 512) vals[i] = bf(x[base + i]); \
    barrier(CLK_LOCAL_MEM_FENCE); \
    const float scale = rsqrt(sumsq_ordered(vals, n, part) / (float)n + eps); \
    for (uint i = lid; i < n; i += 512) y[base + i] = to_bf(vals[i] * scale * LOADW(w[i]));

__attribute__((reqd_work_group_size(512, 1, 1))) __attribute__((intel_reqd_sub_group_size(16)))
__kernel void rmsnorm_r(__global ushort *x, __global const ushort *w, __global ushort *y, uint n, float eps) {
    RMS_ROWS(W_BF)
}

__attribute__((reqd_work_group_size(512, 1, 1))) __attribute__((intel_reqd_sub_group_size(16)))
__kernel void rmsnorm_r_f32w(__global ushort *x, __global const float *w, __global ushort *y, uint n, float eps) {
    RMS_ROWS(W_F32)
}

// ---- Gated DeltaNet window ----

// Extended conv input of one channel: ext(0..2) = the three state rows, ext(3 + t) = qkv row t.
inline ushort ext_at(__global const ushort *qkv, __global const ushort *state, uint cd, uint ch, uint j) {
    return j < 3 ? state[j * cd + ch] : qkv[(ulong)(j - 3) * cd + ch];
}

// Conv + silu of channel ch at window row t: taps cw[ch][0..3] (tap 0 = oldest); bf16 bits.
inline ushort conv_t(__global const ushort *qkv, __global const ushort *state, __global const ushort *cw, uint cd, uint ch, uint t) {
    const ushort4 w = vload4(ch, cw);
    float acc = bf(ext_at(qkv, state, cd, ch, t)) * bf(w.s0);
    acc += bf(ext_at(qkv, state, cd, ch, t + 1)) * bf(w.s1);
    acc += bf(ext_at(qkv, state, cd, ch, t + 2)) * bf(w.s2);
    acc += bf(ext_at(qkv, state, cd, ch, t + 3)) * bf(w.s3);
    return to_bf(silu(acc));
}

// New conv state of one channel after c window rows: ext(c .. c + 2). Reads all three old values before writing.
inline void conv_commit_ch(__global const ushort *qkv, __global ushort *state, uint cd, uint ch, uint c) {
    const ushort n0 = ext_at(qkv, state, cd, ch, c);
    const ushort n1 = ext_at(qkv, state, cd, ch, c + 1);
    const ushort n2 = ext_at(qkv, state, cd, ch, c + 2);
    state[ch] = n0;
    state[cd + ch] = n1;
    state[2 * cd + ch] = n2;
}

// Conv state after c accepted rows for channels [ch0, ch0 + n) (one thread a channel).
__kernel void gdn_conv_commit(__global const ushort *qkv, __global ushort *state, uint cd, uint c, uint ch0, uint n) {
    const uint i = get_global_id(0);
    if (i < n) conv_commit_ch(qkv, state, cd, ch0 + i, c);
}

// q and k channels of window row get_group_id(1): conv + silu, then q = bf16(rms(q) / Dk), k = bf16(rms(k) / sqrt(Dk)) per key head, widened
// to fp32 into qn/kn [rows][2048]. Rows are independent (they only read the state and the window's inputs); the caller shifts the state
// afterwards (gdn_conv_commit). One sub-group a (key head, row).
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void gdn_conv_prep_r(__global const ushort *qkv, __global const ushort *state, __global const ushort *cw, __global float *qn, __global float *kn, uint cd) {
    const uint h = get_group_id(0);
    const uint t = get_group_id(1);
    const uint lane = get_sub_group_local_id();
    for (uint which = 0; which < 2; which++) {
        const uint base = which * (KH * DK) + h * DK + lane * 8;
        float v[8];
        float ss = 0.0f;
        for (int j = 0; j < 8; j++) {
            v[j] = bf(conv_t(qkv, state, cw, cd, base + j, t));
            ss += v[j] * v[j];
        }
        ss = sub_group_reduce_add(ss);
        const float r = rsqrt(ss / (float)DK + 1e-6f);
        const float sc = which == 0 ? (1.0f / (float)DK) : (1.0f / sqrt((float)DK));
        __global float *dst = (which == 0 ? qn : kn) + (ulong)t * (KH * DK) + h * DK + lane * 8;
        for (int j = 0; j < 8; j++) dst[j] = bfr(v[j] * r * sc);
    }
}

// Value-channel conv + silu of window row get_group_id(0): vout[row][6144] = conv of channels 4096 + c (bf16), reading the old state like gdn_conv_prep_r. Grid (rows, 96 groups
// of 64 items).
__kernel void gdn_vconv_r(__global const ushort *qkv, __global const ushort *state, __global const ushort *cw, __global ushort *vout, uint cd) {
    const uint t = get_group_id(0);
    const uint c = get_group_id(1) * 64 + get_local_id(0);
    if (c < VH * DV) vout[(ulong)t * (VH * DV) + c] = conv_t(qkv, state, cw, cd, 2 * KH * DK + c, t);
}

// beta = sigmoid(b) and the decay g = exp(-exp(A_log) * softplus(a + dt_bias)) of window row get_group_id(0), head get_local_id(0) (48 items), as gdn_step.
__kernel void gdn_gates_r(__global const ushort *bp, __global const ushort *ap, __global const float *a_log, __global const float *dt_bias, __global float *beta_o, __global float *g_o) {
    const uint t = get_group_id(0);
    const uint h = get_local_id(0);
    beta_o[t * VH + h] = 1.0f / (1.0f + exp(-bf(bp[t * VH + h])));
    const float av = bf(ap[t * VH + h]) + dt_bias[h];
    const float sp = av > 20.0f ? av : log1p(exp(av));
    g_o[t * VH + h] = exp(-exp(a_log[h]) * sp);
}

// Delta rule over `rows` window rows for value head h, output rows [4*chunk, 4*chunk+4): a 16-lane sub-group owns a row (8 Dk values a lane) and keeps its state in
// registers across the window; the per-row inputs (q, k, the conv'd v, beta, g) are precomputed row-parallel. State goes to s_out (== s_in for in place).
// tiled: GGUF value-head order.
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void gdn_step_r2(__global const float *qn, __global const float *kn, __global const ushort *vconv, __global const float *beta_a, __global const float *g_a,
                          __global const float *s_in, __global float *s_out, __global ushort *y, uint tiled, uint rows) {
    const uint h = get_group_id(0);
    const uint row = get_group_id(1) * 4 + get_sub_group_id();
    const uint lane = get_sub_group_local_id();
    const uint kh = tiled ? h % KH : h / (VH / KH);
    const ulong off = ((ulong)h * DV + row) * DK;
    float8 s = vload8(lane, s_in + off);
    for (uint t = 0; t < rows; t++) {
        const float8 k8 = vload8(lane, kn + (ulong)t * (KH * DK) + kh * DK);
        const float8 q8 = vload8(lane, qn + (ulong)t * (KH * DK) + kh * DK);
        s *= g_a[t * VH + h];
        const float kv = sub_group_reduce_add(dot(s.s0123, k8.s0123) + dot(s.s4567, k8.s4567));
        const float delta = (bf(vconv[(ulong)t * (VH * DV) + h * DV + row]) - kv) * beta_a[t * VH + h];
        s += k8 * delta;
        const float o = sub_group_reduce_add(dot(s.s0123, q8.s0123) + dot(s.s4567, q8.s4567));
        if (lane == 0) y[(ulong)t * (VH * DV) + h * DV + row] = to_bf(o);
    }
    vstore8(s, lane, s_out + off);
}

// gdn_step_r2 with NR value rows a sub-group (rows [NR * (4 * chunk + sg), +NR)): the NR recurrences are independent chains that the sub-group interleaves, and k, q, beta
// and g of a token are loaded once. Every row does exactly the arithmetic and reduction order of gdn_step_r2, so state and y are bit-identical to it. Grid (48, 32 / NR).
#define GDN_STEP_NR(NAME, NR) \
__attribute__((intel_reqd_sub_group_size(16))) \
__kernel void NAME(__global const float *qn, __global const float *kn, __global const ushort *vconv, __global const float *beta_a, __global const float *g_a, \
                   __global const float *s_in, __global float *s_out, __global ushort *y, uint tiled, uint rows) { \
    const uint h = get_group_id(0); \
    const uint row0 = (get_group_id(1) * 4 + get_sub_group_id()) * NR; \
    const uint lane = get_sub_group_local_id(); \
    const uint kh = tiled ? h % KH : h / (VH / KH); \
    const ulong off = ((ulong)h * DV + row0) * DK; \
    float8 s[NR]; \
    _Pragma("unroll") for (int r = 0; r < NR; r++) s[r] = vload8(lane, s_in + off + (ulong)r * DK); \
    for (uint t = 0; t < rows; t++) { \
        const float8 k8 = vload8(lane, kn + (ulong)t * (KH * DK) + kh * DK); \
        const float8 q8 = vload8(lane, qn + (ulong)t * (KH * DK) + kh * DK); \
        const float g = g_a[t * VH + h]; \
        const float beta = beta_a[t * VH + h]; \
        float vr[NR]; \
        _Pragma("unroll") for (int r = 0; r < NR; r++) vr[r] = bf(vconv[(ulong)t * (VH * DV) + h * DV + row0 + r]); \
        float kv[NR]; \
        _Pragma("unroll") for (int r = 0; r < NR; r++) { \
            s[r] *= g; \
            kv[r] = sub_group_reduce_add(dot(s[r].s0123, k8.s0123) + dot(s[r].s4567, k8.s4567)); \
        } \
        _Pragma("unroll") for (int r = 0; r < NR; r++) s[r] += k8 * ((vr[r] - kv[r]) * beta); \
        float o[NR]; \
        _Pragma("unroll") for (int r = 0; r < NR; r++) o[r] = sub_group_reduce_add(dot(s[r].s0123, q8.s0123) + dot(s[r].s4567, q8.s4567)); \
        _Pragma("unroll") for (int r = 0; r < NR; r++) if (lane == 0) y[(ulong)t * (VH * DV) + h * DV + row0 + r] = to_bf(o[r]); \
    } \
    _Pragma("unroll") for (int r = 0; r < NR; r++) vstore8(s[r], lane, s_out + off + (ulong)r * DK); \
}
GDN_STEP_NR(gdn_step_r2x2, 2)
GDN_STEP_NR(gdn_step_r2x4, 4)

// out = bf16(silu(z) * (y * rsqrt(mean(y^2) + eps) * w)) for window row get_group_id(1), value head get_group_id(0).
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void gdn_gate_norm_r(__global const ushort *y, __global const ushort *z, __global const ushort *w, __global ushort *out, float eps) {
    const uint h = get_group_id(0);
    const ulong rb = (ulong)get_group_id(1) * (VH * DV);
    const uint lane = get_sub_group_local_id();
    float v[8];
    float ss = 0.0f;
    for (int j = 0; j < 8; j++) { v[j] = bf(y[rb + h * DV + lane * 8 + j]); ss += v[j] * v[j]; }
    ss = sub_group_reduce_add(ss);
    const float r = rsqrt(ss / (float)DV + eps);
    for (int j = 0; j < 8; j++) {
        const uint d = lane * 8 + j;
        out[rb + h * DV + d] = to_bf(silu(bf(z[rb + h * DV + d])) * (v[j] * r * bf(w[d])));
    }
}

// ---- full attention window ----

// Window row get_group_id(1): heads 0..23 are queries from qg[row][head][0..255] (qg rows are [head][512]: query then gate), heads 24..27
// keys from kraw[row]. Norm, then rotate-half with rope [row][cos(32) | sin(32)]; keys go to kc[k_off + row*1024 ..].
#define ATTN_PREP_ROWS(LOADW) \
    __local float part[16]; \
    __local float nv[HD]; \
    const uint head = get_group_id(0); \
    const uint row = get_group_id(1); \
    const uint i = get_local_id(0); \
    const bool is_q = head < NH; \
    const float x = is_q ? bf(qg[(ulong)row * (NH * 2 * HD) + head * 2 * HD + i]) : bf(kraw[(ulong)row * (NKV * HD) + (head - NH) * HD + i]); \
    float ss = sub_group_reduce_add(x * x); \
    if (get_sub_group_local_id() == 0) part[get_sub_group_id()] = ss; \
    barrier(CLK_LOCAL_MEM_FENCE); \
    float total = 0.0f; \
    for (int j = 0; j < 16; j++) total += part[j]; \
    const float r = rsqrt(total / (float)HD + eps); \
    nv[i] = bfr(x * r * LOADW(is_q ? qw[i] : kw[i])); \
    barrier(CLK_LOCAL_MEM_FENCE); \
    const __global float *rp = rope + row * 64; \
    float o = nv[i]; \
    if (i < RD) o = nv[i] * rp[i] - nv[i + RD] * rp[RD + i]; \
    else if (i < 2 * RD) o = nv[i] * rp[i - RD] + nv[i - RD] * rp[RD + i - RD]; \
    if (is_q) qo[(ulong)row * (NH * HD) + head * HD + i] = to_bf(o); \
    else kc[k_off + (ulong)row * (NKV * HD) + (head - NH) * HD + i] = to_bf(o);

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void attn_prep_r(__global const ushort *qg, __global const ushort *kraw, __global const ushort *qw, __global const ushort *kw,
                          __global const float *rope, __global ushort *qo, __global ushort *kc, ulong k_off, float eps) {
    ATTN_PREP_ROWS(W_BF)
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void attn_prep_r_f32w(__global const ushort *qg, __global const ushort *kraw, __global const float *qw, __global const float *kw,
                               __global const float *rope, __global ushort *qo, __global ushort *kc, ulong k_off, float eps) {
    ATTN_PREP_ROWS(W_F32)
}

// Pass 1 for window row get_group_id(2) at position pos0 + row (keys 0..pos0+row): work-group (kv head, chunk), same arithmetic as
// attn_partial. po/pm/pl are [row][maxc][head]...; chunks past the row's length exit.
__attribute__((reqd_work_group_size(128, 1, 1))) __attribute__((intel_reqd_sub_group_size(16)))
__kernel void attn_partial_r(__global const ushort *q, __global const ushort *kc, __global const ushort *vc,
                             __global float *po, __global float *pm, __global float *pl, uint pos0, float scale, uint maxc) {
    __local float qs[GQ * HD];
    __local float sc[GQ * CH];
    const uint hk = get_group_id(0);
    const uint c = get_group_id(1);
    const uint row = get_group_id(2);
    const uint len = pos0 + row + 1;
    const uint lid = get_local_id(0);
    const uint sg = get_sub_group_id();
    const uint lane = get_sub_group_local_id();
    const uint k0 = c * CH;
    if (k0 >= len) return;
    const uint n = min((uint)CH, len - k0);
    __global const ushort *qrow = q + (ulong)row * (NH * HD);
    __global float *por = po + (ulong)row * maxc * NH * HD;
    __global float *pmr = pm + (ulong)row * maxc * NH;
    __global float *plr = pl + (ulong)row * maxc * NH;
    for (uint i = lid; i < GQ * HD; i += 128) qs[i] = bf(qrow[hk * GQ * HD + i]);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint k = sg; k < n; k += 8) {
        const float16 kv = as_float16(convert_uint16(vload16(lane, kc + ((ulong)(k0 + k) * NKV + hk) * HD)) << 16);
#pragma unroll
        for (int h = 0; h < GQ; h++) {
            const float16 qv = vload16(lane, qs + h * HD);
            const float4 a = qv.s0123 * kv.s0123 + qv.s4567 * kv.s4567 + qv.s89ab * kv.s89ab + qv.scdef * kv.scdef;
            const float s = sub_group_reduce_add(a.x + a.y + a.z + a.w);
            if (lane == 0) sc[h * CH + k] = s * scale;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sg < GQ) {
        float s[CH / 16];
        float m = -INFINITY;
#pragma unroll
        for (int j = 0; j < CH / 16; j++) {
            const uint k = lane + 16 * j;
            s[j] = k < n ? sc[sg * CH + k] : -INFINITY;
            m = fmax(m, s[j]);
        }
        m = sub_group_reduce_max(m);
        float den = 0.0f;
#pragma unroll
        for (int j = 0; j < CH / 16; j++) {
            const float e = exp(s[j] - m);
            sc[sg * CH + lane + 16 * j] = e;
            den += e;
        }
        den = sub_group_reduce_add(den);
        if (lane == 0) {
            pmr[c * NH + hk * GQ + sg] = m;
            plr[c * NH + hk * GQ + sg] = den;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float2 acc[GQ];
#pragma unroll
    for (int h = 0; h < GQ; h++) acc[h] = (float2)(0.0f);
    for (uint k = 0; k < n; k++) {
        const uint vv = ((__global const uint *)vc)[((ulong)(k0 + k) * NKV + hk) * (HD / 2) + lid];
        const float v0 = as_float(vv << 16);
        const float v1 = as_float(vv & 0xffff0000u);
#pragma unroll
        for (int h = 0; h < GQ; h++) {
            const float p = sc[h * CH + k];
            acc[h].x += p * v0;
            acc[h].y += p * v1;
        }
    }
#pragma unroll
    for (int h = 0; h < GQ; h++) vstore2(acc[h], lid, por + ((ulong)c * NH + hk * GQ + h) * HD);
}

// Pass 2 for window row get_group_id(1), query head get_group_id(0): merge chunks, output gate; out [row][head][256].
__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void attn_merge_r(__global const float *po, __global const float *pm, __global const float *pl,
                           __global const ushort *qg, __global ushort *out, uint pos0, uint maxc) {
    const uint head = get_group_id(0);
    const uint row = get_group_id(1);
    const uint d = get_local_id(0);
    const uint len = pos0 + row + 1;
    const uint nch = (len + CH - 1) / CH;
    __global const float *por = po + (ulong)row * maxc * NH * HD;
    __global const float *pmr = pm + (ulong)row * maxc * NH;
    __global const float *plr = pl + (ulong)row * maxc * NH;
    float m = -INFINITY;
    for (uint c = 0; c < nch; c++) m = fmax(m, pmr[c * NH + head]);
    float den = 0.0f, acc = 0.0f;
    for (uint c = 0; c < nch; c++) {
        const float w = exp(pmr[c * NH + head] - m);
        den += plr[c * NH + head] * w;
        acc += por[((ulong)c * NH + head) * HD + d] * w;
    }
    const float gate = bf(qg[(ulong)row * (NH * 2 * HD) + head * 2 * HD + HD + d]);
    out[(ulong)row * (NH * HD) + head * HD + d] = to_bf(bfr(acc / den) * (1.0f / (1.0f + exp(-gate))));
}

// Small fp16-weight projection for prompt chunks: y[row][y_off + o] = bf16(sum_i x[row][i] * w[o][i]); sub-group a (4 outputs a group, row = group y).
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void mv_f16_rows(__global const ushort *w, __global const ushort *x, __global ushort *y, uint in_dim, uint y_off, uint rows) {
    const uint o = get_group_id(0) * 4 + get_sub_group_id();
    const uint m = get_group_id(1);
    if (o >= rows) return;
    const uint lane = get_sub_group_local_id();
    float acc = 0.0f;
    for (uint c = lane; c < in_dim / 8; c += 16) {
        const float8 wv = vload_half8(c, (__global const half *)(w + (ulong)o * in_dim));
        const float8 xv = as_float8(convert_uint8(vload8(c, x + (ulong)m * in_dim)) << 16);
        acc += dot(wv.s0123, xv.s0123) + dot(wv.s4567, xv.s4567);
    }
    acc = sub_group_reduce_add(acc);
    if (lane == 0) y[(ulong)m * rows + y_off + o] = to_bf(acc);
}
