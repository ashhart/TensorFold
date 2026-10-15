// Nemotron-H multi-row (window) kernels: single-row kernels plus a row index; row r matches the one-token path exactly.
#define DS 128
#define SG 16
#define HD 128
#define GQ 16
#define TILE 64
inline float bf(ushort v) { return as_float((uint)v << 16); }
inline ushort to_bf(float f) {
    uint u = as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (ushort)(u >> 16);
}
inline float bfr(float f) { return bf(to_bf(f)); }
inline float bf_round(float f) { return bf(to_bf(f)); }

// The decode's 4-bit dot for up to MAXR rows: lane words l, l+16, ..., each word unpacked once for all rows; x + xo[r].
#ifndef MAXR
#define MAXR 16 // nem_rows_r1 / _r2 / _r4 / _r8: the same kernels built for windows of at most 1, 2, 4, 8 rows
#endif
// One word of a lane against row r's 8 inputs: the decode's dot and sx, then acc[r] += sc * dot + bi * sx.
#define QWORD(PK, SC, BI, WI)                                                                                                  \
    {                                                                                                                          \
        const float q0 = (float)(PK & 15u), q1 = (float)((PK >> 4) & 15u), q2 = (float)((PK >> 8) & 15u);                     \
        const float q3 = (float)((PK >> 12) & 15u), q4 = (float)((PK >> 16) & 15u), q5 = (float)((PK >> 20) & 15u);           \
        const float q6 = (float)((PK >> 24) & 15u), q7 = (float)((PK >> 28) & 15u);                                            \
        _Pragma("unroll") for (uint r = 0; r < MAXR; r++) {                                                                    \
            if (r < n) {                                                                                                       \
                const float8 xv = as_float8(convert_uint8(vload8(WI, x + xo[r])) << 16);                                       \
                float dot = 0.0f, sx = 0.0f;                                                                                   \
                dot += xv.s0 * q0;                                                                                             \
                sx += xv.s0;                                                                                                   \
                dot += xv.s1 * q1;                                                                                             \
                sx += xv.s1;                                                                                                   \
                dot += xv.s2 * q2;                                                                                             \
                sx += xv.s2;                                                                                                   \
                dot += xv.s3 * q3;                                                                                             \
                sx += xv.s3;                                                                                                   \
                dot += xv.s4 * q4;                                                                                             \
                sx += xv.s4;                                                                                                   \
                dot += xv.s5 * q5;                                                                                             \
                sx += xv.s5;                                                                                                   \
                dot += xv.s6 * q6;                                                                                             \
                sx += xv.s6;                                                                                                   \
                dot += xv.s7 * q7;                                                                                             \
                sx += xv.s7;                                                                                                   \
                acc[r] += SC * dot + BI * sx;                                                                                  \
            }                                                                                                                  \
        }                                                                                                                      \
    }

inline void qrows(__global const uint *w, __global const ushort *scales, __global const ushort *biases, __global const ushort *x, const ulong *xo, uint n,
                  ulong row, uint in_dim, float *acc) {
    const uint lane = get_sub_group_local_id();
    const uint words = in_dim / 8, groups = in_dim / 64;
    __global const uint *wr = w + row * words;
    __global const ushort *sr = scales + row * groups, *br = biases + row * groups;
#pragma unroll
    for (uint r = 0; r < MAXR; r++) acc[r] = 0.0f;
    uint wi = lane;
    for (; wi + 3 * SG < words; wi += 4 * SG) { // 4 of the lane's words: loads first, then the words in turn
        uint pk[4];
        float sc[4], bi[4];
#pragma unroll
        for (uint u = 0; u < 4; u++) {
            const uint wu = wi + u * SG;
            pk[u] = wr[wu];
            sc[u] = bf(sr[wu / 8]);
            bi[u] = bf(br[wu / 8]);
        }
#pragma unroll
        for (uint u = 0; u < 4; u++) QWORD(pk[u], sc[u], bi[u], wi + u * SG)
    }
    for (; wi < words; wi += SG) {
        const uint pk = wr[wi];
        const float sc = bf(sr[wi / 8]), bi = bf(br[wi / 8]);
        QWORD(pk, sc, bi, wi)
    }
#pragma unroll
    for (uint r = 0; r < MAXR; r++)
        if (r < n) acc[r] = sub_group_reduce_add(acc[r]);
}

// Rows 0 .. n of a window, x rows `stride` apart.
inline void strided(ulong *xo, uint n, ulong stride) {
#pragma unroll
    for (uint r = 0; r < MAXR; r++) xo[r] = r < n ? (ulong)r * stride : 0;
}

// ---- dense 4-bit matvec (qmv4_bf16 of mamba.cl): a sub-group a weight row; x row r -> y[r * y_stride + y_off + row]
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void qmv4_bf16_r(__global const uint *w, __global const ushort *scales, __global const ushort *biases, __global const ushort *x, __global ushort *y,
                          uint in_dim, uint rows, uint y_stride, uint y_off, uint n) {
    const uint row = get_group_id(0);
    ulong xo[MAXR];
    float acc[MAXR];
    strided(xo, n, in_dim);
    qrows(w, scales, biases, x, xo, n, row, in_dim, acc);
    if (get_sub_group_local_id() == 0)
        for (uint r = 0; r < n; r++) y[(ulong)r * y_stride + y_off + row] = to_bf(acc[r]);
}

// ---- attention-side 4-bit matvec (qmv4_bf / qmv4_f32 of attn.cl): 4 weight rows a work-group; bf16 or fp32 out ----
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void qmv4_bf_r(__global const uint *w, __global const ushort *scales, __global const ushort *biases, __global const ushort *x, __global ushort *y,
                        uint in_dim, uint y_off, uint y_stride, uint rows, uint n) {
    const uint row = get_group_id(0) * 4 + get_sub_group_id();
    if (row >= rows) return;
    ulong xo[MAXR];
    float acc[MAXR];
    strided(xo, n, in_dim);
    qrows(w, scales, biases, x, xo, n, row, in_dim, acc);
    if (get_sub_group_local_id() == 0)
        for (uint r = 0; r < n; r++) y[(ulong)r * y_stride + y_off + row] = to_bf(acc[r]);
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void qmv4_f32_r(__global const uint *w, __global const ushort *scales, __global const ushort *biases, __global const ushort *x, __global float *y,
                         uint in_dim, uint rows, uint n) {
    const uint row = get_group_id(0) * 4 + get_sub_group_id();
    if (row >= rows) return;
    ulong xo[MAXR];
    float acc[MAXR];
    strided(xo, n, in_dim);
    qrows(w, scales, biases, x, xo, n, row, in_dim, acc);
    if (get_sub_group_local_id() == 0)
        for (uint r = 0; r < n; r++) y[(ulong)r * rows + row] = acc[r];
}

// ---- Mamba2 conv1d over n rows, state [3][cd] bf16 sin -> sout; proj rows [n][pd], x part at xoff, out [n][cd] ----
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void conv1d_rows(__global const ushort *proj, uint pd, uint xoff, __global const ushort *sin, __global ushort *sout, __global const ushort *cw,
                          __global const ushort *cb, __global ushort *out, uint cd, uint n) {
    const uint ch = get_global_id(0);
    if (ch >= cd) return;
    const ushort4 w = vload4(ch, cw);
    ushort t0 = sin[ch], t1 = sin[cd + ch], t2 = sin[2 * cd + ch];
    const float b = bf(cb[ch]);
    for (uint r = 0; r < n; r++) {
        const ushort cur = proj[(ulong)r * pd + xoff + ch];
        float acc = b;
        acc = acc + bf(w.s0) * bf(t0);
        acc = acc + bf(w.s1) * bf(t1);
        acc = acc + bf(w.s2) * bf(t2);
        acc = acc + bf(w.s3) * bf(cur);
        const float cv = bfr(acc);
        out[(ulong)r * cd + ch] = to_bf(cv * (1.0f / (1.0f + exp(-cv))));
        t0 = t1;
        t1 = t2;
        t2 = cur;
    }
    sout[ch] = t0;
    sout[cd + ch] = t1;
    sout[2 * cd + ch] = t2;
}

// ---- SSM over n rows (ssm_step per row); a lane's 8 states stay in registers; state [heads][dh][128] fp32 ----
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void ssm_rows(__global const ushort *proj, uint pd, uint dt_off, __global const ushort *xc, uint cd, __global const float *sin, __global float *sout,
                       __global const float *a_log, __global const float *dsk, __global const float *dtb, __global ushort *y, uint dh, uint xd, uint groups,
                       uint per_group, float lo, float hi, uint n) {
    const uint h = get_group_id(0);
    const uint row = get_group_id(1) * 4 + get_sub_group_id();
    const uint lane = get_sub_group_local_id();
    const uint g = h / per_group;
    const ulong sbase = ((ulong)h * dh + row) * DS;
    float st[DS / 16];
    for (uint j = 0; j < DS / 16; j++) st[j] = sin[sbase + lane + 16 * j];
    for (uint r = 0; r < n; r++) {
        const float v = bf(proj[(ulong)r * pd + dt_off + h]) + dtb[h];
        const float dt = fmin(fmax(fmax(v, 0.0f) + log(1.0f + exp(-fabs(v))), lo), hi);
        const float da = exp(-exp(a_log[h]) * dt);
        const float x = bf(xc[(ulong)r * cd + h * dh + row]);
        const float xdt = x * dt;
        __global const ushort *bp = xc + (ulong)r * cd + xd + g * DS;
        __global const ushort *cp = xc + (ulong)r * cd + xd + groups * DS + g * DS;
        float m = 0.0f;
        for (uint j = 0; j < DS / 16; j++) {
            const uint s = lane + 16 * j;
            const float ns = fma(xdt, bf(bp[s]), st[j] * da);
            st[j] = ns;
            m = fma(ns, bf(cp[s]), m);
        }
        m = sub_group_reduce_add(m);
        if (lane == 0) {
            const float z = bf(proj[(ulong)r * pd + h * dh + row]);
            const float gz = bfr(z / (1.0f + exp(-z)));
            y[(ulong)r * xd + h * dh + row] = to_bf(gz * bfr(fma(x, dsk[h], m)));
        }
    }
    for (uint j = 0; j < DS / 16; j++) sout[sbase + lane + 16 * j] = st[j];
}

// Group RMSNorm of rows [n][xd] (group_rmsnorm of mamba.cl): grid (groups, n), 64 items.
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void group_rmsnorm_r(__global const ushort *x, __global const ushort *w, __global ushort *y, uint n, uint xd, float eps) {
    __local float part[4];
    const uint gbase = get_group_id(0) * n;
    const ulong base = (ulong)get_group_id(1) * xd + gbase;
    const uint lid = get_local_id(0);
    float ss = 0.0f;
    for (uint i = lid; i < n; i += 64) {
        const float v = bf(x[base + i]);
        ss += v * v;
    }
    ss = sub_group_reduce_add(ss);
    if (get_sub_group_local_id() == 0) part[get_sub_group_id()] = ss;
    barrier(CLK_LOCAL_MEM_FENCE);
    const float inv = 1.0f / sqrt((part[0] + part[1] + part[2] + part[3]) / (float)n + eps);
    for (uint i = lid; i < n; i += 64) y[base + i] = to_bf(bf(w[gbase + i]) * bfr(bf(x[base + i]) * inv));
}

// ---- MoE: the experts the slots chose, ascending, elist[0 .. nlist) (one work-item) ----
__kernel void moe_groups(__global const uint *ids, __global uint *elist, uint slots, uint n_experts) {
    if (get_global_id(0) != 0) return;
    uchar seen[256];
    for (uint e = 0; e < n_experts; e++) seen[e] = 0;
    for (uint s = 0; s < slots; s++) seen[ids[s]] = 1;
    uint n = 0;
    for (uint e = 0; e < n_experts; e++)
        if (seen[e]) elist[1 + n++] = e;
    elist[0] = n;
}

__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void expert_up_relu2_g(__global const uint *w, __global const ushort *scales, __global const ushort *biases, __global const ushort *x, __global const uint *ids,
                                __global ushort *out, uint in_dim, uint n_rows, uint slots, uint top_k, __global const uint *elist) {
    const uint r = get_group_id(0), j = get_group_id(1);
    if (j >= elist[0]) return;
    const uint e = elist[1 + j];
    ulong xo[MAXR];
    uint sl[MAXR];
    uint m = 0;
    for (uint s = 0; s < slots && m < MAXR; s++)
        if (ids[s] == e) { sl[m] = s; xo[m] = (ulong)(s / top_k) * in_dim; m++; } // an expert serves a window row once: m <= rows <= MAXR
    float acc[MAXR];
    qrows(w, scales, biases, x, xo, m, (ulong)e * n_rows + r, in_dim, acc);
    if (get_sub_group_local_id() == 0)
        for (uint k = 0; k < m; k++) {
            const float u = fmax(bf(to_bf(acc[k])), 0.0f);
            out[(ulong)sl[k] * n_rows + r] = to_bf(u * u);
        }
}

// down: x row of slot s is row s of act [slots][in_dim]; out[s][n_rows] fp32.
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void expert_down_f32_g(__global const uint *w, __global const ushort *scales, __global const ushort *biases, __global const ushort *x, __global const uint *ids,
                                __global float *out, uint in_dim, uint n_rows, uint slots, __global const uint *elist) {
    const uint r = get_group_id(0), j = get_group_id(1);
    if (j >= elist[0]) return;
    const uint e = elist[1 + j];
    ulong xo[MAXR];
    uint sl[MAXR];
    uint m = 0;
    for (uint s = 0; s < slots && m < MAXR; s++)
        if (ids[s] == e) { sl[m] = s; xo[m] = (ulong)s * in_dim; m++; }
    float acc[MAXR];
    qrows(w, scales, biases, x, xo, m, (ulong)e * n_rows + r, in_dim, acc);
    if (get_sub_group_local_id() == 0)
        for (uint k = 0; k < m; k++) out[(ulong)sl[k] * n_rows + r] = acc[k];
}

// Shared expert (dense): grid (n_rows); up with relu2 over the n window rows, down to fp32.
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void dense_up_relu2_r(__global const uint *w, __global const ushort *scales, __global const ushort *biases, __global const ushort *x, __global ushort *out,
                               uint in_dim, uint n_rows, uint n) {
    const uint r = get_group_id(0);
    ulong xo[MAXR];
    float acc[MAXR];
    strided(xo, n, in_dim);
    qrows(w, scales, biases, x, xo, n, r, in_dim, acc);
    if (get_sub_group_local_id() == 0)
        for (uint m = 0; m < n; m++) {
            const float u = fmax(bf(to_bf(acc[m])), 0.0f);
            out[(ulong)m * n_rows + r] = to_bf(u * u);
        }
}

__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void dense_down_f32_r(__global const uint *w, __global const ushort *scales, __global const ushort *biases, __global const ushort *x, __global float *out,
                               uint in_dim, uint n_rows, uint n) {
    const uint r = get_group_id(0);
    ulong xo[MAXR];
    float acc[MAXR];
    strided(xo, n, in_dim);
    qrows(w, scales, biases, x, xo, n, r, in_dim, acc);
    if (get_sub_group_local_id() == 0)
        for (uint m = 0; m < n; m++) out[(ulong)m * n_rows + r] = acc[m];
}

// Block output of row m: bf16(sum_k fma(y[m][k], wts[m][k]) + shared[m]), sum in slot order. Grid (ceil(dim / 64), m).
__kernel void moe_combine_r(__global const float *y, __global const float *wts, __global const float *shared, __global ushort *out, uint dim, uint slots) {
    const uint i = get_global_id(0), m = get_group_id(1);
    if (i >= dim) return;
    float acc = 0.0f;
    for (uint k = 0; k < slots; k++) acc = fma(y[((ulong)m * slots + k) * dim + i], wts[m * slots + k], acc);
    out[(ulong)m * dim + i] = to_bf(acc + shared[(ulong)m * dim + i]);
}

// ---- attention over rows (attn_partial / attn_merge with row z): len0 + z keys, q / out rows at z * q_dim ----
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void attn_partial_r(__global const ushort *q, __global const ushort *kc, __global const ushort *vc, __global float *po, __global float *pm, __global float *pl,
                             uint len0, uint chunk, uint kvh, float scale, uint maxc) {
    __local float4 qs[GQ * HD / 4];
    __local float sc[GQ * TILE];
    __local float ps[GQ * TILE];
    __local float alpha_s[GQ];
    const uint hk = get_group_id(0);
    const uint c = get_group_id(1);
    const uint z = get_group_id(2);
    const uint len = len0 + z;
    const uint lid = get_local_id(0);
    const uint sg = get_sub_group_id();
    const uint lane = get_sub_group_local_id();
    const uint li = lid & 63;
    const uint hg = lid >> 6;
    const uint k0 = c * chunk;
    if (k0 >= len) return;
    const uint k1 = min(len, k0 + chunk);
    const uint heads = kvh * GQ;
    const ulong q_off = (ulong)z * heads * HD;
    const ulong cb = (ulong)z * maxc + c;
    for (uint i = lid; i < GQ * HD; i += 256) ((__local float *)qs)[i] = bf(q[q_off + hk * GQ * HD + i]);
    barrier(CLK_LOCAL_MEM_FENCE);

    float m = -INFINITY, den = 0.0f;
    float2 o[4];
#pragma unroll
    for (int h = 0; h < 4; h++) o[h] = (float2)(0.0f);

    for (uint t0 = k0; t0 < k1; t0 += TILE) {
        const uint key = t0 + li;
        float acc[4];
#pragma unroll
        for (int h = 0; h < 4; h++) acc[h] = 0.0f;
        if (key < k1) {
            for (uint dc = 0; dc < HD / 8; dc++) {
                const float8 kv = as_float8(convert_uint8(vload8(((ulong)key * kvh + hk) * (HD / 8) + dc, kc)) << 16);
#pragma unroll
                for (int h = 0; h < 4; h++) {
                    const float4 a = qs[(hg * 4 + h) * (HD / 4) + dc * 2];
                    const float4 b = qs[(hg * 4 + h) * (HD / 4) + dc * 2 + 1];
                    acc[h] += a.x * kv.s0 + a.y * kv.s1 + a.z * kv.s2 + a.w * kv.s3 + b.x * kv.s4 + b.y * kv.s5 + b.z * kv.s6 + b.w * kv.s7;
                }
            }
        }
#pragma unroll
        for (int h = 0; h < 4; h++) sc[(hg * 4 + h) * TILE + li] = (key < k1) ? acc[h] * scale : -INFINITY;
        barrier(CLK_LOCAL_MEM_FENCE);
        float s[4];
        float tm = -INFINITY;
#pragma unroll
        for (int j = 0; j < 4; j++) { s[j] = sc[sg * TILE + lane + 16 * j]; tm = fmax(tm, s[j]); }
        tm = sub_group_reduce_max(tm);
        const float nm = fmax(m, tm);
        const float a = (m == -INFINITY) ? 0.0f : exp(m - nm);
        float sum = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const float p = exp(s[j] - nm);
            sum += p;
            ps[sg * TILE + lane + 16 * j] = bf_round(p);
        }
        sum = sub_group_reduce_add(sum);
        den = den * a + sum;
        m = nm;
        if (lane == 0) alpha_s[sg] = a;
        barrier(CLK_LOCAL_MEM_FENCE);
        float2 pv[4];
#pragma unroll
        for (int h = 0; h < 4; h++) pv[h] = (float2)(0.0f);
        const uint nk = min((uint)TILE, k1 - t0);
        for (uint k = 0; k < nk; k++) {
            const uint vv = ((__global const uint *)vc)[((ulong)(t0 + k) * kvh + hk) * (HD / 2) + li];
            const float v0 = as_float(vv << 16);
            const float v1 = as_float(vv & 0xffff0000u);
#pragma unroll
            for (int h = 0; h < 4; h++) {
                const float p = ps[(hg * 4 + h) * TILE + k];
                pv[h].x += p * v0;
                pv[h].y += p * v1;
            }
        }
#pragma unroll
        for (int h = 0; h < 4; h++) o[h] = o[h] * alpha_s[hg * 4 + h] + pv[h];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
#pragma unroll
    for (int h = 0; h < 4; h++) vstore2(o[h], li, po + (cb * heads + hk * GQ + hg * 4 + h) * HD);
    if (lane == 0) {
        pm[cb * heads + hk * GQ + sg] = m;
        pl[cb * heads + hk * GQ + sg] = den;
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void attn_merge_r(__global const float *po, __global const float *pm, __global const float *pl, __global ushort *out, uint len0, uint chunk, uint heads, uint maxc) {
    const uint head = get_group_id(0), z = get_group_id(1);
    const uint lane = get_sub_group_local_id();
    const uint len = len0 + z;
    const uint nch = (len + chunk - 1) / chunk;
    const ulong cb = (ulong)z * maxc;
    float m = -INFINITY, den = 0.0f;
    float8 o = (float8)(0.0f);
    for (uint c = 0; c < nch; c++) {
        const float cm = pm[(cb + c) * heads + head];
        const float cl = pl[(cb + c) * heads + head];
        const float8 co = vload8(lane, po + ((cb + c) * heads + head) * HD);
        const bool active = cl > 0.0f;
        const float nm = active ? fmax(m, cm) : m;
        const float a = active ? ((m == -INFINITY) ? 0.0f : exp(m - nm)) : 1.0f;
        const float b = active ? exp(cm - nm) : 0.0f;
        o = o * a + co * b;
        den = den * a + cl * b;
        m = nm;
    }
    const float8 r = o / den;
    __global ushort *dst = out + (ulong)z * heads * HD + head * HD + lane * 8;
    dst[0] = to_bf(r.s0); dst[1] = to_bf(r.s1); dst[2] = to_bf(r.s2); dst[3] = to_bf(r.s3);
    dst[4] = to_bf(r.s4); dst[5] = to_bf(r.s5); dst[6] = to_bf(r.s6); dst[7] = to_bf(r.s7);
}

// cat[r] = [a[r] | b[r]] for rows of n bf16 values: grid (ceil(n / 64), rows).
__kernel void concat_rows(__global const ushort *a, __global const ushort *b, __global ushort *cat, uint n) {
    const uint i = get_global_id(0), r = get_group_id(1);
    if (i >= n) return;
    cat[(ulong)r * 2 * n + i] = a[(ulong)r * n + i];
    cat[(ulong)r * 2 * n + n + i] = b[(ulong)r * n + i];
}
