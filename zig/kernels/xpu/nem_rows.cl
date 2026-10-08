// Nemotron-H multi-row (window) kernels: the single-row kernels of mamba.cl / moe.cl / attn.cl with a row index, so row r of a window computes exactly the bits the
// one-token path computes (same operation order per output value). Recurrent state (conv, SSM) is carried through the rows of the window inside the kernel.
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

// A weight row held in registers for all the window's rows: a lane takes 16-byte loads (32 inputs, half a group) in a strided order (in_dim a multiple of 32, at most MAXQ * 512).
#define MAXQ 11
typedef struct { uint4 u[MAXQ]; float sc[MAXQ]; float bi[MAXQ]; } WRow;

inline void wload(WRow *wr, __global const uint *w, __global const ushort *scales, __global const ushort *biases, ulong row, uint in_dim) {
    const uint lane = get_sub_group_local_id();
    const uint nq = in_dim / 32, groups = in_dim / 64;
    const __global uint4 *wp = (const __global uint4 *)(w + row * (in_dim / 8));
#pragma unroll
    for (int i = 0; i < MAXQ; i++) {
        const uint q = lane + 16 * i;
        if (q < nq) {
            wr->u[i] = wp[q];
            wr->sc[i] = bf(scales[row * groups + (q >> 1)]);
            wr->bi[i] = bf(biases[row * groups + (q >> 1)]);
        }
    }
}

// fp32 sc * dot + bi * sum(x) per half group, lane sums added by a sub-group reduction (valid on every lane).
inline float wdot(const WRow *wr, __global const ushort *x, uint in_dim) {
    const uint lane = get_sub_group_local_id();
    const uint nq = in_dim / 32;
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < MAXQ; i++) {
        const uint q = lane + 16 * i;
        if (q < nq) {
            const uint4 u = wr->u[i];
            const float16 xa = as_float16(convert_uint16(vload16(2 * q, x)) << 16);
            const float16 xb = as_float16(convert_uint16(vload16(2 * q + 1, x)) << 16);
            float dot = 0.0f, sx = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                dot = fma(xa[j], (float)((u.x >> (4 * j)) & 15u), dot);
                sx += xa[j];
            }
#pragma unroll
            for (int j = 0; j < 8; j++) {
                dot = fma(xa[8 + j], (float)((u.y >> (4 * j)) & 15u), dot);
                sx += xa[8 + j];
            }
#pragma unroll
            for (int j = 0; j < 8; j++) {
                dot = fma(xb[j], (float)((u.z >> (4 * j)) & 15u), dot);
                sx += xb[j];
            }
#pragma unroll
            for (int j = 0; j < 8; j++) {
                dot = fma(xb[8 + j], (float)((u.w >> (4 * j)) & 15u), dot);
                sx += xb[8 + j];
            }
            acc += wr->sc[i] * dot + wr->bi[i] * sx;
        }
    }
    return sub_group_reduce_add(acc);
}

// ---- dense 4-bit matvec (qmv4_bf16 of mamba.cl): grid (rows, m); row r of x [m][in_dim] -> y[r * y_stride + y_off + row] (bf16) -------------------------------------
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void qmv4_bf16_r(__global const uint *w, __global const ushort *scales, __global const ushort *biases, __global const ushort *x, __global ushort *y,
                          uint in_dim, uint rows, uint y_stride, uint y_off, uint n) {
    const uint row = get_group_id(0);
    WRow wr;
    wload(&wr, w, scales, biases, row, in_dim);
    for (uint r = 0; r < n; r++) {
        const float acc = wdot(&wr, x + (ulong)r * in_dim, in_dim);
        if (get_sub_group_local_id() == 0) y[(ulong)r * y_stride + y_off + row] = to_bf(acc);
    }
}

// ---- attention-side 4-bit matvec (qmv4_bf / qmv4_f32 of attn.cl): grid (rows / 4, m); bf16 or fp32 output -----------------------------------------------------------
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void qmv4_bf_r(__global const uint *w, __global const ushort *scales, __global const ushort *biases, __global const ushort *x, __global ushort *y,
                        uint in_dim, uint y_off, uint y_stride, uint rows, uint n) {
    const uint row = get_group_id(0) * 4 + get_sub_group_id();
    if (row >= rows) return;
    WRow wr;
    wload(&wr, w, scales, biases, row, in_dim);
    for (uint r = 0; r < n; r++) {
        const float acc = wdot(&wr, x + (ulong)r * in_dim, in_dim);
        if (get_sub_group_local_id() == 0) y[(ulong)r * y_stride + y_off + row] = to_bf(acc);
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void qmv4_f32_r(__global const uint *w, __global const ushort *scales, __global const ushort *biases, __global const ushort *x, __global float *y,
                         uint in_dim, uint rows, uint n) {
    const uint row = get_group_id(0) * 4 + get_sub_group_id();
    if (row >= rows) return;
    WRow wr;
    wload(&wr, w, scales, biases, row, in_dim);
    for (uint r = 0; r < n; r++) {
        const float acc = wdot(&wr, x + (ulong)r * in_dim, in_dim);
        if (get_sub_group_local_id() == 0) y[(ulong)r * rows + row] = acc;
    }
}

// ---- Mamba2 conv1d over n rows (conv1d_step of mamba.cl per row), state [3][cd] bf16 read from sin and written to sout (may be the same buffer) ----------------------------
// proj rows are [n][pd]; the x part starts at xoff. out [n][cd].
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

// ---- SSM over n rows (ssm_step of mamba.cl per row); the 8 states a lane owns stay in registers across the rows. state [heads][dh][128] fp32 from sin to sout -------------
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

// ---- MoE -------------------------------------------------------------------------------------------------------------------------------------------------------------
// The experts the slots chose, ascending: elist[0 .. nlist) (one work-item; the grouped kernels launch min(experts, slots) groups on it).
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
    WRow wr;
    wload(&wr, w, scales, biases, (ulong)e * n_rows + r, in_dim);
    for (uint s = 0; s < slots; s++) {
        if (ids[s] != e) continue;
        const float acc = wdot(&wr, x + (ulong)(s / top_k) * in_dim, in_dim);
        const float u = fmax(bf(to_bf(acc)), 0.0f);
        if (get_sub_group_local_id() == 0) out[(ulong)s * n_rows + r] = to_bf(u * u);
    }
}

// down: x row of slot s is row s of act [slots][in_dim]; out[s][n_rows] fp32.
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void expert_down_f32_g(__global const uint *w, __global const ushort *scales, __global const ushort *biases, __global const ushort *x, __global const uint *ids,
                                __global float *out, uint in_dim, uint n_rows, uint slots, __global const uint *elist) {
    const uint r = get_group_id(0), j = get_group_id(1);
    if (j >= elist[0]) return;
    const uint e = elist[1 + j];
    WRow wr;
    wload(&wr, w, scales, biases, (ulong)e * n_rows + r, in_dim);
    for (uint s = 0; s < slots; s++) {
        if (ids[s] != e) continue;
        const float acc = wdot(&wr, x + (ulong)s * in_dim, in_dim);
        if (get_sub_group_local_id() == 0) out[(ulong)s * n_rows + r] = acc;
    }
}

// Shared expert (dense): grid (n_rows); up with relu2 over the n window rows, down to fp32.
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void dense_up_relu2_r(__global const uint *w, __global const ushort *scales, __global const ushort *biases, __global const ushort *x, __global ushort *out,
                               uint in_dim, uint n_rows, uint n) {
    const uint r = get_group_id(0);
    WRow wr;
    wload(&wr, w, scales, biases, r, in_dim);
    for (uint m = 0; m < n; m++) {
        const float acc = wdot(&wr, x + (ulong)m * in_dim, in_dim);
        const float u = fmax(bf(to_bf(acc)), 0.0f);
        if (get_sub_group_local_id() == 0) out[(ulong)m * n_rows + r] = to_bf(u * u);
    }
}

__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void dense_down_f32_r(__global const uint *w, __global const ushort *scales, __global const ushort *biases, __global const ushort *x, __global float *out,
                               uint in_dim, uint n_rows, uint n) {
    const uint r = get_group_id(0);
    WRow wr;
    wload(&wr, w, scales, biases, r, in_dim);
    for (uint m = 0; m < n; m++) {
        const float acc = wdot(&wr, x + (ulong)m * in_dim, in_dim);
        if (get_sub_group_local_id() == 0) out[(ulong)m * n_rows + r] = acc;
    }
}

// Block output of row m: bf16(sum_k fma(y[m][k], wts[m][k]) + shared[m]), sum in slot order. Grid (ceil(dim / 64), m).
__kernel void moe_combine_r(__global const float *y, __global const float *wts, __global const float *shared, __global ushort *out, uint dim, uint slots) {
    const uint i = get_global_id(0), m = get_group_id(1);
    if (i >= dim) return;
    float acc = 0.0f;
    for (uint k = 0; k < slots; k++) acc = fma(y[((ulong)m * slots + k) * dim + i], wts[m * slots + k], acc);
    out[(ulong)m * dim + i] = to_bf(acc + shared[(ulong)m * dim + i]);
}

// ---- attention over rows: attn_partial / attn_merge of attn.cl with a row z (grid z): the row has len0 + z keys, q / out rows at z * q_dim, partial scratch [z][maxc][..] ---
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
