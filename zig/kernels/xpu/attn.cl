// Nemotron-H attention decode ops (no RoPE in this architecture): 4-bit projections, split-K GQA attention, argmax.
// Activations are bf16, math is fp32; the attention follows the upstream chunked online softmax (p rounded to bf16 for P*V).
#define HD 128   // head dim
#define GQ 16    // query heads per KV head (32 / 2)
#define TILE 64  // keys per online-softmax step

inline float bf(ushort v) { return as_float((uint)v << 16); }
inline ushort to_bf(float f) {
    uint u = as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (ushort)(u >> 16);
}
inline float bf_round(float f) { return as_float((uint)to_bf(f) << 16); }

// y[row] = sum_i x[i] * (q[row][i] * scale + bias), groups of 64; one 16-lane sub-group per row, 4 rows a work-group.
inline float qmv_row(__global const uint *w, __global const ushort *scales, __global const ushort *biases,
                     __global const ushort *x, uint row, uint in_dim) {
    const uint lane = get_sub_group_local_id();
    const uint words = in_dim / 8;
    const uint groups = in_dim / 64;
    float acc = 0.0f;
    for (uint wi = lane; wi < words; wi += 16) {
        const uint g = wi / 8;
        const float sc = bf(scales[row * groups + g]);
        const float bi = bf(biases[row * groups + g]);
        const uint pk = w[row * words + wi];
        const float8 xv = as_float8(convert_uint8(vload8(wi, x)) << 16);
        float dot = 0.0f, sx = 0.0f;
        dot += xv.s0 * (float)(pk & 15u);          sx += xv.s0;
        dot += xv.s1 * (float)((pk >> 4) & 15u);   sx += xv.s1;
        dot += xv.s2 * (float)((pk >> 8) & 15u);   sx += xv.s2;
        dot += xv.s3 * (float)((pk >> 12) & 15u);  sx += xv.s3;
        dot += xv.s4 * (float)((pk >> 16) & 15u);  sx += xv.s4;
        dot += xv.s5 * (float)((pk >> 20) & 15u);  sx += xv.s5;
        dot += xv.s6 * (float)((pk >> 24) & 15u);  sx += xv.s6;
        dot += xv.s7 * (float)((pk >> 28) & 15u);  sx += xv.s7;
        acc += sc * dot + bi * sx;
    }
    return sub_group_reduce_add(acc);
}

// bf16 output at y[y_off + row] (q/k/v land straight in their buffers or KV-cache rows); x read from x[x_off ..].
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void qmv4_bf(__global const uint *w, __global const ushort *scales, __global const ushort *biases,
                      __global const ushort *x, __global ushort *y, uint in_dim, uint x_off, uint y_off, uint rows) {
    const uint row = get_group_id(0) * 4 + get_sub_group_id();
    if (row >= rows) return;
    const float acc = qmv_row(w, scales, biases, x + x_off, row, in_dim);
    if (get_sub_group_local_id() == 0) y[y_off + row] = to_bf(acc);
}

// fp32 output (lm_head logits).
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void qmv4_f32(__global const uint *w, __global const ushort *scales, __global const ushort *biases,
                       __global const ushort *x, __global float *y, uint in_dim, uint x_off, uint y_off, uint rows) {
    const uint row = get_group_id(0) * 4 + get_sub_group_id();
    if (row >= rows) return;
    const float acc = qmv_row(w, scales, biases, x + x_off, row, in_dim);
    if (get_sub_group_local_id() == 0) y[y_off + row] = acc;
}

// Pass 1 of split-K decode attention. Work-group (kv head, chunk) of 256 items: the head's 16 query heads against keys
// [chunk*c, chunk*c+chunk) of a cache laid out [pos][kv head][128] bf16. Writes unnormalised o, running max m and
// denominator l per query head to po[c][head][128], pm[c][head], pl[c][head].
// Score and P*V phases split the 16 heads in four groups of four (item / 64); softmax gives each head a sub-group.
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void attn_partial(__global const ushort *q, __global const ushort *kc, __global const ushort *vc,
                           __global float *po, __global float *pm, __global float *pl,
                           uint len, uint chunk, uint kvh, uint q_off, float scale) {
    __local float4 qs[GQ * HD / 4];
    __local float sc[GQ * TILE];
    __local float ps[GQ * TILE];
    __local float alpha_s[GQ];
    const uint hk = get_group_id(0);
    const uint c = get_group_id(1);
    const uint lid = get_local_id(0);
    const uint sg = get_sub_group_id();  // the query head this sub-group runs the softmax for
    const uint lane = get_sub_group_local_id();
    const uint li = lid & 63;            // key (scores) or dim pair (P*V)
    const uint hg = lid >> 6;            // heads 4*hg .. 4*hg+3
    const uint k0 = c * chunk;
    if (k0 >= len) return;
    const uint k1 = min(len, k0 + chunk);
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
                const float8 kv = as_float8(convert_uint8(vload8((key * kvh + hk) * (HD / 8) + dc, kc)) << 16);
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
        // online softmax for head sg: four keys a lane
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
        // P*V: item owns dims 2*li, 2*li+1 for its four heads
        float2 pv[4];
#pragma unroll
        for (int h = 0; h < 4; h++) pv[h] = (float2)(0.0f);
        const uint nk = min((uint)TILE, k1 - t0);
        for (uint k = 0; k < nk; k++) {
            const uint vv = ((__global const uint *)vc)[((t0 + k) * kvh + hk) * (HD / 2) + li];
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
    const uint heads = kvh * GQ;
#pragma unroll
    for (int h = 0; h < 4; h++) vstore2(o[h], li, po + (c * heads + hk * GQ + hg * 4 + h) * HD);
    if (lane == 0) {
        pm[c * heads + hk * GQ + sg] = m;
        pl[c * heads + hk * GQ + sg] = den;
    }
}

// Pass 2: merges the chunks of one query head in order and writes bf16 o/l to out[out_off + head*128 ..].
// One sub-group per head, 8 dims a lane.
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void attn_merge(__global const float *po, __global const float *pm, __global const float *pl,
                         __global ushort *out, uint len, uint chunk, uint heads, uint out_off) {
    const uint head = get_group_id(0);
    const uint lane = get_sub_group_local_id();
    const uint nch = (len + chunk - 1) / chunk;
    float m = -INFINITY, den = 0.0f;
    float8 o = (float8)(0.0f);
    for (uint c = 0; c < nch; c++) {
        const float cm = pm[c * heads + head];
        const float cl = pl[c * heads + head];
        const float8 co = vload8(lane, po + (c * heads + head) * HD);
        const bool active = cl > 0.0f;
        const float nm = active ? fmax(m, cm) : m;
        const float a = active ? ((m == -INFINITY) ? 0.0f : exp(m - nm)) : 1.0f;
        const float b = active ? exp(cm - nm) : 0.0f;
        o = o * a + co * b;
        den = den * a + cl * b;
        m = nm;
    }
    const float8 r = o / den;
    __global ushort *dst = out + out_off + head * HD + lane * 8;
    dst[0] = to_bf(r.s0); dst[1] = to_bf(r.s1); dst[2] = to_bf(r.s2); dst[3] = to_bf(r.s3);
    dst[4] = to_bf(r.s4); dst[5] = to_bf(r.s5); dst[6] = to_bf(r.s6); dst[7] = to_bf(r.s7);
}

// Greedy argmax over fp32 logits, rows x n. Order matches the upstream CUDA op: the first NaN wins, otherwise the
// largest value with +0 == -0, and ties go to the lowest index. A candidate packs as (key << 32) | ~index so a plain
// unsigned max picks the winner; 0 is the empty candidate.
inline ulong argmax_pack(float f, uint idx) {
    uint w = as_uint(f);
    uint key;
    if ((w & 0x7fffffffu) > 0x7f800000u) key = 0xffffffffu;
    else {
        if ((w & 0x7fffffffu) == 0u) w = 0u;
        key = (w & 0x80000000u) ? ~w : (w ^ 0x80000000u);
    }
    return ((ulong)key << 32) | (ulong)(~idx);
}

// Work-group max of a 256-item group: sub-group reduce, then the 16 sub-group results through local memory.
inline ulong wg_max(ulong v, __local ulong *tmp) {
    v = sub_group_reduce_max(v);
    if (get_sub_group_local_id() == 0) tmp[get_sub_group_id()] = v;
    barrier(CLK_LOCAL_MEM_FENCE);
    ulong r = 0;
    for (int i = 0; i < 16; i++) r = max(r, tmp[i]);
    return r;
}

// Pass 1: work-group (part, row) of 256 items scans its slice of the row.
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void argmax_partial(__global const float *x, __global ulong *part, uint n, uint per_part) {
    __local ulong tmp[16];
    const uint p = get_group_id(0);
    const uint row = get_group_id(1);
    const uint lo = p * per_part;
    const uint hi = min(n, lo + per_part);
    ulong best = 0;
    for (uint i = lo + get_local_id(0); i < hi; i += 256) best = max(best, argmax_pack(x[(ulong)row * n + i], i));
    best = wg_max(best, tmp);
    if (get_local_id(0) == 0) part[row * get_num_groups(0) + p] = best;
}

// Pass 2: one work-group per row merges the parts and writes the winning index.
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void argmax_final(__global const ulong *part, __global int *out, uint parts) {
    __local ulong tmp[16];
    const uint row = get_group_id(0);
    ulong best = 0;
    for (uint i = get_local_id(0); i < parts; i += 256) best = max(best, part[row * parts + i]);
    best = wg_max(best, tmp);
    if (get_local_id(0) == 0) out[row] = (int)(~(uint)(best & 0xffffffffu));
}

// q, k and v projections in one launch (flat grid of 4-row work-groups: q, then k, then v); k and v land at kc[k_off + row], vc[k_off + row] like qmv4_bf.
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void qkv4_bf(__global const uint *qw, __global const ushort *qs, __global const ushort *qb,
                      __global const uint *kw, __global const ushort *ks, __global const ushort *kb,
                      __global const uint *vw, __global const ushort *vs, __global const ushort *vb,
                      __global const ushort *x, __global ushort *qo, __global ushort *kc, __global ushort *vc,
                      uint in_dim, uint k_off, uint q_rows, uint kv_rows) {
    uint g = get_group_id(0);
    const uint qg = q_rows / 4, kg = kv_rows / 4;
    __global const uint *w;
    __global const ushort *s, *b;
    __global ushort *y;
    if (g < qg) { w = qw; s = qs; b = qb; y = qo; }
    else if (g < qg + kg) { g -= qg; w = kw; s = ks; b = kb; y = kc + k_off; }
    else { g -= qg + kg; w = vw; s = vs; b = vb; y = vc + k_off; }
    const uint row = g * 4 + get_sub_group_id();
    const float acc = qmv_row(w, s, b, x, row, in_dim);
    if (get_sub_group_local_id() == 0) y[row] = to_bf(acc);
}
