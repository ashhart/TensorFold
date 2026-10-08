#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
// Nemotron-H attention for a decode row (and for the rows of a window of up to 16): split-K partials of 512 keys on the matrix engine, merged by attn_merge / attn_merge_r as before.
// The 16 query heads of a kv head are the 16 rows of two matrix-engine tiles (8 heads each, one tile per work-group), the keys the columns: a sub-group takes 64 keys (4 tiles of 16, lane = key),
// scores S = Q K^T (8 DPAS over the 128 dims), fp32 softmax statistics per head over its 64 keys, P (rounded to bf16) times V (V read with sub-group block loads); the 8 sub-groups of a
// work-group (512 keys) merge their (m, l, O) in a fixed order through local memory. A row's bits depend on its key count only (absolute chunks and key tiles), so a window row and the same
// token decoded alone are bit-identical, as with attn_partial_r / attn_partial. Output layout of attn_partial_r: po [(z * maxc + chunk) * 32 + head][128] fp32, pm / pl [..][head].
#define HD 128
#define NH 32
#define NKV 2
#define GQ 16
inline float bf(ushort v) { return as_float((uint)v << 16); }
inline ushort to_bf(float f) {
    uint u = as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (ushort)(u >> 16);
}
inline float bfr(float f) { return bf(to_bf(f)); }

// Grid (NKV * 2 head halves, chunks, rows z), 128 items = 8 sub-groups. q [row][32][128] bf16, kc / vc [pos][2][128] bf16; row z has len0 + z keys.
__attribute__((reqd_work_group_size(128, 1, 1))) __attribute__((intel_reqd_sub_group_size(16)))
__kernel void nem_attn_dec_partial(__global const ushort *q, __global const ushort *kc, __global const ushort *vc, __global float *po, __global float *pm, __global float *pl,
                                   uint len0, uint maxc, float scale) {
    __local short qa[8 * 16 * 8]; // [dim block][lane][head of the tile]
    __local float lm[8][8], ls[8][8];
    __local float la[8][8 * HD]; // [sub-group][head * 128 + dim]
    const uint hk = get_group_id(0) >> 1;
    const uint head0 = hk * GQ + (get_group_id(0) & 1) * 8;
    const uint c = get_group_id(1);
    const uint z = get_group_id(2);
    const uint len = len0 + z;
    const uint k0 = c * 512;
    if (k0 >= len) return;
    const uint lid = get_local_id(0);
    const uint sg = get_sub_group_id();
    const uint lane = get_sub_group_local_id();
    for (uint i = lid; i < 8 * 16 * 8; i += 128) {
        const uint t = i >> 7, l = (i >> 3) & 15, r = i & 7;
        qa[i] = (short)q[((ulong)z * NH + head0 + r) * HD + t * 16 + l];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const uint ks = k0 + sg * 64;
    const uint ke = min(ks + 64, len);
    float m[8], ll[8];
    float8 acc[8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        m[i] = -INFINITY;
        ll[i] = 0.0f;
    }
#pragma unroll
    for (int j = 0; j < 8; j++) acc[j] = (float8)(0.0f);
    if (ks < len) { // uniform in the sub-group
        float8 cc[4];
#pragma unroll
        for (int u = 0; u < 4; u++) {
            const uint key = ks + u * 16 + lane;
            const uint kl = min(key, len - 1);
            const __global uint *krow = (const __global uint *)(kc + ((ulong)kl * NKV + hk) * HD);
            float8 s = (float8)(0.0f);
#pragma unroll
            for (int t = 0; t < 8; t++) {
                const int8 b = as_int8(vload8(0, krow + t * 8));
                const short8 a = vload8(t * 16 + lane, qa);
                s = intel_sub_group_bf16_bf16_matrix_mad_k16(a, b, s);
            }
#pragma unroll
            for (int i = 0; i < 8; i++) s[i] = (key < ke) ? s[i] * scale : -INFINITY;
            cc[u] = s;
        }
        short8 pown[4];
#pragma unroll
        for (int i = 0; i < 8; i++) m[i] = sub_group_reduce_max(fmax(fmax(cc[0][i], cc[1][i]), fmax(cc[2][i], cc[3][i])));
#pragma unroll
        for (int u = 0; u < 4; u++) {
            short8 own;
#pragma unroll
            for (int i = 0; i < 8; i++) {
                const float p = (cc[u][i] == -INFINITY) ? 0.0f : bfr(exp(cc[u][i] - m[i]));
                ll[i] += p;
                own[i] = (short)to_bf(p);
            }
            pown[u] = own;
        }
#pragma unroll
        for (int u = 0; u < 4; u++) {
            if (ks + u * 16 >= len) break;
#pragma unroll
            for (int hb = 0; hb < 2; hb++) {
                ushort4 vv[16];
#pragma unroll
                for (int k = 0; k < 16; k++) {
                    const uint vk = min(ks + u * 16 + k, len - 1);
                    vv[k] = intel_sub_group_block_read_us4((const __global ushort *)(vc + ((ulong)vk * NKV + hk) * HD + hb * 64));
                }
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    int8 b;
#pragma unroll
                    for (int d = 0; d < 8; d++) b[d] = (int)vv[2 * d][e] | ((int)vv[2 * d + 1][e] << 16);
                    acc[hb * 4 + e] = intel_sub_group_bf16_bf16_matrix_mad_k16(pown[u], b, acc[hb * 4 + e]);
                }
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const float lsum = sub_group_reduce_add(ll[i]);
        if (lane == 0) {
            lm[sg][i] = m[i];
            ls[sg][i] = lsum;
        }
#pragma unroll
        for (int j = 0; j < 8; j++) la[sg][i * HD + j * 16 + lane] = acc[j][i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const ulong cb = (ulong)z * maxc + c;
    for (uint e = lid; e < 8 * HD; e += 128) {
        const uint i = e / HD;
        float mx = -INFINITY;
        for (int s = 0; s < 8; s++) mx = fmax(mx, lm[s][i]);
        float o = 0.0f;
        for (int s = 0; s < 8; s++) o += (lm[s][i] == -INFINITY) ? 0.0f : exp(lm[s][i] - mx) * la[s][e];
        po[(cb * NH + head0 + i) * HD + (e - i * HD)] = o;
    }
    if (lid < 8) {
        float mx = -INFINITY;
        for (int s = 0; s < 8; s++) mx = fmax(mx, lm[s][lid]);
        float l = 0.0f;
        for (int s = 0; s < 8; s++) l += (lm[s][lid] == -INFINITY) ? 0.0f : exp(lm[s][lid] - mx) * ls[s][lid];
        pm[cb * NH + head0 + lid] = mx;
        pl[cb * NH + head0 + lid] = l;
    }
}

// Merge of the chunks of one (head, row, quarter of the dims): work-group of 256 items = 32 chunk groups x 8 items of 4 output dims (32 dims a work-group); group g takes the chunks g, g + 32, ...
// in order (two passes: the maximum, then the weighted sums), the 32 groups combine in ascending order through local memory. The chunks load in parallel instead of the sequential online
// rescaling of attn_merge. Grid (32 heads, rows, 4). out [row][32][128] bf16 = bf16(o / l); row z has len0 + z keys; po / pm / pl as written by the partial kernel.
__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void nem_attn_dec_merge(__global const float *po, __global const float *pm, __global const float *pl, __global ushort *out, uint len0, uint maxc) {
    __local float lmx[32];
    __local float lo[32][32];
    __local float ld[32];
    const uint head = get_group_id(0);
    const uint z = get_group_id(1);
    const uint qd = get_group_id(2);
    const uint g = get_local_id(0) >> 3;
    const uint dl = get_local_id(0) & 7;
    const uint nch = (len0 + z + 511) / 512;
    const ulong cb = (ulong)z * maxc;
    float mx = -INFINITY;
    for (uint c = g; c < nch; c += 32) mx = fmax(mx, pm[(cb + c) * NH + head]);
    if (dl == 0) lmx[g] = mx;
    barrier(CLK_LOCAL_MEM_FENCE);
    mx = lmx[0];
    for (int i = 1; i < 32; i++) mx = fmax(mx, lmx[i]);
    float den = 0.0f;
    float4 o = (float4)(0.0f);
    for (uint c = g; c < nch; c += 32) {
        const float w = exp(pm[(cb + c) * NH + head] - mx);
        den += w * pl[(cb + c) * NH + head];
        o += w * vload4(qd * 8 + dl, po + ((cb + c) * NH + head) * HD);
    }
    vstore4(o, dl, lo[g]);
    if (dl == 0) ld[g] = den;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (g == 0) {
        float4 r = vload4(dl, lo[0]);
        float dn = ld[0];
        for (int i = 1; i < 32; i++) {
            r += vload4(dl, lo[i]);
            dn += ld[i];
        }
        r /= dn;
        const ushort4 b = (ushort4)(to_bf(r.x), to_bf(r.y), to_bf(r.z), to_bf(r.w));
        vstore4(b, qd * 8 + dl, out + ((ulong)z * NH + head) * HD);
    }
}
