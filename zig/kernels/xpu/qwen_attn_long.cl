#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
// Qwen3.8 full-attention decode at long context: flash-decoding split-K over the bf16 KV cache [pos][4 kv heads][256]. One work-group (8 sub-groups) takes a chunk
// of LC = 1024 keys of one (kv head, row); each sub-group runs an online softmax over its 128 keys in blocks of 16 (lane = key for the scores on the matrix engine,
// lane = 16 dims for P*V in fp32), the sub-groups merge in a fixed order through local memory, and attn_long_merge combines the chunks (<= 128, so 131072 keys).
// A row's bits depend on its key count only (the chunking is by absolute key index), never on the number of rows of the launch.
#define HD 256
#define NH 24
#define NKV 4
#define GQ 6
#define LC 1024
#define KS 128 // keys a sub-group
// KVQ 0: bf16 K/V [pos][kv head][256]. KVQ 8 / 4: quantized records [pos][kv head][REC bytes]: the data (8 bit signed, or 4 bit offset-8 nibbles with the pair layout of
// qwen_kvq.cl) of 8 blocks of 32 dims, then the 8 fp16 block scales (value = q * scale).
#ifndef KVQ
#define KVQ 0
#endif
#if KVQ == 8
#define REC 272
#define DATA 256
#elif KVQ == 4
#define REC 144
#define DATA 128
#endif
inline float bf(ushort v) { return as_float((uint)v << 16); }
inline ushort to_bf(float f) {
    uint u = as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (ushort)(u >> 16);
}
inline float bfr(float f) { return bf(to_bf(f)); }

inline float16 unpack16(uint8 u) {
    float16 f;
    f.s0 = as_float(u.s0 << 16); f.s1 = as_float(u.s0 & 0xffff0000u);
    f.s2 = as_float(u.s1 << 16); f.s3 = as_float(u.s1 & 0xffff0000u);
    f.s4 = as_float(u.s2 << 16); f.s5 = as_float(u.s2 & 0xffff0000u);
    f.s6 = as_float(u.s3 << 16); f.s7 = as_float(u.s3 & 0xffff0000u);
    f.s8 = as_float(u.s4 << 16); f.s9 = as_float(u.s4 & 0xffff0000u);
    f.sa = as_float(u.s5 << 16); f.sb = as_float(u.s5 & 0xffff0000u);
    f.sc = as_float(u.s6 << 16); f.sd = as_float(u.s6 & 0xffff0000u);
    f.se = as_float(u.s7 << 16); f.sf = as_float(u.s7 & 0xffff0000u);
    return f;
}

#if KVQ
inline uint bfpair(int a, int b) { return (as_uint((float)a) >> 16) | (as_uint((float)b) & 0xffff0000u); }

// B operand of one 16-dim block t (dword d = dims 2d, 2d + 1) of a K record.
inline int8 kdeq(const __global uchar *rec, int t) {
#if KVQ == 8
    const uint4 u = vload4(0, (const __global uint *)(rec + t * 16));
    uint o[8];
#pragma unroll
    for (int d = 0; d < 8; d++) {
        const uint x = d < 4 ? (d < 2 ? u.x : u.y) : (d < 6 ? u.z : u.w);
        const int sh = 16 * (d & 1);
        o[d] = bfpair((int)(x << (24 - sh)) >> 24, (int)(x << (16 - sh)) >> 24);
    }
    return (int8)(o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7]);
#else
    const uint2 w = vload2(0, (const __global uint *)(rec + t * 8));
    uint o[8];
#pragma unroll
    for (int p = 0; p < 4; p++) {
        o[p] = ((w.x >> (4 * p)) & 0x000F000Fu) | 0x43004300u;
        o[4 + p] = ((w.y >> (4 * p)) & 0x000F000Fu) | 0x43004300u;
    }
    return (int8)(o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7]);
#endif
}

// The 16 dims of a lane (dims 16 lane ..) of a V record as floats, times the block scale.
inline float16 vdeq(const __global uchar *rec, uint lane) {
#if KVQ == 8
    return convert_float16(as_char16(vload4(0, (const __global uint *)(rec + lane * 16))));
#else
    const uint2 w = vload2(0, (const __global uint *)(rec + lane * 8));
    float16 f;
#pragma unroll
    for (int p = 0; p < 4; p++) {
        f[2 * p] = as_float(((w.x >> (4 * p)) & 0xFu) | 0x4B000000u) - 8388616.0f;
        f[2 * p + 1] = as_float(((w.x >> (4 * p + 16)) & 0xFu) | 0x4B000000u) - 8388616.0f;
        f[8 + 2 * p] = as_float(((w.y >> (4 * p)) & 0xFu) | 0x4B000000u) - 8388616.0f;
        f[8 + 2 * p + 1] = as_float(((w.y >> (4 * p + 16)) & 0xFu) | 0x4B000000u) - 8388616.0f;
    }
    return f;
#endif
}
#endif

// Grid (4 kv heads * rows, chunks); row z has len0 + z keys, q is [row][24][256] bf16. po [row][maxc][24][256] fp32, pm / pl [row][maxc][24].
__attribute__((reqd_work_group_size(128, 1, 1))) __attribute__((intel_reqd_sub_group_size(16)))
__kernel void attn_long_partial(__global const ushort *q, __global const ushort *kc, __global const ushort *vc, __global float *po, __global float *pm,
                                __global float *pl, uint len0, float scale, uint maxc) {
    __local short qa[16 * 16 * 8]; // [dim block][lane][head (6 + 2 zero rows)]
    __local float ms[8 * GQ], ls[8 * GQ], accs[GQ * HD];
#if KVQ == 4
    __local float qsum[8 * 8]; // [block][head]: the sum of q over the 32 dims of a block (the offset 8 of the 4-bit values comes out through it)
#endif
    const uint hk = get_group_id(0) & 3, z = get_group_id(0) >> 2, c = get_group_id(1); // rows beside kv heads: their work-groups of a chunk run together and share the L2
    const uint lid = get_local_id(0), sg = get_sub_group_id(), lane = get_sub_group_local_id();
    const uint len = len0 + z;
    const uint k0 = c * LC;
    if (k0 >= len) return;
    const __global ushort *qr = q + (ulong)z * NH * HD + (ulong)hk * GQ * HD;
    for (uint i = lid; i < 16 * 16 * 8; i += 128) {
        const uint t = i >> 7, l = (i >> 3) & 15, h = i & 7;
        qa[i] = h < GQ ? (short)qr[h * HD + t * 16 + l] : (short)0;
    }
#if KVQ == 4
    if (lid < 64) {
        const uint b = lid >> 3, h = lid & 7;
        float sum = 0.0f;
        if (h < GQ) for (uint i = 0; i < 32; i++) sum += bf(qr[h * HD + b * 32 + i]);
        qsum[lid] = sum;
    }
#endif
    barrier(CLK_LOCAL_MEM_FENCE);
    const uint ks = k0 + sg * KS;
    const uint ke = min(ks + KS, len);
    float m[GQ], ll[GQ];
    float16 acc[GQ];
#pragma unroll
    for (int h = 0; h < GQ; h++) {
        m[h] = -INFINITY;
        ll[h] = 0.0f;
        acc[h] = (float16)(0.0f);
    }
    for (uint kb = ks; kb < ke; kb += 16) {
        const uint key = min(kb + lane, ke - 1);
        float8 cc = (float8)(0.0f);
#if KVQ == 0
        const __global uint *krow = (const __global uint *)(kc + ((ulong)key * NKV + hk) * HD);
#pragma unroll
        for (int t = 0; t < 16; t++) {
            const int8 b = as_int8(vload8(0, krow + t * 8));
            const short8 a = vload8(t * 16 + lane, qa);
            cc = intel_sub_group_bf16_bf16_matrix_mad_k16(a, b, cc);
        }
#else
        const __global uchar *krec = (const __global uchar *)kc + ((ulong)key * NKV + hk) * REC;
        const float8 ksc = vload_half8(0, (const __global half *)(krec + DATA));
#pragma unroll
        for (int bl = 0; bl < 8; bl++) {
            float8 cb = (float8)(0.0f);
#pragma unroll
            for (int tt = 0; tt < 2; tt++) {
                const int t = 2 * bl + tt;
                cb = intel_sub_group_bf16_bf16_matrix_mad_k16(vload8(t * 16 + lane, qa), kdeq(krec, t), cb);
            }
#if KVQ == 4
            cb -= 136.0f * vload8(bl, qsum);
#endif
            cc += cb * ksc[bl];
        }
#endif
        float p[GQ];
        const bool ok = kb + lane < ke;
#pragma unroll
        for (int h = 0; h < GQ; h++) {
            const float s = ok ? cc[h] * scale : -INFINITY;
            const float mn = fmax(m[h], sub_group_reduce_max(s));
            const float alpha = exp(m[h] - mn);
            p[h] = exp(s - mn);
            ll[h] = ll[h] * alpha + p[h];
            acc[h] *= alpha;
            m[h] = mn;
        }
#pragma unroll
        for (int j = 0; j < 16; j++) {
            const uint vk = min(kb + (uint)j, ke - 1);
#if KVQ == 0
            const float16 v = unpack16(vload8(0, (const __global uint *)(vc + ((ulong)vk * NKV + hk) * HD) + lane * 8));
#pragma unroll
            for (int h = 0; h < GQ; h++) acc[h] += sub_group_broadcast(p[h], j) * v;
#else
            const __global uchar *vrec = (const __global uchar *)vc + ((ulong)vk * NKV + hk) * REC;
            const float16 v = vdeq(vrec, lane);
            const float sv = vload_half(lane >> 1, (const __global half *)(vrec + DATA));
#pragma unroll
            for (int h = 0; h < GQ; h++) acc[h] += (sub_group_broadcast(p[h], j) * sv) * v;
#endif
        }
    }
#pragma unroll
    for (int h = 0; h < GQ; h++) {
        const float l = sub_group_reduce_add(ll[h]);
        if (lane == 0) {
            ms[sg * GQ + h] = m[h];
            ls[sg * GQ + h] = l;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float f[GQ];
#pragma unroll
    for (int h = 0; h < GQ; h++) {
        float M = -INFINITY;
        for (int s = 0; s < 8; s++) M = fmax(M, ms[s * GQ + h]);
        f[h] = exp(m[h] - M);
    }
    for (int r = 0; r < 8; r++) {
        if ((int)sg == r) {
#pragma unroll
            for (int h = 0; h < GQ; h++) {
                const float16 s = r == 0 ? f[h] * acc[h] : vload16(lane, accs + h * HD) + f[h] * acc[h];
                vstore16(s, lane, accs + h * HD);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const ulong cb = (ulong)z * maxc + c;
    for (uint i = lid; i < GQ * HD; i += 128) po[(cb * NH + hk * GQ + i / HD) * HD + (i % HD)] = accs[i];
    if (lid < GQ) {
        const uint h = lid;
        float M = -INFINITY;
        for (int s = 0; s < 8; s++) M = fmax(M, ms[s * GQ + h]);
        float den = 0.0f;
        for (int s = 0; s < 8; s++) den += ls[s * GQ + h] * exp(ms[s * GQ + h] - M);
        pm[cb * NH + hk * GQ + h] = M;
        pl[cb * NH + hk * GQ + h] = den;
    }
}

// Merges the chunks of one query head (global max, rescaled sums, in chunk order) and applies the output gate:
// out [row][24][256] = bf16(bf16(o / l) * sigmoid(gate)); qg [row][24][512] (query, gate). Grid (24 heads, rows), 256 items, one dim each.
__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void attn_long_merge(__global const float *po, __global const float *pm, __global const float *pl, __global const ushort *qg, __global ushort *out,
                              uint len0, uint maxc) {
    const uint head = get_group_id(0), z = get_group_id(1), d = get_local_id(0);
    const uint nch = (len0 + z + LC - 1) / LC;
    const ulong cb = (ulong)z * maxc;
    float m = -INFINITY;
    for (uint c = 0; c < nch; c++) m = fmax(m, pm[(cb + c) * NH + head]);
    float den = 0.0f, acc = 0.0f;
    for (uint c = 0; c < nch; c++) {
        const float w = exp(pm[(cb + c) * NH + head] - m);
        den += pl[(cb + c) * NH + head] * w;
        acc += po[((cb + c) * NH + head) * HD + d] * w;
    }
    const float gate = bf(qg[((ulong)z * NH + head) * 2 * HD + HD + d]);
    out[((ulong)z * NH + head) * HD + d] = to_bf(bfr(acc / den) * (1.0f / (1.0f + exp(-gate))));
}
