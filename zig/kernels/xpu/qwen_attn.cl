// Qwen3.8 full-attention decode step: gated q projection, q/k RMSNorm, partial rotate-half RoPE, split-K GQA attention
// over a bf16 KV cache laid out [pos][kv head][256], output gate. Follows families/qwen3_5/cuda/reference.py.
#define HD 256   // head dim
#define NH 24    // query heads
#define NKV 4    // kv heads
#define CH 64    // keys per chunk (a work-group; all GQ query heads of a kv head)
#define GQ 6     // query heads per kv head
#define RD 32    // rotated pairs (partial_rotary_factor 0.25 * 256 / 2)
inline float bf(ushort v) { return as_float((uint)v << 16); }
inline ushort to_bf(float f) {
    uint u = as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (ushort)(u >> 16);
}
inline float bfr(float f) { return bf(to_bf(f)); }

// Work-group (256 items, one dim each) per head: heads 0..23 are queries from qg[head][0..255] (qg rows are 512: query then
// gate), heads 24..27 are keys from kraw. y = bf16(x * rsqrt(mean(x^2) + eps) * w), then rotate-half over the first 2*RD dims
// with rope = [cos(32) | sin(32)] for the position, bf16. Queries go to qo, keys to kc[k_off ..].
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void attn_prep(__global const ushort *qg, __global const ushort *kraw, __global const ushort *qw, __global const ushort *kw,
                        __global const float *rope, __global ushort *qo, __global ushort *kc, uint k_off, float eps) {
    __local float part[16];
    __local float nv[HD];
    const uint head = get_group_id(0);
    const uint i = get_local_id(0);
    const bool is_q = head < NH;
    const float x = is_q ? bf(qg[head * 2 * HD + i]) : bf(kraw[(head - NH) * HD + i]);
    float ss = sub_group_reduce_add(x * x);
    if (get_sub_group_local_id() == 0) part[get_sub_group_id()] = ss;
    barrier(CLK_LOCAL_MEM_FENCE);
    float total = 0.0f;
    for (int j = 0; j < 16; j++) total += part[j];
    const float r = rsqrt(total / (float)HD + eps);
    nv[i] = bfr(x * r * bf(is_q ? qw[i] : kw[i]));
    barrier(CLK_LOCAL_MEM_FENCE);
    float o = nv[i];
    if (i < RD) o = nv[i] * rope[i] - nv[i + RD] * rope[RD + i];
    else if (i < 2 * RD) o = nv[i] * rope[i - RD] + nv[i - RD] * rope[RD + i - RD];
    if (is_q) qo[head * HD + i] = to_bf(o);
    else kc[k_off + (head - NH) * HD + i] = to_bf(o);
}

// Pass 1 of split-K decode attention: work-group (kv head, chunk of CH keys), 128 items. The kv head's GQ query heads share every K/V
// row read: scores (fp32) for all of them, per-head softmax over the chunk, then P*V in fp32. Writes the unnormalised o plus the chunk's
// max and denominator per query head.
__attribute__((reqd_work_group_size(128, 1, 1))) __attribute__((intel_reqd_sub_group_size(16)))
__kernel void attn_partial(__global const ushort *q, __global const ushort *kc, __global const ushort *vc,
                           __global float *po, __global float *pm, __global float *pl, uint len, float scale) {
    __local float qs[GQ * HD];
    __local float sc[GQ * CH];
    const uint hk = get_group_id(0);
    const uint c = get_group_id(1);
    const uint lid = get_local_id(0);
    const uint sg = get_sub_group_id();
    const uint lane = get_sub_group_local_id();
    const uint k0 = c * CH;
    if (k0 >= len) return;
    const uint n = min((uint)CH, len - k0);
    for (uint i = lid; i < GQ * HD; i += 128) qs[i] = bf(q[hk * GQ * HD + i]);
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
            pm[c * NH + hk * GQ + sg] = m;
            pl[c * NH + hk * GQ + sg] = den;
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
    for (int h = 0; h < GQ; h++) vstore2(acc[h], lid, po + ((ulong)c * NH + hk * GQ + h) * HD);
}

// Pass 2: merges the chunks of one query head (global max, rescaled sums) and applies the output gate:
// out = bf16(bf16(o / l) * sigmoid(gate)). One work-group of 256 items a head, one dim each.
__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void attn_merge(__global const float *po, __global const float *pm, __global const float *pl,
                         __global const ushort *qg, __global ushort *out, uint len) {
    const uint head = get_group_id(0);
    const uint d = get_local_id(0);
    const uint nch = (len + CH - 1) / CH;
    float m = -INFINITY;
    for (uint c = 0; c < nch; c++) m = fmax(m, pm[c * NH + head]);
    float den = 0.0f, acc = 0.0f;
    for (uint c = 0; c < nch; c++) {
        const float w = exp(pm[c * NH + head] - m);
        den += pl[c * NH + head] * w;
        acc += po[((ulong)c * NH + head) * HD + d] * w;
    }
    const float gate = bf(qg[head * 2 * HD + HD + d]);
    out[head * HD + d] = to_bf(bfr(acc / den) * (1.0f / (1.0f + exp(-gate))));
}
