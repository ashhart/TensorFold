// Qwen3.8 Gated DeltaNet decode step: causal conv1d + silu, q/k normalisation, delta-rule recurrence (fp32 state), gated RMSNorm.
// Follows families/qwen3_5/cuda/reference.py: conv state [3][C] bf16 oldest first, recurrent state [Hv][Dv][Dk] fp32.
#define DK 128
#define DV 128
#define KH 16     // key heads
#define VH 48     // value heads
inline float bf(ushort v) { return as_float((uint)v << 16); }
inline ushort to_bf(float f) {
    uint u = as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (ushort)(u >> 16);
}
inline float bfr(float f) { return bf(to_bf(f)); }
inline float silu(float x) { return x / (1.0f + exp(-x)); }

// One conv channel: taps cw[ch][0..3] (tap 0 = oldest) over [state0, state1, state2, qkv[ch]], silu, bf16; returns the bf16 bits.
// The caller shifts the state for the channel.
inline ushort conv_ch(__global const ushort *qkv, __global const ushort *state, __global const ushort *cw, uint cd, uint ch) {
    const ushort4 w = vload4(ch, cw);
    float acc = bf(state[ch]) * bf(w.s0);
    acc += bf(state[cd + ch]) * bf(w.s1);
    acc += bf(state[2 * cd + ch]) * bf(w.s2);
    acc += bf(qkv[ch]) * bf(w.s3);
    return to_bf(silu(acc));
}

inline void conv_shift(__global const ushort *qkv, __global ushort *state, uint cd, uint ch) {
    state[ch] = state[cd + ch];
    state[cd + ch] = state[2 * cd + ch];
    state[2 * cd + ch] = qkv[ch];
}

// Causal conv + silu of the q and k channels, then q = bf16(rms(q) / Dk), k = bf16(rms(k) / sqrt(Dk)) per key head (no weight,
// eps 1e-6); stored widened to fp32. One 16-lane sub-group per key head, 8 dims a lane; shifts the state of those channels.
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void gdn_conv_prep(__global const ushort *qkv, __global ushort *state, __global const ushort *cw, __global float *qn, __global float *kn, uint cd) {
    const uint h = get_group_id(0);
    const uint lane = get_sub_group_local_id();
    for (uint which = 0; which < 2; which++) {
        const uint base = which * (KH * DK) + h * DK + lane * 8;
        float v[8];
        float ss = 0.0f;
        for (int j = 0; j < 8; j++) {
            v[j] = bf(conv_ch(qkv, state, cw, cd, base + j));
            ss += v[j] * v[j];
        }
        for (int j = 0; j < 8; j++) conv_shift(qkv, state, cd, base + j);
        ss = sub_group_reduce_add(ss);
        const float r = rsqrt(ss / (float)DK + 1e-6f);
        const float sc = which == 0 ? (1.0f / (float)DK) : (1.0f / sqrt((float)DK));
        __global float *dst = (which == 0 ? qn : kn) + h * DK + lane * 8;
        for (int j = 0; j < 8; j++) dst[j] = bfr(v[j] * r * sc);
    }
}

// One delta-rule step for value head h, output rows [4*chunk, 4*chunk+4); a 16-lane sub-group owns a row (8 Dk values a lane).
// tiled != 0: value heads are in llama.cpp's tiled order. S = S*g; kv = S.k; delta = (v - kv)*beta; S += k (x) delta; y = S.q.
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void gdn_step(__global const float *qn, __global const float *kn, __global const ushort *qkv, __global ushort *cstate, __global const ushort *cw,
                       __global const ushort *bp, __global const ushort *ap, __global const float *a_log, __global const float *dt_bias,
                       __global float *state, __global ushort *y, uint tiled, uint cd) {
    const uint h = get_group_id(0);
    const uint row = get_group_id(1) * 4 + get_sub_group_id();
    const uint lane = get_sub_group_local_id();
    const uint kh = tiled ? h % KH : h / (VH / KH);  // tiled: GGUF order (v head = r * KH + k head), else grouped
    const float beta = 1.0f / (1.0f + exp(-bf(bp[h])));
    const float av = bf(ap[h]) + dt_bias[h];
    const float sp = av > 20.0f ? av : log1p(exp(av));
    const float g = exp(-exp(a_log[h]) * sp);
    const float8 k8 = vload8(lane, kn + kh * DK);
    const float8 q8 = vload8(lane, qn + kh * DK);
    __global float *srow = state + ((ulong)h * DV + row) * DK;
    float8 s = vload8(lane, srow) * g;
    const float kv = sub_group_reduce_add(dot(s.s0123, k8.s0123) + dot(s.s4567, k8.s4567));
    // the v channel of this row: conv + silu computed (and its state shifted) by lane 0, shared with the sub-group
    float vc = 0.0f;
    if (lane == 0) {
        const uint ch = 2 * KH * DK + h * DV + row;
        vc = bf(conv_ch(qkv, cstate, cw, cd, ch));
        conv_shift(qkv, cstate, cd, ch);
    }
    vc = sub_group_broadcast(vc, 0);
    const float delta = (vc - kv) * beta;
    s += k8 * delta;
    vstore8(s, lane, srow);
    const float o = sub_group_reduce_add(dot(s.s0123, q8.s0123) + dot(s.s4567, q8.s4567));
    if (lane == 0) y[h * DV + row] = to_bf(o);
}

// out = bf16(silu(z) * (y * rsqrt(mean(y^2) + eps) * w)) per value head; one sub-group a head, 8 dims a lane.
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void gdn_gate_norm(__global const ushort *y, __global const ushort *z, __global const ushort *w, __global ushort *out, float eps) {
    const uint h = get_group_id(0);
    const uint lane = get_sub_group_local_id();
    float v[8];
    float ss = 0.0f;
    for (int j = 0; j < 8; j++) { v[j] = bf(y[h * DV + lane * 8 + j]); ss += v[j] * v[j]; }
    ss = sub_group_reduce_add(ss);
    const float r = rsqrt(ss / (float)DV + eps);
    for (int j = 0; j < 8; j++) {
        const uint d = lane * 8 + j;
        out[h * DV + d] = to_bf(silu(bf(z[h * DV + d])) * (v[j] * r * bf(w[d])));
    }
}
