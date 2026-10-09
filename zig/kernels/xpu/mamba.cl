// Mamba2 mixer decode step (one token): bf16 matvec, causal conv1d + silu, SSM recurrence, gated group RMSNorm.
// Activations are bf16 with fp32 math; rounding points follow the upstream scan_rows.cu / mamba.py reference.
#define DS 128
inline float bf(ushort v) { return as_float((uint)v << 16); }
inline ushort to_bf(float f) {
    uint u = as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (ushort)(u >> 16);
}
inline float bfr(float f) { return bf(to_bf(f)); }

// 4-bit affine (MLX layout, groups of 64) matvec with a bf16 output; one 16-lane sub-group per output row.
// in_proj output [z(4096) | x,B,C(6144) | dt(64)] is consumed in place by offset, so the split costs nothing.
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void qmv4_bf16(__global const uint *w, __global const ushort *scales, __global const ushort *biases,
                        __global const ushort *x, __global ushort *y, uint in_dim) {
    const uint row = get_group_id(0);
    const uint lane = get_sub_group_local_id();
    const uint words = in_dim / 8;
    const uint groups = in_dim / 64;
    float acc = 0.0f;
    for (uint wi = lane; wi < words; wi += 16) {
        const uint g = wi / 8;
        const float sc = bf(scales[row * groups + g]);
        const float bi = bf(biases[row * groups + g]);
        const uint pk = w[row * words + wi];
        const ushort8 xv = vload8(wi, x);
        float dot = 0.0f, sx = 0.0f;
        for (uint j = 0; j < 8; j++) {
            const float v = bf(xv[j]);
            dot += v * (float)((pk >> (4 * j)) & 15u);
            sx += v;
        }
        acc += sc * dot + bi * sx;
    }
    acc = sub_group_reduce_add(acc);
    if (lane == 0) y[row] = to_bf(acc);
}

// Depthwise causal conv (kernel 4) on proj[xoff .. xoff+cd), bias, bf16, silu, bf16. state is [3][cd] bf16, oldest row first,
// shifted in place. cw is the checkpoint layout [cd][4]; tap k multiplies the input 3-k steps back.
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void conv1d_step(__global const ushort *proj, uint xoff, __global ushort *state, __global const ushort *cw,
                          __global const ushort *cb, __global ushort *out, uint cd) {
    const uint ch = get_global_id(0);
    if (ch >= cd) return;
    const ushort4 w = vload4(ch, cw);
    const ushort t0 = state[ch], t1 = state[cd + ch], t2 = state[2 * cd + ch];
    const ushort cur = proj[xoff + ch];
    float acc = bf(cb[ch]);
    acc = acc + bf(w.s0) * bf(t0);
    acc = acc + bf(w.s1) * bf(t1);
    acc = acc + bf(w.s2) * bf(t2);
    acc = acc + bf(w.s3) * bf(cur);
    const float cv = bfr(acc);
    out[ch] = to_bf(cv * (1.0f / (1.0f + exp(-cv))));
    state[ch] = t1;
    state[cd + ch] = t2;
    state[2 * cd + ch] = cur;
}

// One SSM step. Work-group (head, 4 value rows), one 16-lane sub-group per row, each lane owns 8 of the 128 states.
// xc = [x (heads*dh) | B (groups*128) | C (groups*128)] after conv; state is [heads][dh][128] fp32, updated in place.
// y = bf16(bf16(silu(z)) * bf16(C.s + D x)); a = -exp(a_log), dt = clamp(softplus(proj_dt + dt_bias), lo, hi).
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void ssm_step(__global const ushort *proj, uint dt_off, __global const ushort *xc, __global float *state,
                       __global const float *a_log, __global const float *dsk, __global const float *dtb,
                       __global ushort *y, uint dh, uint xd, uint groups, uint per_group, float lo, float hi) {
    const uint h = get_group_id(0);
    const uint row = get_group_id(1) * 4 + get_sub_group_id();
    const uint lane = get_sub_group_local_id();
    const uint g = h / per_group;
    const float v = bf(proj[dt_off + h]) + dtb[h];
    const float dt = fmin(fmax(fmax(v, 0.0f) + log(1.0f + exp(-fabs(v))), lo), hi);
    const float da = exp(-exp(a_log[h]) * dt);
    const float x = bf(xc[h * dh + row]);
    const float xdt = x * dt;
    __global float *st = state + (h * dh + row) * DS;
    __global const ushort *bp = xc + xd + g * DS;
    __global const ushort *cp = xc + xd + groups * DS + g * DS;
    float m = 0.0f;
    for (uint j = 0; j < DS / 16; j++) {
        const uint s = lane + 16 * j;
        const float ns = fma(xdt, bf(bp[s]), st[s] * da);
        st[s] = ns;
        m = fma(ns, bf(cp[s]), m);
    }
    m = sub_group_reduce_add(m);
    if (lane == 0) {
        const float z = bf(proj[h * dh + row]);
        const float gz = bfr(z / (1.0f + exp(-z)));
        y[h * dh + row] = to_bf(gz * bfr(fma(x, dsk[h], m)));
    }
}

// Group RMSNorm over bf16 (n elements per group, one work-group of 64 per group): out = bf16(w * bf16(x * rsqrt(mean+eps))).
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void group_rmsnorm(__global const ushort *x, __global const ushort *w, __global ushort *y, uint n, float eps) {
    __local float part[4];
    const uint base = get_group_id(0) * n;
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
    for (uint i = lid; i < n; i += 64) y[base + i] = to_bf(bf(w[base + i]) * bfr(bf(x[base + i]) * inv));
}
