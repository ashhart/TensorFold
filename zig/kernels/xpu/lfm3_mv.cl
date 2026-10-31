// LFM3 decode matvecs fused by launch: attn.cl's qmv_row unchanged (same contraction), more per launch.
inline float bf(ushort v) { return as_float((uint)v << 16); }
inline ushort to_bf(float f) {
    uint u = as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (ushort)(u >> 16);
}

// attn.cl's qmv_row, verbatim: one 16-lane sub-group a row, the sum valid on every lane.
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

// Four tables in one launch (q, k, v, gate): rows r0 .. r3 into y0 + o0 .. y3 + o3; 4 rows a work-group.
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void qmv4_bf_x4(__global const uint *w0, __global const ushort *s0, __global const ushort *b0, __global ushort *y0, uint r0, uint o0,
                         __global const uint *w1, __global const ushort *s1, __global const ushort *b1, __global ushort *y1, uint r1, uint o1,
                         __global const uint *w2, __global const ushort *s2, __global const ushort *b2, __global ushort *y2, uint r2, uint o2,
                         __global const uint *w3, __global const ushort *s3, __global const ushort *b3, __global ushort *y3, uint r3, uint o3,
                         __global const ushort *x, uint in_dim) {
    uint row = get_group_id(0) * 4 + get_sub_group_id();
    float acc;
    __global ushort *y;
    if (row < r0) {
        acc = qmv_row(w0, s0, b0, x, row, in_dim);
        y = y0 + o0;
    } else if ((row -= r0) < r1) {
        acc = qmv_row(w1, s1, b1, x, row, in_dim);
        y = y1 + o1;
    } else if ((row -= r1) < r2) {
        acc = qmv_row(w2, s2, b2, x, row, in_dim);
        y = y2 + o2;
    } else if ((row -= r2) < r3) {
        acc = qmv_row(w3, s3, b3, x, row, in_dim);
        y = y3 + o3;
    } else return;
    if (get_sub_group_local_id() == 0) y[row] = to_bf(acc);
}

// w1 and w3 rows and their SwiGLU: act = bf16(bf16(silu(bf16(w1 x))) * bf16(w3 x)), as qmv4_bf + lfm3_swiglu.
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void qmv4_swiglu(__global const uint *w1, __global const ushort *s1, __global const ushort *b1, __global const uint *w3, __global const ushort *s3,
                          __global const ushort *b3, __global const ushort *x, __global ushort *act, uint in_dim, uint rows) {
    const uint row = get_group_id(0) * 4 + get_sub_group_id();
    if (row >= rows) return;
    const float a = bf(to_bf(qmv_row(w1, s1, b1, x, row, in_dim)));
    const float u = bf(to_bf(qmv_row(w3, s3, b3, x, row, in_dim)));
    if (get_sub_group_local_id() == 0) act[row] = to_bf(bf(to_bf(a / (1.0f + exp(-a)))) * u);
}
