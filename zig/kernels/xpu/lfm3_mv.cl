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

// Prompt windows: bf16(x . (q * scale + bias)), one fma chain a 64-wide k block, blocks in order: any cut, same rows.
__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void pgemm(__global const uint *w, __global const ushort *scales, __global const ushort *biases, __global const ushort *x,
                    __global ushort *y, uint k, uint n, uint m, uint ldy, uint y_off, uint qg) {
    __local float xs[16][64];
    __local float ws[16][64];
    const uint lx = get_local_id(0), ly = get_local_id(1), t = ly * 16 + lx;
    const ulong m0 = (ulong)get_group_id(1) * 64, n0 = (ulong)get_group_id(0) * 64;
    const uint lr = t / 4, lk = (t % 4) * 4, words = k / 8, groups = k / qg;
    float acc[4][4];
#pragma unroll
    for (uint i = 0; i < 4; i++)
#pragma unroll
        for (uint j = 0; j < 4; j++) acc[i][j] = 0.0f;
    float part[4][4];
    for (uint k0 = 0; k0 < k; k0 += 16) {
        const ulong xr = m0 + lr, wr = n0 + lr;
        const uint kk = k0 + lk;
        ushort4 xv = (ushort4)(0);
        if (xr < m) xv = vload4(0, x + xr * k + kk);
        float4 wv = (float4)(0.0f);
        if (wr < n) {
            const uint p = w[wr * words + kk / 8] >> (4 * (kk % 8));
            const float sc = bf(scales[wr * groups + kk / qg]), bi = bf(biases[wr * groups + kk / qg]);
            wv = (float4)((float)(p & 15u) * sc + bi, (float)((p >> 4) & 15u) * sc + bi, (float)((p >> 8) & 15u) * sc + bi,
                          (float)((p >> 12) & 15u) * sc + bi);
        }
        xs[lk][lr] = bf(xv.s0);
        xs[lk + 1][lr] = bf(xv.s1);
        xs[lk + 2][lr] = bf(xv.s2);
        xs[lk + 3][lr] = bf(xv.s3);
        ws[lk][lr] = wv.s0;
        ws[lk + 1][lr] = wv.s1;
        ws[lk + 2][lr] = wv.s2;
        ws[lk + 3][lr] = wv.s3;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (k0 % 64 == 0) {
#pragma unroll
            for (uint i = 0; i < 4; i++)
#pragma unroll
                for (uint j = 0; j < 4; j++) part[i][j] = 0.0f;
        }
#pragma unroll
        for (uint q = 0; q < 16; q++) {
            const float4 a = vload4(ly, &xs[q][0]);
            const float4 b = vload4(lx, &ws[q][0]);
            part[0][0] = fma(a.s0, b.s0, part[0][0]);
            part[0][1] = fma(a.s0, b.s1, part[0][1]);
            part[0][2] = fma(a.s0, b.s2, part[0][2]);
            part[0][3] = fma(a.s0, b.s3, part[0][3]);
            part[1][0] = fma(a.s1, b.s0, part[1][0]);
            part[1][1] = fma(a.s1, b.s1, part[1][1]);
            part[1][2] = fma(a.s1, b.s2, part[1][2]);
            part[1][3] = fma(a.s1, b.s3, part[1][3]);
            part[2][0] = fma(a.s2, b.s0, part[2][0]);
            part[2][1] = fma(a.s2, b.s1, part[2][1]);
            part[2][2] = fma(a.s2, b.s2, part[2][2]);
            part[2][3] = fma(a.s2, b.s3, part[2][3]);
            part[3][0] = fma(a.s3, b.s0, part[3][0]);
            part[3][1] = fma(a.s3, b.s1, part[3][1]);
            part[3][2] = fma(a.s3, b.s2, part[3][2]);
            part[3][3] = fma(a.s3, b.s3, part[3][3]);
        }
        if ((k0 + 16) % 64 == 0 || k0 + 16 >= k) {
#pragma unroll
            for (uint i = 0; i < 4; i++)
#pragma unroll
                for (uint j = 0; j < 4; j++) acc[i][j] += part[i][j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
#pragma unroll
    for (uint i = 0; i < 4; i++) {
        const ulong r = m0 + ly * 4 + i;
        if (r >= m) continue;
#pragma unroll
        for (uint j = 0; j < 4; j++) {
            const ulong o = n0 + lx * 4 + j;
            if (o < n) y[r * ldy + y_off + o] = to_bf(acc[i][j]);
        }
    }
}
