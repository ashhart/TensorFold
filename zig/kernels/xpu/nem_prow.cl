// Nemotron-H prompt rows (any count) with the decode step's arithmetic: each output sums as the one-token path does.
#define SG 16
#define PR 8 // prompt rows a sub-group holds
#define PO 8 // output rows a sub-group holds: one load and one sum of a word's inputs serve all of them
#define PSG (16 / PO) // sub-groups a work-group: 16 output rows
inline float bf(ushort v) { return as_float((uint)v << 16); }
inline ushort to_bf(float f) {
    uint u = as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (ushort)(u >> 16);
}

// Epilogues of a projection: bf16, fp32, or relu2 rounded to bf16 (the shared and routed experts' up projection).
#define EP_BF 0
#define EP_F32 1
#define EP_RELU2 2

// One word of a lane, row r, output O: the decode's dot and sum of inputs, then acc += sc * dot + bi * sx.
#define PDOT(O)                                                                                                                \
    {                                                                                                                          \
        float dot = 0.0f;                                                                                                      \
        dot += xv.s0 * q[O][0];                                                                                                \
        dot += xv.s1 * q[O][1];                                                                                                \
        dot += xv.s2 * q[O][2];                                                                                                \
        dot += xv.s3 * q[O][3];                                                                                                \
        dot += xv.s4 * q[O][4];                                                                                                \
        dot += xv.s5 * q[O][5];                                                                                                \
        dot += xv.s6 * q[O][6];                                                                                                \
        dot += xv.s7 * q[O][7];                                                                                                \
        acc[r][O] += sc[O] * dot + bi[O] * sx;                                                                                 \
    }

// acc[r][o] for rows x + xo[r] (r < n) and output rows row0 + o (o < nout): a lane takes words l, l+16, ... in turn.
inline void ptile(__global const uint *w, __global const ushort *scales, __global const ushort *biases, __global const ushort *x, const ulong *xo,
                  uint n, ulong row0, uint nout, uint in_dim, float acc[PR][PO]) {
    const uint lane = get_sub_group_local_id();
    const uint words = in_dim / 8, groups = in_dim / 64;
#pragma unroll
    for (uint r = 0; r < PR; r++)
#pragma unroll
        for (uint o = 0; o < PO; o++) acc[r][o] = 0.0f;
    for (uint wi = lane; wi < words; wi += SG) {
        float q[PO][8], sc[PO], bi[PO];
#pragma unroll
        for (uint o = 0; o < PO; o++) {
            const ulong row = row0 + (o < nout ? o : 0);
            const uint pk = w[row * words + wi];
            sc[o] = bf(scales[row * groups + wi / 8]);
            bi[o] = bf(biases[row * groups + wi / 8]);
#pragma unroll
            for (uint j = 0; j < 8; j++) q[o][j] = (float)((pk >> (4 * j)) & 15u);
        }
#pragma unroll
        for (uint r = 0; r < PR; r++) {
            if (r < n) {
                const float8 xv = as_float8(convert_uint8(vload8(wi, x + xo[r])) << 16);
                float sx = 0.0f;
                sx += xv.s0;
                sx += xv.s1;
                sx += xv.s2;
                sx += xv.s3;
                sx += xv.s4;
                sx += xv.s5;
                sx += xv.s6;
                sx += xv.s7;
#pragma unroll
                for (uint o = 0; o < PO; o++) PDOT(o)
            }
        }
    }
#pragma unroll
    for (uint r = 0; r < PR; r++)
#pragma unroll
        for (uint o = 0; o < PO; o++) acc[r][o] = sub_group_reduce_add(acc[r][o]);
}

// acc[r][o] of the valid rows and outputs to y[yr[r] * y_stride + y_off + row0 + o], spread over the lanes.
inline void pstore(__global void *y, const ulong *yr, uint n, ulong row0, uint nout, ulong y_stride, ulong y_off, uint ep, float acc[PR][PO]) {
    const uint lane = get_sub_group_local_id();
#pragma unroll
    for (uint r = 0; r < PR; r++)
#pragma unroll
        for (uint o = 0; o < PO; o++) {
            if (r < n && o < nout && (r * PO + o) % SG == lane) {
                const ulong at = yr[r] * y_stride + y_off + row0 + o;
                if (ep == EP_F32) {
                    ((__global float *)y)[at] = acc[r][o];
                } else if (ep == EP_RELU2) {
                    const float u = fmax(bf(to_bf(acc[r][o])), 0.0f);
                    ((__global ushort *)y)[at] = to_bf(u * u);
                } else {
                    ((__global ushort *)y)[at] = to_bf(acc[r][o]);
                }
            }
        }
}

// Dense projection of n rows: x [n][in_dim] -> y[r * y_stride + y_off + o]; grid (rows / (PO * PSG), n / PR).
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void pdense(__global const uint *w, __global const ushort *scales, __global const ushort *biases, __global const ushort *x, __global void *y,
                     uint in_dim, uint rows, uint n, uint y_stride, uint y_off, uint ep) {
    const ulong row0 = ((ulong)get_group_id(0) * PSG + get_sub_group_id()) * PO;
    const uint r0 = get_group_id(1) * PR;
    if (row0 >= rows) return;
    const uint nout = min((uint)(rows - row0), (uint)PO), m = min(n - r0, (uint)PR);
    ulong xo[PR], yr[PR];
#pragma unroll
    for (uint r = 0; r < PR; r++) {
        yr[r] = r0 + (r < m ? r : 0);
        xo[r] = yr[r] * in_dim;
    }
    float acc[PR][PO];
    ptile(w, scales, biases, x, xo, m, row0, nout, in_dim, acc);
    pstore(y, yr, m, row0, nout, y_stride, y_off, ep, acc);
}

// The slots of each expert, ascending: list[e * slots + i], count[e]. Grid (experts), one work-item each.
__kernel void pexpert_lists(__global const uint *ids, __global uint *list, __global uint *count, uint slots) {
    const uint e = get_global_id(0);
    uint c = 0;
    for (uint s = 0; s < slots; s++)
        if (ids[s] == e) list[(ulong)e * slots + c++] = s;
    count[e] = c;
}

// Routed expert e (grid z), its slots in blocks of PR (grid y): x row of slot s is s / top_k, out [slots][rows].
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void pexpert(__global const uint *w, __global const ushort *scales, __global const ushort *biases, __global const ushort *x, __global void *y,
                      __global const uint *list, __global const uint *count, uint in_dim, uint rows, uint slots, uint top_k, uint ep) {
    const uint e = get_group_id(2);
    const ulong row0 = ((ulong)get_group_id(0) * PSG + get_sub_group_id()) * PO;
    const uint r0 = get_group_id(1) * PR, c = count[e];
    if (row0 >= rows || r0 >= c) return;
    const uint nout = min((uint)(rows - row0), (uint)PO), m = min(c - r0, (uint)PR);
    ulong xo[PR], yr[PR];
#pragma unroll
    for (uint r = 0; r < PR; r++) {
        yr[r] = list[(ulong)e * slots + r0 + (r < m ? r : 0)];
        xo[r] = (yr[r] / top_k) * in_dim;
    }
    float acc[PR][PO];
    ptile(w, scales, biases, x, xo, m, (ulong)e * rows + row0, nout, in_dim, acc);
    pstore(y, yr, m, row0, nout, rows, 0, ep, acc);
}
