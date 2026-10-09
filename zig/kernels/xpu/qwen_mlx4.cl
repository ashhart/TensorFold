// MLX 4-bit affine (group 64) decode matvec for the Qwen3.8 dense model, tuned for the B70: y[row] = sum_i x[i] * (q[row][i] * scale + bias).
// One sub-group is one work-group (no shared memory, no barriers) and owns one output row; lane l owns the groups l, l + 16, ... of the row:
// The sum order differs from kernels/xpu/qwen_basic.cl (qmv4_bf): groups are summed per lane, 4 partial dots per group, then a sub-group reduce.
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
inline float bf(ushort v) { return as_float((uint)v << 16); }
inline ushort to_bf(float f) {
    uint u = as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (ushort)(u >> 16);
}

// The 8 floats of two uint4 halves of x (4 uint = 8 bf16 each): x16 = x + 8 * q, q = 0..3 of a half group of 32 columns.
#define INL inline __attribute__((always_inline))
INL void load_x32(__global const uint4 *xp, float xf[32], float *sx) {
    float s = 0.0f;
#pragma unroll
    for (int q = 0; q < 4; q++) {
        const uint4 v = xp[q];
        const uint u[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int k = 0; k < 4; k++) {
            xf[8 * q + 2 * k] = as_float(u[k] << 16);
            xf[8 * q + 2 * k + 1] = as_float(u[k] & 0xFFFF0000u);
            s += xf[8 * q + 2 * k] + xf[8 * q + 2 * k + 1];
        }
    }
    *sx = s;
}

// sum over 32 columns of x * nibble for the 4 packed words (one uint4).
INL float dot32(const uint4 wq, const float xf[32]) {
    const uint u[4] = {wq.x, wq.y, wq.z, wq.w};
    float d0 = 0.0f, d1 = 0.0f;
#pragma unroll
    for (int k = 0; k < 4; k++) {
#pragma unroll
        for (int j = 0; j < 8; j += 2) {
            d0 = fma(xf[8 * k + j], (float)((u[k] >> (4 * j)) & 15u), d0);
            d1 = fma(xf[8 * k + j + 1], (float)((u[k] >> (4 * j + 4)) & 15u), d1);
        }
    }
    return d0 + d1;
}

// One output row: sum over the lane's groups, then a sub-group reduce. in_dim a multiple of 64, x 16-byte aligned.
INL float qmv_row(__global const uint *w, __global const ushort *scales, __global const ushort *biases, __global const ushort *x, uint in_dim, uint row) {
    const uint lane = get_sub_group_local_id();
    const uint groups = in_dim / 64, words = in_dim / 8;
    float acc = 0.0f;
    for (uint g = lane; g < groups; g += 16) {
        float xa[32], xb[32], sxa, sxb;
        load_x32((__global const uint4 *)(x + g * 64), xa, &sxa);
        load_x32((__global const uint4 *)(x + g * 64 + 32), xb, &sxb);
        const __global uint4 *wp = (const __global uint4 *)(w + (ulong)row * words + g * 8);
        const uint4 wa = wp[0], wb = wp[1];
        const float sc = bf(scales[(ulong)row * groups + g]), bi = bf(biases[(ulong)row * groups + g]);
        acc += sc * (dot32(wa, xa) + dot32(wb, xb)) + bi * (sxa + sxb);
    }
    return sub_group_reduce_add(acc);
}

// y[y_off + row] (bf16, or fp32 for the logits when f32out); x read from x[x_off ..]. Grid (rows, 1, 1), local size 16.
__attribute__((intel_reqd_sub_group_size(16))) __kernel void qmv4x(
    __global const uint *w, __global const ushort *scales, __global const ushort *biases, __global const ushort *x, __global void *y,
    uint in_dim, uint x_off, uint y_off, uint rows, uint f32out) {
    const uint row = get_group_id(0);
    const float v = qmv_row(w, scales, biases, x + x_off, in_dim, row);
    if (get_sub_group_local_id() == 0) {
        if (f32out) ((__global float *)y)[y_off + row] = v;
        else ((__global ushort *)y)[y_off + row] = to_bf(v);
    }
}

// act[row] = bf16(silu(gate[row]) * up[row]) for the SwiGLU MLP: one sub-group computes the gate row and the up row (both bf16-rounded as the separate
// matvecs stored them, then the same silu expression as qwen_basic.cl swiglu), sharing the x loads. Grid (rows, 1, 1), local size 16.
__attribute__((intel_reqd_sub_group_size(16))) __kernel void qmv4x_gateup(
    __global const uint *wg, __global const ushort *sg, __global const ushort *bg, __global const uint *wu, __global const ushort *su,
    __global const ushort *bu, __global const ushort *x, __global ushort *act, uint in_dim, uint rows) {
    const uint row = get_group_id(0), lane = get_sub_group_local_id();
    const uint groups = in_dim / 64, words = in_dim / 8;
    float accg = 0.0f, accu = 0.0f;
    for (uint g = lane; g < groups; g += 16) {
        float xa[32], xb[32], sxa, sxb;
        load_x32((__global const uint4 *)(x + g * 64), xa, &sxa);
        load_x32((__global const uint4 *)(x + g * 64 + 32), xb, &sxb);
        const float sx = sxa + sxb;
        const __global uint4 *pg = (const __global uint4 *)(wg + (ulong)row * words + g * 8);
        const __global uint4 *pu = (const __global uint4 *)(wu + (ulong)row * words + g * 8);
        const uint4 ga = pg[0], gb = pg[1], ua = pu[0], ub = pu[1];
        accg += bf(sg[(ulong)row * groups + g]) * (dot32(ga, xa) + dot32(gb, xb)) + bf(bg[(ulong)row * groups + g]) * sx;
        accu += bf(su[(ulong)row * groups + g]) * (dot32(ua, xa) + dot32(ub, xb)) + bf(bu[(ulong)row * groups + g]) * sx;
    }
    accg = sub_group_reduce_add(accg);
    accu = sub_group_reduce_add(accu);
    if (lane == 0) {
        const float a = bf(to_bf(accg));
        act[row] = to_bf(a / (1.0f + exp(-a)) * bf(to_bf(accu)));
    }
}

// Row-invariant multi-row matvec: m <= R activation rows (x [m][in_dim], bf16) against the same weights, y[r * y_stride + y_off + row]. A sub-group owns W weight
// rows and reads each weight tile once for all m rows; every (weight row, activation row) pair runs the exact operation sequence of qmv_row (per lane over its groups,
// then the sub-group reduce), so a row's bits do not depend on m, on its index or on R and W. R in {1,2,4,8,16}; rows past m are skipped.
INL void qmv_rw(const int R, const int W, __global const uint *w, __global const ushort *scales, __global const ushort *biases, __global const ushort *x,
                uint in_dim, uint row0, uint rows, uint m, float acc[16]) {
    const uint lane = get_sub_group_local_id();
    const uint groups = in_dim / 64, words = in_dim / 8;
#pragma unroll
    for (int i = 0; i < 16; i++) acc[i] = 0.0f;
    for (uint g = lane; g < groups; g += 16) {
        uint4 wa[4], wb[4];
        float sc[4], bi[4];
#pragma unroll
        for (int k = 0; k < W; k++) {
            const uint row = min(row0 + k, rows - 1);
            const __global uint4 *wp = (const __global uint4 *)(w + (ulong)row * words + g * 8);
            wa[k] = wp[0];
            wb[k] = wp[1];
            sc[k] = bf(scales[(ulong)row * groups + g]);
            bi[k] = bf(biases[(ulong)row * groups + g]);
        }
#pragma unroll
        for (int r = 0; r < R; r++) {
            if (r < m) {
                float xa[32], xb[32], sxa, sxb;
                load_x32((__global const uint4 *)(x + (ulong)r * in_dim + g * 64), xa, &sxa);
                load_x32((__global const uint4 *)(x + (ulong)r * in_dim + g * 64 + 32), xb, &sxb);
#pragma unroll
                for (int k = 0; k < W; k++) acc[k * R + r] += sc[k] * (dot32(wa[k], xa) + dot32(wb[k], xb)) + bi[k] * (sxa + sxb);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < R * W; i++) acc[i] = sub_group_reduce_add(acc[i]);
}

#define QMVR(R, W) \
    __attribute__((intel_reqd_sub_group_size(16))) __kernel void qmv4r_##R##_##W( \
        __global const uint *wt, __global const ushort *scales, __global const ushort *biases, __global const ushort *x, __global void *y, \
        uint in_dim, uint y_off, uint y_stride, uint rows, uint m, uint f32out) { \
        const uint row0 = get_group_id(0) * W; \
        float acc[16]; \
        qmv_rw(R, W, wt, scales, biases, x, in_dim, row0, rows, m, acc); \
        if (get_sub_group_local_id() == 0) { \
            _Pragma("unroll") for (int k = 0; k < W; k++) { \
                _Pragma("unroll") for (int r = 0; r < R; r++) { \
                    if (r < m && row0 + k < rows) { \
                        const ulong at = (ulong)r * y_stride + y_off + row0 + k; \
                        if (f32out) ((__global float *)y)[at] = acc[k * R + r]; else ((__global ushort *)y)[at] = to_bf(acc[k * R + r]); \
                    } \
                } \
            } \
        } \
    }
QMVR(1, 1) QMVR(1, 2) QMVR(1, 4) QMVR(2, 1) QMVR(2, 2) QMVR(2, 4) QMVR(4, 1) QMVR(4, 2) QMVR(4, 4) QMVR(8, 1) QMVR(8, 2) QMVR(16, 1)


// act[r * rows + row] = bf16(silu(gate) * up) for m <= R rows: the gate and up rows through qmv_rw (the exact operation sequence of qmv4x_gateup per row).
#define QMVGU(R) \
    __attribute__((intel_reqd_sub_group_size(16))) __kernel void qmv4r_gu_##R( \
        __global const uint *wg, __global const ushort *sg, __global const ushort *bg, __global const uint *wu, __global const ushort *su, \
        __global const ushort *bu, __global const ushort *x, __global ushort *act, uint in_dim, uint rows, uint m) { \
        const uint row = get_group_id(0); \
        float accg[16], accu[16]; \
        qmv_rw(R, 1, wg, sg, bg, x, in_dim, row, rows, m, accg); \
        qmv_rw(R, 1, wu, su, bu, x, in_dim, row, rows, m, accu); \
        if (get_sub_group_local_id() == 0) { \
            _Pragma("unroll") for (int r = 0; r < R; r++) { \
                if (r < m) { \
                    const float a = bf(to_bf(accg[r])); \
                    act[(ulong)r * rows + row] = to_bf(a / (1.0f + exp(-a)) * bf(to_bf(accu[r]))); \
                } \
            } \
        } \
    }
QMVGU(2) QMVGU(4) QMVGU(8) QMVGU(16)

// ---- systolic matvec: the MLX 4-bit kernel for every row count (decode m = 1 included) ------------------------------------------------------------------
// y[r][row] for m <= 16 activation rows. 16 weight rows a sub-group (lane = weight row = DPAS column), bf16 x bf16 -> fp32 on the systolic array.
// Weights are repacked at load time into 16-row blocks: block rb, group g holds [lane][8 words] at ((rb * groups + g) * 16 + lane) * 8 and the scale and bias of
// (rb, g, lane) at (rb * groups + g) * 16 + lane, so a sub-group's loads are contiguous 512 B runs.
// A nibble q is the bf16 128 + q (bits 0x4300 | q), two nibbles (j and j + 4 of a word) per dword with one shift, one mask and one or; the offset comes back out
// with the group's sum of x, which a second DPAS against ones gives: acc += sc * C1 + (bias - 128 sc) * Cones. A row's bits depend on that row only (same B,
// same chain; the array rows are independent, the M = 1, 2, 4, 8 forms of the instruction give the same row), not on m, its slot or the variant.
// Activations come pre-arranged by qmv4_xprep as xt [row group of M][in_dim][M] bf16 in the k order of the nibble pairs.
__attribute__((intel_reqd_sub_group_size(16))) __kernel void qmv4_xprep(__global const ushort *x, __global ushort *xt, uint in_dim, uint m, uint Mw) {
    const uint n = get_global_id(0), row = get_global_id(1); // natural input index, row of x (rows past m are zero)
    if (n >= in_dim) return;
    const uint i = n & 15, b = n >> 4, wi = i >> 3, jj = i & 7, j = jj & 3, h = jj >> 2;
    const uint pos = b * 16 + 2 * (wi * 4 + j) + h;
    xt[((ulong)(row / Mw) * in_dim + pos) * Mw + (row % Mw)] = row < m ? x[(ulong)row * in_dim + n] : (ushort)0;
}

INL short8 load_a(const int M, __global const short *xt, ulong at) {
    short8 a = (short8)(0);
    if (M == 1) a.s0 = xt[at];
    else if (M == 2) a.s01 = vload2(0, xt + at * 2);
    else if (M == 4) a.s0123 = vload4(0, xt + at * 4);
    else a = vload8(0, xt + at * 8);
    return a;
}

INL float8 dpas_m(const int M, short8 a, int8 b, float8 c) {
    if (M == 1) c.s0 = intel_sub_group_bf16_bf16_matrix_mad_k16(a.s0, b, c.s0);
    else if (M == 2) c.s01 = intel_sub_group_bf16_bf16_matrix_mad_k16(a.s01, b, c.s01);
    else if (M == 4) c.s0123 = intel_sub_group_bf16_bf16_matrix_mad_k16(a.s0123, b, c.s0123);
    else c = intel_sub_group_bf16_bf16_matrix_mad_k16(a, b, c);
    return c;
}

// Groups g0 .. g0 + gper (a multiple of 4) of NB consecutive 16-row blocks starting at block rb * NB; G row groups of M rows (G = 2 only for M = 8, NB = 2 only for
// M <= 8): an A fragment is loaded once for the NB blocks. acc[nb + q].
INL void qmv_dpas(const int M, const int G, const int NB, __global const uint *w, __global const ushort *scales, __global const ushort *biases,
                  __global const short *xt, uint in_dim, uint rb, uint g0, uint gper, uint one, float8 acc[2]) {
    const uint lane = get_sub_group_local_id();
    const uint groups = in_dim / 64;
    const int UG = 4 / NB; // groups a step: four blocks-groups of weights in flight either way
    const int8 ones = (int8)((int)(one | (lane & (one >> 31)))); // bf16 1.0 pairs, made lane-dependent (the term is 0) so the 8 registers are built once, not splatted before every use
    acc[0] = (float8)(0.0f);
    acc[1] = (float8)(0.0f);
    for (uint gi = 0; gi < gper; gi += UG) {
        uint4 wq[2][8];
        float sc[2][4], bi[2][4];
#pragma unroll
        for (int u = 0; u < UG; u++)
#pragma unroll
            for (int nb = 0; nb < NB; nb++) {
                const ulong blk = (ulong)(rb * NB + nb) * groups + g0 + gi + u;
                const __global uint4 *wp = (const __global uint4 *)(w + (blk * 16 + lane) * 8);
                wq[nb][2 * u] = wp[0];
                wq[nb][2 * u + 1] = wp[1];
                sc[nb][u] = bf(scales[blk * 16 + lane]);
                bi[nb][u] = bf(biases[blk * 16 + lane]);
            }
#pragma unroll
        for (int u = 0; u < UG; u++) {
            float8 c1[2][2], cs[2][2]; // two independent chains (blocks 0, 1 and 2, 3) so the array's latency overlaps; added below
#pragma unroll
            for (int ch = 0; ch < 2; ch++)
#pragma unroll
                for (int q = 0; q < 2; q++) c1[ch][q] = cs[ch][q] = (float8)(0.0f);
#pragma unroll
            for (int blk = 0; blk < 4; blk++) {
#pragma unroll
                for (int q = 0; q < G; q++) {
                    const short8 a = load_a(M, xt, (ulong)q * in_dim + (g0 + gi + u) * 64 + blk * 16 + lane);
                    cs[blk >> 1][q] = dpas_m(M, a, ones, cs[blk >> 1][q]);
#pragma unroll
                    for (int nb = 0; nb < NB; nb++) {
                        const uint ww[8] = {wq[nb][2 * u].x, wq[nb][2 * u].y, wq[nb][2 * u].z, wq[nb][2 * u].w, wq[nb][2 * u + 1].x, wq[nb][2 * u + 1].y, wq[nb][2 * u + 1].z, wq[nb][2 * u + 1].w};
                        int8 b;
#pragma unroll
                        for (int d = 0; d < 8; d++) b[d] = (int)(((ww[blk * 2 + (d >> 2)] >> (4 * (d & 3))) & 0x000F000Fu) | 0x43004300u);
                        c1[blk >> 1][nb + q] = dpas_m(M, a, b, c1[blk >> 1][nb + q]);
                    }
                }
            }
#pragma unroll
            for (int q = 0; q < 2; q++) {
                c1[0][q] += c1[1][q];
                cs[0][q] += cs[1][q];
            }
#pragma unroll
            for (int nb = 0; nb < NB; nb++) {
#pragma unroll
                for (int q = 0; q < G; q++) {
                    const float bb = bi[nb][u] - 128.0f * sc[nb][u];
                    const int ix = nb + q;
                    // only the M live rows: the narrow forms keep the idle lanes of the accumulators out of the loop
                    if (M == 1) acc[ix].s0 = fma(c1[0][ix].s0, sc[nb][u], fma(cs[0][q].s0, bb, acc[ix].s0));
                    else if (M == 2) acc[ix].s01 = fma(c1[0][ix].s01, (float2)(sc[nb][u]), fma(cs[0][q].s01, (float2)(bb), acc[ix].s01));
                    else if (M == 4) acc[ix].s0123 = fma(c1[0][ix].s0123, (float4)(sc[nb][u]), fma(cs[0][q].s0123, (float4)(bb), acc[ix].s0123));
                    else acc[ix] = fma(c1[0][ix], (float8)(sc[nb][u]), fma(cs[0][q], (float8)(bb), acc[ix]));
                }
            }
        }
    }
}

// grid (rows / (16 NB), S): split s sums groups s * gper ...; S = 1 stores the result, S > 1 stores partials z [S][16][rows] for qmv4_finish.
#define QMVD(M, G, NB) \
    __attribute__((intel_reqd_sub_group_size(16))) __kernel void qmv4d_##M##_##G##_##NB( \
        __global const uint *wt, __global const ushort *scales, __global const ushort *biases, __global const short *xt, __global void *y, __global float *z, \
        uint in_dim, uint y_off, uint y_stride, uint rows, uint m, uint f32out, uint S, uint one) { \
        const uint rb = get_group_id(0), lane = get_sub_group_local_id(), split = get_group_id(1); \
        const uint gper = (in_dim / 64) / S; \
        float8 acc[2]; \
        qmv_dpas(M, G, NB, wt, scales, biases, xt, in_dim, rb, split * gper, gper, one, acc); \
        _Pragma("unroll") for (int nb = 0; nb < NB; nb++) { \
            const uint orow = (rb * NB + nb) * 16 + lane; \
            if (orow < rows) { \
                _Pragma("unroll") for (int q = 0; q < G; q++) { \
                    _Pragma("unroll") for (int r = 0; r < M; r++) { \
                        if (q * M + r < m) { \
                            if (S == 1) { \
                                const ulong at = (ulong)(q * M + r) * y_stride + y_off + orow; \
                                if (f32out) ((__global float *)y)[at] = acc[nb + q][r]; else ((__global ushort *)y)[at] = to_bf(acc[nb + q][r]); \
                            } else z[((ulong)split * 16 + q * M + r) * rows + orow] = acc[nb + q][r]; \
                        } \
                    } \
                } \
            } \
        } \
    }
QMVD(1, 1, 1) QMVD(2, 1, 1) QMVD(4, 1, 1) QMVD(8, 1, 1) QMVD(8, 2, 1) QMVD(1, 1, 2) QMVD(2, 1, 2) QMVD(4, 1, 2) QMVD(8, 1, 2)

// Adds the S partials of every (row, output) in order and stores. Grid (ceil(rows / 16), m), local size 16.
__kernel void qmv4_finish(__global const float *z, __global void *y, uint rows, uint S, uint y_off, uint y_stride, uint f32out) {
    const uint o = get_global_id(0), r = get_global_id(1);
    if (o >= rows) return;
    float s = 0.0f;
    for (uint q = 0; q < S; q++) s += z[((ulong)q * 16 + r) * rows + o];
    const ulong at = (ulong)r * y_stride + y_off + o;
    if (f32out) ((__global float *)y)[at] = s; else ((__global ushort *)y)[at] = to_bf(s);
}

// act[r][o] = bf16(silu(g) * u) with g, u the in-order sums of the S partials of the gate and up projections, each rounded to bf16 first. Grid (ceil(rows / 16), m).
__kernel void qmv4_finish_gu(__global const float *zg, __global const float *zu, __global ushort *act, uint rows, uint S) {
    const uint o = get_global_id(0), r = get_global_id(1);
    if (o >= rows) return;
    float sg = 0.0f, su = 0.0f;
    for (uint q = 0; q < S; q++) {
        sg += zg[((ulong)q * 16 + r) * rows + o];
        su += zu[((ulong)q * 16 + r) * rows + o];
    }
    const float a = bf(to_bf(sg));
    act[(ulong)r * rows + o] = to_bf(a / (1.0f + exp(-a)) * bf(to_bf(su)));
}
