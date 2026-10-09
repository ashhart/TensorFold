// Qwen3.8 dense building blocks: 4-bit affine matvec (group 64), embedding, RMSNorm, residual add, SwiGLU, greedy argmax.
// Activations are bf16 with fp32 math; every stored activation is rounded to bf16 as in the upstream reference.
inline float bf(ushort v) { return as_float((uint)v << 16); }
inline ushort to_bf(float f) {
    uint u = as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (ushort)(u >> 16);
}
inline float bfr(float f) { return bf(to_bf(f)); }

// y[row] = sum_i x[i] * (q[row][i] * scale + bias); one 16-lane sub-group per row, 4 rows a work-group.
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

// bf16 output at y[y_off + row]; x read from x[x_off ..].
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

// One work-group (64 items, 4 sub-groups) per row; y = bf16(x * rsqrt(mean(x^2) + eps) * w).
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void rmsnorm(__global const ushort *x, __global const ushort *w, __global ushort *y, uint n, float eps) {
    __local float part[4];
    const uint row = get_group_id(0);
    const uint lid = get_local_id(0);
    float ss = 0.0f;
    for (uint i = lid; i < n; i += 64) {
        const float v = bf(x[row * n + i]);
        ss += v * v;
    }
    ss = sub_group_reduce_add(ss);
    if (get_sub_group_local_id() == 0) part[get_sub_group_id()] = ss;
    barrier(CLK_LOCAL_MEM_FENCE);
    const float total = part[0] + part[1] + part[2] + part[3];
    const float scale = rsqrt(total / (float)n + eps);
    for (uint i = lid; i < n; i += 64) y[row * n + i] = to_bf(bf(x[row * n + i]) * scale * bf(w[i]));
}

// Gathers rows of a 4-bit table by token id: one work-group (64 items) per id.
__kernel void embed4(__global const uint *w, __global const ushort *scales, __global const ushort *biases,
                     __global const uint *ids, __global ushort *y, uint dim) {
    const uint t = get_group_id(0);
    const uint row = ids[t];
    const uint words = dim / 8;
    const uint groups = dim / 64;
    for (uint wi = get_local_id(0); wi < words; wi += 64) {
        const uint g = wi / 8;
        const float sc = bf(scales[row * groups + g]);
        const float bi = bf(biases[row * groups + g]);
        const uint pk = w[row * words + wi];
        for (uint j = 0; j < 8; j++) y[t * dim + wi * 8 + j] = to_bf(sc * (float)((pk >> (4 * j)) & 15u) + bi);
    }
}

// Residual add in place: x = bf16(x + d).
__kernel void add_bf16(__global ushort *x, __global const ushort *d, uint n) {
    const uint i = get_global_id(0);
    if (i < n) x[i] = to_bf(bf(x[i]) + bf(d[i]));
}

// In place: fp32 values rounded to bf16 and widened back (upstream logits are bf16). NaN stays NaN.
__kernel void round_bf16_f32(__global float *x, uint n) {
    const uint i = get_global_id(0);
    if (i < n && !isnan(x[i])) x[i] = bfr(x[i]);
}

// act = bf16(silu(gate) * up) with gate and up already bf16 matvec outputs.
__kernel void swiglu(__global const ushort *g, __global const ushort *u, __global ushort *act, uint n) {
    const uint i = get_global_id(0);
    if (i >= n) return;
    const float a = bf(g[i]);
    act[i] = to_bf(a / (1.0f + exp(-a)) * bf(u[i]));
}

// Greedy argmax over fp32 logits (first NaN wins, +0 == -0, ties go to the lowest index). A candidate packs as
// (key << 32) | ~index so a plain unsigned max picks the winner; 0 is the empty candidate.
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

// fp16-weight matvec (unquantized small projections, e.g. in_proj_a/b): y[y_off + row] = bf16(sum x[i] * w[row][i]);
// one 16-lane sub-group per row, 4 rows a work-group, fp32 accumulation.
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void mv_f16(__global const ushort *w, __global const ushort *x, __global ushort *y, uint in_dim, uint y_off, uint rows) {
    const uint row = get_group_id(0) * 4 + get_sub_group_id();
    if (row >= rows) return;
    const uint lane = get_sub_group_local_id();
    float acc = 0.0f;
    for (uint c = lane; c < in_dim / 8; c += 16) {
        const float8 wv = vload_half8(c, (__global const half *)(w + (ulong)row * in_dim));
        const float8 xv = as_float8(convert_uint8(vload8(c, x)) << 16);
        acc += dot(wv.s0123, xv.s0123) + dot(wv.s4567, xv.s4567);
    }
    acc = sub_group_reduce_add(acc);
    if (lane == 0) y[y_off + row] = to_bf(acc);
}

// Plain bf16 embedding rows by token id: one work-group (64 items) per id.
__kernel void embed_bf16(__global const ushort *w, __global const uint *ids, __global ushort *y, uint dim) {
    const uint t = get_group_id(0);
    for (uint i = get_local_id(0); i < dim; i += 64) y[t * dim + i] = w[(ulong)ids[t] * dim + i];
}

// Sum of squares of vals[0..n) in the order of the original 64-item rmsnorm: lane L adds elements L, L+64, ... in order (SUMSQ_STEP), the
// 16-lane sub-group reduce, then the four sub-group parts left to right. The loads and squares-input stores run on all 512 items; only the
// ordered adds use 64 lanes, so results are bit-identical to the 64-item kernel. Needs a barrier after vals is written.
#define SUMSQ_STEP(ss, f) ss = fma(f, f, ss)
// n == 5120 (the hidden size): the same sums with a static trip count (80 reads a lane, issued in batches) instead of a loop that waits for every read.
inline float sumsq_5120(__local float *vals, __local float *part) {
    const uint lid = get_local_id(0);
    if (lid < 64) {
        float ss = 0.0f;
#pragma unroll 16
        for (uint k = 0; k < 80; k++) {
            const float f = vals[lid + 64 * k];
            SUMSQ_STEP(ss, f);
        }
        ss = sub_group_reduce_add(ss);
        if (get_sub_group_local_id() == 0) part[get_sub_group_id()] = ss;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    return part[0] + part[1] + part[2] + part[3];
}

inline float sumsq_ordered(__local float *vals, uint n, __local float *part) {
    const uint lid = get_local_id(0);
    if (lid < 64) {
        float ss = 0.0f;
        for (uint i = lid; i < n; i += 64) {
            const float f = vals[i];
            SUMSQ_STEP(ss, f);
        }
        ss = sub_group_reduce_add(ss);
        if (get_sub_group_local_id() == 0) part[get_sub_group_id()] = ss;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    return part[0] + part[1] + part[2] + part[3];
}

// Fused residual add and RMSNorm for one row of n <= 5120 values with a 512-item work-group: x = bf16(x + d); y = bf16(x * rsqrt(mean(x^2) + eps) * w).
// Same arithmetic, same summation order as add_bf16 followed by rmsnorm.
#define ADD_RMS_BODY(WT, LOADW) \
    __local float part[4]; \
    __local float vals[5120]; \
    const uint lid = get_local_id(0); \
    if (n == 5120) { /* static trip counts: every load of a work-item is issued before it is used (same values, same sums) */ \
        ushort xa[10], da[10]; \
        WT wa[10]; \
        _Pragma("unroll") for (uint k = 0; k < 10; k++) { xa[k] = x[lid + 512 * k]; da[k] = d[lid + 512 * k]; wa[k] = w[lid + 512 * k]; } \
        _Pragma("unroll") for (uint k = 0; k < 10; k++) { \
            const ushort v = to_bf(bf(xa[k]) + bf(da[k])); \
            x[lid + 512 * k] = v; \
            vals[lid + 512 * k] = bf(v); \
        } \
        barrier(CLK_LOCAL_MEM_FENCE); \
        const float scale = rsqrt(sumsq_5120(vals, part) / 5120.0f + eps); \
        _Pragma("unroll") for (uint k = 0; k < 10; k++) y[lid + 512 * k] = to_bf(vals[lid + 512 * k] * scale * LOADW(wa[k])); \
        return; \
    } \
    for (uint i = lid; i < n; i += 512) { \
        const ushort v = to_bf(bf(x[i]) + bf(d[i])); \
        x[i] = v; \
        vals[i] = bf(v); \
    } \
    barrier(CLK_LOCAL_MEM_FENCE); \
    const float scale = rsqrt(sumsq_ordered(vals, n, part) / (float)n + eps); \
    for (uint i = lid; i < n; i += 512) y[i] = to_bf(vals[i] * scale * LOADW(w[i]));
#define W_BF(v) bf(v)
#define W_F32(v) (v)

__attribute__((reqd_work_group_size(512, 1, 1))) __attribute__((intel_reqd_sub_group_size(16)))
__kernel void add_rmsnorm(__global ushort *x, __global const ushort *d, __global const ushort *w, __global ushort *y, uint n, float eps) {
    ADD_RMS_BODY(ushort, W_BF)
}

// The same with fp32 norm weights (GGUF checkpoints).
__attribute__((reqd_work_group_size(512, 1, 1))) __attribute__((intel_reqd_sub_group_size(16)))
__kernel void add_rmsnorm_f32w(__global ushort *x, __global const ushort *d, __global const float *w, __global ushort *y, uint n, float eps) {
    ADD_RMS_BODY(float, W_F32)
}

// RMSNorm of one row (n <= 5120) with a 512-item work-group (no residual add); same summation order as rmsnorm.
__attribute__((reqd_work_group_size(512, 1, 1))) __attribute__((intel_reqd_sub_group_size(16)))
__kernel void rmsnorm512(__global ushort *x, __global const ushort *w, __global ushort *y, uint n, float eps) {
    __local float part[4];
    __local float vals[5120];
    const uint lid = get_local_id(0);
    if (n == 5120) {
        ushort xa[10], wa[10];
#pragma unroll
        for (uint k = 0; k < 10; k++) { xa[k] = x[lid + 512 * k]; wa[k] = w[lid + 512 * k]; }
#pragma unroll
        for (uint k = 0; k < 10; k++) vals[lid + 512 * k] = bf(xa[k]);
        barrier(CLK_LOCAL_MEM_FENCE);
        const float scale = rsqrt(sumsq_5120(vals, part) / 5120.0f + eps);
#pragma unroll
        for (uint k = 0; k < 10; k++) y[lid + 512 * k] = to_bf(vals[lid + 512 * k] * scale * bf(wa[k]));
        return;
    }
    for (uint i = lid; i < n; i += 512) vals[i] = bf(x[i]);
    barrier(CLK_LOCAL_MEM_FENCE);
    const float scale = rsqrt(sumsq_ordered(vals, n, part) / (float)n + eps);
    for (uint i = lid; i < n; i += 512) y[i] = to_bf(vals[i] * scale * bf(w[i]));
}
