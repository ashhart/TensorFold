// Sage-A1.5 (qwen3_5_moe) MoE decode ops: fp16 router, softmax top-k with renormalised weights, the shared expert's
// sigmoid gate, 4-bit affine (MLX, groups of 64) SwiGLU experts on stacked tables, and the gated combine.
// x: the post-attention norm's fp16 rows [rows, dim] (forward.zig's h); out: fp16 [rows, dim], the residual's r.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
inline float bf(ushort v) { return as_float((uint)v << 16); }
inline ushort to_bf(float f) {
    uint u = as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (ushort)(u >> 16);
}

#define SG 16
#define MAX_PER_LANE 16 // SG * MAX_PER_LANE = 256 experts at most
#define MAX_TOPK 8

// logits[row][e] = x[row] . gate[e] in fp32 (gate fp16 [n_experts, dim]); slot n_experts is the shared expert's gate.
// Grid (n_experts + 1, rows), one sub-group a work-group; dim: a multiple of 8.
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void sage_router(__global const half *x, __global const half *gate, __global const half *shared_gate,
                          __global float *logits, uint dim) {
    const uint e = get_group_id(0), row = get_group_id(1), n = get_num_groups(0);
    const uint lane = get_sub_group_local_id();
    __global const half *w = e + 1 == n ? shared_gate : gate + (ulong)e * dim;
    float acc = 0.0f;
    for (uint i = lane; i < dim / 8; i += SG) {
        const float8 xv = convert_float8(vload8(i, x + (ulong)row * dim)), wv = convert_float8(vload8(i, w));
        for (uint j = 0; j < 8; j++) acc = fma(xv[j], wv[j], acc);
    }
    acc = sub_group_reduce_add(acc);
    if (lane == 0) logits[row * n + e] = acc;
}

// Per row: softmax over n_experts, top_k by probability (ties to the lower id), weights renormalised to sum 1.
// shared_w[row] = sigmoid(the shared gate's logit). Grid (rows), one sub-group a work-group.
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void sage_route(__global const float *logits, __global uint *ids, __global float *wts, __global float *shared_w,
                         uint n_experts, uint top_k) {
    const uint row = get_group_id(0), lane = get_sub_group_local_id();
    const uint stride = n_experts + 1, per = n_experts / SG;
    __global const float *l = logits + (ulong)row * stride;
    float v[MAX_PER_LANE];
    float m = -INFINITY;
    for (uint j = 0; j < per; j++) {
        v[j] = l[lane + SG * j];
        m = fmax(m, v[j]);
    }
    m = sub_group_reduce_max(m);
    // Renormalised top-k of a softmax only needs exp(l - max): the softmax's denominator cancels.
    float picked[MAX_TOPK];
    float total = 0.0f;
    for (uint k = 0; k < top_k; k++) {
        float best = -INFINITY;
        uint best_e = 0xffffffffu;
        for (uint j = 0; j < per; j++)
            if (v[j] > best) { best = v[j]; best_e = lane + SG * j; }
        const float top = sub_group_reduce_max(best);
        const uint winner = sub_group_reduce_min(best == top ? best_e : 0xffffffffu);
        for (uint j = 0; j < per; j++)
            if (lane + SG * j == winner) v[j] = -INFINITY;
        picked[k] = exp(top - m);
        total += picked[k];
        if (lane == 0) ids[row * top_k + k] = winner;
    }
    if (lane == 0) {
        for (uint k = 0; k < top_k; k++) wts[row * top_k + k] = picked[k] / total;
        shared_w[row] = 1.0f / (1.0f + exp(-l[n_experts]));
    }
}

// 4-bit affine dot of row `row` of a stacked table with fp16 x; the sum is valid on every lane of the sub-group.
// A lane reads 4 words (32 values, half a group of 64) at a time, with one scale and one bias. in_dim: a multiple of 32.
inline float qdot(__global const uint *w, __global const ushort *scales, __global const ushort *biases, __global const half *x,
                  ulong row, uint in_dim) {
    const uint lane = get_sub_group_local_id();
    const uint quads = in_dim / 32, groups = in_dim / 64;
    __global const uint *wr = w + row * (in_dim / 8);
    float acc = 0.0f;
    for (uint c = lane; c < quads; c += SG) {
        const uint4 pk = vload4(c, wr);
        const float sc = bf(scales[row * groups + c / 2]);
        const float bi = bf(biases[row * groups + c / 2]);
        float dot = 0.0f, sx = 0.0f;
        for (uint q = 0; q < 4; q++) {
            const uint u = ((const uint *)&pk)[q];
            const float8 xv = convert_float8(vload8(c * 4 + q, x));
            for (uint j = 0; j < 8; j++) {
                dot = fma(xv[j], (float)((u >> (4 * j)) & 15u), dot);
                sx += xv[j];
            }
        }
        acc += sc * dot + bi * sx;
    }
    return sub_group_reduce_add(acc);
}

// qdot of two rows of one table (gate and up) against the same x, split over the work-group; valid on every work-item.
inline float2 qdot2(__global const uint *w, __global const ushort *scales, __global const ushort *biases, __global const half *x,
                    ulong r0, ulong r1, uint in_dim) {
    const uint quads = in_dim / 32, groups = in_dim / 64;
    __global const uint *w0 = w + r0 * (in_dim / 8), *w1 = w + r1 * (in_dim / 8);
    float2 acc = 0.0f;
    for (uint c = get_local_id(0); c < quads; c += get_local_size(0)) {
        const uint4 p0 = vload4(c, w0), p1 = vload4(c, w1);
        const float s0 = bf(scales[r0 * groups + c / 2]), b0 = bf(biases[r0 * groups + c / 2]);
        const float s1 = bf(scales[r1 * groups + c / 2]), b1 = bf(biases[r1 * groups + c / 2]);
        float d0 = 0.0f, d1 = 0.0f, sx = 0.0f;
        for (uint q = 0; q < 4; q++) {
            const uint u0 = ((const uint *)&p0)[q], u1 = ((const uint *)&p1)[q];
            const float8 xv = convert_float8(vload8(c * 4 + q, x));
            for (uint j = 0; j < 8; j++) {
                d0 = fma(xv[j], (float)((u0 >> (4 * j)) & 15u), d0);
                d1 = fma(xv[j], (float)((u1 >> (4 * j)) & 15u), d1);
                sx += xv[j];
            }
        }
        acc.x += s0 * d0 + b0 * sx;
        acc.y += s1 * d1 + b1 * sx;
    }
    return (float2)(work_group_reduce_add(acc.x), work_group_reduce_add(acc.y));
}

// gate and up of one expert pair and SwiGLU: act[p][r] = fp16(bf16(silu(bf16(gate)) * bf16(up))). One work-group a
// row r (K split over its work-items); grid (ff, pairs).
// Pair p is (row p / top_k, expert ids[p]); pairs at or past n_pairs are the shared expert of row p - n_pairs, whose
// tables are the `sh_` ones. gate_up holds 2 * ff rows an expert: gate rows 0..ff, then up rows.
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void sage_expert_swiglu(__global const uint *w, __global const ushort *scales, __global const ushort *biases,
                                 __global const uint *sh_w, __global const ushort *sh_scales, __global const ushort *sh_biases,
                                 __global const half *x, __global const uint *ids, __global half *act,
                                 uint dim, uint ff, uint top_k, uint n_pairs) {
    const uint r = get_group_id(0), p = get_group_id(1);
    const bool shared = p >= n_pairs;
    const uint row = shared ? p - n_pairs : p / top_k;
    const ulong base = shared ? 0 : (ulong)ids[p] * 2 * ff;
    __global const half *xr = x + (ulong)row * dim;
    const float2 gu = shared ? qdot2(sh_w, sh_scales, sh_biases, xr, r, ff + r, dim) : qdot2(w, scales, biases, xr, base + r, base + ff + r, dim);
    const float gb = bf(to_bf(gu.x)), ub = bf(to_bf(gu.y));
    if (get_local_id(0) == 0) act[(ulong)p * ff + r] = convert_half_rte(bf(to_bf(gb / (1.0f + exp(-gb)) * ub)));
}

// down of one expert pair to fp32: y[p][r]. One sub-group a row; grid (dim / sub-groups a group, pairs).
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void sage_expert_down(__global const uint *w, __global const ushort *scales, __global const ushort *biases,
                               __global const uint *sh_w, __global const ushort *sh_scales, __global const ushort *sh_biases,
                               __global const half *act, __global const uint *ids, __global float *y,
                               uint dim, uint ff, uint n_pairs) {
    const uint r = get_group_id(0) * get_num_sub_groups() + get_sub_group_id(), p = get_group_id(1);
    const bool shared = p >= n_pairs;
    __global const half *a = act + (ulong)p * ff;
    const float acc = shared ? qdot(sh_w, sh_scales, sh_biases, a, r, ff) : qdot(w, scales, biases, a, (ulong)ids[p] * dim + r, ff);
    if (get_sub_group_local_id() == 0) y[(ulong)p * dim + r] = acc;
}

// out[row][i] = fp16(bf16(sum_k wts[k] * y[k][i] + shared_w * y_shared[i])), the sum in pick order. Grid (dim, rows).
__kernel void sage_moe_combine(__global const float *y, __global const float *wts, __global const float *shared_w,
                               __global half *out, uint dim, uint top_k, uint n_pairs) {
    const uint i = get_global_id(0), row = get_global_id(1);
    if (i >= dim) return;
    float acc = 0.0f;
    for (uint k = 0; k < top_k; k++) {
        const uint p = row * top_k + k;
        acc = fma(y[(ulong)p * dim + i], wts[p], acc);
    }
    acc = fma(y[(ulong)(n_pairs + row) * dim + i], shared_w[row], acc);
    out[(ulong)row * dim + i] = convert_half_rte(bf(to_bf(acc)));
}

// ---------------------------------------------------------------- grouped prompt MoE (XMX)
// A prompt chunk's pairs sorted by expert into tiles of at most MC pairs, so a tile reads its expert's weights once.
// The shared expert is expert n_experts, one pair a row (pair n_pairs + row). act and y keep the pair index, so
// sage_moe_combine is the same as in decode.
#define MC 16
#define DPH(a, b, c) intel_sub_group_f16_f16_matrix_mad_k16(as_short8(a), as_int8(b), c)

// sums[row][g] = the sum of src[row] over group g of 64 (the bias term of the affine dot). Grid (k / 64, rows).
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void sage_group_sums(__global const half *src, __global float *sums, uint k) {
    const uint g = get_group_id(0), row = get_group_id(1), lane = get_sub_group_local_id();
    __global const half *s = src + (ulong)row * k + g * 64;
    const float v = (float)s[lane] + (float)s[lane + 16] + (float)s[lane + 32] + (float)s[lane + 48];
    const float t = sub_group_reduce_add(v);
    if (lane == 0) sums[(ulong)row * (k / 64) + g] = t;
}

// order: the pair indices grouped by expert (shared expert last); tiles: (expert, first slot, count) uint4 records;
// ntiles[0]: the tile count. One work-group of 256 work-items.
__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void sage_moe_sort(__global const uint *ids, __global uint *order, __global uint4 *tiles, __global uint *ntiles,
                   uint n_experts, uint n_pairs, uint rows) {
    __local uint cnt[257], start[257], tbase[257], fill[257];
    const uint lid = get_local_id(0);
    for (uint e = lid; e <= n_experts; e += 256) cnt[e] = e == n_experts ? rows : 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint p = lid; p < n_pairs; p += 256) atomic_inc(&cnt[ids[p]]);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0) {
        uint s = 0, t = 0;
        for (uint e = 0; e <= n_experts; e++) {
            start[e] = s;
            tbase[e] = t;
            fill[e] = s;
            s += cnt[e];
            t += (cnt[e] + MC - 1) / MC;
        }
        ntiles[0] = t;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint e = lid; e <= n_experts; e += 256)
        for (uint i = 0; i * MC < cnt[e]; i++) tiles[tbase[e] + i] = (uint4)(e, start[e] + i * MC, min((uint)MC, cnt[e] - i * MC), 0);
    for (uint p = lid; p < n_pairs; p += 256) order[atomic_inc(&fill[ids[p]])] = p;
    for (uint r = lid; r < rows; r += 256) order[start[n_experts] + r] = n_pairs + r;
}

// The x row a pair reads: its token's row (routed) or row p - n_pairs (shared) for gate|up; the pair's own row for down.
inline uint pair_row(uint p, uint n_pairs, uint top_k, bool own) { return own ? p : p >= n_pairs ? p - n_pairs : p / top_k; }

// acc[rb][c] += x[tile rows] . W[col block c]ᵀ over in_dim, 4-bit affine (MLX 4/64): per group of 64, the DPAS dot of
// x with the raw nibbles, then scale * dot + bias * sum(x). Lane l owns column cols[c] + l; rows of the tile past `n`
// read the tile's first row and are not stored.
#define TILE_DOT(NB)                                                                                                    \
    for (uint g = 0; g < groups; g++) {                                                                                \
        float8 t[2][NB];                                                                                               \
        for (uint rb = 0; rb < 2; rb++)                                                                                \
            for (uint c = 0; c < NB; c++) t[rb][c] = 0.0f;                                                             \
        uint8 q[NB];                                                                                                   \
        for (uint c = 0; c < NB; c++) q[c] = vload8(0, wt + (wrow[c] * words + g * 8));                                \
        for (uint s = 0; s < 4; s++) {                                                                                 \
            short8 a[2];                                                                                               \
            for (uint rb = 0; rb < 2; rb++)                                                                            \
                for (uint r = 0; r < 8; r++) a[rb][r] = as_short(xr[rb * 8 + r][g * 64 + s * 16 + lane]);              \
            for (uint c = 0; c < NB; c++) {                                                                            \
                const uint lo = ((uint *)&q[c])[2 * s], hi = ((uint *)&q[c])[2 * s + 1];                                \
                int8 b;                                                                                                \
                for (uint j = 0; j < 4; j++) {                                                                         \
                    ((int *)&b)[j] = as_int((half2)((half)((lo >> (8 * j)) & 15u), (half)((lo >> (8 * j + 4)) & 15u))); \
                    ((int *)&b)[4 + j] = as_int((half2)((half)((hi >> (8 * j)) & 15u), (half)((hi >> (8 * j + 4)) & 15u))); \
                }                                                                                                      \
                for (uint rb = 0; rb < 2; rb++) t[rb][c] = DPH(a[rb], b, t[rb][c]);                                    \
            }                                                                                                          \
        }                                                                                                              \
        for (uint c = 0; c < NB; c++) {                                                                                \
            const float sc = bf(scales[wrow[c] * groups + g]), bi = bf(biases[wrow[c] * groups + g]);                  \
            for (uint rb = 0; rb < 2; rb++)                                                                            \
                for (uint r = 0; r < 8; r++)                                                                           \
                    acc[rb][c][r] += sc * t[rb][c][r] + bi * xs[rb * 8 + r][g];                                        \
        }                                                                                                              \
    }

// Grouped gate|up and SwiGLU: act[p][r] as sage_expert_swiglu. Grid (ff / 16, tile bound), one sub-group a group.
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void sage_tile_swiglu(__global const uint *w, __global const ushort *ex_scales, __global const ushort *ex_biases,
                               __global const uint *sh_w, __global const ushort *sh_scales, __global const ushort *sh_biases,
                               __global const half *x, __global const float *x_sums, __global const uint *order,
                               __global const uint4 *tiles, __global const uint *ntiles, __global half *act,
                               uint dim, uint ff, uint top_k, uint n_pairs, uint n_experts) {
    const uint tile = get_group_id(1), lane = get_sub_group_local_id();
    if (tile >= ntiles[0]) return;
    const uint4 tl = tiles[tile];
    const bool shared = tl.x == n_experts;
    __global const uint *wt = shared ? sh_w : w;
    __global const ushort *scales = shared ? sh_scales : ex_scales;
    __global const ushort *biases = shared ? sh_biases : ex_biases;
    const uint words = dim / 8, groups = dim / 64;
    const ulong base = shared ? 0 : (ulong)tl.x * 2 * ff;
    const uint col = get_group_id(0) * 16 + lane;
    const ulong wrow[2] = {base + col, base + ff + col};
    __global const half *xr[MC];
    __global const float *xs[MC];
    uint pr[MC];
    for (uint i = 0; i < MC; i++) {
        pr[i] = order[tl.y + (i < tl.z ? i : 0)];
        const uint row = pair_row(pr[i], n_pairs, top_k, false);
        xr[i] = x + (ulong)row * dim;
        xs[i] = x_sums + (ulong)row * groups;
    }
    float8 acc[2][2];
    for (uint rb = 0; rb < 2; rb++) acc[rb][0] = acc[rb][1] = 0.0f;
    TILE_DOT(2)
    for (uint rb = 0; rb < 2; rb++)
        for (uint r = 0; r < 8; r++) {
            const uint i = rb * 8 + r;
            if (i >= tl.z) break;
            const float gb = bf(to_bf(acc[rb][0][r])), ub = bf(to_bf(acc[rb][1][r]));
            act[(ulong)pr[i] * ff + col] = convert_half_rte(bf(to_bf(gb / (1.0f + exp(-gb)) * ub)));
        }
}

// Grouped down to fp32: y[p][r] as sage_expert_down. Grid (dim / 32, tile bound), one sub-group a group.
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void sage_tile_down(__global const uint *w, __global const ushort *ex_scales, __global const ushort *ex_biases,
                             __global const uint *sh_w, __global const ushort *sh_scales, __global const ushort *sh_biases,
                             __global const half *a_in, __global const float *a_sums, __global const uint *order,
                             __global const uint4 *tiles, __global const uint *ntiles, __global float *y,
                             uint dim, uint ff, uint n_experts) {
    const uint tile = get_group_id(1), lane = get_sub_group_local_id();
    if (tile >= ntiles[0]) return;
    const uint4 tl = tiles[tile];
    const bool shared = tl.x == n_experts;
    __global const uint *wt = shared ? sh_w : w;
    __global const ushort *scales = shared ? sh_scales : ex_scales;
    __global const ushort *biases = shared ? sh_biases : ex_biases;
    const uint words = ff / 8, groups = ff / 64;
    const ulong base = shared ? 0 : (ulong)tl.x * dim;
    const uint col = get_group_id(0) * 32 + lane;
    const ulong wrow[2] = {base + col, base + col + 16};
    __global const half *xr[MC];
    __global const float *xs[MC];
    uint pr[MC];
    for (uint i = 0; i < MC; i++) {
        pr[i] = order[tl.y + (i < tl.z ? i : 0)];
        xr[i] = a_in + (ulong)pr[i] * ff;
        xs[i] = a_sums + (ulong)pr[i] * groups;
    }
    float8 acc[2][2];
    for (uint rb = 0; rb < 2; rb++) acc[rb][0] = acc[rb][1] = 0.0f;
    TILE_DOT(2)
    for (uint rb = 0; rb < 2; rb++)
        for (uint r = 0; r < 8; r++) {
            const uint i = rb * 8 + r;
            if (i >= tl.z) break;
            y[(ulong)pr[i] * dim + col] = acc[rb][0][r];
            y[(ulong)pr[i] * dim + col + 16] = acc[rb][1][r];
        }
}

// sage_router for prompt rows on XMX: 16 rows x 16 experts a sub-group, fp16 DPAS into fp32. Column n_experts is the
// shared expert's gate. Grid ((n_experts + 16) / 16, (rows + 15) / 16), one sub-group a group.
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void sage_router_tile(__global const half *x, __global const half *gate, __global const half *shared_gate,
                               __global float *logits, uint dim, uint n_experts, uint rows) {
    const uint lane = get_sub_group_local_id(), e = get_group_id(0) * 16 + lane, r0 = get_group_id(1) * 16;
    __global const uint *wr = (__global const uint *)(e == n_experts ? shared_gate : gate + (ulong)min(e, n_experts - 1) * dim);
    __global const half *xr[16];
    for (uint i = 0; i < 16; i++) xr[i] = x + (ulong)min(r0 + i, rows - 1) * dim;
    float8 acc[2] = {0.0f, 0.0f};
    for (uint k0 = 0; k0 < dim; k0 += 16) {
        const int8 b = as_int8(vload8(0, wr + k0 / 2));
        for (uint rb = 0; rb < 2; rb++) {
            short8 a;
            for (uint r = 0; r < 8; r++) a[r] = as_short(xr[rb * 8 + r][k0 + lane]);
            acc[rb] = DPH(a, b, acc[rb]);
        }
    }
    if (e > n_experts) return;
    for (uint rb = 0; rb < 2; rb++)
        for (uint r = 0; r < 8; r++) {
            const uint row = r0 + rb * 8 + r;
            if (row < rows) logits[(ulong)row * (n_experts + 1) + e] = acc[rb][r];
        }
}

// ---------------------------------------------------------------- grouped decode MoE (several streams)
// A decode step of `rows` rows (one a stream, at most MAX_G). sage_group_plan groups the step's pairs by expert, and
// the group kernels do every pair of a group, so a step reads each expert's weights once. The shared expert is a
// group of all rows. act and y keep the pair index, as in decode.
#define MAX_G 16
#define GREC (2 + MAX_G) // a group record: expert (n_experts: the shared one), count, pair indices

// plan[0]: the group count; record g at plan + 1 + g * GREC. One work-group of 256 work-items, n_pairs <= 255.
__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void sage_group_plan(__global const uint *ids, __global uint *plan, uint n_experts, uint n_pairs, uint rows) {
    __local uint e_of[256];
    __local uint count;
    const uint k = get_local_id(0);
    if (k == 0) count = 0;
    e_of[k] = k < n_pairs ? ids[k] : 0xffffffffu;
    barrier(CLK_LOCAL_MEM_FENCE);
    bool lead = k < n_pairs;
    for (uint j = 0; j < k && lead; j++)
        if (e_of[j] == e_of[k]) lead = false;
    if (k == n_pairs) lead = true; // the shared expert's group
    if (lead) {
        __global uint *rec = plan + 1 + atomic_inc(&count) * GREC;
        uint n = 0;
        if (k == n_pairs) {
            for (uint i = 0; i < rows; i++) rec[2 + n++] = n_pairs + i;
            rec[0] = n_experts;
        } else {
            for (uint j = k; j < n_pairs; j++)
                if (e_of[j] == e_of[k] && n < MAX_G) rec[2 + n++] = j;
            rec[0] = e_of[k];
        }
        rec[1] = n;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (k == 0) plan[0] = count;
}

// Group k's pairs (in pr) and its expert; 0 past the last group.
inline uint group_of(__global const uint *plan, uint k, uint *e, uint *pr) {
    if (k >= plan[0]) return 0;
    __global const uint *rec = plan + 1 + k * GREC;
    *e = rec[0];
    const uint n = rec[1];
    for (uint i = 0; i < MAX_G; i++) pr[i] = i < n ? rec[2 + i] : 0;
    return n;
}

// The 32 values of quad c of a row (4 words), as fp32.
inline void unpack32(uint4 pk, float *v) {
    for (uint q = 0; q < 4; q++) {
        const uint u = ((const uint *)&pk)[q];
        for (uint j = 0; j < 8; j++) v[q * 8 + j] = (float)((u >> (4 * j)) & 15u);
    }
}

// gate|up and SwiGLU for every pair of a group: act[p][r] as sage_expert_swiglu. Grid (ff, n_pairs + 1: the most groups).
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void sage_group_swiglu(__global const uint *w, __global const ushort *ex_scales, __global const ushort *ex_biases,
                                __global const uint *sh_w, __global const ushort *sh_scales, __global const ushort *sh_biases,
                                __global const half *x, __global const uint *plan, __global half *act,
                                uint dim, uint ff, uint top_k, uint n_pairs, uint n_experts) {
    const uint r = get_group_id(0), k = get_group_id(1), lane = get_sub_group_local_id();
    uint pr[MAX_G], e;
    const uint n = group_of(plan, k, &e, pr);
    if (n == 0) return;
    const bool shared = e == n_experts;
    __global const uint *wt = shared ? sh_w : w;
    __global const ushort *scales = shared ? sh_scales : ex_scales;
    __global const ushort *biases = shared ? sh_biases : ex_biases;
    const ulong base = shared ? 0 : (ulong)e * 2 * ff, r0 = base + r, r1 = base + ff + r;
    const uint quads = dim / 32, groups = dim / 64;
    float a0[MAX_G], a1[MAX_G];
    for (uint i = 0; i < MAX_G; i++) a0[i] = a1[i] = 0.0f;
    for (uint c = lane; c < quads; c += SG) {
        float v0[32], v1[32];
        unpack32(vload4(c, wt + r0 * (dim / 8)), v0);
        unpack32(vload4(c, wt + r1 * (dim / 8)), v1);
        const float s0 = bf(scales[r0 * groups + c / 2]), b0 = bf(biases[r0 * groups + c / 2]);
        const float s1 = bf(scales[r1 * groups + c / 2]), b1 = bf(biases[r1 * groups + c / 2]);
        __attribute__((opencl_unroll_hint))
        for (uint i = 0; i < MAX_G; i++) {
            if (i >= n) break;
            __global const half *xr = x + (ulong)(pr[i] >= n_pairs ? pr[i] - n_pairs : pr[i] / top_k) * dim;
            float d0 = 0.0f, d1 = 0.0f, sx = 0.0f;
            for (uint q = 0; q < 4; q++) {
                const float8 xv = convert_float8(vload8(c * 4 + q, xr));
                for (uint j = 0; j < 8; j++) {
                    d0 = fma(xv[j], v0[q * 8 + j], d0);
                    d1 = fma(xv[j], v1[q * 8 + j], d1);
                    sx += xv[j];
                }
            }
            a0[i] += s0 * d0 + b0 * sx;
            a1[i] += s1 * d1 + b1 * sx;
        }
    }
    __attribute__((opencl_unroll_hint))
    for (uint i = 0; i < MAX_G; i++) {
        if (i >= n) break;
        const float gb = bf(to_bf(sub_group_reduce_add(a0[i]))), ub = bf(to_bf(sub_group_reduce_add(a1[i])));
        if (lane == 0) act[(ulong)pr[i] * ff + r] = convert_half_rte(bf(to_bf(gb / (1.0f + exp(-gb)) * ub)));
    }
}

// down for every pair of a group to fp32: y[p][r] as sage_expert_down. A lane owns one output row and reads all of it,
// and the act values are the same for the whole sub-group. Grid (dim / 16, n_pairs + 1: the most groups).
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void sage_group_down(__global const uint *w, __global const ushort *ex_scales, __global const ushort *ex_biases,
                              __global const uint *sh_w, __global const ushort *sh_scales, __global const ushort *sh_biases,
                              __global const half *a_in, __global const uint *plan, __global float *y,
                              uint dim, uint ff, uint n_experts) {
    const uint k = get_group_id(1), lane = get_sub_group_local_id(), r = get_group_id(0) * SG + lane;
    uint pr[MAX_G], e;
    const uint n = group_of(plan, k, &e, pr);
    if (n == 0) return;
    const bool shared = e == n_experts;
    __global const uint *wt = shared ? sh_w : w;
    __global const ushort *scales = shared ? sh_scales : ex_scales;
    __global const ushort *biases = shared ? sh_biases : ex_biases;
    const ulong row = (shared ? 0 : (ulong)e * dim) + r;
    const uint quads = ff / 32, groups = ff / 64;
    __global const uint *wr = wt + row * (ff / 8);
    float acc[MAX_G];
    for (uint i = 0; i < MAX_G; i++) acc[i] = 0.0f;
    for (uint c = 0; c < quads; c++) {
        float v[32];
        unpack32(vload4(c, wr), v);
        const float sc = bf(scales[row * groups + c / 2]), bi = bf(biases[row * groups + c / 2]);
        __attribute__((opencl_unroll_hint))
        for (uint i = 0; i < MAX_G; i++) {
            if (i >= n) break;
            __global const half *ar = a_in + (ulong)pr[i] * ff;
            float d = 0.0f, sx = 0.0f;
            for (uint u = 0; u < 4; u++) {
                const float8 xv = convert_float8(vload8(c * 4 + u, ar));
                for (uint j = 0; j < 8; j++) {
                    d = fma(xv[j], v[u * 8 + j], d);
                    sx += xv[j];
                }
            }
            acc[i] += sc * d + bi * sx;
        }
    }
    __attribute__((opencl_unroll_hint))
    for (uint i = 0; i < MAX_G; i++) {
        if (i >= n) break;
        y[(ulong)pr[i] * dim + r] = acc[i];
    }
}
