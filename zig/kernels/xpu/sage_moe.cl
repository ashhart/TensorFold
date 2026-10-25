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
// Grid (n_experts + 1, rows), one sub-group a work-group.
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void sage_router(__global const half *x, __global const half *gate, __global const half *shared_gate,
                          __global float *logits, uint dim) {
    const uint e = get_group_id(0), row = get_group_id(1), n = get_num_groups(0);
    const uint lane = get_sub_group_local_id();
    __global const half *w = e + 1 == n ? shared_gate : gate + (ulong)e * dim;
    float acc = 0.0f;
    for (uint i = lane; i < dim; i += SG) acc = fma((float)x[row * dim + i], vload_half(i, w), acc);
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
inline float qdot(__global const uint *w, __global const ushort *scales, __global const ushort *biases, __global const half *x,
                  ulong row, uint in_dim) {
    const uint lane = get_sub_group_local_id();
    const uint words = in_dim / 8, groups = in_dim / 64;
    float acc = 0.0f;
    for (uint wi = lane; wi < words; wi += SG) {
        const uint g = wi / 8;
        const float sc = bf(scales[row * groups + g]);
        const float bi = bf(biases[row * groups + g]);
        const uint pk = w[row * words + wi];
        float dot = 0.0f, sx = 0.0f;
        for (uint j = 0; j < 8; j++) {
            const float xv = (float)x[wi * 8 + j];
            dot += xv * (float)((pk >> (4 * j)) & 15u);
            sx += xv;
        }
        acc += sc * dot + bi * sx;
    }
    return sub_group_reduce_add(acc);
}

// gate and up of one expert pair and SwiGLU: act[p][r] = fp16(bf16(silu(bf16(gate)) * bf16(up))). Grid (ff, pairs).
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
    const float g = shared ? qdot(sh_w, sh_scales, sh_biases, xr, r, dim) : qdot(w, scales, biases, xr, base + r, dim);
    const float u = shared ? qdot(sh_w, sh_scales, sh_biases, xr, ff + r, dim) : qdot(w, scales, biases, xr, base + ff + r, dim);
    const float gb = bf(to_bf(g)), ub = bf(to_bf(u));
    if (get_sub_group_local_id() == 0) act[(ulong)p * ff + r] = convert_half_rte(bf(to_bf(gb / (1.0f + exp(-gb)) * ub)));
}

// down of one expert pair to fp32: y[p][r]. Grid (dim, pairs); pairs as in sage_expert_swiglu.
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void sage_expert_down(__global const uint *w, __global const ushort *scales, __global const ushort *biases,
                               __global const uint *sh_w, __global const ushort *sh_scales, __global const ushort *sh_biases,
                               __global const half *act, __global const uint *ids, __global float *y,
                               uint dim, uint ff, uint n_pairs) {
    const uint r = get_group_id(0), p = get_group_id(1);
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
