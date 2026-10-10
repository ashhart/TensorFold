// Nemotron MoE decode ops: router logits, top-k routing, 4-bit expert matvecs (stacked tables, expert id from a slot list), combine.
// Activations are bf16, math is fp32; bf16 rounding points follow the CUDA reference (tensorfold experts.cuh, glue.py).
inline float bf(ushort v) { return as_float((uint)v << 16); }
inline ushort to_bf(float f) {
    uint u = as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (ushort)(u >> 16);
}

#define SG 16
#define MAX_PER_LANE 16 // router supports up to SG * MAX_PER_LANE = 256 experts
#define MAX_TOPK 8

// bf16 logits = x . gate_row, one sub-group per (expert, row); fp32 accumulation, rounded to bf16 once. A lane adds its elements lane, lane + 16, ... in turn; for dim == 2688 the loads
// of RB elements are issued together (static trip counts: the plain loop waits for every load in turn), the fmas stay in the same order.
#define RB 24
inline void router_logit_row(__global const ushort *x, __global const ushort *gate, __global ushort *logits, uint dim, uint n_experts, uint e, uint row) {
    const uint lane = get_sub_group_local_id();
    float acc = 0.0f;
    if (dim == 2688) {
        for (uint b = 0; b < 168 / RB; b++) {
            ushort xa[RB], ga[RB];
#pragma unroll
            for (uint j = 0; j < RB; j++) {
                const uint i = lane + SG * (b * RB + j);
                xa[j] = x[row * dim + i];
                ga[j] = gate[e * dim + i];
            }
#pragma unroll
            for (uint j = 0; j < RB; j++) acc = fma(bf(xa[j]), bf(ga[j]), acc);
        }
    } else {
        for (uint i = lane; i < dim; i += SG) acc = fma(bf(x[row * dim + i]), bf(gate[e * dim + i]), acc);
    }
    acc = sub_group_reduce_add(acc);
    if (lane == 0) logits[row * n_experts + e] = to_bf(acc);
}

__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void router_logits(__global const ushort *x, __global const ushort *gate, __global ushort *logits, uint dim, uint n_experts) {
    router_logit_row(x, gate, logits, dim, n_experts, get_group_id(0), get_group_id(1));
}

// Per row: sigmoid scores, top-k of score + bias (ties to the lower id), weights = score / (sum + 1e-20) * scaling in pick order.
// One sub-group per row; lane l holds experts l, l + 16, ...; n_experts must be a multiple of 16.
inline void route_row(__global const ushort *logits, __global const float *bias, __global uint *ids, __global float *wts,
                      uint n_experts, uint top_k, uint scaling_bits, uint row) {
    const uint lane = get_sub_group_local_id();
    const uint per = n_experts / SG;
    float sel[MAX_PER_LANE], prob[MAX_PER_LANE], picked[MAX_TOPK];
    for (uint j = 0; j < per; j++) {
        const uint e = lane + SG * j;
        prob[j] = 1.0f / (1.0f + exp(-bf(logits[row * n_experts + e])));
        sel[j] = prob[j] + bias[e];
    }
    float total = 0.0f;
    for (uint k = 0; k < top_k; k++) {
        float best = -INFINITY;
        uint best_e = 0xffffffffu;
        for (uint j = 0; j < per; j++) {
            if (sel[j] > best) { best = sel[j]; best_e = lane + SG * j; }
        }
        const float top = sub_group_reduce_max(best);
        const uint winner = sub_group_reduce_min(best == top ? best_e : 0xffffffffu);
        float p = 0.0f;
        for (uint j = 0; j < per; j++) {
            if (lane + SG * j == winner) { p = prob[j]; sel[j] = -INFINITY; }
        }
        p = sub_group_reduce_add(p);
        picked[k] = p;
        total += p;
        if (lane == 0) ids[row * top_k + k] = winner;
    }
    if (lane == 0) {
        const float denom = total + 1e-20f;
        const float scaling = as_float(scaling_bits);
        for (uint k = 0; k < top_k; k++) wts[row * top_k + k] = picked[k] / denom * scaling;
    }
}

__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void moe_route(__global const ushort *logits, __global const float *bias, __global uint *ids, __global float *wts,
                        uint n_experts, uint top_k, uint scaling_bits) {
    route_row(logits, bias, ids, wts, n_experts, top_k, scaling_bits, get_group_id(0));
}

// 4-bit affine dot of one row of table `row` with bf16 x (a sub-group cooperates; the sum is valid on every lane).
inline float qdot(__global const uint *w, __global const ushort *scales, __global const ushort *biases, __global const ushort *x,
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
            const float xv = bf(x[wi * 8 + j]);
            dot += xv * (float)((pk >> (4 * j)) & 15u);
            sx += xv;
        }
        acc += sc * dot + bi * sx;
    }
    return sub_group_reduce_add(acc);
}

// Expert up projection with relu2 epilogue: grid (n_rows, slots). Slot s uses expert slot_ids[s] of the stacked tables
// [experts][n_rows][in_dim]; x is shared by all slots (x_stride 0) or one row per slot. out[s][n_rows] bf16 = bf16(relu(bf16(acc))^2).
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void expert_up_relu2(__global const uint *w, __global const ushort *scales, __global const ushort *biases,
                              __global const ushort *x, __global const uint *slot_ids, __global ushort *out,
                              uint in_dim, uint n_rows, uint x_stride) {
    const uint r = get_group_id(0), s = get_group_id(1);
    const ulong row = (ulong)slot_ids[s] * n_rows + r;
    const float acc = qdot(w, scales, biases, x + (ulong)s * x_stride, row, in_dim);
    const float u = fmax(bf(to_bf(acc)), 0.0f);
    if (get_sub_group_local_id() == 0) out[(ulong)s * n_rows + r] = to_bf(u * u);
}

// Expert down projection to fp32: grid (n_rows, slots); x holds one bf16 row of in_dim per slot, out[s][n_rows] fp32.
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void expert_down_f32(__global const uint *w, __global const ushort *scales, __global const ushort *biases,
                              __global const ushort *x, __global const uint *slot_ids, __global float *out,
                              uint in_dim, uint n_rows, uint x_stride) {
    const uint r = get_group_id(0), s = get_group_id(1);
    const ulong row = (ulong)slot_ids[s] * n_rows + r;
    const float acc = qdot(w, scales, biases, x + (ulong)s * x_stride, row, in_dim);
    if (get_sub_group_local_id() == 0) out[(ulong)s * n_rows + r] = acc;
}

// Block output for one row: bf16(sum_k fma(y[k], wts[k]) + shared), sum in slot order. Grid: ceil(dim / 64) groups of 64.
__kernel void moe_combine(__global const float *y, __global const float *wts, __global const float *shared, __global ushort *out,
                          uint dim, uint slots) {
    const uint i = get_global_id(0);
    if (i >= dim) return;
    float acc = 0.0f;
    for (uint k = 0; k < slots; k++) acc = fma(y[(ulong)k * dim + i], wts[k], acc);
    out[i] = to_bf(acc + shared[i]);
}

// The routed experts and the shared expert in one launch (flat grid: n_rows * slots routed rows, then sh_rows shared rows); every row is the value
// expert_up_relu2 computes (shared: expert 0 of its own table, x shared by all rows).
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void expert_up_relu2_sh(__global const uint *w, __global const ushort *scales, __global const ushort *biases,
                                 __global const uint *sw, __global const ushort *sscales, __global const ushort *sbiases,
                                 __global const ushort *x, __global const uint *slot_ids, __global ushort *out, __global ushort *sh_out,
                                 uint in_dim, uint n_rows, uint slots) {
    const uint g = get_group_id(0), routed = n_rows * slots;
    float acc;
    __global ushort *dst;
    if (g < routed) {
        const uint s = g / n_rows, r = g % n_rows;
        acc = qdot(w, scales, biases, x, (ulong)slot_ids[s] * n_rows + r, in_dim);
        dst = out + (ulong)s * n_rows + r;
    } else {
        const uint r = g - routed;
        acc = qdot(sw, sscales, sbiases, x, r, in_dim);
        dst = sh_out + r;
    }
    const float u = fmax(bf(to_bf(acc)), 0.0f);
    if (get_sub_group_local_id() == 0) *dst = to_bf(u * u);
}

// Down projections of the routed experts (act row per slot, stride in_dim) and of the shared expert (sact) in one launch: fp32 out[s][n_rows], then sh_out[n_rows].
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void expert_down_f32_sh(__global const uint *w, __global const ushort *scales, __global const ushort *biases,
                                 __global const uint *sw, __global const ushort *sscales, __global const ushort *sbiases,
                                 __global const ushort *x, __global const ushort *sx, __global const uint *slot_ids, __global float *out, __global float *sh_out,
                                 uint in_dim, uint n_rows, uint slots, uint sh_in) {
    const uint g = get_group_id(0), routed = n_rows * slots;
    float acc;
    __global float *dst;
    if (g < routed) {
        const uint s = g / n_rows, r = g % n_rows;
        acc = qdot(w, scales, biases, x + (ulong)s * in_dim, (ulong)slot_ids[s] * n_rows + r, in_dim);
        dst = out + (ulong)s * n_rows + r;
    } else {
        const uint r = g - routed;
        acc = qdot(sw, sscales, sbiases, sx, r, sh_in);
        dst = sh_out + r;
    }
    if (get_sub_group_local_id() == 0) *dst = acc;
}

// Decode (one row): the router logits and the shared expert's up projection in one launch (work-group g < n_experts: logit g; the others: shared row g - n_experts, relu2). The two are independent,
// so the small router kernel no longer sits alone in the dependency chain.
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void router_up_sh(__global const ushort *x, __global const ushort *gate, __global ushort *logits, uint dim, uint n_experts,
                           __global const uint *sw, __global const ushort *sscales, __global const ushort *sbiases, __global ushort *sact) {
    const uint g = get_group_id(0);
    if (g < n_experts) {
        router_logit_row(x, gate, logits, dim, n_experts, g, 0);
        return;
    }
    const uint r = g - n_experts;
    const float acc = qdot(sw, sscales, sbiases, x, r, dim);
    const float u = fmax(bf(to_bf(acc)), 0.0f);
    if (get_sub_group_local_id() == 0) sact[r] = to_bf(u * u);
}

// Decode (one row): top-k routing (work-group 0) and the shared expert's down projection (work-group r + 1: output row r, fp32) in one launch.
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void route_down_sh(__global const ushort *logits, __global const float *bias, __global uint *ids, __global float *wts, uint n_experts, uint top_k, uint scaling_bits,
                            __global const uint *sw, __global const ushort *sscales, __global const ushort *sbiases, __global const ushort *sact, __global float *sy, uint sh_in) {
    const uint g = get_group_id(0);
    if (g == 0) {
        route_row(logits, bias, ids, wts, n_experts, top_k, scaling_bits, 0);
        return;
    }
    const float acc = qdot(sw, sscales, sbiases, sact, g - 1, sh_in);
    if (get_sub_group_local_id() == 0) sy[g - 1] = acc;
}

// The MoE block's combine and the next layer's residual add + RMSNorm in one launch (one work-group of 64 per row): delta = bf16(sum_k fma(ey[k], wts[k]) + sy) as moe_combine, then
// x = bf16(x + delta) and y = rmsnorm(x) as add_rmsnorm (same sums in the same order). Row r reads ey[(r * top_k + k) * n + i], wts[r * top_k + k], sy[r * n + i]. For n == 64 * CK and
// top_k == 6 the loads of a group of EK elements are issued together (the plain loops wait for each load in turn); other shapes take the plain loops.
#define CK 42
#define EK 7
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void add_comb_rmsnorm(__global ushort *x, __global const float *ey, __global const float *wts, __global const float *sy, __global const ushort *w, __global ushort *y,
                               uint n, uint top_k, float eps) {
    __local float part[4];
    const uint row = get_group_id(0);
    const uint lid = get_local_id(0);
    float ss = 0.0f;
    ushort xa[CK], wa[CK];
    if (n == 64 * CK && top_k == 6) {
        float wt[6];
#pragma unroll
        for (uint k = 0; k < 6; k++) wt[k] = wts[row * 6 + k];
#pragma unroll
        for (uint g = 0; g < CK / EK; g++) {
            float ya[EK][6], sa[EK];
            ushort xl[EK];
#pragma unroll
            for (uint j = 0; j < EK; j++) {
                const uint i = lid + 64 * (g * EK + j);
#pragma unroll
                for (uint k = 0; k < 6; k++) ya[j][k] = ey[((ulong)(row * 6 + k)) * n + i];
                sa[j] = sy[(ulong)row * n + i];
                xl[j] = x[(ulong)row * n + i];
                wa[g * EK + j] = w[i];
            }
#pragma unroll
            for (uint j = 0; j < EK; j++) {
                const uint i = lid + 64 * (g * EK + j);
                float acc = 0.0f;
#pragma unroll
                for (uint k = 0; k < 6; k++) acc = fma(ya[j][k], wt[k], acc);
                const ushort delta = to_bf(acc + sa[j]);
                const ushort s = to_bf(bf(xl[j]) + bf(delta));
                xa[g * EK + j] = s;
                x[(ulong)row * n + i] = s;
                const float v = bf(s);
                ss += v * v;
            }
        }
    } else {
        for (uint i = lid; i < n; i += 64) {
            float acc = 0.0f;
            for (uint k = 0; k < top_k; k++) acc = fma(ey[((ulong)(row * top_k + k)) * n + i], wts[row * top_k + k], acc);
            const ushort delta = to_bf(acc + sy[(ulong)row * n + i]);
            const ushort s = to_bf(bf(x[(ulong)row * n + i]) + bf(delta));
            x[(ulong)row * n + i] = s;
            const float v = bf(s);
            ss += v * v;
        }
    }
    ss = sub_group_reduce_add(ss);
    if (get_sub_group_local_id() == 0) part[get_sub_group_id()] = ss;
    barrier(CLK_LOCAL_MEM_FENCE);
    const float total = part[0] + part[1] + part[2] + part[3];
    const float scale = rsqrt(total / (float)n + eps);
    if (n == 64 * CK && top_k == 6) {
#pragma unroll
        for (uint k = 0; k < CK; k++) y[(ulong)row * n + lid + 64 * k] = to_bf(bf(xa[k]) * scale * bf(wa[k]));
    } else {
        for (uint i = lid; i < n; i += 64) y[(ulong)row * n + i] = to_bf(bf(x[(ulong)row * n + i]) * scale * bf(w[i]));
    }
}
