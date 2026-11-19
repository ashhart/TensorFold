// DeepSeek-V4.1's MoE route: sqrt(softplus(logit)) scores, the top-k of score plus bias. Variants via TF_DSV41_E/TOPK.
#include <metal_stdlib>
using namespace metal;

#ifndef TF_DSV41_PER
#define TF_DSV41_PER ((TF_DSV41_E + 31) / 32)
#endif

#ifndef TF_CTL
#define TF_CTL 0 // planted faults: 1 a higher id wins a tie, 2 a NaN key wins, 3 the bias in the weight, 4 a write past k
#endif

// The reference's softplus: the linear tail above 20 and the quadratic one below -10, as the CUDA router computes it.
inline float tf_dsv41_softplus(float z) {
    if (z != z) return z;
    const float e = metal::precise::exp(fmin(z, 20.0f));
    if (z > 20.0f) return z;
    if (z < -10.0f) return e - 0.5f * e * e;
    return metal::precise::log(1.0f + e);
}

// A candidate beats the incumbent when its key is larger, or the keys tie and its id is lower; NaN keys never win.
inline bool tf_dsv41_better(float as, int ai, float bs, int bi) {
    if (ai >= TF_DSV41_E) return false;
    if (bi >= TF_DSV41_E) return true;
    if (as != as) return TF_CTL == 2; // control 2: a NaN key beats every other
    if (bs != bs) return TF_CTL != 2;
    if (as > bs) return true;
    if (as < bs) return false;
    return TF_CTL == 1 ? ai > bi : ai < bi;
}

struct Dsv41RouteArgs {
    int rows;
    int topk;
    float scale;
    float eps;
};

// One simdgroup a row: the top-k of score plus bias (ties to the lower id), then each picked score over the sum.
[[kernel]] void tf_dsv41_route_select(const device float* LOGITS [[buffer(0)]], const device float* BIAS [[buffer(1)]],
                                      const device float* BIAS_VL [[buffer(2)]], const device uchar* IMAGE [[buffer(3)]],
                                      constant Dsv41RouteArgs& a [[buffer(4)]], device int* PICK [[buffer(5)]],
                                      device float* WTS [[buffer(6)]], uint row [[threadgroup_position_in_grid]],
                                      uint lane [[thread_index_in_simdgroup]]) {
    const int r = int(row);
    if (r >= a.rows || a.topk > TF_DSV41_TOPK) return;
    const device float* bias = IMAGE[r] != 0 ? BIAS_VL : BIAS;
    float s[TF_DSV41_PER], ch[TF_DSV41_PER];
    int eid[TF_DSV41_PER];
    for (int j = 0; j < TF_DSV41_PER; j++) {
        const int e = j * 32 + int(lane);
        if (e < TF_DSV41_E) {
            s[j] = metal::precise::sqrt(tf_dsv41_softplus(LOGITS[size_t(r) * TF_DSV41_E + e]));
            ch[j] = s[j] + bias[e];
            eid[j] = e;
        } else {
            s[j] = 0.0f;
            ch[j] = -INFINITY;
            eid[j] = -1;
        }
    }
    float w[TF_DSV41_TOPK];
    for (int k = 0; k < a.topk; k++) {
        // The lane's own best: its ids ascend with j, so a strict > keeps the lower id on a tie.
        float best = -INFINITY;
        int bid = TF_DSV41_E; // nothing yet
        for (int j = 0; j < TF_DSV41_PER; j++) {
            const bool nan_wins = TF_CTL == 2 && ch[j] != ch[j]; // control 2: a NaN key beats every other
            if (eid[j] >= 0 && (nan_wins || ch[j] > best || (ch[j] == best && eid[j] < bid))) { best = ch[j]; bid = eid[j]; }
        }
        // The butterfly: key descending then id ascending is a total order, so every lane ends with the same pick.
        for (int off = 16; off > 0; off /= 2) {
            const float ob = simd_shuffle_xor(best, off);
            const int oi = simd_shuffle_xor(bid, off);
            if (tf_dsv41_better(ob, oi, best, bid)) { best = ob; bid = oi; }
        }
        // The winner's own unbiased score: its lane holds it, and the other lanes take it from there.
        float mine = 0.0f;
        if (bid < TF_DSV41_E) {
            for (int j = 0; j < TF_DSV41_PER; j++) {
                if (eid[j] == bid) { mine = s[j]; eid[j] = -1; ch[j] = -INFINITY; } // the slot is spent
            }
        }
        w[k] = bid < TF_DSV41_E ? simd_shuffle(mine, uint(bid & 31)) : 0.0f;
        if (lane == 0) PICK[size_t(r) * a.topk + k] = bid < TF_DSV41_E ? bid : -1;
    }
    if (lane == 0) {
        float total = w[0];
        for (int k = 1; k < a.topk; k++) total = total + w[k];
        for (int k = 0; k < a.topk; k++) WTS[size_t(r) * a.topk + k] = (w[k] / (total + a.eps)) * a.scale;
        if (TF_CTL == 3) WTS[size_t(r) * a.topk] += bias[0]; // control 3: the bias added to the chosen weight
        if (TF_CTL == 4 && r == a.rows - 1) WTS[size_t(r) * a.topk + a.topk] = 1.0f; // control 4: one weight past the last row's picks
    }
}

struct Dsv41PlanArgs {
    int rows;
    int topk;
    float scale;
    float eps;
};

#define TF_DSV41_NOWHERE 0xFFFFFFFFu
// Simdgroups a row's scores take (two experts a lane), and rows routed at once in a group of 1024.
#define TF_DSV41_RSG ((TF_DSV41_E + 63) / 64)
#define TF_DSV41_RAT (32 / TF_DSV41_RSG)

// A candidate of a simdgroup's top-k: its key, expert id (TF_DSV41_E: none) and unbiased score.
struct Dsv41Cand {
    float key;
    int id;
    float s;
};

// Whether (bk, bi) comes before (ak, ai) in the route's order: key descending, then id ascending (both real keys).
inline bool tf_dsv41_before(float bk, int bi, float ak, int ai) {
    if (bk > ak) return true;
    if (bk < ak) return false;
    return TF_CTL == 1 ? bi > ai : bi < ai;
}

// The route's threadgroup scratch: candidates, picks, the plan's counters.
struct Dsv41PlanTg {
    Dsv41Cand cand[TF_DSV41_RAT][TF_DSV41_RSG * TF_DSV41_TOPK];
    Dsv41Cand picked[TF_DSV41_RAT][TF_DSV41_TOPK];
    uint ids[16 * TF_DSV41_TOPK];
    atomic_uint count[1024], cursor[1024];
    uint start[1024], part_n[32], part_s[32];
};

// route_select's picks and weights by rank (simdgroups' top-k, then their union's), then the plan; atomic logits.
inline void tf_dsv41_route_plan_body(const device float* LOGITS, const device float* BIAS, const device float* BIAS_VL,
                                     const device uchar* IMAGE, constant Dsv41PlanArgs& a, device int* PICK,
                                     device float* WTS, device uint* slots, device uint* nslots, device uint* pairs,
                                     device uint* where, threadgroup Dsv41PlanTg& m, uint t, uint sg, uint lane) {
    if (a.topk > TF_DSV41_TOPK) return;
    const int topk = a.topk;
    const uint rg = sg / TF_DSV41_RSG, sr = sg % TF_DSV41_RSG, tr = t - rg * TF_DSV41_RSG * 32;
    for (int r0 = 0; r0 < a.rows; r0 += TF_DSV41_RAT) {
        const int r = r0 + int(rg);
        const bool live = rg < TF_DSV41_RAT && r < a.rows && r < 16;
        // the simdgroup's top-k by rank among its 64 experts (lane l holds 64 sr + l and + 32); empty slots hold none
        if (live) {
            const device float* bias = IMAGE[r] != 0 ? BIAS_VL : BIAS;
            float s[2], key[2];
            bool real[2];
            int id[2];
            for (int j = 0; j < 2; ++j) {
                id[j] = int(sr * 64 + 32 * j + lane);
                float ch = NAN;
                s[j] = 0.0f;
                if (id[j] < TF_DSV41_E) {
                    // other groups of the launch wrote the logits: an atomic load never meets a stale cache line
                    const float z = as_type<float>(atomic_load_explicit((const device atomic_uint*)(LOGITS + size_t(r) * TF_DSV41_E + id[j]), memory_order_relaxed));
                    s[j] = metal::precise::sqrt(tf_dsv41_softplus(z));
                    ch = s[j] + bias[id[j]];
                }
                // control 2: a NaN key ranks first
                const bool nanw = TF_CTL == 2 && id[j] < TF_DSV41_E && ch != ch;
                real[j] = id[j] < TF_DSV41_E && (ch == ch || nanw);
                key[j] = nanw ? INFINITY : ch;
            }
            int rank[2] = {0, 0};
            for (ushort l = 0; l < 32; ++l) {
                for (int i = 0; i < 2; ++i) {
                    const float ok = simd_shuffle(key[i], l);
                    const bool oreal = simd_shuffle(real[i] ? 1u : 0u, l) != 0u;
                    const int oid = int(sr * 64 + 32 * i + l);
                    for (int j = 0; j < 2; ++j) rank[j] += (oreal && oid != id[j] && tf_dsv41_before(ok, oid, key[j], id[j])) ? 1 : 0;
                }
            }
            const int nreal = int(simd_sum((real[0] ? 1u : 0u) + (real[1] ? 1u : 0u)));
            for (int j = 0; j < 2; ++j)
                if (real[j] && rank[j] < topk) m.cand[rg][sr * topk + rank[j]] = Dsv41Cand{key[j], id[j], s[j]};
            if (int(lane) < topk && int(lane) >= nreal) m.cand[rg][sr * topk + lane] = Dsv41Cand{-INFINITY, TF_DSV41_E, 0.0f};
            if (sr == 0 && int(lane) < topk) m.picked[rg][lane] = Dsv41Cand{0.0f, -1, 0.0f};
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // the union's ranks: a candidate's place among every real candidate of its row
        if (live && tr < uint(TF_DSV41_RSG * topk)) {
            const Dsv41Cand me = m.cand[rg][tr];
            if (me.id < TF_DSV41_E) {
                int rank = 0;
                for (int j = 0; j < TF_DSV41_RSG * topk; ++j) {
                    const Dsv41Cand o = m.cand[rg][j];
                    rank += (o.id < TF_DSV41_E && j != int(tr) && tf_dsv41_before(o.key, o.id, me.key, me.id)) ? 1 : 0;
                }
                if (rank < topk) m.picked[rg][rank] = me;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (live && tr == 0) {
            const device float* bias = IMAGE[r] != 0 ? BIAS_VL : BIAS;
            float w[TF_DSV41_TOPK];
            for (int k = 0; k < topk; k++) {
                const Dsv41Cand q = m.picked[rg][k];
                w[k] = q.id >= 0 ? q.s : 0.0f;
                PICK[size_t(r) * a.topk + k] = q.id;
                m.ids[r * topk + k] = uint(q.id);
            }
            float total = w[0];
            for (int k = 1; k < topk; k++) total = total + w[k];
            for (int k = 0; k < topk; k++) WTS[size_t(r) * a.topk + k] = (w[k] / (total + a.eps)) * a.scale;
            if (TF_CTL == 3) WTS[size_t(r) * a.topk] += bias[0];
            if (TF_CTL == 4 && r == a.rows - 1) WTS[size_t(r) * a.topk + a.topk] = 1.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // experts_kimi's plan (tf_dsv41_route_plan) of the picks over every expert
    const uint np = uint(min(a.rows, 16) * topk), nt = 1024;
    for (uint x = t; x < 1024; x += nt) {
        atomic_store_explicit(&m.count[x], 0u, memory_order_relaxed);
        atomic_store_explicit(&m.cursor[x], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint p = t; p < np; p += nt) {
        const uint x = m.ids[p];
        if (x < uint(TF_DSV41_E)) atomic_fetch_add_explicit(&m.count[x], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint n = t < uint(TF_DSV41_E) ? atomic_load_explicit(&m.count[t], memory_order_relaxed) : 0u;
    const uint before_n = simd_prefix_exclusive_sum(n);
    const uint before_s = simd_prefix_exclusive_sum(n > 0 ? 1u : 0u);
    if (lane == 31) m.part_n[sg] = before_n + n, m.part_s[sg] = before_s + (n > 0 ? 1u : 0u);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint base_n = 0, base_s = 0;
    for (uint q = 0; q < sg; ++q) base_n += m.part_n[q], base_s += m.part_s[q];
    if (t < uint(TF_DSV41_E)) m.start[t] = base_n + before_n;
    if (t < uint(TF_DSV41_E) && n > 0) {
        slots[3 * (base_s + before_s)] = t;
        slots[3 * (base_s + before_s) + 1] = base_n + before_n;
        slots[3 * (base_s + before_s) + 2] = n;
    }
    if (t == nt - 1) {
        uint total = 0;
        for (uint q = 0; q < nt / 32; ++q) total += m.part_s[q];
        nslots[0] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint p = t; p < np; p += nt) {
        const uint x = m.ids[p];
        uint pos = TF_DSV41_NOWHERE;
        if (x < uint(TF_DSV41_E)) {
            pos = m.start[x] + atomic_fetch_add_explicit(&m.cursor[x], 1u, memory_order_relaxed);
            pairs[pos] = p;
        }
        where[p] = pos;
    }
}

struct Dsv41GateArgs {
    uint k, rows_group; // a group's rows: groups of 32 columns repeat for each rows_group-row slice
};

// The gate matvec's faults (the route's are TF_CTL): 1 non-finite x as 0, 2 steps lost, 3 fp16 sums, 4 a write past.
#ifndef TF_GCTL
#define TF_GCTL 0
#endif

inline float tf_dsv41_gx(bfloat v) {
    const float f = float(v);
    return (TF_GCTL == 1 && !isfinite(f)) ? 0.0f : f;
}

inline float tf_dsv41_gstep(float acc, float x, float w) {
    const float r = fma(x, w, acc);
    return TF_GCTL == 3 ? float(half(r)) : r;
}

// The gate's logits as mv.metal's matvec (k 0: given), a simdgroup a column; the last group to finish routes and plans.
[[kernel, max_total_threads_per_threadgroup(1024)]] void tf_dsv41_route_gate_plan(
    device float* LOGITS [[buffer(0)]], const device float* BIAS [[buffer(1)]],
    const device float* BIAS_VL [[buffer(2)]], const device uchar* IMAGE [[buffer(3)]],
    constant Dsv41PlanArgs& a [[buffer(4)]], device int* PICK [[buffer(5)]], device float* WTS [[buffer(6)]],
    device uint* slots [[buffer(7)]], device uint* nslots [[buffer(8)]], device uint* pairs [[buffer(9)]],
    device uint* where [[buffer(10)]], const device bfloat* X [[buffer(11)]], const device bfloat* GW [[buffer(12)]],
    constant Dsv41GateArgs& ga [[buffer(13)]], device atomic_uint* SYNC [[buffer(14)]],
    uint g [[threadgroup_position_in_grid]], uint ng [[threadgroups_per_grid]], uint t [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup Dsv41PlanTg m;
    threadgroup uint last[1];
    const uint col = (g % ((TF_DSV41_E + 31) / 32)) * 32 + sg, k = ga.k;
    const int rlo = int(g / ((TF_DSV41_E + 31) / 32) * ga.rows_group), rhi = min(a.rows, rlo + int(ga.rows_group));
    if (col < uint(TF_DSV41_E) && k > 0) {
        // up to four rows a pass share each step's weight loads; every row's lane chain keeps its order
        const device bfloat* w = GW + ulong(col) * k;
        for (int r0 = rlo; r0 < rhi; r0 += 4) {
            const int rn = min(4, rhi - r0);
            if (rn == 1) {
                // one row: 32 steps' loads a batch, as the matvec's one-row launch
                const device bfloat* x = X + ulong(r0) * k;
                float acc = 0.0f;
                uint i = lane;
                for (; i + 32u * 31u < k; i += 32u * 32u) {
                    if (TF_GCTL == 2 && i + 32u * 32u >= k) break;
                    float wv[32], xv[32];
                    for (uint u = 0; u < 32u; ++u) wv[u] = float(w[i + 32u * u]);
                    for (uint u = 0; u < 32u; ++u) xv[u] = tf_dsv41_gx(x[i + 32u * u]);
                    for (uint u = 0; u < 32u; ++u) acc = tf_dsv41_gstep(acc, xv[u], wv[u]);
                }
                for (; i < k; i += 32u) {
                    if (TF_GCTL == 2 && i - lane + 32u >= k) break;
                    acc = tf_dsv41_gstep(acc, tf_dsv41_gx(x[i]), float(w[i]));
                }
                for (ushort q = 1; q < 32; q *= 2) acc += simd_shuffle_xor(acc, q);
                if (lane == 0) atomic_store_explicit((device atomic_uint*)(LOGITS + size_t(r0) * TF_DSV41_E + col), as_type<uint>(acc), memory_order_relaxed);
                continue;
            }
            float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            uint i = lane;
            for (; i + 32u * 15u < k; i += 32u * 16u) {
                float wv[16];
                if (TF_GCTL == 2 && i + 32u * 16u >= k) break;
                for (uint u = 0; u < 16u; ++u) wv[u] = float(w[i + 32u * u]);
                for (int r = 0; r < 4; ++r) {
                    if (r >= rn) break;
                    const device bfloat* x = X + ulong(r0 + r) * k;
                    float xv[16];
                    for (uint u = 0; u < 16u; ++u) xv[u] = tf_dsv41_gx(x[i + 32u * u]);
                    for (uint u = 0; u < 16u; ++u) acc[r] = tf_dsv41_gstep(acc[r], xv[u], wv[u]);
                }
            }
            for (; i < k; i += 32u) {
                if (TF_GCTL == 2 && i - lane + 32u >= k) break;
                const float wv = float(w[i]);
                for (int r = 0; r < rn; ++r) acc[r] = tf_dsv41_gstep(acc[r], tf_dsv41_gx(X[ulong(r0 + r) * k + i]), wv);
            }
            for (int r = 0; r < rn; ++r) {
                float v = acc[r];
                for (ushort q = 1; q < 32; q *= 2) v += simd_shuffle_xor(v, q);
                if (lane == 0) atomic_store_explicit((device atomic_uint*)(LOGITS + size_t(r0 + r) * TF_DSV41_E + col), as_type<uint>(v), memory_order_relaxed);
            }
        }
        if (TF_GCTL == 4 && col == 0 && lane == 0) LOGITS[size_t(a.rows) * TF_DSV41_E] = 1.0f;
    }
    // the last group to finish routes: every other group's logits are visible after its acquire
    threadgroup_barrier(mem_flags::mem_device);
    if (t == 0) {
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        const uint done = atomic_fetch_add_explicit(SYNC, 1u, memory_order_relaxed);
        last[0] = done == ng - 1 ? 1u : 0u;
        if (last[0] != 0u) atomic_store_explicit(SYNC, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (last[0] == 0u) return;
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    tf_dsv41_route_plan_body(LOGITS, BIAS, BIAS_VL, IMAGE, a, PICK, WTS, slots, nslots, pairs, where, m, t, sg, lane);
}
