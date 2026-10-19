// Register-slot tree scan: the common CPU schedule preserves parent states and original output rows.
#pragma once
struct SlotStep { unsigned row; int read_slot, write_slot; };
// R adjacent value rows a warp, each with its own state and the same arithmetic; the rows share a step's loads.
template <int SLOTS, int R>
__device__ __forceinline__ void q35_slots(Geo geo, const __nv_bfloat16* q, const __nv_bfloat16* k,
        const __nv_bfloat16* v, const float* g, const __nv_bfloat16* beta, const Seg* segs,
        const SlotStep* steps, int layer, __nv_bfloat16* y, const int* P) {
    const int NK = geo.nk, NV = geo.nv;
    if (past_windows(P, blockIdx.z / NV)) return;
    const int hv = blockIdx.z % NV, hk = hv / (NV / NK);
    const Seg sg = segs[blockIdx.z / NV];
    const int dk = threadIdx.x, dv0 = (blockIdx.y * 4 + threadIdx.y) * R;
    constexpr int PT = DK / 32;
    const size_t at = ((size_t)hv * DK + dv0) * DK + PT * dk;  // row r's entries are r * DK further
    float* st = reinterpret_cast<float*>(sg.rec) + (size_t)layer * geo.rec() + at;
    float root[R][PT], kept[SLOTS][R][PT], state[R][PT];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int i = 0; i < PT; ++i) {
            root[r][i] = st[r * DK + i];
            if (sg.rows == 1) reinterpret_cast<float*>(sg.before)[(size_t)layer * geo.rec() + at + r * DK + i] = root[r][i];
        }
#pragma unroll
    for (int slot = 0; slot < SLOTS; ++slot)
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int i = 0; i < PT; ++i) kept[slot][r][i] = 0.f;
    const bool record = recorded(P);
    // row inputs as stored, the next row's loaded while this one computes
    struct In { uint2 k, q; __nv_bfloat16 v[R], beta; float g; };
    auto load = [&](unsigned row, In& x) {
        const size_t key = ((size_t)row * NK + hk) * DK + PT * dk;
        x.k = *reinterpret_cast<const uint2*>(k + key);
        x.q = *reinterpret_cast<const uint2*>(q + key);
#pragma unroll
        for (int r = 0; r < R; ++r) x.v[r] = v[((size_t)row * NV + hv) * DK + dv0 + r];
        x.g = g[row * NV + hv];
        x.beta = beta[row * NV + hv];
    };
    In next;
    SlotStep step_next;
    if (sg.rows > 0) {
        step_next = steps[sg.row0];
        load(step_next.row, next);
    }
    for (int item = 0; item < sg.rows; ++item) {
        const SlotStep step = step_next;
        const In in = next;
        if (item + 1 < sg.rows) {
            step_next = steps[sg.row0 + item + 1];
            load(step_next.row, next);
        }
        const int row = step.row, rr = row - sg.row0;
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int i = 0; i < PT; ++i) state[r][i] = root[r][i];
#pragma unroll
        for (int slot = 0; slot < SLOTS; ++slot) if (slot == step.read_slot)
#pragma unroll
            for (int r = 0; r < R; ++r)
#pragma unroll
                for (int i = 0; i < PT; ++i) state[r][i] = kept[slot][r][i];
        const __nv_bfloat16* kb = reinterpret_cast<const __nv_bfloat16*>(&in.k);
        const __nv_bfloat16* qb = reinterpret_cast<const __nv_bfloat16*>(&in.q);
        float kc[PT], qc[PT];
#pragma unroll
        for (int i = 0; i < PT; i++) {
            kc[i] = bf(kb[i]);
            qc[i] = bf(qb[i]);
        }
        const float gc = in.g;
        const float bc = bf(in.beta);
        float kv_mem[R], delta[R], out[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            kv_mem[r] = 0.0f;
#pragma unroll
            for (int i = 0; i < PT; i++) {
                state[r][i] = state[r][i] * gc;
                kv_mem[r] += state[r][i] * kc[i];
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) kv_mem[r] = warp_sum(kv_mem[r]);
#pragma unroll
        for (int r = 0; r < R; ++r) {
            delta[r] = (bf(in.v[r]) - kv_mem[r]) * bc;
            out[r] = 0.0f;
#pragma unroll
            for (int i = 0; i < PT; i++) {
                state[r][i] = fmaf(kc[i], delta[r], state[r][i]);
                out[r] += state[r][i] * qc[i];
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) out[r] = warp_sum(out[r]);
        if (dk == 0)
#pragma unroll
            for (int r = 0; r < R; ++r) y[((size_t)row * NV + hv) * DK + dv0 + r] = __float2bfloat16_rn(out[r]);
        if (record) {
            const LogRow l = log_row(geo, sg, layer, rr);
            if (dk == 0) {
#pragma unroll
                for (int r = 0; r < R; ++r) l.delta[hv * DK + dv0 + r] = delta[r];
                if (dv0 == 0) l.g[hv] = gc;
            }
            if (dv0 == 0 && hv % (NV / NK) == 0) *reinterpret_cast<uint2*>(l.k + hk * DK + PT * dk) = in.k;
        }
#pragma unroll
        for (int slot = 0; slot < SLOTS; ++slot) if (slot == step.write_slot)
#pragma unroll
            for (int r = 0; r < R; ++r)
#pragma unroll
                for (int i = 0; i < PT; ++i) kept[slot][r][i] = state[r][i];
    }
    if (record && sg.rows != 1) return;
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int i = 0; i < PT; ++i) st[r * DK + i] = state[r][i];
}
#define TF_Q35_SLOTS(N, R) \
extern "C" __global__ void __launch_bounds__(128) tf_q35_gdn_slots_##N(Geo geo, const __nv_bfloat16* q, \
 const __nv_bfloat16* k, const __nv_bfloat16* v, const float* g, const __nv_bfloat16* beta, const Seg* segs, \
 const SlotStep* steps, int layer, __nv_bfloat16* y, const int* P) { q35_slots<N, R>(geo, q, k, v, g, beta, segs, steps, layer, y, P); }
TF_Q35_SLOTS(1, 2)
TF_Q35_SLOTS(2, 2)
TF_Q35_SLOTS(4, 2)
TF_Q35_SLOTS(8, 1)
TF_Q35_SLOTS(16, 1)
