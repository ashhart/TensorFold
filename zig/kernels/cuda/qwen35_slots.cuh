// Register-slot tree scan: the common CPU schedule preserves parent states and original output rows.
#pragma once
struct SlotStep { unsigned row; int read_slot, write_slot; };
template <int SLOTS>
__device__ __forceinline__ void q35_slots(Geo geo, const __nv_bfloat16* q, const __nv_bfloat16* k,
        const __nv_bfloat16* v, const float* g, const __nv_bfloat16* beta, const Seg* segs,
        const SlotStep* steps, int layer, __nv_bfloat16* y, const int* P) {
    const int NK = geo.nk, NV = geo.nv;
    if (past_windows(P, blockIdx.z / NV)) return;
    const int hv = blockIdx.z % NV, hk = hv / (NV / NK);
    const Seg sg = segs[blockIdx.z / NV];
    const int dk = threadIdx.x, dv = blockIdx.y * 4 + threadIdx.y;
    constexpr int PT = DK / 32;
    const size_t at = ((size_t)hv * DK + dv) * DK + PT * dk;
    float* st = reinterpret_cast<float*>(sg.rec) + (size_t)layer * geo.rec() + at;
    float root[PT], kept[SLOTS][PT], state[PT];
#pragma unroll
    for (int i = 0; i < PT; ++i) {
        root[i] = st[i];
        if (sg.rows == 1) reinterpret_cast<float*>(sg.before)[(size_t)layer * geo.rec() + at + i] = root[i];
    }
#pragma unroll
    for (int slot = 0; slot < SLOTS; ++slot)
#pragma unroll
        for (int i = 0; i < PT; ++i) kept[slot][i] = 0.f;
    const bool record = recorded(P);
    // row inputs as stored, the next row's loaded while this one computes
    struct In { uint2 k, q; __nv_bfloat16 v, beta; float g; };
    auto load = [&](unsigned row, In& x) {
        const size_t key = ((size_t)row * NK + hk) * DK + PT * dk;
        x.k = *reinterpret_cast<const uint2*>(k + key);
        x.q = *reinterpret_cast<const uint2*>(q + key);
        x.v = v[((size_t)row * NV + hv) * DK + dv];
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
        const int row = step.row, r = row - sg.row0;
#pragma unroll
        for (int i = 0; i < PT; ++i) state[i] = root[i];
#pragma unroll
        for (int slot = 0; slot < SLOTS; ++slot) if (slot == step.read_slot)
#pragma unroll
            for (int i = 0; i < PT; ++i) state[i] = kept[slot][i];
        const __nv_bfloat16* kb = reinterpret_cast<const __nv_bfloat16*>(&in.k);
        const __nv_bfloat16* qb = reinterpret_cast<const __nv_bfloat16*>(&in.q);
        float kc[PT], qc[PT];
#pragma unroll
        for (int i = 0; i < PT; i++) {
            kc[i] = bf(kb[i]);
            qc[i] = bf(qb[i]);
        }
        const float vc = bf(in.v);
        const float gc = in.g;
        const float bc = bf(in.beta);
        float kv_mem = 0.0f;
#pragma unroll
        for (int i = 0; i < PT; i++) {
            state[i] = state[i] * gc;
            kv_mem += state[i] * kc[i];
        }
        kv_mem = warp_sum(kv_mem);
        const float delta = (vc - kv_mem) * bc;
        float out = 0.0f;
#pragma unroll
        for (int i = 0; i < PT; i++) {
            state[i] = fmaf(kc[i], delta, state[i]);
            out += state[i] * qc[i];
        }
        out = warp_sum(out);
        if (dk == 0) y[((size_t)row * NV + hv) * DK + dv] = __float2bfloat16_rn(out);
        if (record) {
            const LogRow l = log_row(geo, sg, layer, r);
            if (dk == 0) {
                l.delta[hv * DK + dv] = delta;
                if (dv == 0) l.g[hv] = gc;
            }
            if (dv == 0 && hv % (NV / NK) == 0) *reinterpret_cast<uint2*>(l.k + hk * DK + PT * dk) = in.k;
        }
#pragma unroll
        for (int slot = 0; slot < SLOTS; ++slot) if (slot == step.write_slot)
#pragma unroll
            for (int i = 0; i < PT; ++i) kept[slot][i] = state[i];
    }
    if (record && sg.rows != 1) return;
#pragma unroll
    for (int i = 0; i < PT; ++i) st[i] = state[i];
}
#define TF_Q35_SLOTS(N) \
extern "C" __global__ void __launch_bounds__(128) tf_q35_gdn_slots_##N(Geo geo, const __nv_bfloat16* q, \
 const __nv_bfloat16* k, const __nv_bfloat16* v, const float* g, const __nv_bfloat16* beta, const Seg* segs, \
 const SlotStep* steps, int layer, __nv_bfloat16* y, const int* P) { q35_slots<N>(geo, q, k, v, g, beta, segs, steps, layer, y, P); }
TF_Q35_SLOTS(1)
TF_Q35_SLOTS(2)
TF_Q35_SLOTS(4)
TF_Q35_SLOTS(8)
TF_Q35_SLOTS(16)
