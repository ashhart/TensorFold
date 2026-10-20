// Register-slot tree scan: the common CPU schedule preserves parent states and original output rows.
#pragma once
struct SlotStep { unsigned row; int read_slot, write_slot; };
// spread_sum: N values' warp sums in one butterfly (warp_sum's pairs), value j where lane >> (5 - log2 N) == j.
template <int N>
__device__ __forceinline__ float spread_sum(float (&v)[N], int lane) {
#pragma unroll
    for (int level = 0; level < 5; ++level) {
        const int m = 16 >> level, h = N >> (level + 1);
        if (h >= 1) {
            const bool up = lane & m;
#pragma unroll
            for (int i = 0; i < h; ++i) {
                const float give = up ? v[i] : v[h + i], keep = up ? v[h + i] : v[i];
                v[i] = keep + __shfl_xor_sync(0xffffffffu, give, m);
            }
        } else {
            v[0] += __shfl_xor_sync(0xffffffffu, v[0], m);
        }
    }
    return v[0];
}

template <int N>
__host__ __device__ constexpr int q35_log2() { return N <= 1 ? 0 : 1 + q35_log2<N / 2>(); }

// R value rows a warp share a step's loads; its k . state sums reduce with the last step's outputs; slots in shared.
template <int SLOTS, int R>
__device__ __forceinline__ void q35_slots(Geo geo, const __nv_bfloat16* q, const __nv_bfloat16* k,
        const __nv_bfloat16* v, const float* g, const __nv_bfloat16* beta, const Seg* segs,
        const SlotStep* steps, int layer, __nv_bfloat16* y, const int* P) {
    __shared__ float4 kept[4][SLOTS * R * 32];
    const int NK = geo.nk, NV = geo.nv;
    if (past_windows(P, blockIdx.z / NV)) return;
    const int hv = blockIdx.z % NV, hk = hv / (NV / NK);
    const Seg sg = segs[blockIdx.z / NV];
    const int dk = threadIdx.x, dv0 = (blockIdx.y * 4 + threadIdx.y) * R;
    constexpr int PT = DK / 32;
    static_assert(PT == 4, "a lane's four state entries are one float4 slot");
    const size_t at = ((size_t)hv * DK + dv0) * DK + PT * dk;  // row r's entries are r * DK further
    float* st = reinterpret_cast<float*>(sg.rec) + (size_t)layer * geo.rec() + at;
    float4* slots = kept[threadIdx.y];
    float root[R][PT], state[R][PT];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int i = 0; i < PT; ++i) {
            root[r][i] = st[r * DK + i];
            if (sg.rows == 1) reinterpret_cast<float*>(sg.before)[(size_t)layer * geo.rec() + at + r * DK + i] = root[r][i];
        }
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
    constexpr int SH = 5 - q35_log2<2 * R>();
    const bool ylane = (dk & ((1 << SH) - 1)) == 0 && (dk >> SH) >= R;
    const int yr = (dk >> SH) - R;
    float pend[R];
    int pend_row = -1;
#pragma unroll
    for (int r = 0; r < R; ++r) pend[r] = 0.0f;
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int i = 0; i < PT; ++i) state[r][i] = root[r][i];
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
        for (int r = 0; r < R; ++r) {
            if (step.read_slot >= 0) {
                const float4 t = slots[(step.read_slot * R + r) * 32 + dk];
                state[r][0] = t.x;
                state[r][1] = t.y;
                state[r][2] = t.z;
                state[r][3] = t.w;
            } else {
#pragma unroll
                for (int i = 0; i < PT; ++i) state[r][i] = root[r][i];
            }
        }
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
        float both[2 * R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            float kv_mem = 0.0f;
#pragma unroll
            for (int i = 0; i < PT; i++) {
                state[r][i] = state[r][i] * gc;
                kv_mem += state[r][i] * kc[i];
            }
            both[r] = kv_mem;
            both[R + r] = pend[r];
        }
        const float tot = spread_sum<2 * R>(both, dk);
        if (pend_row >= 0 && ylane) y[((size_t)pend_row * NV + hv) * DK + dv0 + yr] = __float2bfloat16_rn(tot);
        float delta[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            delta[r] = (bf(in.v[r]) - __shfl_sync(0xffffffffu, tot, r << SH)) * bc;
            pend[r] = 0.0f;
#pragma unroll
            for (int i = 0; i < PT; i++) {
                state[r][i] = fmaf(kc[i], delta[r], state[r][i]);
                pend[r] += state[r][i] * qc[i];
            }
            if (step.write_slot >= 0) slots[(step.write_slot * R + r) * 32 + dk] = make_float4(state[r][0], state[r][1], state[r][2], state[r][3]);
        }
        pend_row = row;
        if (record) {
            const LogRow l = log_row(geo, sg, layer, rr);
            if (dk == 0) {
#pragma unroll
                for (int r = 0; r < R; ++r) l.delta[hv * DK + dv0 + r] = delta[r];
                if (dv0 == 0) l.g[hv] = gc;
            }
            if (dv0 == 0 && hv % (NV / NK) == 0) *reinterpret_cast<uint2*>(l.k + hk * DK + PT * dk) = in.k;
        }
        __syncwarp();
    }
    if (pend_row >= 0) {
        constexpr int SR = 5 - q35_log2<R>();
        const float tot = spread_sum<R>(pend, dk);
        if ((dk & ((1 << SR) - 1)) == 0) y[((size_t)pend_row * NV + hv) * DK + dv0 + (dk >> SR)] = __float2bfloat16_rn(tot);
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
