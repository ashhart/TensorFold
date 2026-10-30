#pragma once

// A decode step's expert picks and the router launch that also picks; decode.hip includes it after its router.

namespace {

constexpr int kSelectThreads = 512;
constexpr int kSelectMax = 1024;

__device__ __forceinline__ float round_bf16_f(float v) { return round_kind(v, 2); }

// act.hip's moe_select with picks by rank: expert i is in the top k when fewer than k beat it; `items` writes the plan.
__device__ void select_body(const float* logits, int* pick, float* wts, int* items, int* members, int capacity,
                            int experts, int top_k) {
    __shared__ float v[kSelectMax];
    __shared__ float top_v[32];
    __shared__ int top_i[32];
    const int row = blockIdx.x;
    const int slots = top_k + 1;
    const float* l = logits + static_cast<long long>(row) * (experts + 1);
    for (int i = threadIdx.x; i < experts; i += kSelectThreads) v[i] = l[i];
    __syncthreads();
    for (int i = threadIdx.x; i < experts; i += kSelectThreads) {
        const float vi = v[i];
        int rank = 0;
#pragma unroll 16
        for (int j = 0; j < experts; ++j) {
            const float vj = v[j];
            rank += (vj > vi) || (vj == vi && j < i);
        }
        if (rank < top_k) {
            top_v[rank] = vi;
            top_i[rank] = i;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const float top = top_v[0];
        float ex[32];
        float total = 0.f;
        for (int k = 0; k < top_k; ++k) {
            ex[k] = expf(top_v[k] - top);
            total += ex[k];
        }
        for (int k = 0; k < top_k; ++k) {
            pick[row * slots + k] = top_i[k];
            wts[row * slots + k] = round_bf16_f(ex[k] / total);
        }
        const float gate = round_bf16_f(l[experts]);
        pick[row * slots + top_k] = experts;
        wts[row * slots + top_k] = round_bf16_f(1.f / (1.f + expf(-gate)));
    }
    if (items != nullptr) {
        __syncthreads();
        for (int k = threadIdx.x; k < capacity; k += kSelectThreads) {
            const bool real = k < slots;
            items[3 * k] = real ? pick[k] : 0;
            items[3 * k + 1] = real ? k : 0;
            items[3 * k + 2] = real ? 1 : 0;
            if (real) members[k] = k;
        }
    }
}

// select_body's pick of a row by the router's block, k argmax passes (NaN: rank loop); coherent loads: others wrote it.
__device__ void select_row(const float* logits, int row, int* pick, float* wts, int* items, int* members, int capacity,
                           int experts, int top_k) {
    constexpr int kThreads = 32 * kRouterWaves;
    __shared__ float v[kSelectMax];
    __shared__ unsigned char taken[kSelectMax];
    __shared__ float top_v[32];
    __shared__ int top_i[32];
    __shared__ float wave_v[kRouterWaves];
    __shared__ int wave_i[kRouterWaves];
    __shared__ int has_nan;
    const int slots = top_k + 1;
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    float* l = const_cast<float*>(logits) + static_cast<long long>(row) * (experts + 1);
    if (threadIdx.x == 0) has_nan = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < experts; i += kThreads) {
        const float vi = __hip_atomic_load(l + i, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        v[i] = vi;
        taken[i] = 0;
        if (vi != vi) has_nan = 1;
    }
    __syncthreads();
    if (has_nan) {
        for (int i = threadIdx.x; i < experts; i += kThreads) {
            const float vi = v[i];
            int rank = 0;
            for (int j = 0; j < experts; ++j) {
                const float vj = v[j];
                rank += (vj > vi) || (vj == vi && j < i);
            }
            if (rank < top_k) {
                top_v[rank] = vi;
                top_i[rank] = i;
            }
        }
    } else {
        for (int k = 0; k < top_k; ++k) {
            float bv = 0.f;
            int bi = 0x7fffffff;
            for (int i = threadIdx.x; i < experts; i += kThreads) {
                const float vi = v[i];
                if (!taken[i] && (bi == 0x7fffffff || vi > bv || (vi == bv && i < bi))) {
                    bv = vi;
                    bi = i;
                }
            }
#pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                const float ov = __shfl_xor(bv, mask, 32);
                const int oi = __shfl_xor(bi, mask, 32);
                if (oi != 0x7fffffff && (bi == 0x7fffffff || ov > bv || (ov == bv && oi < bi))) {
                    bv = ov;
                    bi = oi;
                }
            }
            if (lane == 0) {
                wave_v[wave] = bv;
                wave_i[wave] = bi;
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                float cv = wave_v[0];
                int ci = wave_i[0];
                for (int w = 1; w < kRouterWaves; ++w) {
                    const float ov = wave_v[w];
                    const int oi = wave_i[w];
                    if (oi != 0x7fffffff && (ci == 0x7fffffff || ov > cv || (ov == cv && oi < ci))) {
                        cv = ov;
                        ci = oi;
                    }
                }
                top_v[k] = cv;
                top_i[k] = ci;
                taken[ci] = 1;
            }
            __syncthreads();
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const float top = top_v[0];
        float ex[32];
        float total = 0.f;
        for (int k = 0; k < top_k; ++k) {
            ex[k] = expf(top_v[k] - top);
            total += ex[k];
        }
        for (int k = 0; k < top_k; ++k) {
            pick[row * slots + k] = top_i[k];
            wts[row * slots + k] = round_bf16_f(ex[k] / total);
        }
        const float gate = round_bf16_f(__hip_atomic_load(l + experts, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT));
        pick[row * slots + top_k] = experts;
        wts[row * slots + top_k] = round_bf16_f(1.f / (1.f + expf(-gate)));
    }
    if (items != nullptr) {
        __syncthreads();
        for (int k = threadIdx.x; k < capacity; k += kThreads) {
            const bool real = k < slots;
            items[3 * k] = real ? pick[k] : 0;
            items[3 * k + 1] = real ? k : 0;
            items[3 * k + 2] = real ? 1 : 0;
            if (real) members[k] = k;
        }
    }
    __syncthreads();
}

// select_row's bytes by LANES (32 or 16) lanes a row; KPER * LANES >= experts, top_k <= LANES; true: NaN, none written.
template <int KPER, int LANES>
__device__ bool select_row_lanes(const float* logits, int row, bool live, int* pick, float* wts, int experts, int top_k) {
    const int lane = threadIdx.x & 31;
    const int hl = lane & (LANES - 1);
    const int base = lane & ~(LANES - 1);
    const unsigned group = LANES == 32 ? 0xffffffffu : ((1u << LANES) - 1u) << base;
    const int slots = top_k + 1;
    const float* l = logits + static_cast<long long>(live ? row : 0) * (experts + 1);
    float v[KPER];
    unsigned taken = 0;
    bool nan = false;
#pragma unroll
    for (int j = 0; j < KPER; ++j) {
        const int i = hl + LANES * j;
        v[j] = i < experts ? __hip_atomic_load(l + i, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) : 0.f;
        nan |= i < experts && v[j] != v[j];
    }
    const bool bad = (static_cast<unsigned>(__ballot(nan)) & group) != 0u;
    // the k-th pick, held by the group's lane k
    float mine_v = 0.f;
    int mine_i = 0;
    for (int k = 0; k < top_k; ++k) {
        float bv = 0.f;
        int bi = 0x7fffffff;
#pragma unroll
        for (int j = 0; j < KPER; ++j) {
            const int i = hl + LANES * j;
            if (i < experts && !((taken >> j) & 1u) && (bi == 0x7fffffff || v[j] > bv || (v[j] == bv && i < bi))) {
                bv = v[j];
                bi = i;
            }
        }
#pragma unroll
        for (int mask = LANES / 2; mask > 0; mask >>= 1) {
            const float ov = __shfl_xor(bv, mask, 32);
            const int oi = __shfl_xor(bi, mask, 32);
            if (oi != 0x7fffffff && (bi == 0x7fffffff || ov > bv || (ov == bv && oi < bi))) {
                bv = ov;
                bi = oi;
            }
        }
        if (bi % LANES == hl) taken |= 1u << (bi / LANES);
        if (hl == k) {
            mine_v = bv;
            mine_i = bi;
        }
    }
    // thread 0's arithmetic in its order; each exp is taken again for its weight (the same value)
    const float top = __shfl(mine_v, base, 32);
    float total = 0.f;
    for (int k = 0; k < top_k; ++k) total += expf(__shfl(mine_v, base + k, 32) - top);
    if (live && !bad) {
        const int at = row * slots;
        if (hl < top_k) {
            pick[at + hl] = mine_i;
            wts[at + hl] = round_bf16_f(expf(mine_v - top) / total);
        }
        if (hl == 0) {
            const float gate = round_bf16_f(__hip_atomic_load(l + experts, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT));
            pick[at + top_k] = experts;
            wts[at + top_k] = round_bf16_f(1.f / (1.f + expf(-gate)));
        }
    }
    return bad;
}

// tf_moe_route's members and items for a few rows' pairs (at most kPlanPairs) by one block, the same plan as it makes.
constexpr int kPlanPairs = 32 * kRouterWaves;  // a pair a thread: 8 rows of top 10 + the shared expert (88)
// The shared memory it takes from `pool`, in ints: 3 * experts + 2 * 32 * kRouterWaves + kPlanPairs.
__device__ void route_plan(const int* picks, int pairs, int experts, int tile, int* members, int* items, int capacity,
                           int* pool) {
    constexpr int kThreads = 32 * kRouterWaves;
    int* counts = pool;
    int* starts = counts + experts;
    int* ends = starts + experts;
    int* scan_c = ends + experts;
    int* scan_n = scan_c + kThreads;
    int* pk = scan_n + kThreads;
    const int t = threadIdx.x;
    for (int e = t; e < experts; e += kThreads) counts[e] = 0;
    if (t < pairs) pk[t] = picks[t];
    __syncthreads();
    if (t < pairs) atomicAdd(&counts[pk[t]], 1);
    __syncthreads();
    const int per = (experts + kThreads - 1) / kThreads;
    const int lo = min(experts, t * per), hi = min(experts, lo + per);
    int own_c = 0, own_n = 0;
    for (int e = lo; e < hi; ++e) {
        own_c += counts[e];
        own_n += (counts[e] + tile - 1) / tile;
    }
    scan_c[t] = own_c;
    scan_n[t] = own_n;
    __syncthreads();
    for (int off = 1; off < kThreads; off <<= 1) {
        const int vc = t >= off ? scan_c[t - off] : 0, vn = t >= off ? scan_n[t - off] : 0;
        __syncthreads();
        scan_c[t] += vc;
        scan_n[t] += vn;
        __syncthreads();
    }
    int start = scan_c[t] - own_c, end = scan_n[t] - own_n;
    for (int e = lo; e < hi; ++e) {
        starts[e] = start;
        start += counts[e];
        end += (counts[e] + tile - 1) / tile;
        ends[e] = end;
    }
    __syncthreads();
    if (t < pairs) {
        const int e = pk[t];
        int before = 0;
        for (int q = 0; q < t; ++q) before += pk[q] == e;
        members[starts[e] + before] = t;
    }
    for (int slot = t; slot < capacity; slot += kThreads) {
        int a = 0, b = experts;
        while (a < b) {  // searchsorted(ends, slot, right=True)
            const int mid = (a + b) / 2;
            if (ends[mid] <= slot) a = mid + 1;
            else b = mid;
        }
        const int e = a < experts - 1 ? a : experts - 1;
        const int reps = (counts[e] + tile - 1) / tile;
        const int within = slot - (ends[e] - reps);
        const int count = counts[e] - within * tile;
        items[slot * 3] = e;
        items[slot * 3 + 1] = starts[e] + within * tile;
        items[slot * 3 + 2] = count < 0 ? 0 : (count > tile ? tile : count);
    }
}

// One launch: tf_router_decode's logits from bf16 rows, then the last block's picks and plan; `done` is 0 after it.
template <int KIND>
__device__ void router_select(const void* x, const void* rows, float* logits, int r, int d, int e, int* pick, float* wts,
                              int* items, int* members, int capacity, int top_k, int tile, int staged, unsigned* done) {
    __shared__ bool last;
    // the dynamic shared memory the host sized (none where it stages nothing and plans nothing; launches/moe.zig)
    extern __shared__ __attribute__((aligned(16))) uint4 router_shared[];
    // a verify window's few rows in shared memory once a block: each wave (an expert) reads them there, not from L2
    const void* xs = router_shared;
    if (staged) {
        const uint4* from = static_cast<const uint4*>(x);
        for (int i = threadIdx.x; i < r * d / 8; i += 32 * kRouterWaves) router_shared[i] = from[i];
        __syncthreads();
    }
    const int expert = blockIdx.x * kRouterWaves + (threadIdx.x >> 5);
    if (expert < e) router_decode<KIND, true>(staged ? xs : x, rows, logits, r, d, e, staged);
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(done, 1u) == gridDim.x - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    if (r == 1) {
        select_row(logits, 0, pick, wts, items, members, capacity, e - 1, top_k);
    } else if (r <= 2 * kRouterWaves && e - 1 <= kSelectMax) {
        // a wave a row (half a wave past kRouterWaves rows), NaN rows by select_row; the plan reuses the staged memory
        __shared__ int fallback[2 * kRouterWaves];
        int* pool = reinterpret_cast<int*>(router_shared) + 2 * 32 * kRouterWaves;
        const int wave = threadIdx.x >> 5;
        __syncthreads();
        const int lane = threadIdx.x & 31;
        if (r > kRouterWaves && e - 1 <= 16 * 32 && top_k <= 16) {
            // 5 to 8 rows: half a wave a row, rows w and w + kRouterWaves in wave w's two halves
            const int row = wave + (lane >= 16 ? kRouterWaves : 0);
            const bool live = row < r;
            const bool bad = select_row_lanes<32, 16>(logits, row, live, pick, wts, e - 1, top_k);
            if (live && (lane & 15) == 0) fallback[row] = bad;
        } else {
            for (int row = wave; row < r; row += kRouterWaves) {
                const bool bad = e - 1 <= 16 * 32 ? select_row_lanes<16, 32>(logits, row, true, pick, wts, e - 1, top_k)
                                                  : select_row_lanes<kSelectMax / 32, 32>(logits, row, true, pick, wts, e - 1, top_k);
                if (lane == 0) fallback[row] = bad;
            }
        }
        __syncthreads();
        for (int row = 0; row < r; ++row)
            if (fallback[row]) select_row(logits, row, pick, wts, nullptr, nullptr, 0, e - 1, top_k);
        __threadfence_block();
        __syncthreads();
        if (items != nullptr) route_plan(pick, r * (top_k + 1), e, tile, members, items, capacity, pool);
    } else {
        for (int row = 0; row < r; ++row) select_row(logits, row, pick, wts, nullptr, nullptr, 0, e - 1, top_k);
    }
    if (threadIdx.x == 0) *done = 0u;
}

}  // namespace
