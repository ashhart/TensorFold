// Host-tier K/V for Flash Next (EXL3): a layer's 4-row slots in lent frames; kvt_plan, kvt_fetch, kvt_remap.
#include <cuda_bf16.h>
#include <stdint.h>

// One block of 1024 threads.
__device__ __forceinline__ void kvt_plan_body(const int* __restrict__ ids, const int* __restrict__ nkr,
                                                            const int* __restrict__ spr, int m, int idw, int top,
                                                            int ratio, int win0, unsigned* clock, int* blk2slot,
                                                            int* blk2slot_all, int key_off,
                                                            int* __restrict__ slot2blk, unsigned* __restrict__ stamp,
                                                            unsigned* __restrict__ mark, int nslots, int* __restrict__ ctl,
                                                            int* __restrict__ need, int* __restrict__ miss,
                                                            int* __restrict__ fetch) {
    __shared__ int s_need, s_miss, s_avail;
    __shared__ int hist[256];
    __shared__ unsigned cut_s;
    __shared__ int left_s;
    __shared__ unsigned epoch;
    (void)nkr;
    if (threadIdx.x == 0) {
        s_need = 0;
        s_miss = 0;
        s_avail = 0;
        if (*clock == UINT32_MAX) {
            ctl[2] = 1;
            epoch = 0;
        } else {
            epoch = ++*clock;
        }
        ctl[6] = static_cast<int>(epoch);
        ctl[1] = 0;
    }
    __syncthreads();
    if (epoch == 0 || ctl[2] != 0) return;
    bool any = false;
    for (int r = 0; r < m; ++r) any |= spr[r] != 0;
    if (!any) {
        if (threadIdx.x == 0) ctl[1] = 0;
        return;
    }
    // the call's unique blocks before the window (a row lists its 512 blocks as 4 consecutive keys each)
    for (int e = threadIdx.x; e < m * top; e += blockDim.x) {
        const int r = e / top, j = e - r * top;
        if (spr[r] == 0) continue;
        const int b = ids[static_cast<long long>(r) * idw + j * ratio] / ratio;
        if (b * ratio >= win0) continue;
        if (atomicExch(&mark[b], epoch) != epoch) need[atomicAdd(&s_need, 1)] = b;
    }
    __syncthreads();
    const int n_need = s_need;
    for (int i = threadIdx.x; i < n_need; i += blockDim.x) {
        const int b = need[i];
        const int s = blk2slot[b];
        bool hit = s >= 0 && s < nslots;
        if (hit)
            stamp[s] = epoch;
        else {
            blk2slot[b] = -1;
            miss[atomicAdd(&s_miss, 1)] = b;
        }
    }
    __syncthreads();
    int n_miss = s_miss;
    // slots this call does not use
    for (int s = threadIdx.x; s < nslots; s += blockDim.x)
        if (stamp[s] != epoch) atomicAdd(&s_avail, 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        if (n_miss > s_avail) ctl[2] = 1;
        ctl[1] = n_miss > s_avail ? 0 : n_miss;
        unsigned* counters = reinterpret_cast<unsigned*>(ctl);
        counters[3] += ctl[1];
        counters[4] += n_need;
        counters[5] += 1;
    }
    if (n_miss > s_avail) return;
    n_miss = min(n_miss, s_avail);
    if (n_miss == 0) return;
    // the n_miss smallest stamps among unused slots: the largest keys of ~stamp (used slots key 0, never needed)
    auto key = [&](int s) -> unsigned { const unsigned t = stamp[s]; return t == epoch ? 0u : ~t; };
    unsigned cut = 0;
    int left = n_miss;
    for (int p = 0; p < 4; ++p) {
        const int shift = 24 - 8 * p;
        for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0;
        __syncthreads();
        for (int s = threadIdx.x; s < nslots; s += blockDim.x) {
            const unsigned k = key(s);
            if (p == 0 || (k >> (shift + 8)) == (cut >> (shift + 8))) atomicAdd(&hist[(k >> shift) & 0xFF], 1);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int above = 0;
            int d = 255;
            for (; d > 0; --d) {
                if (above + hist[d] >= left) break;
                above += hist[d];
            }
            left_s = left - above;
            cut_s = cut | (static_cast<unsigned>(d) << shift);
        }
        __syncthreads();
        cut = cut_s;
        left = left_s;
        __syncthreads();
    }
    // every slot above the cut and `left` of those equal to it (any of them: equal stamps are equally old) take the
    __shared__ int s_eq, s_place;
    if (threadIdx.x == 0) {
        s_eq = 0;
        s_place = 0;
    }
    __syncthreads();
    for (int s = threadIdx.x; s < nslots; s += blockDim.x) {
        const unsigned k = key(s);
        bool chosen = k > cut;
        if (!chosen && k == cut) chosen = atomicAdd(&s_eq, 1) < left;
        if (!chosen) continue;
        const int place = atomicAdd(&s_place, 1);
        if (place >= n_miss) continue;
        const int b = miss[place];
        const int old = slot2blk[s];
        if (old >= 0) blk2slot_all[old] = -1;
        blk2slot[b] = s;
        slot2blk[s] = b;
        stamp[s] = epoch;
        fetch[2 * place] = b;
        fetch[2 * place + 1] = s;
    }
}

extern "C" __global__ void __launch_bounds__(1024) kvt_plan(
    const int* ids, const int* counts, const int* sparse, int rows, int id_width, int top_blocks,
    int ratio, int window_origin, unsigned* clock, int* inverse, int* all_inverse, int key_offset,
    int* owners, unsigned* stamps, unsigned* marks, int slots, int* controls, int* needed,
    int* missing, int* fetched) {
    kvt_plan_body(ids, counts, sparse, rows, id_width, top_blocks, ratio, window_origin,
        clock, inverse, all_inverse, key_offset, owners, stamps, marks, slots, controls, needed,
        missing, fetched);
}

// Each miss's 4 rows of K and V: host store (hk, hv) to slot s: row (s % 64) * 4 of entry vpage + s / 64.
extern "C" __global__ void __launch_bounds__(256) kvt_fetch(const uint4* __restrict__ hk, const uint4* __restrict__ hv,
                                                            uint4* __restrict__ dk, uint4* __restrict__ dv,
                                                            const int* __restrict__ ctl, const int* __restrict__ fetch,
                                                            const int* __restrict__ lids, const int* __restrict__ pt,
                                                            int vpage, long long ps, int row16, int ratio) {
    const int n = ctl[1];
    const long long per = static_cast<long long>(ratio) * row16;
    for (int i = blockIdx.x; i < n; i += gridDim.x) {
        const long long b = fetch[2 * i], s = fetch[2 * i + 1];
        const long long page = b * ratio / 256, in_page = b * ratio % 256;
        const long long src = static_cast<long long>(lids[page]) * ps + in_page * row16;
        const long long dst = static_cast<long long>(pt[vpage + (s >> 6)]) * ps + (s & 63) * per;
        for (int e = threadIdx.x; e < per; e += blockDim.x) {
            dk[dst + e] = hk[src + e];
            dv[dst + e] = hv[src + e];
        }
    }
}

// A row a block: its key list from positions to table rows: the window's stay, older ones take their slot's.
extern "C" __global__ void __launch_bounds__(256) kvt_remap(int* __restrict__ ids, int* __restrict__ nkr,
                                                            const int* __restrict__ spr, int idw, int win0, int vrow,
                                                            const int* __restrict__ blk2slot, int ratio, int* ctl) {
    const int r = blockIdx.x;
    if (spr[r] == 0) return;
    const int n = nkr[r];
    int* row = ids + static_cast<long long>(r) * idw;
    __shared__ int missing;
    if (threadIdx.x == 0) missing = atomicAdd(&ctl[2], 0) != 0;
    __syncthreads();
    for (int k = threadIdx.x; k < n; k += blockDim.x) {
        const int p = row[k];
        if (p < win0 && blk2slot[p / ratio] < 0) atomicExch(&missing, 1);
    }
    __syncthreads();
    if (missing) {
        if (threadIdx.x == 0) {
            atomicExch(&ctl[2], 1);
            nkr[r] = 0;
        }
        return;
    }
    for (int k = threadIdx.x; k < n; k += blockDim.x) {
        const int p = row[k];
        row[k] = p >= win0 ? p : vrow + blk2slot[p / ratio] * ratio + p % ratio;
    }
}
