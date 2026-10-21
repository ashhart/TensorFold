// Dynamic KV paging for Flash Next attention (ours; cuda_exl3_kvpage.zig, docs/plans/exl3-kv-paging.md).
//
// A sequence's full K/V of each attention layer lives in (mapped, pinned) host memory, [pos][2][256] bf16 each. The
// card keeps one compact buffer for K and one for V, [12 windows of WR rows | S slots of 4 rows]: a layer's window
// holds its positions [win0, win0 + WR) densely; the slots, shared by the 12 layers, hold 4-key blocks the sparse
// indexer selected (slot2blk keys a slot by layer * nblocks + block). A layer's kernels see the buffer from its own
// window: window rows 0..WR-1, slot s at row SB + 4 s with SB = (12 - layer) * WR. After
// upstream's fn_qsa_select listed each row's keys (positions), these kernels
//   kvp_plan   (one block): the call's unique blocks outside the window; hits are stamped with the call's epoch; the
//              misses take the least recently used slots (radix selection of the smallest stamps, as fn_qsa_select
//              picks the largest scores; equal stamps claimed in any order), and the block <-> slot tables updated;
//   kvp_fetch  : each miss's 4 rows of K and V read from host memory into its slot;
//   kvp_remap  : each sparse row's key list rewritten from positions into rows of the compact buffer;
// so upstream's fn_attn_chunks / fn_attn_merge run unchanged on the compact buffer. Dense rows (<= 2051 keys) read
// rows 0..n-1, which is the window while win0 == 0 (the window is never moved before position WR - chunk).
//
// ctl (ints, one set a layer): [0] sequence clock (layer 0 only), [1] misses of the last call, [2] sticky failure,
// [3] misses in total, [4] needed blocks in total, [5] calls with a sparse row, [6] last executed epoch.
#include <cuda_bf16.h>
#include <stdint.h>

// One block of 1024 threads.
template <bool Shared>
__device__ __forceinline__ void kvp_plan_body(const int* __restrict__ ids, const int* __restrict__ nkr,
                                                            const int* __restrict__ spr, int m, int idw, int top,
                                                            int ratio, int win0, unsigned* clock, int* blk2slot,
                                                            int* blk2slot_all, int key_off,
                                                            int* __restrict__ slot2blk, unsigned* __restrict__ stamp,
                                                            unsigned* __restrict__ mark, int nslots, int* __restrict__ ctl,
                                                            int* __restrict__ need, int* __restrict__ miss,
                                                            int* __restrict__ fetch, unsigned long long* owner_generation,
                                                            const unsigned long long* lane_generation,
                                                            unsigned long long generation, int lane_stride, int all_blocks) {
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
    if (Shared && lane_generation[key_off / lane_stride] != generation) {
        if (threadIdx.x == 0) ctl[2] = 1;
        return;
    }
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
        if (Shared && hit) hit = owner_generation[s] == generation && slot2blk[s] == key_off + b;
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
        ctl[3] += ctl[1];
        ctl[4] += n_need;
        ctl[5] += 1;
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
    // misses; claims through two shared counters, so no block-wide scan
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
        if (Shared) {
            if (old >= 0 && old < all_blocks && owner_generation[s] == lane_generation[old / lane_stride] && blk2slot_all[old] == s) blk2slot_all[old] = -1;
            owner_generation[s] = generation;
        } else if (old >= 0) blk2slot_all[old] = -1;
        blk2slot[b] = s;
        slot2blk[s] = key_off + b;
        stamp[s] = epoch;
        fetch[2 * place] = b;
        fetch[2 * place + 1] = s;
    }
}

extern "C" __global__ void __launch_bounds__(1024) kvp_plan(
    const int* ids, const int* counts, const int* sparse, int rows, int id_width, int top_blocks,
    int ratio, int window_origin, unsigned* clock, int* inverse, int* all_inverse, int key_offset,
    int* owners, unsigned* stamps, unsigned* marks, int slots, int* controls, int* needed,
    int* missing, int* fetched) {
    kvp_plan_body<false>(ids, counts, sparse, rows, id_width, top_blocks, ratio, window_origin,
        clock, inverse, all_inverse, key_offset, owners, stamps, marks, slots, controls, needed,
        missing, fetched, nullptr, nullptr, 0, 1, 0);
}

extern "C" __global__ void __launch_bounds__(1024) kvp_shared_plan(
    const int* ids, const int* counts, const int* sparse, int rows, int id_width, int top_blocks,
    int ratio, int window_origin, unsigned* clock, int* inverse, int* all_inverse, int key_offset,
    int* owners, unsigned* stamps, unsigned* marks, int slots, int* controls, int* needed,
    int* missing, int* fetched, unsigned long long* owner_generation,
    const unsigned long long* lane_generation, unsigned long long generation, int lane_stride,
    int all_blocks) {
    kvp_plan_body<true>(ids, counts, sparse, rows, id_width, top_blocks, ratio, window_origin,
        clock, inverse, all_inverse, key_offset, owners, stamps, marks, slots, controls, needed,
        missing, fetched, owner_generation, lane_generation, generation, lane_stride, all_blocks);
}

extern "C" __global__ void __launch_bounds__(256) kvp_shared_release(
    int* all_inverse, int* owners, unsigned* stamps, unsigned long long* owner_generation,
    int slots, int key_begin, int key_end, unsigned long long generation) {
    for (int slot = blockIdx.x * blockDim.x + threadIdx.x; slot < slots; slot += gridDim.x * blockDim.x) {
        const int previous = owners[slot];
        if (previous < key_begin || previous >= key_end || owner_generation[slot] != generation) continue;
        if (all_inverse[previous] == slot) all_inverse[previous] = -1;
        owners[slot] = -1;
        owner_generation[slot] = 0;
        stamps[slot] = 0;
    }
}

// Each miss's block: 4 rows of K and of V (ROW16 16-byte words a row) from host memory into its slot (dk / dv: the
// slots' first row, WR 0; or a layer's window base with WR its slot base rows).
extern "C" __global__ void __launch_bounds__(256) kvp_fetch(const uint4* __restrict__ hk, const uint4* __restrict__ hv,
                                                            uint4* __restrict__ dk, uint4* __restrict__ dv,
                                                            const int* __restrict__ ctl, const int* __restrict__ fetch,
                                                            int wr, int row16, int ratio) {
    const int n = ctl[1];
    const long long per = static_cast<long long>(ratio) * row16;
    for (int i = blockIdx.x; i < n; i += gridDim.x) {
        const long long b = fetch[2 * i], s = fetch[2 * i + 1];
        const uint4* sk = hk + b * per;
        const uint4* sv = hv + b * per;
        uint4* tk = dk + static_cast<long long>(wr) * row16 + s * per;
        uint4* tv = dv + static_cast<long long>(wr) * row16 + s * per;
        for (int e = threadIdx.x; e < per; e += blockDim.x) {
            tk[e] = sk[e];
            tv[e] = sv[e];
        }
    }
}

// A row a block: its sparse key list from positions into rows from the layer's window base (WR: the slot base SB).
extern "C" __global__ void __launch_bounds__(256) kvp_remap(int* __restrict__ ids, int* __restrict__ nkr,
                                                            const int* __restrict__ spr, int idw, int win0, int wr,
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
        row[k] = p >= win0 ? p - win0 : wr + blk2slot[p / ratio] * ratio + p % ratio;
    }
}

// n rows of 16-byte words: dst[i] = src[i] for i < n (a window move within one buffer, issued in non-overlapping
// pieces by the host, or a write-out to host memory).
extern "C" __global__ void __launch_bounds__(256) kvp_copy(const uint4* __restrict__ src, uint4* __restrict__ dst, long long n) {
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += static_cast<long long>(gridDim.x) * blockDim.x) dst[i] = src[i];
}
