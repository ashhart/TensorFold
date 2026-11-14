// window_rows.cuh's decode partials over a row's live chunks; included inside namespace tf_wrows, after part_one.
#pragma once

// Chunks blockIdx.z, then every gridDim.z-th, up to the row's live ones; the merge reads no unwritten one past them.
template <int D, int FAULT, bool PG = false>
__device__ __forceinline__ void part(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kc,
                                     const __nv_bfloat16* __restrict__ vc, const int* __restrict__ pos,
                                     const int* __restrict__ slot, float* __restrict__ po, float* __restrict__ pm,
                                     float* __restrict__ pl, int h, int hk, int win, long long ring, float scale,
                                     int chunks, Pg pg = Pg{}) {
    const int live = liveChunks(chunks, win, pos[blockIdx.x], FAULT == ALL);
    for (int c = blockIdx.z; c < live; c += gridDim.z) {
        part_one<D, FAULT, PG>(q, kc, vc, pos, slot, po, pm, pl, h, hk, win, ring, scale, c, chunks, pg);
        __syncthreads(); // the next chunk reuses this one's shared memory
    }
}
