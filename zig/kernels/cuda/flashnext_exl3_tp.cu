// The Flash Next family's EXL3 expert-parallel glue (cuda_exl3_tp.zig): a rank's masked picks, and the summed MoE
// branch placed where the pass's write-back reads the per-slot outputs.

// A rank holds the expert ids [LO, HI) (global ids: rank 0 the routed 0..255, rank 1 the routed 256..511 and the
// shared expert 512). Another rank's slot becomes pick -1 (Experts.run skips it) with weight 0, so the rank's combined
// output is the fp32 fmaf chain over its own slots in slot order (a skipped slot adds weight 0 times a finite value).
extern "C" __global__ void fn_exl3_tp_mask(const int* __restrict__ pick, const float* __restrict__ wts,
                                           int* __restrict__ mpick, float* __restrict__ mwts, int n, int lo, int hi) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int p = pick[i];
    const bool own = p >= lo && p < hi;
    mpick[i] = own ? p : -1;
    mwts[i] = own ? wts[i] : 0.0f;
}

// The ranks' summed branch [rows, D] into the write-back's slot layout: y[r, 0, :] = sum, the other slots 0, and the
// weights {1, 0, ..., 0}. hc_writeback (MODE 2) then reads fma(sum, 1, 0) = sum and adds 0 * 0 for every other slot,
// so it writes the sum's bf16 exactly. Program (row, slot), 256 threads over D.
extern "C" __global__ void __launch_bounds__(256) fn_exl3_tp_place(const float* __restrict__ sum, float* __restrict__ y,
                                                                   float* __restrict__ wts, int d, int slots) {
    const int r = blockIdx.x, k = blockIdx.y;
    float* out = y + (static_cast<long long>(r) * slots + k) * d;
    const float* in = sum + static_cast<long long>(r) * d;
    for (int i = threadIdx.x; i < d; i += blockDim.x) out[i] = k == 0 ? in[i] : 0.0f;
    if (k == 0 && threadIdx.x < slots) wts[r * slots + threadIdx.x] = threadIdx.x == 0 ? 1.0f : 0.0f;
}
