// The Flash Next family's EXL3 expert-parallel glue (cuda_exl3_tp.zig): a rank's masked picks.

// The other rank's slots (outside [LO, HI)) become -1, which the experts' kernels skip; slots keep their positions.
extern "C" __global__ void fn_exl3_tp_mask(const int* __restrict__ pick, int* __restrict__ mpick, int n, int lo, int hi) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int p = pick[i];
    mpick[i] = (p >= lo && p < hi) ? p : -1;
}
