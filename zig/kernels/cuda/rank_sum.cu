// The rank-order sum of a reduce-scatter (rank_scatter.zig): rank 0's fp32 term first, then 1, 2, ... rounded each add.

extern "C" __global__ void __launch_bounds__(256)
    tf_rank_sum(const float* __restrict__ peers, const float* __restrict__ own, int world, int me, long long n,
                float* __restrict__ out) {
  for (long long i = blockIdx.x * 256LL + threadIdx.x; i < n; i += (long long)gridDim.x * 256) {
    float acc = me == 0 ? own[i] : peers[i];
    for (int r = 1; r < world; ++r) acc = __fadd_rn(acc, r == me ? own[i] : peers[(long long)r * n + i]);
    out[i] = acc;
  }
}
