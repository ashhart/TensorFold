__device__ __forceinline__ float block_sum(float v, float* scratch) {
    v = warp_sum(v);
    const int warps = blockDim.x >> 5;
    if ((threadIdx.x & 31) == 0) scratch[threadIdx.x >> 5] = v;
    __syncthreads();
    float total = 0.0f;
    for (int w = 0; w < warps; ++w) total = __fadd_rn(total, scratch[w]);
    __syncthreads();
    return total;
}
