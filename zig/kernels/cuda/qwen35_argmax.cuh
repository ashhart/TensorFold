// Greedy tokens, a block a slice then a warp a row: the highest logit (no NaN or -inf), ties to the lowest id.
constexpr int ARGMAX_SLICE = 4096;

__device__ __forceinline__ void argmax_take(float v, int i, float& best, int& id) {
    if (v > best || (v == best && i < id)) {
        best = v;
        id = i;
    }
}

extern "C" __global__ void __launch_bounds__(256) tf_q35_argmax_slices(const __nv_bfloat16* __restrict__ L, int V, float* __restrict__ PV,
                                                                        int* __restrict__ PI) {
    __shared__ float bv[8];
    __shared__ int bi[8];
    const int base = blockIdx.x * ARGMAX_SLICE;
    const int end = min(V, base + ARGMAX_SLICE);
    const __nv_bfloat16* x = L + (size_t)blockIdx.y * V;
    float best = -INFINITY;
    int id = 0x7fffffff;
    for (int i = base + threadIdx.x; i < end; i += blockDim.x) {
        const float v = bf(x[i]);
        if (v > best) {
            best = v;
            id = i;
        }
    }
#pragma unroll
    for (int m = 16; m; m >>= 1) argmax_take(__shfl_xor_sync(0xffffffffu, best, m), __shfl_xor_sync(0xffffffffu, id, m), best, id);
    if ((threadIdx.x & 31) == 0) {
        bv[threadIdx.x >> 5] = best;
        bi[threadIdx.x >> 5] = id;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < 8; w++) argmax_take(bv[w], bi[w], best, id);
        PV[blockIdx.y * gridDim.x + blockIdx.x] = best;
        PI[blockIdx.y * gridDim.x + blockIdx.x] = id;
    }
}

extern "C" __global__ void __launch_bounds__(32) tf_q35_argmax_merge(const float* __restrict__ PV, const int* __restrict__ PI, int slices,
                                                                       unsigned* __restrict__ TOK) {
    float best = -INFINITY;
    int id = 0x7fffffff;
    for (int s = threadIdx.x; s < slices; s += 32) argmax_take(PV[blockIdx.x * slices + s], PI[blockIdx.x * slices + s], best, id);
#pragma unroll
    for (int m = 16; m; m >>= 1) argmax_take(__shfl_xor_sync(0xffffffffu, best, m), __shfl_xor_sync(0xffffffffu, id, m), best, id);
    if (threadIdx.x == 0) TOK[blockIdx.x] = (unsigned)id;
}

// Each row's greedy token: the largest logit, a tie to the lower id (the host draw's rule). Grid (rows), 1024 threads.
extern "C" __global__ void __launch_bounds__(1024) tf_q35_argmax(const __nv_bfloat16* __restrict__ L, int V, unsigned* __restrict__ TOK) {
    __shared__ float bv[32];
    __shared__ int bi[32];
    const __nv_bfloat16* x = L + (size_t)blockIdx.x * V;
    float best = -INFINITY;
    int id = 0x7fffffff;
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        const float v = bf(x[i]);
        if (v > best) {
            best = v;
            id = i;
        }
    }
#pragma unroll
    for (int m = 16; m; m >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, best, m);
        const int oi = __shfl_xor_sync(0xffffffffu, id, m);
        if (ov > best || (ov == best && oi < id)) {
            best = ov;
            id = oi;
        }
    }
    if ((threadIdx.x & 31) == 0) {
        bv[threadIdx.x >> 5] = best;
        bi[threadIdx.x >> 5] = id;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < 32; w++)
            if (bv[w] > best || (bv[w] == best && bi[w] < id)) {
                best = bv[w];
                id = bi[w];
            }
        TOK[blockIdx.x] = (unsigned)id;
    }
}
