// Two-rank one-shot RoCE all-gather after b12x (github.com/local-inference-lab/b12x, Apache-2.0): docs/roce-gather.md.

#include <stdint.h>

namespace {

constexpr int C_SEQ = 0, C_NBYTES = 1, C_ERROR = 3, C_HCA = 4;  // ctrl words (C_NBYTES + slot)
constexpr int FLAG_STRIDE = 128;
// counters (device memory, u32): epoch, poison, then stage and tail arrivals for each grid size class
constexpr int K_EPOCH = 0, K_POISON = 1, K_CLASSES = 4;

__device__ __forceinline__ uint32_t ld_relaxed_gpu(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ uint32_t ld_relaxed_sys(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ uint32_t ld_acquire_sys(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ uint4 ld_relaxed_sys4(const uint4* p) {
    uint4 v;
    asm volatile("ld.relaxed.sys.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void st_relaxed_sys(uint32_t* p, uint32_t v) {
    asm volatile("st.relaxed.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void st_release_gpu(uint32_t* p, uint32_t v) {
    asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ uint32_t arrive(uint32_t* p) {
    uint32_t old;
    asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], 1;" : "=r"(old) : "l"(p) : "memory");
    return old;
}
__device__ __forceinline__ void fence_sys() { asm volatile("fence.sc.sys;" ::: "memory"); }
__device__ __forceinline__ void fence_gpu() { asm volatile("fence.sc.gpu;" ::: "memory"); }

}  // namespace

// in: this rank's shard (packs of 16 bytes); out: [rank 0's shard | rank 1's shard]. klass: log2(gridDim.x).
extern "C" __global__ void __launch_bounds__(512) roce_gather(
    const uint4* __restrict__ in, uint4* __restrict__ out, int packs, int rank, char* recv, char* flags, char* send,
    uint32_t* ctrl, long long slot_bytes, uint32_t* counters, int klass, uint32_t spin_limit, int n_hca) {
    const int tid = threadIdx.x;
    uint32_t* stage = counters + 2 + klass;
    uint32_t* tail = counters + 2 + K_CLASSES + klass;
    if (ld_relaxed_gpu(counters + K_POISON) != 0u) return;
    const uint32_t seq = ld_relaxed_gpu(counters + K_EPOCH) + 1u;
    const uint32_t slot = seq & 1u;
    const int at = blockIdx.x * blockDim.x + tid, stride = gridDim.x * blockDim.x;

    uint4* mine = reinterpret_cast<uint4*>(send + (long long)slot * slot_bytes);  // 1. stage
    for (int i = at; i < packs; i += stride) mine[i] = in[i];
    __syncthreads();
    if (tid == 0) {  // 2. the doorbell, by the last block to stage
        fence_sys();
        if ((arrive(stage) + 1u) % gridDim.x == 0u) {
            st_relaxed_sys(ctrl + C_NBYTES + slot, (uint32_t)packs * 16u);
            fence_sys();
            st_relaxed_sys(ctrl + C_SEQ, seq);
        }
    }
    if (tid < n_hca) {  // 3. the peer's stripes
        const uint32_t* flag = reinterpret_cast<const uint32_t*>(flags + ((long long)slot * n_hca + tid) * FLAG_STRIDE);
        uint32_t polls = 0;
        while (ld_acquire_sys(flag) != seq) {
            if (++polls >= spin_limit) {
                st_relaxed_sys(ctrl + C_HCA, (uint32_t)tid);
                st_relaxed_sys(ctrl + C_ERROR, seq);
                st_release_gpu(counters + K_POISON, seq);
                break;
            }
        }
    }
    __syncthreads();
    if (ld_relaxed_gpu(counters + K_POISON) == 0u) {  // 4. both shards in rank order
        const uint4* theirs = reinterpret_cast<const uint4*>(recv + (long long)slot * slot_bytes);
        uint4* own_out = out + (long long)rank * packs;
        uint4* peer_out = out + (long long)(1 - rank) * packs;
        for (int i = at; i < packs; i += stride) {
            own_out[i] = in[i];
            peer_out[i] = ld_relaxed_sys4(theirs + i);
        }
    }
    fence_gpu();  // 5. the epoch, by the last block to finish
    __syncthreads();
    if (tid == 0 && (arrive(tail) + 1u) % gridDim.x == 0u) {
        fence_gpu();
        if (ld_relaxed_sys(ctrl + C_ERROR) == 0u) st_release_gpu(counters + K_EPOCH, seq);
    }
}
