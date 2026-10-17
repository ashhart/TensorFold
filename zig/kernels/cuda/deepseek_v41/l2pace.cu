// Copy of src/tensorfold/families/deepseek_v41/cuda/l2pace.cu (blob fcf123ce92e9): sass_check.sh



// Paced L2 prefetch of a site (TF_DSV41_L2PF_PACE_GBPS), so it stops starving the RoCE exchange beside it.

// One thread per CTA issues piece k at t0 + delay + k * ns_per_piece: ~rate x latency bytes in flight, not the site.

// Exactness: only reads weight memory into L2 (cp.async.bulk.prefetch.L2); writes nothing.

#include <cuda_runtime.h>
#include <stdint.h>

namespace dsv41_l2pace {

__device__ __forceinline__ uint64_t now_ns() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__device__ __forceinline__ void bulk_prefetch(const void *p, uint32_t bytes) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p), "r"(bytes) : "memory");
#else
    for (uint32_t o = 0; o < bytes; o += 128) asm volatile("prefetch.global.L2 [%0];" ::"l"((const char *)p + o));
#endif
}

__device__ __forceinline__ void wait_until(uint64_t due) {
    while (true) {
        const uint64_t t = now_ns();
        if (t >= due) {
            return;
        }
        const uint64_t left = due - t;
        __nanosleep(static_cast<unsigned>(left > 2000u ? 1000u : (left > 64u ? left / 2u : 32u)));
    }
}

// table: n rows of int64 (address, bytes): 16-byte aligned addresses, byte counts multiples of 16
__global__ void __launch_bounds__(32) paced_kernel(const int64_t *table, int n, uint32_t chunk, uint64_t ns_per_piece,
                                                   uint64_t delay_ns) {
    if (threadIdx.x != 0) {
        return;
    }
    const uint64_t c = blockIdx.x;
    const uint64_t g = gridDim.x;
    const uint64_t t0 = now_ns() + delay_ns;
    uint64_t k = 0;                                     // the piece's index over the whole table
    for (int s = 0; s < n; s++) {
        const uint8_t *p = reinterpret_cast<const uint8_t *>(static_cast<uintptr_t>(table[2 * s]));
        const uint64_t bytes = static_cast<uint64_t>(table[2 * s + 1]);
        for (uint64_t off = 0; off < bytes; off += chunk, k++) {
            if (k % g != c) {
                continue;
            }
            wait_until(t0 + k * ns_per_piece);
            const uint64_t left = bytes - off;
            bulk_prefetch(p + off, static_cast<uint32_t>(left < chunk ? left : chunk));
        }
    }
}

cudaError_t launch_paced(const int64_t *table, int n, int ctas, uint32_t chunk, uint64_t ns_per_piece,
                         uint64_t delay_ns, cudaStream_t stream) {
    paced_kernel<<<ctas, 32, 0, stream>>>(table, n, chunk, ns_per_piece, delay_ns);
    return cudaGetLastError();
}

}  // namespace dsv41_l2pace
