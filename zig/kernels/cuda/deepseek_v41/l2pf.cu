// Copy of src/tensorfold/families/glm5_next/spark/l2pf.cu (blob ace57d9cab36): sass_check.sh

// L2 prefetch for idle windows: reads weights into L2 and writes nothing the model reads, so results cannot change.

// bulk = cp.async.bulk.prefetch.L2 per chunk (sm_90+), lines = prefetch.global.L2 per 128-byte line (fallback).

// prefetch_segments: a static (address, bytes) table; prefetch_experts: the router's picked experts, graph-capturable.

#include <cuda_runtime.h>
#include <stdint.h>

namespace tfl2pf {

constexpr uint32_t LINE = 128;

__device__ __forceinline__ void bulk_prefetch(const void *p, uint32_t bytes) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p), "r"(bytes) : "memory");
#else   // pre-sm_90 targets (never run on GB10): a line prefetch per 128 bytes
    for (uint32_t o = 0; o < bytes; o += LINE) asm volatile("prefetch.global.L2 [%0];" ::"l"((const char *)p + o));
#endif
}

__device__ __forceinline__ void line_prefetch(const void *p) {
    asm volatile("prefetch.global.L2 [%0];" ::"l"(p));
}

// [p, p + bytes) in 16-multiple pieces of `chunk`, piece i by thread i of `stride`; addresses 16-byte aligned.
template <bool Bulk>
__device__ __forceinline__ void range(const uint8_t *p, uint64_t bytes, uint32_t chunk, uint64_t index,
                                      uint64_t stride) {
    if (Bulk) {
        const uint64_t pieces = (bytes + chunk - 1) / chunk;
        for (uint64_t i = index; i < pieces; i += stride) {
            const uint64_t off = i * chunk;
            const uint64_t left = bytes - off;
            bulk_prefetch(p + off, static_cast<uint32_t>(left < chunk ? left : chunk));
        }
    } else {
        const uint64_t lines = (bytes + LINE - 1) / LINE;
        for (uint64_t i = index; i < lines; i += stride) {
            line_prefetch(p + i * LINE);
        }
    }
}

// table: n pairs of int64 (device address, bytes)
template <bool Bulk>
__global__ void __launch_bounds__(256) segments_kernel(const int64_t *table, int n, uint32_t chunk) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (int s = 0; s < n; s++) {
        const uint8_t *p = reinterpret_cast<const uint8_t *>(static_cast<uintptr_t>(table[2 * s]));
        const uint64_t bytes = static_cast<uint64_t>(table[2 * s + 1]);
        // segment s's first piece goes to thread s: many one-piece heads spread over the threads, not all on thread 0
        const uint64_t first = (index + stride - static_cast<uint64_t>(s) % stride) % stride;
        range<Bulk>(p, bytes, chunk, first, stride);
    }
}

// Block u prefetches the first `bytes` of expert ids[u] in each of nmat matrices; ids/count = router device output.
template <bool Bulk>
__global__ void __launch_bounds__(256) experts_kernel(const int32_t *ids, const int32_t *count, int maxu,
                                                      const int64_t *bases, int nmat, int64_t stride_bytes,
                                                      int64_t bytes, int64_t experts, uint32_t chunk) {
    const int n = min(static_cast<int>(*count), maxu);
    for (int u = blockIdx.x; u < n; u += gridDim.x) {
        const int64_t e = static_cast<int64_t>(ids[u]);
        if (e < 0 || e >= experts) {
            continue;                    // never read outside the expert tensors
        }
        for (int m = 0; m < nmat; m++) {
            const uint8_t *p = reinterpret_cast<const uint8_t *>(static_cast<uintptr_t>(bases[m])) + e * stride_bytes;
            range<Bulk>(p, static_cast<uint64_t>(bytes), chunk, threadIdx.x, blockDim.x);
        }
    }
}

cudaError_t launch_segments(const int64_t *table, int n, int grid, int threads, uint32_t chunk, bool bulk,
                            cudaStream_t stream) {
    if (bulk) {
        segments_kernel<true><<<grid, threads, 0, stream>>>(table, n, chunk);
    } else {
        segments_kernel<false><<<grid, threads, 0, stream>>>(table, n, chunk);
    }
    return cudaGetLastError();
}

cudaError_t launch_experts(const int32_t *ids, const int32_t *count, int maxu, const int64_t *bases, int nmat,
                           int64_t stride_bytes, int64_t bytes, int64_t experts, int grid, int threads, uint32_t chunk,
                           bool bulk, cudaStream_t stream) {
    if (bulk) {
        experts_kernel<true><<<grid, threads, 0, stream>>>(ids, count, maxu, bases, nmat, stride_bytes, bytes, experts,
                                                           chunk);
    } else {
        experts_kernel<false><<<grid, threads, 0, stream>>>(ids, count, maxu, bases, nmat, stride_bytes, bytes, experts,
                                                            chunk);
    }
    return cudaGetLastError();
}

}  // namespace tfl2pf
