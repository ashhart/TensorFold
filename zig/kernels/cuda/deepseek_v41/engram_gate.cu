// Copy of src/tensorfold/families/deepseek_v41/cuda/engram_gate.cu (blob 6c2e75a4f4de): sass_check.sh



// Engram gate: the GPU, not the host, waits for a round's Engram rows, so a CUDA graph needs no host round trip.

// ctl (pinned, mapped): ctl[0] = armed round, ctl[ready] = last published round; on timeout ctl[err] = layer slot.
#include <cstdint>
#include <cuda_runtime.h>

namespace dsv41_engram_gate {

__device__ __forceinline__ long long ld_acquire_sys(const long long* p) {
    long long v;
    asm volatile("ld.acquire.sys.s64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ unsigned long long now_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__global__ void gate_copy_kernel(long long* ctl, int ready, int err_slot, const unsigned char* base, long long stride,
                                 unsigned char* dst, long long nbytes, long long timeout_ns, int vec) {
    __shared__ long long want_s;
    if (threadIdx.x == 0) {
        const long long want = ld_acquire_sys(ctl);
        const unsigned long long t0 = now_ns();
        while (ld_acquire_sys(ctl + ready) < want) {
            if ((long long)(now_ns() - t0) > timeout_ns) {
                *reinterpret_cast<volatile long long*>(ctl + err_slot) = ready;
                __threadfence_system();
                break;
            }
            __nanosleep(256);
        }
        want_s = want;
    }
    __syncthreads();
    const unsigned char* src = base + (want_s & 1) * stride;
    const long long step = (long long)gridDim.x * blockDim.x;
    const long long first = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long done = 0;
    if (vec) {
        const long long n16 = nbytes >> 4;
        const uint4* s4 = reinterpret_cast<const uint4*>(src);
        uint4* d4 = reinterpret_cast<uint4*>(dst);
        for (long long i = first; i < n16; i += step) d4[i] = __ldcv(s4 + i);
        done = n16 << 4;
    }
    for (long long i = done + first; i < nbytes; i += step) dst[i] = __ldcv(src + i);
}

}  // namespace dsv41_engram_gate
