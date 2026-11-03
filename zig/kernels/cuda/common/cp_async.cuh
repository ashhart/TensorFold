// cp.async groups: commit the copies issued so far as one group, wait until at most N groups are still in flight.
#pragma once

#if defined(__CUDACC__) || defined(__CUDA__)
#include <cuda_runtime.h>
#include <stdint.h>
#endif

namespace tf_common {

#if defined(__CUDACC__) || defined(__CUDA__)

__device__ __forceinline__ uint32_t smem(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem(dst)), "l"(src));
}

__device__ __forceinline__ void cp16z(void* dst, const void* src, bool read) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem(dst)), "l"(src), "r"(read ? 16 : 0));
}

__device__ __forceinline__ void cp8(void* dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(smem(dst)), "l"(src));
}

__device__ __forceinline__ void cp4(void* dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(smem(dst)), "l"(src));
}

__device__ __forceinline__ uint4 ld_nc(const uint4* p) {
    uint4 r;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
                 : "l"(p));
    return r;
}

#endif

__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

#if defined(__CUDACC__) || defined(__CUDA__)
__device__ __forceinline__ void cp16z(unsigned d, const void* src, bool ok) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(ok ? 16 : 0));
}

__device__ __forceinline__ uint2 ld_nc2(const void* p) {
    uint2 r;
    asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0, %1}, [%2];\n" : "=r"(r.x), "=r"(r.y) : "l"(p));
    return r;
}

__device__ __forceinline__ uint32_t ld_nc32(const void* p) {
    uint32_t r;
    asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];\n" : "=r"(r) : "l"(p));
    return r;
}

#endif

} // namespace tf_common
