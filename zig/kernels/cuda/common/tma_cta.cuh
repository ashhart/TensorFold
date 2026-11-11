// CTA-shared TMA tile transfers retain their descriptor coordinates and completion barrier.
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "qmm_frag.cuh"

namespace tf_tma_cta {
__device__ __forceinline__ void copy2(void* dst, const CUtensorMap* map, int x, int y, uint64_t* bar) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2,%3}], [%4];" :: "r"(qmm_frag::smem(dst)),
                 "l"(reinterpret_cast<uint64_t>(map)), "r"(x), "r"(y),
                 "r"(qmm_frag::smem(bar)) : "memory");
}

__device__ __forceinline__ void copy3(void* dst, const CUtensorMap* map,
                                     int x, int y, int z, uint64_t* bar) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2,%3,%4}], [%5];" :: "r"(qmm_frag::smem(dst)),
                 "l"(reinterpret_cast<uint64_t>(map)), "r"(x), "r"(y), "r"(z),
                 "r"(qmm_frag::smem(bar)) : "memory");
}

__device__ __forceinline__ void copy4(void* dst, const CUtensorMap* map,
                                     int x, int y, int z, int w, uint64_t* bar) {
    asm volatile("cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2,%3,%4,%5}], [%6];" :: "r"(qmm_frag::smem(dst)),
                 "l"(reinterpret_cast<uint64_t>(map)), "r"(x), "r"(y), "r"(z), "r"(w),
                 "r"(qmm_frag::smem(bar)) : "memory");
}

}
