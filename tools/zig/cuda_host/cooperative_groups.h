#pragma once
#include "cuda_runtime.h"
namespace cooperative_groups {
struct cluster_group {
    __device__ void sync(); __device__ unsigned block_rank(); __device__ unsigned num_blocks();
    template<class T> __device__ T* map_shared_rank(T*,unsigned);
};
__device__ cluster_group this_cluster();
}
