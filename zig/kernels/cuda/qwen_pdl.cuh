// Qwen3.8 glue kernels as dependent launches (sm_90 on).
#pragma once

#ifdef __CUDACC__  // the host emulator compiles Qwen kernels without CUDA headers
#include "common/qmm_frag.cuh"
#endif

namespace qwen_pdl {

// The next grid may launch once every CTA of this one has run this (or ended).
__device__ __forceinline__ void trigger() {
#ifdef __CUDACC__
  qmm_frag::grid_launch();
#endif
}

// Returns once the grid before has completed and its writes are visible; read those coherently (no const __restrict__).
__device__ __forceinline__ void wait() {
#ifdef __CUDACC__
  qmm_frag::grid_wait();
#endif
}

// Every Qwen glue kernel's first statement: a dependent launch then reads and writes exactly as a plain one.
__device__ __forceinline__ void enter() {
  trigger();
  wait();
}

}  // namespace qwen_pdl
