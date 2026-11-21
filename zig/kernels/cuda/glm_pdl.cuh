// GLM kernels as dependent launches: each one lets the next kernel in, then waits for the one before (sm_90 on).
#pragma once

#ifdef __CUDACC__  // the host emulator (tools/glm_index_emulate) compiles GLM kernels without CUDA headers
#include "common/qmm_frag.cuh"
#endif

namespace glm_pdl {

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

// Every GLM kernel's first statement: a dependent launch then reads and writes exactly as a plain one.
__device__ __forceinline__ void enter() {
  trigger();
  wait();
}

// A kernel that writes weights waits but lets no dependent in early: qmmf and glm_mm read weights before their wait.
__device__ __forceinline__ void enter_weights() { wait(); }

}  // namespace glm_pdl
