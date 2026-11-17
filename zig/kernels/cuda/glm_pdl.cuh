// GLM kernels as dependent launches: each one lets the next kernel in, then waits for the one before (sm_90 on).
#pragma once

#include "common/qmm_frag.cuh"

namespace glm_pdl {

// The next grid may launch once every CTA of this one has run this (or ended).
__device__ __forceinline__ void trigger() { qmm_frag::grid_launch(); }

// Returns once the grid before has completed and its writes are visible; read those coherently (no const __restrict__).
__device__ __forceinline__ void wait() { qmm_frag::grid_wait(); }

// Every GLM kernel's first statement: a dependent launch then reads and writes exactly as a plain one.
__device__ __forceinline__ void enter() {
  trigger();
  wait();
}

// A kernel that writes weights waits but lets no dependent in early: qmmf and glm_mm read weights before their wait.
__device__ __forceinline__ void enter_weights() { wait(); }

}  // namespace glm_pdl
