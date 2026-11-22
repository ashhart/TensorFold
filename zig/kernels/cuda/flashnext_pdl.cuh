// A guarded Flash Next entry releases its successor, then waits before reading any predecessor output.
#pragma once
#include "common/qmm_frag.cuh"
namespace flashnext_pdl {
__device__ __forceinline__ void enter() {
    qmm_frag::grid_launch();
    qmm_frag::grid_wait();
}
}  // namespace flashnext_pdl
