// DSpark's pick merge, one body for every family: the ranks' gathered (value, id) picks to the walk's next token.
#pragma once

#include <math_constants.h>

namespace tf_common {

// Thread 0, picks stride bytes apart: the largest value, the lower id on a tie, kNone when none beats -Inf.
template <int kNone>
__device__ __forceinline__ void dspark_pick(const unsigned char* __restrict__ picks, int world, int stride,
                                            int* __restrict__ walk, int j) {
  if (threadIdx.x != 0) return;
  float best = -CUDART_INF_F;
  int id = kNone;
  for (int q = 0; q < world; ++q) {
    const float v = *reinterpret_cast<const float*>(picks + (size_t)q * stride);
    const int k = *reinterpret_cast<const int*>(picks + (size_t)q * stride + 4);
    if (v > best || (v == best && k < id)) {
      best = v;
      id = k;
    }
  }
  walk[j + 1] = id;
}

}  // namespace tf_common
