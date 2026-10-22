// Picks as the next forward's ids: a pick's token, or `fallback` for none (its flag stays); the control drops it.
#pragma once

#include <stdint.h>

namespace tf_round_ids {

enum : int { FALLBACK = 1 };
constexpr int ALL = 1;

__device__ __forceinline__ void pick_ids(const uint32_t* __restrict__ picks, int* __restrict__ ids, unsigned rows,
                                         unsigned vocab, int fallback, int keep) {
    const unsigned r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const uint32_t t = picks[2 * r];  // Pick: token, then its nonfinite kinds
    ids[r] = ((keep & FALLBACK) && t >= vocab) ? fallback : static_cast<int>(t);
}

}  // namespace tf_round_ids
