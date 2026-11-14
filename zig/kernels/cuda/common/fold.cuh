// Serial folds: n terms added in index order, one IEEE add each, so every caller's chain keeps its bits.
#pragma once

namespace tf_common {

// at(0) + at(1) + ... + at(n - 1) from 0.0f; U > 0 unrolls by U, 0 leaves the unroll to the compiler.
template <int U = 0, typename At>
__device__ __forceinline__ float sum_rn(int n, At at) {
    float total = 0.0f;
    if constexpr (U == 0) {
        for (int i = 0; i < n; ++i) total = __fadd_rn(total, at(i));
    } else {
#pragma unroll (U)
        for (int i = 0; i < n; ++i) total = __fadd_rn(total, at(i));
    }
    return total;
}

}  // namespace tf_common
