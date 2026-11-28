// Hy4-flash tiled row-dot helper (`rows.cu` and planted controls): bit-equal to `tf_dense_rows_1`.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include "../common/cp_async.cuh"
#include "../common/reduce.cuh"
#include "../common/vec.cuh"

namespace tf_hy4_rows {

constexpr int WARPS = 8;
constexpr int VEC = 8;  // bf16 per 16-byte load

// Lane l adds k = 8*l + 256*j in order, then fixed warp shuffle tree; CONTROL 1..4 plant contract faults.
template <int RT, int C, int CONTROL>
__device__ __forceinline__ void tile(const __nv_bfloat16* x, int64_t x_stride, const __nv_bfloat16* w, float* out,
                                     int64_t out_stride, int rows, int n, int k, int repeat) {
    const int lane = threadIdx.x & 31;
    const int col0 = (blockIdx.x * WARPS + (threadIdx.x >> 5)) * C;
    const int r0 = blockIdx.y * RT;
    if (col0 >= n) return;
    const int nr = min(RT, rows - r0) - (CONTROL == 3 && r0 + RT >= rows ? 1 : 0);
    const __nv_bfloat16* wr[C];
#pragma unroll
    for (int c = 0; c < C; ++c) wr[c] = w + static_cast<int64_t>(min(col0 + c, n - 1)) * k;
    float acc[RT][C];
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
        for (int c = 0; c < C; ++c) acc[r][c] = 0.f;
#pragma unroll 2
    for (int k0 = lane * VEC; k0 < k; k0 += 32 * VEC) {
        float wf[C][VEC];
#pragma unroll
        for (int c = 0; c < C; ++c) {
            tf_common::unpack8(tf_common::ld_nc(reinterpret_cast<const uint4*>(wr[c] + k0)), wf[c]);
        }
#pragma unroll
        for (int r = 0; r < RT; ++r) {
            if (r >= nr) break;
            float xf[VEC];
            tf_common::unpack8(*reinterpret_cast<const uint4*>(x + (r0 + r) * x_stride + k0), xf);
#pragma unroll
            for (int c = 0; c < C; ++c)
#pragma unroll
                for (int i = 0; i < VEC; ++i) {
                    acc[r][c] = __fmaf_rn(xf[i], wf[c][i], acc[r][c]);
                    if constexpr (CONTROL == 4) acc[r][c] = __half2float(__float2half_rn(acc[r][c]));
                }
        }
    }
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
        for (int c = 0; c < C; ++c) {
            float v = acc[r][c];
            for (int m = 16; m > 0; m >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, m));
            if constexpr (CONTROL == 1) v = v != v ? 0.f : v;
            if constexpr (CONTROL == 2) v = __fadd_rn(v, static_cast<float>(repeat));
            if (lane == 0 && r < nr && col0 + c < n) out[(r0 + r) * out_stride + col0 + c] = v;
        }
}

}  // namespace tf_hy4_rows
