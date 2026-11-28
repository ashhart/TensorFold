// Inkling 2's prompt-row kernels: the router's row dots in tiles (tf_ink2_rowdot's bits), the dense GEMM's epilogue.
#include <cstdint>

#include "exl3_candidate.cuh"

namespace ink2r {

constexpr int kRows = 32, kOuts = 16, kCols = 128;

__device__ __forceinline__ float bf(uint16_t u) { return __uint_as_float(static_cast<uint32_t>(u) << 16); }

}  // namespace ink2r

// fp32 sigmoid?(x[row] . W[j]): lane l sums columns l, l + 32, ... then the xor tree, as tf_ink2_rowdot; C % 128 == 0.
extern "C" __global__ void __launch_bounds__(256) tf_ink2_rowdot_tiles(const uint16_t* __restrict__ x,
                                                                       const uint16_t* __restrict__ W,
                                                                       float* __restrict__ out, int rows, int J,
                                                                       int C, int act) {
    using namespace ink2r;
    __shared__ float xs[kRows][kCols];
    __shared__ float ws[kOuts][kCols];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int r0 = blockIdx.x * kRows, j0 = blockIdx.y * kOuts;
    float acc[4][kOuts];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < kOuts; ++j) acc[i][j] = 0.0f;
    for (int c0 = 0; c0 < C; c0 += kCols) {
        __syncthreads();
        for (int e = threadIdx.x; e < (kRows + kOuts) * kCols / 8; e += 256) {
            const int r = e / (kCols / 8), c = (e % (kCols / 8)) * 8;
            const bool wrow = r >= kRows;
            const int src = wrow ? j0 + r - kRows : r0 + r;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (src < (wrow ? J : rows))
                v = *reinterpret_cast<const uint4*>((wrow ? W : x) + static_cast<size_t>(src) * C + c0 + c);
            float* d = wrow ? &ws[r - kRows][c] : &xs[r][c];
            const uint32_t u[4] = {v.x, v.y, v.z, v.w};
            for (int k = 0; k < 4; ++k) {
                d[2 * k] = ink2r::bf(static_cast<uint16_t>(u[k] & 0xffffu));
                d[2 * k + 1] = ink2r::bf(static_cast<uint16_t>(u[k] >> 16));
            }
        }
        __syncthreads();
        for (int m = 0; m < kCols / 32; ++m) {
            const int c = m * 32 + lane;
            float xv[4], wv[kOuts];
#pragma unroll
            for (int i = 0; i < 4; ++i) xv[i] = xs[warp * 4 + i][c];
#pragma unroll
            for (int j = 0; j < kOuts; ++j) wv[j] = ws[j][c];
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < kOuts; ++j) acc[i][j] += xv[i] * wv[j];
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row = r0 + warp * 4 + i;
#pragma unroll
        for (int j = 0; j < kOuts; ++j) {
            float s = acc[i][j];
            for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
            if (lane != 0 || row >= rows || j0 + j >= J) continue;
            out[static_cast<size_t>(row) * J + j0 + j] = act == 1 ? 1.0f / (1.0f + expf(-s)) : s;
        }
    }
}

// A warp a (row, 128-column block) of a prompt GEMM's fp32 sums: transform, svh, 2^-3 (tf_exl3_prompt_job's tail).
extern "C" __global__ void __launch_bounds__(32) tf_ink2_prompt_tail(const float* __restrict__ z, const __half* svh,
                                                                     unsigned n, uint8_t* y, unsigned y_f32) {
    const unsigned lane = threadIdx.x, row = blockIdx.y, ob = blockIdx.x;
    float v[4];
    for (int q = 0; q < 4; ++q) v[q] = z[static_cast<uint64_t>(row) * n + ob * 128 + lane + 32 * q];
    tf_exl3_fwht(v, lane);
    for (int q = 0; q < 4; ++q) {
        const unsigned c = ob * 128 + lane + 32 * q;
        tf_exl3_store(y, static_cast<uint64_t>(row) * n + c, __fmul_rn(v[q], __fmul_rn(__half2float(svh[c]), 0.125f)), y_f32);
    }
}
