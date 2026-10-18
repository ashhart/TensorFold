// bf16 rows times bf16 weight rows into fp32 for every CUDA family: routers and output heads (core/dense_ref.zig).

// A warp an output column, RT rows a launch tile; each row's sum runs in one order whatever RT and the row count.

#include <cuda_bf16.h>
#include <stdint.h>

namespace tf_dense {

constexpr int WARPS = 8;
constexpr int VEC = 8; // bf16 a 16-byte load

__device__ __forceinline__ void unpack(const uint4& u, float* f) {
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 t = __bfloat1622float2(h[i]);
        f[2 * i] = t.x;
        f[2 * i + 1] = t.y;
    }
}

// out[r, n] = x[r] . w[n] for a tile's rows: lane l sums k = 8 l + 256 j in order, then the fixed shuffle tree.
template <int RT>
__device__ __forceinline__ void tile(const __nv_bfloat16* x, int64_t x_stride, const __nv_bfloat16* w, float* out,
                                     int64_t out_stride, int rows, int n, int k) {
    const int lane = threadIdx.x & 31;
    const int col = blockIdx.x * WARPS + (threadIdx.x >> 5);
    const int r0 = blockIdx.y * RT;
    if (col >= n) return;
    float acc[RT];
#pragma unroll
    for (int r = 0; r < RT; ++r) acc[r] = 0.f;
    const __nv_bfloat16* wr = w + static_cast<int64_t>(col) * k;
#pragma unroll 4 // several 16-byte weight loads in flight a lane (memory-level parallelism)
    for (int k0 = lane * VEC; k0 < k; k0 += 32 * VEC) {
        float wf[VEC];
        unpack(*reinterpret_cast<const uint4*>(wr + k0), wf);
#pragma unroll
        for (int r = 0; r < RT; ++r) {
            if (r0 + r >= rows) break;
            float xf[VEC];
            unpack(*reinterpret_cast<const uint4*>(x + (r0 + r) * x_stride + k0), xf);
#pragma unroll
            for (int i = 0; i < VEC; ++i) acc[r] = __fmaf_rn(xf[i], wf[i], acc[r]);
        }
    }
#pragma unroll
    for (int r = 0; r < RT; ++r) {
        float v = acc[r];
        for (int m = 16; m > 0; m >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, m));
        if (lane == 0 && r0 + r < rows) out[(r0 + r) * out_stride + col] = v;
    }
}

}  // namespace tf_dense

using namespace tf_dense;

#define TF_DENSE_ROWS(RT)                                                                                              \
    extern "C" __global__ void __launch_bounds__(WARPS * 32) tf_dense_rows_##RT(                                     \
        const __nv_bfloat16* x, long long x_stride, const __nv_bfloat16* w, float* out, long long out_stride,          \
        int rows, int n, int k) {                                                                                      \
        tile<RT>(x, x_stride, w, out, out_stride, rows, n, k);                                                         \
    }

TF_DENSE_ROWS(1)
TF_DENSE_ROWS(4)
TF_DENSE_ROWS(16)
TF_DENSE_ROWS(32)
