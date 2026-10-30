// Qwen3.5 bf16-weight pieces: token rows of a bf16 table and small dense projections (GDN in_proj_a / in_proj_b).

#include <cuda_bf16.h>
#include <stdint.h>

// P: device counts as qwen35.cu's; rows at or past P[2] are padding and left untouched; may be null.
__device__ __forceinline__ bool q35b_past_rows(const int *P, int m) { return P != nullptr && m >= P[2]; }

// out[r] = table[ids[r]], D wide, a bit copy; grid: rows; 16-byte vectors when D % 8 == 0 and aligned.
extern "C" __global__ void __launch_bounds__(256)
    tf_q35_embed_bf16(const int *__restrict__ ids, const __nv_bfloat16 *__restrict__ table,
                      __nv_bfloat16 *__restrict__ out, int D, const int *__restrict__ P) {
    if (q35b_past_rows(P, blockIdx.x))
        return;
    const size_t id = (size_t)ids[blockIdx.x];
    const __nv_bfloat16 *src = table + id * (size_t)D;
    __nv_bfloat16 *dst = out + (size_t)blockIdx.x * D;
    if ((D & 7) == 0 && ((reinterpret_cast<uintptr_t>(src) | reinterpret_cast<uintptr_t>(dst)) & 15) == 0) {
        for (int i = threadIdx.x; i < D / 8; i += blockDim.x)
            reinterpret_cast<uint4 *>(dst)[i] = reinterpret_cast<const uint4 *>(src)[i];
    } else {
        for (int i = threadIdx.x; i < D; i += blockDim.x)
            dst[i] = src[i];
    }
}

// y[r][j] = bf16(sum_k x[r][k] * W[j][k]) in a fixed order whatever the row count; grid (n, rows), k % 8 == 0, aligned.
extern "C" __global__ void __launch_bounds__(256)
    tf_q35_dense_bf16(const __nv_bfloat16 *__restrict__ X, const __nv_bfloat16 *__restrict__ W,
                      __nv_bfloat16 *__restrict__ Y, int n, int k) {
    __shared__ float red[8];
    const int j = blockIdx.x;
    const size_t r = blockIdx.y;
    const uint4 *x = reinterpret_cast<const uint4 *>(X + r * (size_t)k);
    const uint4 *w = reinterpret_cast<const uint4 *>(W + (size_t)j * k);
    float acc = 0.0f;
    for (int c = threadIdx.x; c < k / 8; c += 256) {
        const uint4 xv = x[c], wv = w[c];
        const __nv_bfloat16 *xb = reinterpret_cast<const __nv_bfloat16 *>(&xv);
        const __nv_bfloat16 *wb = reinterpret_cast<const __nv_bfloat16 *>(&wv);
#pragma unroll
        for (int e = 0; e < 8; e++)
            acc = __fmaf_rn(__bfloat162float(xb[e]), __bfloat162float(wb[e]), acc);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if ((threadIdx.x & 31) == 0)
        red[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = red[0];
#pragma unroll
        for (int i = 1; i < 8; i++)
            s += red[i];
        Y[r * (size_t)n + j] = __float2bfloat16_rn(s);
    }
}
