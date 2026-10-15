// GLM-5.3-Flash on CUDA: the small steps the Python engine runs as torch ops (integers and casts, so the same values).

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

// sparse.select_tokens: row r's 512 pools' tokens ascending, then the last pool's visible ones; 0 count when dense.
extern "C" __global__ void glm_sparse_tokens(const int64_t* __restrict__ pools, const int* __restrict__ pos, int rows,
                                             int* __restrict__ tokens, int* __restrict__ counts) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    const int64_t q = (int64_t)pos[0] + r;
    const int64_t npool = (q + 1) / 4;
    int* out = tokens + (int64_t)r * 2051;
    for (int i = threadIdx.x; i < 2048; i += blockDim.x) out[i] = (int)(pools[(int64_t)r * 512 + i / 4] * 4 + i % 4);
    if (threadIdx.x < 3) {
        const int64_t t = npool * 4 + threadIdx.x;
        out[2048 + threadIdx.x] = t <= q ? (int)t : -1;
    }
    if (threadIdx.x == 0) {
        int tail = 0;
        for (int j = 0; j < 3; ++j) tail += (npool * 4 + j) <= q;
        counts[r] = npool > 512 ? 2048 + tail : 0;
    }
}

// dst[r * dst_stride + c] = bf16(src[r * src_stride + c]) (torch's fp32 -> bf16 copy_: round to nearest even)
extern "C" __global__ void glm_cast_bf16(const float* __restrict__ src, long long src_stride, __nv_bfloat16* __restrict__ dst,
                                         long long dst_stride, int cols) {
    const int r = blockIdx.y;
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < cols; c += gridDim.x * blockDim.x)
        dst[r * dst_stride + c] = __float2bfloat16_rn(src[r * src_stride + c]);
}

// out[i] = base[0] + off + i * step (a block's first-row position on the device: Python's pos_dev + r0)
extern "C" __global__ void glm_positions(const int* __restrict__ base, int off, int step, int n, int* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = base[0] + off + i * step;
}

// rows of ``words`` 32-bit words from src (row pitch spitch words) to dst (dpitch words): torch's strided copy_
extern "C" __global__ void glm_copy_rows(const uint32_t* __restrict__ src, long long spitch, uint32_t* __restrict__ dst,
                                         long long dpitch, int words) {
    const long long r = blockIdx.y;
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < words; c += gridDim.x * blockDim.x) dst[r * dpitch + c] = src[r * spitch + c];
}

// -- packed dense linears (TF_GLM_DENSE_BITS): K / 16 chunks of 16 offset BITS-bit values, fp16 scale a 64 ----------

template <int BITS>
__device__ __forceinline__ void glm_unpack16(const uint32_t* __restrict__ p, float v[16]) {
    if constexpr (BITS == 4) {
        const uint32_t a = p[0], b = p[1];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            v[i] = (float)((int)((a >> (4 * i)) & 15u) - 8);
            v[8 + i] = (float)((int)((b >> (4 * i)) & 15u) - 8);
        }
    } else if constexpr (BITS == 8) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t w = p[j];
#pragma unroll
            for (int i = 0; i < 4; ++i) v[4 * j + i] = (float)((int)((w >> (8 * i)) & 255u) - 128);
        }
    } else {
        const uint32_t w0 = p[0], w1 = p[1], w2 = p[2];
#pragma unroll
        for (int i = 0; i < 5; ++i) {
            v[i] = (float)((int)((w0 >> (6 * i)) & 63u) - 32);
            v[5 + i] = (float)((int)((w1 >> (6 * i)) & 63u) - 32);
            v[10 + i] = (float)((int)((w2 >> (6 * i)) & 63u) - 32);
        }
        v[15] = (float)((int)((w0 >> 30) | ((w1 >> 30) << 2) | ((w2 >> 30) << 4)) - 32);
    }
}

template <int BITS>
__device__ __forceinline__ void glm_load_chunk(const uint32_t* __restrict__ p, uint32_t* w) {
    if constexpr (BITS == 4) {
        const uint2 a = __ldg(reinterpret_cast<const uint2*>(p));
        w[0] = a.x; w[1] = a.y;
    } else if constexpr (BITS == 8) {
        const uint4 a = __ldg(reinterpret_cast<const uint4*>(p));
        w[0] = a.x; w[1] = a.y; w[2] = a.z; w[3] = a.w;
    } else {
        w[0] = __ldg(p); w[1] = __ldg(p + 1); w[2] = __ldg(p + 2);
    }
}

// out[m, n] = sum_k x[m, k] q[n, k] s[n, k / 64], M <= 16: a warp a row, fp32 chunks, xor reduce; independent of M.
template <int BITS>
__device__ __forceinline__ void glm_qgemv(const __nv_bfloat16* __restrict__ x, long long xs, const uint32_t* __restrict__ w,
                                          const __half* __restrict__ s, void* __restrict__ out, int f32, int M, int N, int K) {
    constexpr int WPC = BITS / 2;
    const int lane = threadIdx.x & 31;
    const int n = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (n >= N) return;
    const int chunks = K >> 4;
    const uint32_t* wr = w + (long long)n * chunks * WPC;
    const __half* sr = s + (long long)n * (K >> 6);
    float acc[16];
#pragma unroll
    for (int m = 0; m < 16; ++m) acc[m] = 0.f;
    for (int c = lane; c < chunks; c += 32) {
        uint32_t pw[WPC];
        glm_load_chunk<BITS>(wr + (long long)c * WPC, pw);
        float v[16];
        glm_unpack16<BITS>(pw, v);
        const float sc = __half2float(sr[c >> 2]);
#pragma unroll
        for (int m = 0; m < 16; ++m) {
            if (m < M) {
                const uint4* xp = reinterpret_cast<const uint4*>(x + m * xs + (c << 4));
                const uint4 a = __ldg(xp), b = __ldg(xp + 1);
                const __nv_bfloat162* ha = reinterpret_cast<const __nv_bfloat162*>(&a);
                const __nv_bfloat162* hb = reinterpret_cast<const __nv_bfloat162*>(&b);
                float p = 0.f;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float2 f = __bfloat1622float2(ha[i]);
                    p = fmaf(v[2 * i], f.x, p);
                    p = fmaf(v[2 * i + 1], f.y, p);
                }
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float2 f = __bfloat1622float2(hb[i]);
                    p = fmaf(v[8 + 2 * i], f.x, p);
                    p = fmaf(v[8 + 2 * i + 1], f.y, p);
                }
                acc[m] = fmaf(p, sc, acc[m]);
            }
        }
    }
#pragma unroll
    for (int m = 0; m < 16; ++m) {
        if (m < M) {
            float a = acc[m];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) a += __shfl_xor_sync(0xffffffffu, a, off);
            if (lane == 0) {
                if (f32) reinterpret_cast<float*>(out)[(long long)m * N + n] = a;
                else reinterpret_cast<__nv_bfloat16*>(out)[(long long)m * N + n] = __float2bfloat16_rn(a);
            }
        }
    }
}

// the rows as bf16 [N, K] (rounded to nearest even from value * scale), for prompt-sized calls on the bf16 matmul
template <int BITS>
__device__ __forceinline__ void glm_qdequant(const uint32_t* __restrict__ w, const __half* __restrict__ s,
                                             __nv_bfloat16* __restrict__ out, long long chunks) {
    const long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= chunks) return;
    uint32_t pw[BITS / 2];
    glm_load_chunk<BITS>(w + c * (BITS / 2), pw);
    float v[16];
    glm_unpack16<BITS>(pw, v);
    const float sc = __half2float(s[c >> 2]);
    __align__(16) __nv_bfloat16 o[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) o[i] = __float2bfloat16_rn(v[i] * sc);
    uint4* dst = reinterpret_cast<uint4*>(out + c * 16);
    dst[0] = reinterpret_cast<const uint4*>(o)[0];
    dst[1] = reinterpret_cast<const uint4*>(o)[1];
}

#define GLM_Q(B)                                                                                                         \
    extern "C" __global__ void __launch_bounds__(256) glm_qgemv##B(const __nv_bfloat16* __restrict__ x, long long xs,    \
        const uint32_t* __restrict__ w, const __half* __restrict__ s, void* __restrict__ out, int f32, int M, int N,     \
        int K) { glm_qgemv<B>(x, xs, w, s, out, f32, M, N, K); }                                                         \
    extern "C" __global__ void glm_qdequant##B(const uint32_t* __restrict__ w, const __half* __restrict__ s,             \
        __nv_bfloat16* __restrict__ out, long long chunks) { glm_qdequant<B>(w, s, out, chunks); }
GLM_Q(4)
GLM_Q(6)
GLM_Q(8)

// x rows 0 .. rows - 1 (``copies`` streams of 4096 bf16 each) take feats [rows, 4096]: image features for placeholders
extern "C" __global__ void glm_inject_rows(const __nv_bfloat16* __restrict__ feats, __nv_bfloat16* __restrict__ x, int rows,
                                           int copies) {
    const long long n = (long long)rows * copies * 4096;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        const long long r = i / (copies * 4096), c = i % 4096;
        x[i] = feats[r * 4096 + c];
    }
}
