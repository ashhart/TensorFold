// GLM vision tower's steps around cuBLASLt, as mlx-vlm glm5_next/vision.py: fp32 residual, bf16 outputs rounded once.

#include <cuda_bf16.h>
#include <stdint.h>

namespace {
__device__ __forceinline__ float bf(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ __nv_bfloat16 tobf(float x) { return __float2bfloat16_rn(x); }

// a block's sum (blockDim.x a multiple of 32, at most 1024); every thread gets it
__device__ float block_sum(float v) {
    __shared__ float part[32];
    __shared__ float total;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) part[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float s = lane < (int)(blockDim.x >> 5) ? part[lane] : 0.f;
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (lane == 0) total = s;
    }
    __syncthreads();
    return total;
}

__device__ __forceinline__ float limited_swiglu(float g, float u, float limit) {
    g = fminf(g, limit);
    u = fminf(fmaxf(u, -limit), limit);
    return g / (1.f + expf(-g)) * u;
}
}  // namespace

// dst[i] = bf16(src[i]) (the processor's fp32 patches)
extern "C" __global__ void glm_vis_to_bf16(const float* __restrict__ src, __nv_bfloat16* __restrict__ dst, long long n) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
        dst[i] = tobf(src[i]);
}

// out[r, c] = bf16(acc[r, c] + bias[c]) (bias may be null); with x32: x32[r, c] += that bf16 value (a residual add)
extern "C" __global__ void glm_vis_bias(const float* __restrict__ acc, const __nv_bfloat16* __restrict__ bias,
                                        __nv_bfloat16* __restrict__ out, float* __restrict__ x32, long long n, int cols) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        const float v = acc[i] + (bias ? bf(bias[i % cols]) : 0.f);
        const __nv_bfloat16 h = tobf(v);
        if (out) out[i] = h;
        if (x32) x32[i] += bf(h);
    }
}

// x32[i] = acc[i] + bias[i % cols] (the patch embedding: the residual stream's first value, in fp32)
extern "C" __global__ void glm_vis_embed32(const float* __restrict__ acc, const __nv_bfloat16* __restrict__ bias,
                                          float* __restrict__ x32, long long n, int cols) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
        x32[i] = bf(tobf(acc[i] + bf(bias[i % cols])));
}

// RMSNorm of fp32 rows: y = bf16(x rsqrt(mean(x^2) + eps) w), a block a row
extern "C" __global__ void glm_vis_rmsnorm(const float* __restrict__ x, const __nv_bfloat16* __restrict__ w,
                                           __nv_bfloat16* __restrict__ y, int cols, float eps) {
    const float* row = x + (long long)blockIdx.x * cols;
    float s = 0.f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) s += row[c] * row[c];
    const float inv = rsqrtf(block_sum(s) / cols + eps);
    for (int c = threadIdx.x; c < cols; c += blockDim.x) y[(long long)blockIdx.x * cols + c] = tobf(row[c] * inv * bf(w[c]));
}

// attention inputs: q, k head-RMSNormed and rotated (fp32, 2D angles) into qh, kh; v into vt; a warp a (row, head).
extern "C" __global__ void glm_vis_qkv(const __nv_bfloat16* __restrict__ qkv, const __nv_bfloat16* __restrict__ qn,
                                       const __nv_bfloat16* __restrict__ kn, const int* __restrict__ pos,
                                       const float* __restrict__ inv_freq, __nv_bfloat16* __restrict__ qh,
                                       __nv_bfloat16* __restrict__ kh, __nv_bfloat16* __restrict__ vt, int N, int seg,
                                       int H, float eps) {
    const long long item = blockIdx.x * (long long)(blockDim.x >> 5) + (threadIdx.x >> 5);
    if (item >= (long long)N * H) return;
    const int lane = threadIdx.x & 31;
    const int n = (int)(item / H), h = (int)(item % H);
    const int g = n / seg, j = n % seg;
    const long long blk = (long long)g * H + h;
    const __nv_bfloat16* base = qkv + (long long)n * 3 * H * 64 + h * 64;
    const int p = lane < 16 ? pos[2 * n] : pos[2 * n + 1];
    const float ang = (float)p * inv_freq[lane & 15];
    const float c = cosf(ang), s = sinf(ang);
    for (int which = 0; which < 2; ++which) {
        const __nv_bfloat16* src = base + which * H * 64;
        const __nv_bfloat16* wn = which ? kn : qn;
        float a = bf(src[lane]), b = bf(src[lane + 32]);
        float ss = a * a + b * b;
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        const float inv = rsqrtf(ss / 64.f + eps);
        a = bf(tobf(a * inv * bf(wn[lane])));
        b = bf(tobf(b * inv * bf(wn[lane + 32])));
        // rotate_half: dims l < 32 take -x[l + 32], dims l + 32 take x[l]
        const float ra = a * c - b * s, rb = b * c + a * s;
        __nv_bfloat16* dst = (which ? kh : qh) + (blk * seg + j) * 64;
        dst[lane] = tobf(ra);
        dst[lane + 32] = tobf(rb);
    }
    const __nv_bfloat16* v = base + 2 * H * 64;
    vt[(blk * 64 + lane) * seg + j] = v[lane];
    vt[(blk * 64 + lane + 32) * seg + j] = v[lane + 32];
}

// A scores row's softmax: pr = bf16(exp(scale (s - max))) and its sum (of the bf16 values), a block a row
extern "C" __global__ void glm_vis_softmax(const float* __restrict__ sc, __nv_bfloat16* __restrict__ pr,
                                           float* __restrict__ sums, int n, float scale) {
    const float* row = sc + (long long)blockIdx.x * n;
    __shared__ float mx_s[32];
    __shared__ float mx;
    float m = -INFINITY;
    for (int c = threadIdx.x; c < n; c += blockDim.x) m = fmaxf(m, row[c]);
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if ((threadIdx.x & 31) == 0) mx_s[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < (blockDim.x >> 5) ? mx_s[threadIdx.x] : -INFINITY;
        for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
        if (threadIdx.x == 0) mx = v;
    }
    __syncthreads();
    float s = 0.f;
    for (int c = threadIdx.x; c < n; c += blockDim.x) {
        const __nv_bfloat16 e = tobf(expf((row[c] - mx) * scale));
        pr[(long long)blockIdx.x * n + c] = e;
        s += bf(e);
    }
    s = block_sum(s);
    if (threadIdx.x == 0) sums[blockIdx.x] = s;
}

// att[r, h * 64 + d] = bf16(oh[r, d] / sums[r]) (a head's values product, normalized, into its columns)
extern "C" __global__ void glm_vis_head_out(const float* __restrict__ oh, const float* __restrict__ sums,
                                            __nv_bfloat16* __restrict__ att, int rows, int H, int h) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * 64) return;
    const int r = i / 64, d = i % 64;
    att[(long long)r * H * 64 + h * 64 + d] = tobf(oh[i] / sums[r]);
}

// out = bf16(limited_swiglu(bf16(g + gb), bf16(u + ub))) from the two fp32 products (biases may be null)
extern "C" __global__ void glm_vis_swiglu(const float* __restrict__ g, const float* __restrict__ u,
                                          const __nv_bfloat16* __restrict__ gb, const __nv_bfloat16* __restrict__ ub,
                                          __nv_bfloat16* __restrict__ out, long long n, int cols, float limit) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        const int c = (int)(i % cols);
        const float gv = bf(tobf(g[i] + (gb ? bf(gb[c]) : 0.f)));
        const float uv = bf(tobf(u[i] + (ub ? bf(ub[c]) : 0.f)));
        out[i] = tobf(limited_swiglu(gv, uv, limit));
    }
}

// The merger's norm: y = bf16(gelu_erf(LayerNorm(bf16(acc)) w + b)), a block a row of ``cols``
extern "C" __global__ void glm_vis_ln_gelu(const float* __restrict__ acc, const __nv_bfloat16* __restrict__ w,
                                           const __nv_bfloat16* __restrict__ b, __nv_bfloat16* __restrict__ y, int cols,
                                           float eps) {
    const float* row = acc + (long long)blockIdx.x * cols;
    float s = 0.f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) s += bf(tobf(row[c]));
    const float mean = block_sum(s) / cols;
    float v = 0.f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        const float d = bf(tobf(row[c])) - mean;
        v += d * d;
    }
    const float inv = rsqrtf(block_sum(v) / cols + eps);
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        const float x = bf(tobf((bf(tobf(row[c])) - mean) * inv * bf(w[c]) + bf(b[c])));
        y[(long long)blockIdx.x * cols + c] = tobf(0.5f * x * (1.f + erff(x * 0.70710678118654752f)));
    }
}
