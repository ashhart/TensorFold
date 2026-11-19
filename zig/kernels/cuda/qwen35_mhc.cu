#include <cuda_bf16.h>
#include <math.h>
#include <stddef.h>

static __device__ float bf(__nv_bfloat16 x) { return __bfloat162float(x); }
static __device__ float round_bf(float x) { return bf(__float2bfloat16_rn(x)); }
static __device__ bool past(const int* p, int row) { return p && row >= p[2]; }
static __device__ float lse(float a, float b) {
    float m = fmaxf(a, b);
    return m + logf(expf(a - m) + expf(b - m));
}

// Grid (rows), block (256).
extern "C" __global__ void qwen35_mhc_enter(__nv_bfloat16* h, const __nv_bfloat16* add,
                                              __nv_bfloat16* r, int d, const int* p) {
    int row = blockIdx.x;
    if (past(p, row)) return;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        size_t hidx = (size_t)row * d + i, ridx = (size_t)row * 2 * d + i;
#ifdef MHC_CONTROL
        __nv_bfloat16 v = h[hidx];
#else
        __nv_bfloat16 v = add ? __float2bfloat16_rn(bf(h[hidx]) + bf(add[hidx])) : h[hidx];
#endif
        h[hidx] = v;
        r[ridx] = v;
        r[ridx + d] = v;
    }
}

// Grid (rows), block (256); reduction uses fixed thread and tree order.
extern "C" __global__ void qwen35_mhc_route(const __nv_bfloat16* r, const float* w, const float* b,
                                              __nv_bfloat16* branch, float* post, float* mix,
                                              int d, int iterations, float pre_scale, const int* p) {
    int row = blockIdx.x, t = threadIdx.x, width = 2 * d;
    if (past(p, row)) return;
    const __nv_bfloat16* x = r + (size_t)row * width;
    __shared__ float scratch[256], logits[8], inv, pre[2];
    float sum = 0.0f;
    for (int i = t; i < width; i += 256) { float v = bf(x[i]); sum = __fadd_rn(sum, __fmul_rn(v, v)); }
    scratch[t] = sum;
    __syncthreads();
    for (int stride = 128; stride; stride >>= 1) {
        if (t < stride) scratch[t] = __fadd_rn(scratch[t], scratch[t + stride]);
        __syncthreads();
    }
    if (!t) inv = rsqrtf(scratch[0] / width + 1e-6f);
    __syncthreads();
    for (int k = 0; k < 8; ++k) {
        sum = 0.0f;
        for (int i = t; i < width; i += 256) sum = __fadd_rn(sum, __fmul_rn(__fmul_rn(bf(x[i]), inv), w[(size_t)k * width + i]));
        scratch[t] = sum;
        __syncthreads();
        for (int stride = 128; stride; stride >>= 1) {
            if (t < stride) scratch[t] = __fadd_rn(scratch[t], scratch[t + stride]);
            __syncthreads();
        }
        if (!t) logits[k] = scratch[0] + b[k];
        __syncthreads();
    }
    if (!t) {
        for (int s = 0; s < 2; ++s) {
            pre[s] = round_bf(1.0f / (1.0f + expf(-logits[s])));
            post[row * 2 + s] = 2.0f / (1.0f + expf(-logits[s + 2]));
#ifdef MHC_CONTROL
            post[row * 2 + s] += 0.125f;
#endif
        }
        float a = logits[4], b0 = logits[5], c = logits[6], d0 = logits[7];
        for (int j = 0; j < iterations; ++j) {
            float u = lse(a, b0), v = lse(c, d0);
            a -= u; b0 -= u; c -= v; d0 -= v;
            u = lse(a, c); v = lse(b0, d0);
            a -= u; c -= u; b0 -= v; d0 -= v;
        }
        mix[row * 4] = expf(a); mix[row * 4 + 1] = expf(b0);
        mix[row * 4 + 2] = expf(c); mix[row * 4 + 3] = expf(d0);
    }
    __syncthreads();
    for (int i = t; i < d; i += 256) {
        __nv_bfloat16 v = __float2bfloat16_rn(
            __fadd_rn(__fmul_rn(pre[0], bf(x[i])), __fmul_rn(pre[1], bf(x[d + i]))));
        if (pre_scale != 1.0f) v = __float2bfloat16_rn(__fmul_rn(bf(v), pre_scale));
        branch[(size_t)row * d + i] = v;
    }
}

// Grid (rows), block (256).
extern "C" __global__ void qwen35_mhc_merge(__nv_bfloat16* r, const __nv_bfloat16* branch,
                                              const float* post, const float* mix, int d, const int* p) {
    int row = blockIdx.x;
    if (past(p, row)) return;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        size_t at = (size_t)row * 2 * d + i;
        float x = bf(r[at]), y = bf(r[at + d]), v = bf(branch[(size_t)row * d + i]);
        float o[2];
        for (int s = 0; s < 2; ++s) {
            float m = fmaf(round_bf(mix[row * 4 + s * 2 + 1]), y,
                           __fadd_rn(__fmul_rn(round_bf(mix[row * 4 + s * 2]), x), 0.0f));
#ifdef MHC_CONTROL
            o[s] = round_bf(m);
#else
            o[s] = round_bf(m) + round_bf(round_bf(post[row * 2 + s]) * v);
#endif
        }
        r[at] = __float2bfloat16_rn(o[0]); r[at + d] = __float2bfloat16_rn(o[1]);
    }
}

// Grid (rows), block (256).
extern "C" __global__ void qwen35_mhc_leave(const __nv_bfloat16* r, __nv_bfloat16* h, int d, const int* p) {
    int row = blockIdx.x;
    if (past(p, row)) return;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        size_t at = (size_t)row * 2 * d + i;
#ifdef MHC_CONTROL
        h[(size_t)row * d + i] = r[at];
#else
        h[(size_t)row * d + i] = __float2bfloat16_rn((bf(r[at]) + bf(r[at + d])) * 0.5f);
#endif
    }
}
