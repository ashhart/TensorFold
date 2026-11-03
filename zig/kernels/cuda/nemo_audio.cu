// Nemotron 3.5 Transcribe's audio encoder kernels (families/nemo_audio), rounding as torch's bf16 path does op for op.

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

#include "common/glue_math.cuh"

using namespace tf_glue;

namespace {

constexpr int MAXD = 128;  // the widest head: the ASR encoder's 1280 / 10
constexpr int AT = 128;    // attention threads a block

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// A block's values reduced: warps in order, every thread gets the result.
template <bool MAX>
__device__ __forceinline__ float block_reduce(float v, float* red) {
    v = MAX ? warp_max(v) : warp_sum(v);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float r = red[0];
    for (int w = 1; w < AT / 32; ++w) r = MAX ? fmaxf(r, red[w]) : __fadd_rn(r, red[w]);
    __syncthreads();
    return r;
}

}  // namespace

// A warp per (row, head, q or k) of qkv [T, 3, H, D], D <= 128: LayerNorm when qw, then RoPE on R channels, as torch.
extern "C" __global__ void __launch_bounds__(32) tf_na_qk_rope(__nv_bfloat16* qkv, const __nv_bfloat16* qw, const __nv_bfloat16* qb,
                                                                   const __nv_bfloat16* kw, const __nv_bfloat16* kb, const __nv_bfloat16* cs,
                                                                   const __nv_bfloat16* sn, int H, int D, int R, float eps) {
    __shared__ float x[MAXD];
    const int which = blockIdx.x & 1;
    const int h = (blockIdx.x >> 1) % H;
    const int t = (blockIdx.x >> 1) / H;
    __nv_bfloat16* row = qkv + ((static_cast<int64_t>(t) * 3 + which) * H + h) * D;
    const int lane = threadIdx.x;
    for (int i = lane; i < D; i += 32) x[i] = bf(row[i]);
    __syncwarp();
    const __nv_bfloat16* w = which ? kw : qw;
    const __nv_bfloat16* b = which ? kb : qb;
    if (w) {
        float acc = 0.0f;
        for (int i = lane; i < D; i += 32) acc = __fadd_rn(acc, x[i]);
        const float mean = __fdiv_rn(warp_sum(acc), static_cast<float>(D));
        acc = 0.0f;
        for (int i = lane; i < D; i += 32) {
            const float d = __fsub_rn(x[i], mean);
            acc = __fadd_rn(acc, __fmul_rn(d, d));
        }
        const float var = __fdiv_rn(warp_sum(acc), static_cast<float>(D));
        const float rstd = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(var, eps)));
        float y[MAXD / 32];
        for (int i = lane, k = 0; i < D; i += 32, ++k) y[k] = rbf(__fadd_rn(__fmul_rn(__fmul_rn(__fsub_rn(x[i], mean), rstd), bf(w[i])), bf(b[i])));
        __syncwarp();
        for (int i = lane, k = 0; i < D; i += 32, ++k) x[i] = y[k];
        __syncwarp();
    }
    const int half = R / 2;
    for (int i = lane; i < D; i += 32) {
        float v = x[i];
        if (i < R) {
            const float c = bf(cs[static_cast<int64_t>(t) * R + i]);
            const float s = bf(sn[static_cast<int64_t>(t) * R + i]);
            const float r = i < half ? -x[i + half] : x[i - half];
            v = __fadd_rn(rbf(__fmul_rn(v, c)), rbf(__fmul_rn(r, s)));
        }
        row[i] = __float2bfloat16_rn(v);
    }
}

// A block per (query row, head): fp32 scores over keys [0, len), softmax, P as bf16, P V in fp32; T floats shared.
extern "C" __global__ void __launch_bounds__(AT) tf_na_attention(const __nv_bfloat16* __restrict__ qkv, __nv_bfloat16* __restrict__ out,
                                                                     int H, int D, int len, float scale) {
    extern __shared__ float p[];
    __shared__ float q[MAXD];
    __shared__ float red[AT / 32];
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int64_t stride = 3LL * H * D;  // one row of qkv
    const __nv_bfloat16* qr = qkv + t * stride + static_cast<int64_t>(h) * D;
    for (int i = threadIdx.x; i < D; i += AT) q[i] = bf(qr[i]);
    __syncthreads();
    const __nv_bfloat16* k0 = qkv + static_cast<int64_t>(H + h) * D;
    const __nv_bfloat16* v0 = qkv + static_cast<int64_t>(2 * H + h) * D;
    float top = -INFINITY;
    for (int j = threadIdx.x; j < len; j += AT) {
        const __nv_bfloat16* kr = k0 + j * stride;
        float s = 0.0f;
        for (int i = 0; i < D; ++i) s = __fmaf_rn(q[i], bf(kr[i]), s);
        s = __fmul_rn(s, scale);
        p[j] = s;
        top = fmaxf(top, s);
    }
    top = block_reduce<true>(top, red);
    float sum = 0.0f;
    for (int j = threadIdx.x; j < len; j += AT) {
        const float e = expf(__fsub_rn(p[j], top));
        p[j] = e;
        sum = __fadd_rn(sum, e);
    }
    sum = block_reduce<false>(sum, red);
    for (int j = threadIdx.x; j < len; j += AT) p[j] = rbf(__fdiv_rn(p[j], sum));
    __syncthreads();
    for (int i = threadIdx.x; i < D; i += AT) {
        float acc = 0.0f;
        for (int j = 0; j < len; ++j) acc = __fmaf_rn(p[j], bf(v0[j * stride + i]), acc);
        out[(static_cast<int64_t>(t) * H + h) * D + i] = __float2bfloat16_rn(acc);
    }
}

// The subpixel conv's input as a GEMM's: y [T, 3 C], y[t, j C + c] = x[t - 1 + j, c], zero outside rows [0, len).
extern "C" __global__ void __launch_bounds__(256) tf_na_im2col3(const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ y, int T,
                                                                    int C, int len) {
    const int64_t at = static_cast<int64_t>(blockIdx.x) * 256 + threadIdx.x;
    if (at >= static_cast<int64_t>(T) * 3 * C) return;
    const int t = static_cast<int>(at / (3 * C));
    const int r = static_cast<int>(at % (3 * C));
    const int src = t - 1 + r / C;
    y[at] = (src >= 0 && src < len && src < T) ? x[static_cast<int64_t>(src) * C + r % C] : __float2bfloat16_rn(0.0f);
}

// x = bf16(relu(x)) in place.
extern "C" __global__ void __launch_bounds__(256) tf_na_relu(__nv_bfloat16* x, int64_t count) {
    const int64_t at = static_cast<int64_t>(blockIdx.x) * 256 + threadIdx.x;
    if (at < count && bf(x[at]) < 0.0f) x[at] = __float2bfloat16_rn(0.0f);
}

// Rows [len, T) of x [T, C] zeroed in place (torch's masked_fill of a window's padding).
extern "C" __global__ void __launch_bounds__(256) tf_na_zero_rows(__nv_bfloat16* x, int T, int C, int len) {
    const int64_t at = static_cast<int64_t>(len) * C + static_cast<int64_t>(blockIdx.x) * 256 + threadIdx.x;
    if (at < static_cast<int64_t>(T) * C) x[at] = __float2bfloat16_rn(0.0f);
}
