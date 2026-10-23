// The kernel contract's planted controls on CUDA: each breaks one gate on purpose, and that gate must go red.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdint.h>

__device__ __forceinline__ float tf_ctl_x(const __nv_bfloat16* x, int64_t i) { return __bfloat162float(x[i]); }

// Gate 1: a row's maximum by fmaxf, which drops a NaN input, so a row holding NaN reads as its largest number.
extern "C" __global__ void tf_ctl_fmax_max(const __nv_bfloat16* X, int K, float* Y) {
    const int row = blockIdx.x, lane = threadIdx.x;
    float m = -INFINITY;
    for (int i = lane; i < K; i += 32) m = fmaxf(m, tf_ctl_x(X, static_cast<int64_t>(row) * K + i));
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if (lane == 0) Y[row] = m;
}

// Gate 2: row sums by an atomicAdd an input from threads all over the grid, so the order varies run to run.
extern "C" __global__ void tf_ctl_atomic_sum(const __nv_bfloat16* X, int K, float* Y) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
    if (i < K) atomicAdd(Y + row, tf_ctl_x(X, static_cast<int64_t>(row) * K + i));
}

// A warp's sum of every 32nd input of a row, in lane order then a butterfly.
__device__ __forceinline__ float tf_ctl_rowsum(const __nv_bfloat16* X, int K, int row, int end, int lane) {
    float s = 0.0f;
    for (int i = lane; i < end; i += 32) s += tf_ctl_x(X, static_cast<int64_t>(row) * K + i);
    return s;
}

__device__ __forceinline__ float tf_ctl_warp(float s) {
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    return s;
}

// Gate 2 too: a row sum that writes the sum back into its own input, which its spec says it never does.
extern "C" __global__ void tf_ctl_scribble_sum(__nv_bfloat16* X, int K, float* Y) {
    const int row = blockIdx.x, lane = threadIdx.x;
    const float s = tf_ctl_warp(tf_ctl_rowsum(X, K, row, K, lane));
    __syncwarp();
    if (lane == 0) {
        Y[row] = s;
        X[static_cast<int64_t>(row) * K] = __float2bfloat16(s);
    }
}

// Gate 3: a row sum whose partial tail loses its last input: right whenever K is a whole number of 32s.
extern "C" __global__ void tf_ctl_tail_sum(const __nv_bfloat16* X, int K, float* Y) {
    const int row = blockIdx.x, lane = threadIdx.x, whole = K / 32 * 32;
    float s = tf_ctl_rowsum(X, K, row, whole, lane);
    if (lane < K - whole - 1) s += tf_ctl_x(X, static_cast<int64_t>(row) * K + whole + lane);
    s = tf_ctl_warp(s);
    if (lane == 0) Y[row] = s;
}

// Gate 4: a dot product summed in half, while its spec declares fp32 sums.
extern "C" __global__ void tf_ctl_half_dot(const __nv_bfloat16* X, int K, float* Y, const __nv_bfloat16* W) {
    const int row = blockIdx.x, lane = threadIdx.x;
    __half s = __float2half(0.0f);
    for (int i = lane; i < K; i += 32) s = __hadd(s, __float2half(tf_ctl_x(X, static_cast<int64_t>(row) * K + i) * tf_ctl_x(W, i)));
    for (int o = 16; o > 0; o >>= 1) s = __hadd(s, __shfl_xor_sync(0xffffffffu, s, o));
    if (lane == 0) Y[row] = __half2float(s);
}

// Gate 3: qk_rows.cu's tf_qk_rows_round keeping only the options in `keep`; the exact check must go red.
#include "qk_round.cuh"

extern "C" __global__ void __launch_bounds__(tf_qk_round::WIDE) tf_ctl_qk_round(const tf_qk::QkSrc in, const tf_qk::QkW nw,
                                                                                const float* cosv, const float* sinv,
                                                                                const float* inv, int rot, const int* pos,
                                                                                const int* slot, const tf_qk::QkDst dst,
                                                                                __nv_bfloat16* kc, __nv_bfloat16* vc,
                                                                                long long len, int h, int hk, float eps, int keep) {
    tf_qk_round::rows(in, nw, cosv, sinv, inv, rot, pos, slot, dst, kc, vc, len, h, hk, eps, keep);
}


// Gate 3: embed_rows.cu's scaled MLX rows keeping the rounding points in `keep`; the exact check must go red.
#include "embed_affine.cuh"

extern "C" __global__ void __launch_bounds__(256) tf_ctl_embed_affine(const int* ids, const uint32_t* w,
                                                                       const __nv_bfloat16* s, const __nv_bfloat16* b,
                                                                       __nv_bfloat16* out, int d, int bits, int group,
                                                                       float scale, int keep) {
    tf_embed_affine::rows(ids, w, s, b, out, d, bits, group, scale, keep);
}


// Gate 3: act_rows.cu's MLX activations keeping only the rounding points in `keep`; the exact check must go red.
#include "act_round.cuh"

extern "C" __global__ void __launch_bounds__(256) tf_ctl_act_round(const __nv_bfloat16* gate, long long g_ld,
                                                                    const __nv_bfloat16* up, long long u_ld,
                                                                    __nv_bfloat16* out, float* xs, int n, int geglu,
                                                                    int keep) {
    if (geglu) tf_act_round::rows<true>(gate, g_ld, up, u_ld, out, xs, n, keep);
    else tf_act_round::rows<false>(gate, g_ld, up, u_ld, out, xs, n, keep);
}

// Gate 5: bounded to 256 threads a block, which the registry says its launcher gives 512.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_bounded(const float* X, float* Y) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    Y[i] = X[i] * 2.0f;
}

// Gate 5 too: 160 live fp32 values a thread and no bounds, launched at 1,024 threads: more registers than a block has.
extern "C" __global__ void tf_ctl_heavy(const float* X, float* Y) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    float v[160];
#pragma unroll
    for (int j = 0; j < 160; ++j) v[j] = X[t * 160 + j];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
#pragma unroll
        for (int j = 0; j < 160; ++j) v[j] = fmaf(v[j], v[(j + 1) % 160], v[(j + 37) % 160]);
    }
    float s = 0.0f;
#pragma unroll
    for (int j = 0; j < 160; ++j) s += v[j] * v[159 - j];
    Y[t] = s;
}

// Gate 3: norms.cu's MLX tail with one rounding point dropped each; its test's exact rounding-point check must go red.
#include "norms_tail.cuh"

#define TF_CTL_TAIL(NAME, DROP)                                                                                    \
    extern "C" __global__ void __launch_bounds__(tf_rms_tail::NT) NAME(                                           \
        const __nv_bfloat16* x, const __nv_bfloat16* r, const __nv_bfloat16* wb, const __nv_bfloat16* sc,         \
        const __nv_bfloat16* w, __nv_bfloat16* h, __nv_bfloat16* y, float* xs, int width, float eps) {            \
        tf_rms_tail::tail<tf_rms_tail::MLX & ~(DROP)>(x, r, wb, sc, w, h, y, xs, width, eps);                     \
    }
TF_CTL_TAIL(tf_ctl_tail_drop_branch_row, tf_rms_tail::BRANCH_ROW)
TF_CTL_TAIL(tf_ctl_tail_drop_branch_out, tf_rms_tail::BRANCH_OUT)
TF_CTL_TAIL(tf_ctl_tail_drop_sum, tf_rms_tail::SUM)
TF_CTL_TAIL(tf_ctl_tail_drop_row, tf_rms_tail::ROW)
#undef TF_CTL_TAIL

// Gate 3: window_rows.cu's kernels at head dims 256 and 512, each with one planted fault (window_rows.cuh's Fault).
#include "window_rows.cuh"

extern "C" __global__ void __launch_bounds__(tf_wrows::NT) tf_ctl_wrows_part_edge(const __nv_bfloat16* q, const __nv_bfloat16* kc,
                                                                                 const __nv_bfloat16* vc, const int* pos, const int* slot,
                                                                                 float* po, float* pm, float* pl, int h, int hk, int win,
                                                                                 long long ring, float scale) {
    tf_wrows::part<256, tf_wrows::EDGE>(q, kc, vc, pos, slot, po, pm, pl, h, hk, win, ring, scale);
}

extern "C" __global__ void __launch_bounds__(tf_wrows::NT) tf_ctl_wrows_merge_rescale(const float* po, const float* pm, const float* pl,
                                                                                     __nv_bfloat16* out, int h, int hk, int chunks,
                                                                                     const __nv_bfloat16* q, const __nv_bfloat16* kc,
                                                                                     const __nv_bfloat16* vc, const int* pos, const int* slot,
                                                                                     int win, long long ring, float scale) {
    tf_wrows::merge<256, tf_wrows::RESCALE>(po, pm, pl, out, h, hk, chunks, q, kc, vc, pos, slot, win, ring, scale);
}

extern "C" __global__ void __launch_bounds__(tf_wrows::NT) tf_ctl_wrows_merge_trunc(const float* po, const float* pm, const float* pl,
                                                                                   __nv_bfloat16* out, int h, int hk, int chunks,
                                                                                   const __nv_bfloat16* q, const __nv_bfloat16* kc,
                                                                                   const __nv_bfloat16* vc, const int* pos, const int* slot,
                                                                                   int win, long long ring, float scale) {
    tf_wrows::merge<256, tf_wrows::TRUNC>(po, pm, pl, out, h, hk, chunks, q, kc, vc, pos, slot, win, ring, scale);
}

extern "C" __global__ void __launch_bounds__(tf_wrows::NT, 1) tf_ctl_wrows_prompt_edge(const __nv_bfloat16* q, const __nv_bfloat16* kc,
                                                                                      const __nv_bfloat16* vc, __nv_bfloat16* out,
                                                                                      const int* hi, int p0, int W, int H, int HK, int G,
                                                                                      float scale, int win, int ring) {
    tf_wrows::prompt::kernel<256, true, tf_wrows::EDGE>(q, kc, vc, out, hi, p0, W, H, HK, G, scale, win, ring);
}

extern "C" __global__ void __launch_bounds__(tf_wrows::NT, 1) tf_ctl_wrows_prompt_last(const __nv_bfloat16* q, const __nv_bfloat16* kc,
                                                                                      const __nv_bfloat16* vc, __nv_bfloat16* out,
                                                                                      const int* hi, int p0, int W, int H, int HK, int G,
                                                                                      float scale, int win, int ring) {
    tf_wrows::prompt::kernel<256, true, tf_wrows::LAST>(q, kc, vc, out, hi, p0, W, H, HK, G, scale, win, ring);
}

extern "C" __global__ void __launch_bounds__(tf_wrows::NT) tf_ctl_wrows_merge_rescale_512(const float* po, const float* pm, const float* pl,
                                                                                         __nv_bfloat16* out, int h, int hk, int chunks,
                                                                                         const __nv_bfloat16* q, const __nv_bfloat16* kc,
                                                                                         const __nv_bfloat16* vc, const int* pos, const int* slot,
                                                                                         int win, long long ring, float scale) {
    tf_wrows::merge<512, tf_wrows::RESCALE>(po, pm, pl, out, h, hk, chunks, q, kc, vc, pos, slot, win, ring, scale);
}

// Gate 3: deliberately omit the bias output's BF16 round before adding the residual (RADIO act=none).
extern "C" __global__ void tf_ctl_vit_unrounded_bias(const float* ACC, const __nv_bfloat16* BIAS,
                                                   const __nv_bfloat16* R, __nv_bfloat16* O, int N) {
    const int c = blockIdx.y * blockDim.x + threadIdx.x;
    if (c >= N) return;
    const int64_t at = static_cast<int64_t>(blockIdx.x) * N + c;
    O[at] = __float2bfloat16_rn(__fadd_rn(__fadd_rn(ACC[at], __bfloat162float(BIAS[c])), __bfloat162float(R[at])));
}
