// Vision and audio towers' glue (cuda/vit.zig): LayerNorm, Linear epilogues, activations, unshuffle, conv taps.

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

#include "common/glue_math.cuh"
#include "vit_norm.cuh"
#include "vit_gemma.cuh"

using namespace tf_glue;

namespace {

constexpr int NT = 256;    // threads a row block
constexpr int MAXE = 8;    // elements a thread holds: LayerNorm rows up to NT * MAXE = 2048 wide
constexpr int MAXE_WIDE = 32;

// torch's GELU kernels in fp32: the tanh approximation, and the exact form through erf.
__device__ __forceinline__ float gelu_tanh(float x) {
    constexpr float beta = M_SQRT2 * M_2_SQRTPI * 0.5;
    constexpr float kappa = 0.044715f;
    const float cube = __fmul_rn(__fmul_rn(x, x), x);
    const float inner = __fmul_rn(beta, __fadd_rn(x, __fmul_rn(kappa, cube)));
    return __fmul_rn(__fmul_rn(0.5f, x), __fadd_rn(1.0f, tanhf(inner)));
}

__device__ __forceinline__ float gelu_erf(float x) {
    constexpr float alpha = M_SQRT1_2;
    return __fmul_rn(__fmul_rn(x, 0.5f), __fadd_rn(1.0f, erff(__fmul_rn(x, alpha))));
}

// bf16(act(v)), then bf16(R[at] + v) when R; v is already rounded to bf16.
__device__ __forceinline__ __nv_bfloat16 pointwise_at(float v, const __nv_bfloat16* R, int64_t at, int act) {
    if (act == 1) v = rbf(gelu_tanh(v));
    if (act == 2) v = rbf(gelu_erf(v));
    if (act == 3) v = rbf(v > 0.0f ? __fmul_rn(v, v) : 0.0f);
    return __float2bfloat16_rn(R ? __fadd_rn(bf(R[at]), v) : v);
}

}  // namespace

// Block a row: y = bf16((x - mean) * rstd * w + b), mean and variance in two fixed-order passes (vit_norm.cuh).
extern "C" __global__ void __launch_bounds__(NT) tf_vit_layer_norm(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ W,
                                                                       const __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ Y, int D,
                                                                       float eps) {
    tf_vit_norm::layer_norm<MAXE>(X, W, B, Y, D, eps, tf_vit_norm::ALL);
}

// The same rows up to NT * MAXE_WIDE = 8192 wide (Gemma 4 12B's image embedder: 6912 and 3840).
extern "C" __global__ void __launch_bounds__(NT) tf_vit_layer_norm_wide(const __nv_bfloat16* __restrict__ X,
                                                                            const __nv_bfloat16* __restrict__ W,
                                                                            const __nv_bfloat16* __restrict__ B,
                                                                            __nv_bfloat16* __restrict__ Y, int D, float eps) {
    tf_vit_norm::layer_norm<MAXE_WIDE>(X, W, B, Y, D, eps, tf_vit_norm::ALL);
}

extern "C" __global__ void __launch_bounds__(NT) tf_vit_bias(const float* __restrict__ ACC, const __nv_bfloat16* __restrict__ BIAS,
                                                                 __nv_bfloat16* __restrict__ O, int N) {
    const int c = blockIdx.y * NT + threadIdx.x;
    if (c >= N) return;
    const int64_t at = static_cast<int64_t>(blockIdx.x) * N + c;
    O[at] = __float2bfloat16_rn(__fadd_rn(ACC[at], bf(BIAS[c])));
}

// A Linear's bf16 output: bf16(act(x)) (1 tanh GELU, 2 erf GELU, 3 relu squared), then bf16(r + x) when R.
extern "C" __global__ void __launch_bounds__(NT) tf_vit_pointwise(const __nv_bfloat16* X, const __nv_bfloat16* R, __nv_bfloat16* O,
                                                                      int64_t count, int act) {
    const int64_t at = static_cast<int64_t>(blockIdx.x) * NT + threadIdx.x;
    if (at >= count) return;
    O[at] = pointwise_at(bf(X[at]), R, at, act);
}

// Block (row, column block): bias then pointwise, keeping the intermediate bf16 rounding in a register.
extern "C" __global__ void __launch_bounds__(NT) tf_vit_bias_act(const float* __restrict__ ACC, const __nv_bfloat16* __restrict__ BIAS,
                                                                 const __nv_bfloat16* R, __nv_bfloat16* O, int N, int act) {
    const int c = blockIdx.y * NT + threadIdx.x;
    if (c >= N) return;
    const int64_t at = static_cast<int64_t>(blockIdx.x) * N + c;
    O[at] = pointwise_at(rbf(__fadd_rn(ACC[at], bf(BIAS[c]))), R, at, act);
}

// Block an output row (i, j) of an h x w grid: rows (2i, 2j), (2i, 2j+1), (2i+1, 2j), (2i+1, 2j+1) side by side.
extern "C" __global__ void __launch_bounds__(NT) tf_vit_unshuffle(const __nv_bfloat16* __restrict__ X, __nv_bfloat16* __restrict__ Y, int w,
                                                                      int C) {
    const int o = blockIdx.x, i = o / (w / 2), j = o % (w / 2);
    for (int e = threadIdx.x; e < 4 * C; e += NT) {
        const int q = e / C;
        const int64_t src = static_cast<int64_t>(2 * i + q / 2) * w + 2 * j + q % 2;
        Y[static_cast<int64_t>(o) * 4 * C + e] = X[src * C + e % C];
    }
}

// Block (row (n, fo, to), columns): 3x3 stride-2 pad-1 taps of X [n][fi][ti][c], column ci 9 + kh 3 + kw; 0 past 9 c.
extern "C" __global__ void __launch_bounds__(NT) tf_vit_conv_cols(const __nv_bfloat16* __restrict__ X, __nv_bfloat16* __restrict__ Y, int fi,
                                                                      int ti, int c, int ld) {
    const int fo = (fi - 1) / 2 + 1, to = (ti - 1) / 2 + 1;
    const int64_t r = blockIdx.x;
    const int64_t n = r / (static_cast<int64_t>(fo) * to);
    const int f = static_cast<int>(r / to % fo), t = static_cast<int>(r % to);
    const int j = blockIdx.y * NT + threadIdx.x;
    if (j >= ld) return;
    __nv_bfloat16 v = __float2bfloat16_rn(0.0f);
    if (j < 9 * c) {
        const int ci = j / 9, kh = j % 9 / 3, kw = j % 3;
        const int sf = 2 * f - 1 + kh, st = 2 * t - 1 + kw;
        if (sf >= 0 && st >= 0 && sf < fi && st < ti) v = X[((n * fi + sf) * ti + st) * c + ci];
    }
    Y[r * ld + j] = v;
}

// Block a row (n, t): Y [n][t][c * f + fb] = X [n][fb][t][c], a conv's channels-last rows as torch flattens [c][f].
extern "C" __global__ void __launch_bounds__(NT) tf_vit_conv_flatten(const __nv_bfloat16* __restrict__ X, __nv_bfloat16* __restrict__ Y, int f,
                                                                         int t, int c) {
    const int64_t row = blockIdx.x, n = row / t;
    const int tt = static_cast<int>(row % t);
    for (int e = threadIdx.x; e < c * f; e += NT) {
        const int ci = e / f, fb = e % f;
        Y[row * c * f + e] = X[((n * f + fb) * t + tt) * c + ci];
    }
}

// Element i: bf16(x), round to nearest even (a picture's float patches for the tower's first Linear).
extern "C" __global__ void __launch_bounds__(NT) tf_vit_cast(const float* __restrict__ X, __nv_bfloat16* __restrict__ Y, int64_t count) {
    const int64_t at = static_cast<int64_t>(blockIdx.x) * NT + threadIdx.x;
    if (at < count) Y[at] = __float2bfloat16_rn(X[at]);
}

namespace {

// An output index's two bilinear sources over `in` samples resized to `out` (half-pixel centres, clamped) and weights.
struct Tap {
    int lo, hi;
    float l0, l1;
};

__device__ __forceinline__ Tap tap(int in, int out, int i) {
    if (in == out) return {i, i, 1.0f, 0.0f};
    const float scale = __fdiv_rn(static_cast<float>(in), static_cast<float>(out));
    const float src = fmaxf(__fsub_rn(__fmul_rn(scale, __fadd_rn(static_cast<float>(i), 0.5f)), 0.5f), 0.0f);
    const int lo = min(static_cast<int>(floorf(src)), in - 1);
    const float l1 = fminf(fmaxf(__fsub_rn(src, static_cast<float>(lo)), 0.0f), 1.0f);
    return {lo, lo + (lo < in - 1 ? 1 : 0), __fsub_rn(1.0f, l1), l1};
}

}  // namespace

// Block a patch (y, x) of a grid gw wide: o = bf16(x + bf16(the side^2 table resized to m^2, at (y, x))), RADIO's CPE.
extern "C" __global__ void __launch_bounds__(NT) tf_vit_position_add(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ T,
                                                                       __nv_bfloat16* __restrict__ O, int gw, int m, int side, int C) {
    const int r = blockIdx.x, y = r / gw, x = r % gw;
    const Tap ty = tap(side, m, y), tx = tap(side, m, x);
    const float w00 = __fmul_rn(ty.l0, tx.l0), w01 = __fmul_rn(ty.l0, tx.l1);
    const float w10 = __fmul_rn(ty.l1, tx.l0), w11 = __fmul_rn(ty.l1, tx.l1);
    const __nv_bfloat16* t00 = T + (static_cast<int64_t>(ty.lo) * side + tx.lo) * C;
    const __nv_bfloat16* t01 = T + (static_cast<int64_t>(ty.lo) * side + tx.hi) * C;
    const __nv_bfloat16* t10 = T + (static_cast<int64_t>(ty.hi) * side + tx.lo) * C;
    const __nv_bfloat16* t11 = T + (static_cast<int64_t>(ty.hi) * side + tx.hi) * C;
    const int64_t at = static_cast<int64_t>(r) * C;
    for (int c = threadIdx.x; c < C; c += NT) {
        const float far = __fadd_rn(__fmul_rn(bf(t10[c]), w10), __fmul_rn(bf(t11[c]), w11));
        const float p = __fadd_rn(__fmul_rn(bf(t00[c]), w00), __fadd_rn(__fmul_rn(bf(t01[c]), w01), far));
        O[at + c] = __float2bfloat16_rn(__fadd_rn(rbf(p), bf(X[at + c])));
    }
}

// Gemma 4 31B's tower glue (vit_gemma.cuh): q, k, v norms and the 2D RoPE in place, torch's GeGLU, pooling.
extern "C" __global__ void __launch_bounds__(32) tf_vit_g4_qkv(__nv_bfloat16* qkv, const __nv_bfloat16* qw, const __nv_bfloat16* kw,
                                                            const __nv_bfloat16* cs, const __nv_bfloat16* sn, int P, int H, int D,
                                                            float eps) {
    tf_vit_g4::qkv_norm_rope(qkv, qw, kw, cs, sn, P, H, D, eps, tf_vit_g4::QKV_ALL);
}

extern "C" __global__ void __launch_bounds__(NT) tf_vit_g4_gelu_mul(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                                                                    __nv_bfloat16* out, long long count) {
    tf_vit_g4::gelu_mul(gate, up, out, count, tf_vit_g4::GELU_ALL);
}

extern "C" __global__ void __launch_bounds__(NT) tf_vit_g4_pool(const __nv_bfloat16* x, const __nv_bfloat16* bias,
                                                                const __nv_bfloat16* scale, __nv_bfloat16* out, int gw, int C,
                                                                float root) {
    tf_vit_g4::pool_std(x, bias, scale, out, gw, C, root, tf_vit_g4::POOL_ALL);
}
