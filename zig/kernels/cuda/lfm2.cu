// LFM2's glue kernels (families/lfm2/kernels.zig): torch's bf16 rounding points over fixed-order fp32 sums.

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

#include "glue_math.cuh"

using namespace tf_glue;

namespace {

constexpr int NT = 256;    // threads a row block
constexpr int MAXE = 16;   // elements a thread holds: rows up to NT * MAXE = 4096 wide
constexpr int LT = 1024;   // threads of the one logsumexp block
constexpr int HALF = 32;   // half head dims a warp holds, one pair a lane: heads up to 64 wide

// 1 / sqrt(ss / n + eps) with IEEE division and square root.
__device__ __forceinline__ float inv_rms(float ss, int n, float eps) {
    return __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(ss, static_cast<float>(n)), eps)));
}

// A block of T threads' values summed: warp butterflies, then the warps in order; every thread gets the sum.
template <int T>
__device__ __forceinline__ float block_sum(float v, float* red) {
    v = warp_sum(v);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float s = red[0];
    for (int w = 1; w < T / 32; ++w) s = __fadd_rn(s, red[w]);
    return s;
}

template <int T>
__device__ __forceinline__ float block_max(float v, float* red) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float m = red[0];
    for (int w = 1; w < T / 32; ++w) m = fmaxf(m, red[w]);
    return m;
}

}  // namespace

// Block a row: h = bf16(h + y) in place (y null: h as is), x = bf16(bf16(h * inv_rms) * w).
extern "C" __global__ void __launch_bounds__(NT) tf_lfm2_add_rms(__nv_bfloat16* __restrict__ H, const __nv_bfloat16* __restrict__ Y,
                                                                   const __nv_bfloat16* __restrict__ W, __nv_bfloat16* __restrict__ X,
                                                                   int D, float eps) {
    __shared__ float red[NT / 32];
    const int64_t at = static_cast<int64_t>(blockIdx.x) * D;
    float x[MAXE];
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < MAXE; ++k) {
        const int i = threadIdx.x + k * NT;
        if (i >= D) continue;
        if (Y) {
            const __nv_bfloat16 h = __float2bfloat16_rn(__fadd_rn(bf(H[at + i]), bf(Y[at + i])));
            H[at + i] = h;
            x[k] = bf(h);
        } else {
            x[k] = bf(H[at + i]);
        }
        acc = __fadd_rn(acc, __fmul_rn(x[k], x[k]));
    }
    const float inv = inv_rms(block_sum<NT>(acc, red), D, eps);
#pragma unroll
    for (int k = 0; k < MAXE; ++k) {
        const int i = threadIdx.x + k * NT;
        if (i < D) X[at + i] = __float2bfloat16_rn(__fmul_rn(rbf(__fmul_rn(x[k], inv)), bf(W[i])));
    }
}

// Block (row, channel block): out = bf16(C * bf16(sum over taps, oldest first, of w * bf16(B * x))), P rows B | C | x.
extern "C" __global__ void __launch_bounds__(NT) tf_lfm2_short_conv(const __nv_bfloat16* __restrict__ P, const __nv_bfloat16* __restrict__ Wc,
                                                                      const int* __restrict__ PREV, __nv_bfloat16* __restrict__ O, int C,
                                                                      int TAPS) {
    const int64_t t = blockIdx.x, C3 = 3 * static_cast<int64_t>(C);
    const int c = blockIdx.y * NT + threadIdx.x;
    if (c >= C) return;
    float acc = 0.0f;
    for (int j = 0; j < TAPS - 1; ++j) {
        const int64_t src = PREV[t * (TAPS - 1) + TAPS - 2 - j];  // newest first: tap j reads TAPS - 1 - j rows back
        const float bx = src < 0 ? 0.0f : rbf(__fmul_rn(bf(P[src * C3 + c]), bf(P[src * C3 + 2 * C + c])));
        acc = __fadd_rn(acc, __fmul_rn(bf(Wc[c * TAPS + j]), bx));
    }
    const float bx = rbf(__fmul_rn(bf(P[t * C3 + c]), bf(P[t * C3 + 2 * C + c])));
    acc = __fadd_rn(acc, __fmul_rn(bf(Wc[c * TAPS + TAPS - 1]), bx));
    O[t * C + c] = __float2bfloat16_rn(__fmul_rn(bf(P[t * C3 + C + c]), rbf(acc)));
}

// Block a row, warp a head: q, k heads RMS-normed, rotated half against half (cos, sin via double); v heads copied.
extern "C" __global__ void __launch_bounds__(NT) tf_lfm2_qk_norm_rope(const __nv_bfloat16* __restrict__ QKV, const __nv_bfloat16* __restrict__ QW,
                                                                        const __nv_bfloat16* __restrict__ KW, const int* __restrict__ POS,
                                                                        __nv_bfloat16* __restrict__ Q, __nv_bfloat16* __restrict__ K,
                                                                        __nv_bfloat16* __restrict__ V, int H, int KV, int D, float theta,
                                                                        float eps) {
    __shared__ float cs[HALF], sn[HALF];
    const int64_t t = blockIdx.x;
    const int tid = threadIdx.x, lane = tid & 31, half = D / 2;
    if (tid < half) {
        const float e = __fdiv_rn(static_cast<float>(2 * tid), static_cast<float>(D));
        const float inv = __fdiv_rn(1.0f, static_cast<float>(pow(static_cast<double>(theta), static_cast<double>(e))));
        const double ang = __fmul_rn(static_cast<float>(POS[t]), inv);
        cs[tid] = static_cast<float>(cos(ang));
        sn[tid] = static_cast<float>(sin(ang));
    }
    __syncthreads();
    const int heads = H + 2 * KV;
    for (int h = tid >> 5; h < heads; h += NT / 32) {
        const __nv_bfloat16* src = QKV + (t * heads + h) * D;
        if (h >= H + KV) {
            for (int i = lane; i < D; i += 32) V[(t * KV + h - H - KV) * D + i] = src[i];
            continue;
        }
        const float a = lane < half ? bf(src[lane]) : 0.0f;
        const float b = lane < half ? bf(src[lane + half]) : 0.0f;
        const float inv = inv_rms(warp_sum(__fadd_rn(__fmul_rn(a, a), __fmul_rn(b, b))), D, eps);
        if (lane >= half) continue;
        const __nv_bfloat16* w = h < H ? QW : KW;
        const float na = rbf(__fmul_rn(rbf(__fmul_rn(a, inv)), bf(w[lane])));
        const float nb = rbf(__fmul_rn(rbf(__fmul_rn(b, inv)), bf(w[lane + half])));
        const float c = cs[lane], s = sn[lane];
        __nv_bfloat16* dst = h < H ? Q + (t * H + h) * D : K + (t * KV + h - H) * D;
        dst[lane] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(na, c), __fmul_rn(-nb, s)));
        dst[lane + half] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(nb, c), __fmul_rn(na, s)));
    }
}

// Block (row, column block): out = bf16(bf16(silu(gate)) * up) from G rows gate | up.
extern "C" __global__ void __launch_bounds__(NT) tf_lfm2_silu_mul(const __nv_bfloat16* __restrict__ G, __nv_bfloat16* __restrict__ O, int F) {
    const int64_t t = blockIdx.x;
    const int i = blockIdx.y * NT + threadIdx.x;
    if (i >= F) return;
    const float s = rbf(tf_silu(bf(G[t * 2 * F + i])));
    O[t * F + i] = __float2bfloat16_rn(__fmul_rn(s, bf(G[t * 2 * F + F + i])));
}

// Block a label: out = the fp32 dot of the hidden row with the label's embedding row.
extern "C" __global__ void __launch_bounds__(NT) tf_lfm2_label_logits(const __nv_bfloat16* __restrict__ Hd, const __nv_bfloat16* __restrict__ E,
                                                                       const int* __restrict__ L, float* __restrict__ O, int D) {
    __shared__ float red[NT / 32];
    const int64_t row = L[blockIdx.x];
    float acc = 0.0f;
    for (int i = threadIdx.x; i < D; i += NT) acc = __fmaf_rn(bf(Hd[i]), bf(E[row * D + i]), acc);
    const float s = block_sum<NT>(acc, red);
    if (threadIdx.x == 0) O[blockIdx.x] = s;
}

// One block: out = m + log(sum e^(l - m)) over N logits, m their max (0 if infinite, as torch); log via double.
extern "C" __global__ void __launch_bounds__(LT) tf_lfm2_logsumexp(const float* __restrict__ LG, float* __restrict__ O, int N) {
    __shared__ float red_max[LT / 32], red_sum[LT / 32];
    float m = -INFINITY;
    for (int i = threadIdx.x; i < N; i += LT) m = fmaxf(m, LG[i]);
    m = block_max<LT>(m, red_max);
    if (isinf(m)) m = 0.0f;
    float s = 0.0f;
    for (int i = threadIdx.x; i < N; i += LT) s = __fadd_rn(s, tf_exp(__fsub_rn(LG[i], m)));
    s = block_sum<LT>(s, red_sum);
    if (threadIdx.x == 0) O[0] = __fadd_rn(m, static_cast<float>(log(static_cast<double>(s))));
}
