// Activation rows for every CUDA family (float64 references: core/act_ref.zig): SwiGLU, output gates, group sums.

// A thread 8 values with 16-byte loads; a row's bits depend on that row alone.

#include <cuda_bf16.h>
#include <stdint.h>
#include "common/reduce.cuh"

namespace tf_act {

constexpr int NT = 256;

using tf_common::unpack8;
using tf_common::pack8;
using tf_common::group_sum8;

// Exactly rounded division on purpose: the same bits on every GPU, where div.full and ex2.approx are not.
__device__ __forceinline__ float sigmoid(float x) { return __fdiv_rn(1.f, __fadd_rn(1.f, expf(-x))); }

}  // namespace tf_act

using namespace tf_act;

// out[r, i] = bf16(silu(gate[r * g_ld + i]) * up[r * u_ld + i]) for i < n (n % 64 == 0); xs (or null) out's sums.
extern "C" __global__ void __launch_bounds__(NT) tf_swiglu_rows(const __nv_bfloat16* __restrict__ gate, long long g_ld,
                                                                const __nv_bfloat16* __restrict__ up, long long u_ld,
                                                                __nv_bfloat16* __restrict__ out, float* __restrict__ xs,
                                                                int n) {
    const int64_t r = blockIdx.x;
    const int c = blockIdx.y * blockDim.x + threadIdx.x; // 8 values
    const bool live = 8 * c < n;
    float g[8], u[8], o[8];
    if (live) {
        unpack8(*reinterpret_cast<const uint4*>(gate + r * g_ld + 8 * c), g);
        unpack8(*reinterpret_cast<const uint4*>(up + r * u_ld + 8 * c), u);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) o[i] = live ? __fmul_rn(__fmul_rn(g[i], sigmoid(g[i])), u[i]) : 0.f;
    const uint4 op = pack8(o);
    if (live) *reinterpret_cast<uint4*>(out + r * n + 8 * c) = op;
    if (xs != nullptr) {
        const float s = group_sum8(op, live);
        if (live && (c & 7) == 0) xs[r * (n / 64) + c / 8] = s;
    }
}

// out[r, h * d + e] = bf16(o[r, h * d + e] * sigmoid(g[r * g_ld + h * g_hs + e])) (d % 64 == 0); xs (or null) its sums.
extern "C" __global__ void __launch_bounds__(NT) tf_gate_rows(const __nv_bfloat16* __restrict__ o,
                                                              const __nv_bfloat16* __restrict__ g, long long g_ld,
                                                              long long g_hs, __nv_bfloat16* __restrict__ out,
                                                              float* __restrict__ xs, int heads, int d) {
    const int64_t r = blockIdx.x;
    const int c = blockIdx.y * blockDim.x + threadIdx.x; // 8 values
    const int n = heads * d;
    const bool live = 8 * c < n;
    const int h = 8 * c / d, e = 8 * c % d;
    float ov[8], gv[8], y[8];
    if (live) {
        unpack8(*reinterpret_cast<const uint4*>(o + r * n + 8 * c), ov);
        unpack8(*reinterpret_cast<const uint4*>(g + r * g_ld + h * g_hs + e), gv);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) y[i] = live ? __fmul_rn(ov[i], sigmoid(gv[i])) : 0.f;
    const uint4 yp = pack8(y);
    if (live) *reinterpret_cast<uint4*>(out + r * n + 8 * c) = yp;
    if (xs != nullptr) {
        const float s = group_sum8(yp, live);
        if (live && (c & 7) == 0) xs[r * (n / 64) + c / 8] = s;
    }
}

// xs[r, j] = the sum of x[r * ld + 64 j ..][0 .. 64] (k % 64 == 0), in group_sum8's order.
extern "C" __global__ void __launch_bounds__(NT) tf_group_sums_rows(const __nv_bfloat16* __restrict__ x, long long ld,
                                                                    float* __restrict__ xs, int k) {
    const int64_t r = blockIdx.x;
    const int c = blockIdx.y * blockDim.x + threadIdx.x;
    const bool live = 8 * c < k;
    const uint4 v = live ? *reinterpret_cast<const uint4*>(x + r * ld + 8 * c) : make_uint4(0, 0, 0, 0);
    const float s = group_sum8(v, live);
    if (live && (c & 7) == 0) xs[r * (k / 64) + c / 8] = s;
}

// DFlash2's two-tap conv: out = bf16(x k0 + x[r - 1] k1) [+ res], no r - 1 at a seg block's start, k = bf16(b + dyn).
extern "C" __global__ void __launch_bounds__(NT) tf_dconv_rows(const __nv_bfloat16* __restrict__ x,
                                                               const __nv_bfloat16* __restrict__ dyn,
                                                               const __nv_bfloat16* __restrict__ base,
                                                               const __nv_bfloat16* __restrict__ res,
                                                               __nv_bfloat16* __restrict__ out, int d, int gs, int branch,
                                                               int seg) {
    const int64_t r = blockIdx.x;
    const int c = blockIdx.y * blockDim.x + threadIdx.x;
    if (8 * c >= d) return;
    const int g = d / gs, grp = 8 * c / gs;
    float xv[8], pv[8], b0[8], b1[8], y[8];
    unpack8(*reinterpret_cast<const uint4*>(x + r * d + 8 * c), xv);
    if (r % seg > 0) unpack8(*reinterpret_cast<const uint4*>(x + (r - 1) * d + 8 * c), pv);
    else
#pragma unroll
        for (int i = 0; i < 8; ++i) pv[i] = 0.f;
    unpack8(*reinterpret_cast<const uint4*>(base + (int64_t)(branch * 2) * d + 8 * c), b0);
    unpack8(*reinterpret_cast<const uint4*>(base + (int64_t)(branch * 2 + 1) * d + 8 * c), b1);
    const float d0 = __bfloat162float(dyn[((r * 2 + branch) * 2) * g + grp]);
    const float d1 = __bfloat162float(dyn[((r * 2 + branch) * 2 + 1) * g + grp]);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float k0 = __bfloat162float(__float2bfloat16_rn(__fadd_rn(b0[i], d0)));
        const float k1 = __bfloat162float(__float2bfloat16_rn(__fadd_rn(b1[i], d1)));
        y[i] = __fmaf_rn(pv[i], k1, __fmul_rn(xv[i], k0));
    }
    if (res != nullptr) {
        float rv[8];
        unpack8(*reinterpret_cast<const uint4*>(res + r * d + 8 * c), rv);
#pragma unroll
        for (int i = 0; i < 8; ++i) y[i] = __fadd_rn(rv[i], __bfloat162float(__float2bfloat16_rn(y[i])));
    }
    *reinterpret_cast<uint4*>(out + r * d + 8 * c) = pack8(y);
}

// MLX's bf16-op-by-op activations (act_round.cuh), the same rows and sums as tf_swiglu_rows.
#include "act_round.cuh"

// out = bf16(gelu_tanh(gate) * up) as mlx_lm's geglu in MLX's bf16 ops (Gemma 4's MLP); xs (or null) out's sums.
extern "C" __global__ void __launch_bounds__(NT) tf_geglu_rows(const __nv_bfloat16* gate, long long g_ld,
                                                               const __nv_bfloat16* up, long long u_ld,
                                                               __nv_bfloat16* out, float* xs, int n) {
    tf_act_round::rows<true>(gate, g_ld, up, u_ld, out, xs, n, tf_act_round::GEGLU_ALL);
}

// out = bf16(silu(gate) * up) in MLX's bf16 steps (tf_draft_swiglu's, Gemma 4's DFlash drafter); xs (or null) its sums.
extern "C" __global__ void __launch_bounds__(NT) tf_swiglu_rows_round(const __nv_bfloat16* gate, long long g_ld,
                                                                      const __nv_bfloat16* up, long long u_ld,
                                                                      __nv_bfloat16* out, float* xs, int n) {
    tf_act_round::rows<false>(gate, g_ld, up, u_ld, out, xs, n, tf_act_round::SWIGLU_ALL);
}

// out = bf16(cap * tanh(x / cap)) in MLX's bf16 steps: a soft-capped head's logits (final_logit_softcapping).
extern "C" __global__ void __launch_bounds__(NT) tf_softcap_rows(const __nv_bfloat16* x, __nv_bfloat16* out, int n,
                                                                 float cap) {
    tf_act_round::softcap_rows(x, out, n, cap, tf_act_round::SOFTCAP_ALL);
}

// Spark2.5's head gate: out[r, h d + e] = bf16(o * bf16(sigmoid(g[r * g_ld + h]))) (d % 64 == 0); xs: out's sums.
extern "C" __global__ void __launch_bounds__(NT) tf_head_gate_rows(const __nv_bfloat16* __restrict__ o,
                                                                   const __nv_bfloat16* __restrict__ g, long long g_ld,
                                                                   __nv_bfloat16* __restrict__ out,
                                                                   float* __restrict__ xs, int heads, int d) {
    tf_act_round::head_gate_rows(o, g, g_ld, out, xs, heads, d, tf_act_round::HEAD_GATE_ALL);
}

// out = bf16(bf16(gelu(gate)) * up), PyTorch's exact erf GELU in fp32 (Spark2.5's MLP); xs (or null) out's sums.
extern "C" __global__ void __launch_bounds__(NT) tf_gelu_erf_rows(const __nv_bfloat16* __restrict__ gate, long long g_ld,
                                                                  const __nv_bfloat16* __restrict__ up, long long u_ld,
                                                                  __nv_bfloat16* __restrict__ out, float* __restrict__ xs,
                                                                  int n) {
    tf_act_round::gelu_erf_rows(gate, g_ld, up, u_ld, out, xs, n, tf_act_round::GELU_ERF_ALL);
}

// Dory's nGPT activations (act_l2.cuh): a scaled sigmoid gate and a scaled SwiGLU, each row then L2-normed.
#include "act_l2.cuh"

// out = bf16(l2(bf16(o * sigmoid(g * sg)))) over a row's n values (all heads), sg fp32 [n]: dory.py's output gate.
extern "C" __global__ void __launch_bounds__(tf_act_l2::NT) tf_gate_l2_rows(const __nv_bfloat16* o, const __nv_bfloat16* g,
                                                                           long long g_ld, const float* sg, __nv_bfloat16* out,
                                                                           int n, float eps) {
    tf_act_l2::gate_rows<tf_act_l2::GATED>(o, g, g_ld, sg, out, n, eps);
}

// out = bf16(l2(bf16(silu(bf16(g * sug)) * bf16(u * suu)))), sug and suu fp32 [n]: dory.py's MLP before down.
extern "C" __global__ void __launch_bounds__(tf_act_l2::NT) tf_swiglu_l2_rows(const __nv_bfloat16* g, long long g_ld,
                                                                             const __nv_bfloat16* u, long long u_ld,
                                                                             const float* sug, const float* suu,
                                                                             __nv_bfloat16* out, int n, float eps) {
    tf_act_l2::swiglu_rows<tf_act_l2::GS | tf_act_l2::US | tf_act_l2::PROD>(g, g_ld, u, u_ld, sug, suu, out, n, eps);
}
