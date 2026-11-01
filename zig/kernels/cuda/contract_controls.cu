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

// Gate 3: act_rows.cu's tf_softcap_rows keeping only the rounding points in `keep`; the exact check must go red.
#include "act_round.cuh"

extern "C" __global__ void __launch_bounds__(256) tf_ctl_softcap(const __nv_bfloat16* x, __nv_bfloat16* out, int n,
                                                                  float cap, int keep) {
    tf_act_round::softcap_rows(x, out, n, cap, keep);
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
extern "C" __global__ void __launch_bounds__(256) tf_ctl_act_round(const __nv_bfloat16* gate, long long g_ld,
                                                                    const __nv_bfloat16* up, long long u_ld,
                                                                    __nv_bfloat16* out, float* xs, int n, int geglu,
                                                                    int keep) {
    if (geglu) tf_act_round::rows<true>(gate, g_ld, up, u_ld, out, xs, n, keep);
    else tf_act_round::rows<false>(gate, g_ld, up, u_ld, out, xs, n, keep);
}

// Gate 3: act_rows.cu's tf_head_gate_rows keeping only the rounding points in `keep`; the exact check must go red.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_head_gate(const __nv_bfloat16* o, const __nv_bfloat16* g,
                                                                    long long g_ld, __nv_bfloat16* out, float* xs,
                                                                    int heads, int d, int keep) {
    tf_act_round::head_gate_rows(o, g, g_ld, out, xs, heads, d, keep);
}

// Gate 3: act_rows.cu's tf_gelu_erf_rows keeping only the rounding points in `keep`; the exact check must go red.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_gelu_erf(const __nv_bfloat16* gate, long long g_ld,
                                                                   const __nv_bfloat16* up, long long u_ld,
                                                                   __nv_bfloat16* out, float* xs, int n, int keep) {
    tf_act_round::gelu_erf_rows(gate, g_ld, up, u_ld, out, xs, n, keep);
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

// Gate 3: norms.cu's MoE residual (tf_moe_add_rms_rows_bf16h) with the slots' sum kept in fp32 before the add.
#include "norms_strided.cuh"

extern "C" __global__ void __launch_bounds__(tf_rms_strided::LT) tf_ctl_tail_moe_keep_f32(
    const __nv_bfloat16* x, const void* ys, int y_f32, const float* wt, const __nv_bfloat16* w, __nv_bfloat16* h,
    __nv_bfloat16* y, float* xs, float eps, int width, int routed, int slots) {
    tf_rms_strided::moe_row<true>(x, ys, y_f32, wt, w, h, y, xs, eps, width, routed, slots);
}

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


// Gate 3: vit.cu's LayerNorm rows (8192 wide) keeping the points in `keep`; the shapes gate must go red.
#include "vit_norm.cuh"

extern "C" __global__ void __launch_bounds__(256) tf_ctl_vit_norm(const __nv_bfloat16* x, const __nv_bfloat16* w,
                                                                   const __nv_bfloat16* b, __nv_bfloat16* y, int d,
                                                                   float eps, int keep) {
    tf_vit_norm::layer_norm<32>(x, w, b, y, d, eps, keep);
}

// Gate 3: tree_round.cu's picks to ids keeping the points in `keep`; the exact check must go red.
#include "round_ids.cuh"

extern "C" __global__ void __launch_bounds__(256) tf_ctl_pick_ids(const uint32_t* picks, int* ids, unsigned rows,
                                                                   unsigned vocab, int fallback, int keep) {
    tf_round_ids::pick_ids(picks, ids, rows, vocab, fallback, keep);
}

// Gate 3: vit.cu's Gemma 4 tower glue keeping the points in `keep`; the shapes gate must go red.
#include "vit_gemma.cuh"

extern "C" __global__ void __launch_bounds__(32) tf_ctl_vit_g4_qkv(__nv_bfloat16* qkv, const __nv_bfloat16* qw, const __nv_bfloat16* kw,
                                                                const __nv_bfloat16* cs, const __nv_bfloat16* sn, int P, int H, int D,
                                                                float eps, int keep) {
    tf_vit_g4::qkv_norm_rope(qkv, qw, kw, cs, sn, P, H, D, eps, keep);
}

extern "C" __global__ void __launch_bounds__(256) tf_ctl_vit_g4_gelu_mul(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                                                                        __nv_bfloat16* out, long long count, int keep) {
    tf_vit_g4::gelu_mul(gate, up, out, count, keep);
}

extern "C" __global__ void __launch_bounds__(256) tf_ctl_vit_g4_pool(const __nv_bfloat16* x, const __nv_bfloat16* bias,
                                                                    const __nv_bfloat16* scale, __nv_bfloat16* out, int gw, int C,
                                                                    float root, int keep) {
    tf_vit_g4::pool_std(x, bias, scale, out, gw, C, root, keep);
}

// Gate 4: deliberately sum the BF16 dense projection in fp16 instead of its required fp32 accumulator.
extern "C" __global__ void tf_ctl_q35_dense_half(const __nv_bfloat16 *X, const __nv_bfloat16 *W, __nv_bfloat16 *Y,
                                                 int N, int K) {
    if (threadIdx.x != 0)
        return;
    const int64_t row = blockIdx.y, col = blockIdx.x;
    __half sum = __float2half_rn(0.0f);
    for (int i = 0; i < K; ++i)
        sum = __hadd(sum, __hmul(__float2half_rn(__bfloat162float(X[row * K + i])),
                                 __float2half_rn(__bfloat162float(W[col * K + i]))));
    Y[row * N + col] = __float2bfloat16_rn(__half2float(sum));
}
// The pre-template GLM BF16 operators, kept apart from the candidate math for byte equality.
#include "contract_glm_bf16_baseline.cuh"
// Gate 4: flashnext_hc's fn_hc_dense_mix8 with each 32-column group summed in bf16, while its spec declares fp32 sums.
#include "common/glue_math.cuh"

extern "C" __global__ void __launch_bounds__(256) tf_ctl_hc_mix8_bf16(const __nv_bfloat16* UP, const __nv_bfloat16* NORMED,
                                                                      __nv_bfloat16* MIXED, float* XSM, int D, int SS) {
    using tf_glue::bf;
    using tf_glue::rbf;
    __shared__ float row[8][32];
    const int r = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int group = blockIdx.y * 8 + warp, d = group * 32 + lane;
    if (group >= D / 32) return;
    float total = 0.0f;
    for (int stream = 0; stream < SS; ++stream) {
        const long long at = (static_cast<long long>(r) * SS + stream) * D + d;
        total = __fadd_rn(total, rbf(__fmul_rn(rbf(tf_glue::tf_sigmoid(bf(UP[at]))), bf(NORMED[at]))));
    }
    const float mixed = rbf(__fdiv_rn(total, static_cast<float>(SS)));
    MIXED[static_cast<long long>(r) * D + d] = __float2bfloat16_rn(mixed);
    row[warp][lane] = mixed;
    __syncwarp();
    if (lane == 0) {
        float sum = 0.0f;
        for (int i = 0; i < 32; ++i) sum = rbf(__fadd_rn(sum, row[warp][i]));
        XSM[static_cast<long long>(r) * (D / 32) + group] = sum;
    }
}

// Gate 3: norms.cu's nGPT rows with one rounding point dropped each; their range check must go red.
#include "norms_l2.cuh"

extern "C" __global__ void __launch_bounds__(tf_l2::NT) tf_ctl_l2_drop_sum(const __nv_bfloat16* x, const __nv_bfloat16* r,
                                                                           __nv_bfloat16* y, int width, float eps) {
    tf_l2::l2_rows<0>(x, r, y, width, eps);
}

#define TF_CTL_SLERP(NAME, KEEP)                                                                                   \
    extern "C" __global__ void __launch_bounds__(tf_l2::NT) NAME(const __nv_bfloat16* res, const __nv_bfloat16* out, \
                                                                 const __nv_bfloat16* alpha, float scale,          \
                                                                 __nv_bfloat16* h, int width, float eps) {         \
        tf_l2::slerp_rows<KEEP>(res, out, alpha, scale, h, width, eps);                                            \
    }
TF_CTL_SLERP(tf_ctl_slerp_drop_a, tf_l2::B)
TF_CTL_SLERP(tf_ctl_slerp_drop_b, tf_l2::A)
#undef TF_CTL_SLERP

// Gate 3: qk_rows.cu's tf_qk_rows_l2 with one rounding point dropped each; its range check must go red.
#include "qk_l2.cuh"

#define TF_CTL_QK_L2(NAME, KEEP)                                                                                   \
    extern "C" __global__ void __launch_bounds__(tf_qk_l2::WIDE) NAME(                                             \
        const tf_qk::QkSrc in, const float* qs, const float* cosv, const float* sinv, const float* inv, int rot,   \
        const int* pos, const int* slot, const tf_qk::QkDst dst, __nv_bfloat16* kc, __nv_bfloat16* vc,            \
        long long len, int h, int hk, float eps) {                                                                 \
        tf_qk_l2::rows<KEEP>(in, qs, cosv, sinv, inv, rot, pos, slot, dst, kc, vc, len, h, hk, eps);               \
    }
TF_CTL_QK_L2(tf_ctl_qk_l2_drop_rope, tf_qk_l2::NORM)
TF_CTL_QK_L2(tf_ctl_qk_l2_drop_norm, tf_qk_l2::ROPE)
#undef TF_CTL_QK_L2

// Gate 3: act_rows.cu's nGPT activations with one rounding point dropped each; their range checks must go red.
#include "act_l2.cuh"

extern "C" __global__ void __launch_bounds__(tf_act_l2::NT) tf_ctl_gate_l2_drop_gated(const __nv_bfloat16* o, const __nv_bfloat16* g,
                                                                                     long long g_ld, const float* sg,
                                                                                     __nv_bfloat16* out, int n, float eps) {
    tf_act_l2::gate_rows<0>(o, g, g_ld, sg, out, n, eps);
}

#define TF_CTL_SWIGLU_L2(NAME, KEEP)                                                                               \
    extern "C" __global__ void __launch_bounds__(tf_act_l2::NT) NAME(                                              \
        const __nv_bfloat16* g, long long g_ld, const __nv_bfloat16* u, long long u_ld, const float* sug,          \
        const float* suu, __nv_bfloat16* out, int n, float eps) {                                                  \
        tf_act_l2::swiglu_rows<KEEP>(g, g_ld, u, u_ld, sug, suu, out, n, eps);                                     \
    }
TF_CTL_SWIGLU_L2(tf_ctl_swiglu_l2_drop_gs, tf_act_l2::US | tf_act_l2::PROD)
TF_CTL_SWIGLU_L2(tf_ctl_swiglu_l2_drop_us, tf_act_l2::GS | tf_act_l2::PROD)
TF_CTL_SWIGLU_L2(tf_ctl_swiglu_l2_drop_prod, tf_act_l2::GS | tf_act_l2::US)
#undef TF_CTL_SWIGLU_L2

// Gate 1: tf_pocket_in (a warp a neuron, act rows 64 wide) with relu as fmaxf(s, 0), which turns a NaN sum into 0.
extern "C" __global__ void __launch_bounds__(128) tf_ctl_pocket_in_fmax(const __nv_bfloat16* y, int d, const float* up,
                                                                       const int* used, float* act, int rows) {
    const int lane = threadIdx.x & 31, j = blockIdx.x * 4 + (threadIdx.x >> 5), r = blockIdx.y;
    if (j >= *used || r >= rows) return;
    float s = 0.0f;
    for (int i = lane; i < d; i += 32) s += up[static_cast<int64_t>(j) * d + i] * tf_ctl_x(y, static_cast<int64_t>(r) * d + i);
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (lane == 0) {
        const float z = fmaxf(s, 0.0f);
        act[static_cast<int64_t>(r) * 64 + j] = z * z;
    }
}
// Gate 3: moss_image.cu's steps, each with one of the publisher's bf16 points dropped; their range checks must go red.
namespace tf_ctl_moss {
typedef __nv_bfloat16 bf16;
__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float ld(const bf16* p) { return __bfloat162float(*p); }
__device__ __forceinline__ float warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
// tf_moss_norm; DROP 0: bf16(x inv), 1: bf16(y s).
template <int DROP>
__device__ void norm(const bf16* X, const bf16* W, const bf16* S, const bf16* B, bf16* Y, int width, float eps,
                     int modulated) {
  __shared__ float part[8];
  const bf16* x = X + (size_t)blockIdx.x * width;
  bf16* y = Y + (size_t)blockIdx.x * width;
  float a = 0.0f;
  for (int i = threadIdx.x; i < width; i += 256) a += ld(x + i) * ld(x + i);
  a = warp_sum(a);
  if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = a;
  __syncthreads();
  float total = 0.0f;
  for (int w = 0; w < 8; ++w) total += part[w];
  const float inv = 1.0f / sqrtf(total / (float)width + eps);
  for (int i = threadIdx.x; i < width; i += 256) {
    const float xn = ld(x + i) * inv;
    float v = bf((DROP == 0 ? xn : bf(xn)) * ld(W + i));
    if (modulated) v = (DROP == 1 ? v * ld(S + i) : bf(v * ld(S + i))) + ld(B + i);
    y[i] = __float2bfloat16_rn(v);
  }
}
}  // namespace tf_ctl_moss

extern "C" __global__ void __launch_bounds__(256) tf_ctl_moss_norm_inv(const __nv_bfloat16* X, const __nv_bfloat16* W,
                                                                       const __nv_bfloat16* S, const __nv_bfloat16* B,
                                                                       __nv_bfloat16* Y, int width, float eps,
                                                                       int modulated) {
  tf_ctl_moss::norm<0>(X, W, S, B, Y, width, eps, modulated);
}

extern "C" __global__ void __launch_bounds__(256) tf_ctl_moss_norm_scale(const __nv_bfloat16* X, const __nv_bfloat16* W,
                                                                         const __nv_bfloat16* S, const __nv_bfloat16* B,
                                                                         __nv_bfloat16* Y, int width, float eps,
                                                                         int modulated) {
  tf_ctl_moss::norm<1>(X, W, S, B, Y, width, eps, modulated);
}

// tf_moss_qk without bf16(q cos).
extern "C" __global__ void __launch_bounds__(256) tf_ctl_moss_qk_cos(const __nv_bfloat16* QR, const __nv_bfloat16* KR,
                                                                     const __nv_bfloat16* VR, const __nv_bfloat16* NQ,
                                                                     const __nv_bfloat16* NK, const float* CS,
                                                                     const float* SN, __nv_bfloat16* QKV, int rows,
                                                                     int heads, int stride, float eps) {
  using namespace tf_ctl_moss;
  const int r = blockIdx.x, h = blockIdx.y * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (h >= heads || r >= rows) return;
  const size_t from = (size_t)r * stride + h * 128 + lane * 4;
  float q[4], k[4], qq = 0.0f, kk = 0.0f;
  for (int i = 0; i < 4; ++i) {
    q[i] = ld(QR + from + i);
    k[i] = ld(KR + from + i);
    qq += q[i] * q[i];
    kk += k[i] * k[i];
  }
  const float qn = 1.0f / sqrtf(warp_sum(qq) / 128.0f + eps);
  const float kn = 1.0f / sqrtf(warp_sum(kk) / 128.0f + eps);
  for (int i = 0; i < 4; ++i) {
    q[i] = bf(bf(q[i] * qn) * ld(NQ + lane * 4 + i));
    k[i] = bf(bf(k[i] * kn) * ld(NK + lane * 4 + i));
  }
  const float sign = lane < 16 ? -1.0f : 1.0f;
  bf16* qo = QKV + (size_t)r * 3 * heads * 128 + h * 128 + lane * 4;
  bf16* ko = qo + heads * 128;
  bf16* vo = ko + heads * 128;
  for (int i = 0; i < 4; ++i) {
    const float qp = sign * __shfl_xor_sync(0xffffffffu, q[i], 16);
    const float kp = sign * __shfl_xor_sync(0xffffffffu, k[i], 16);
    const int j = (lane & 15) * 4 + i;
    const float c = CS[(size_t)r * 64 + j], s = SN[(size_t)r * 64 + j];
    qo[i] = __float2bfloat16_rn(q[i] * c + bf(qp * s));
    ko[i] = __float2bfloat16_rn(bf(k[i] * c) + bf(kp * s));
    vo[i] = VR[from + i];
  }
}

// tf_moss_situ without bf16(sigmoid(g)).
extern "C" __global__ void __launch_bounds__(256) tf_ctl_moss_situ_sigmoid(const __nv_bfloat16* H, const __nv_bfloat16* G,
                                                                           __nv_bfloat16* O, int width, int stride,
                                                                           int rows) {
  using namespace tf_ctl_moss;
  const int c = blockIdx.y * 256 + threadIdx.x, r = blockIdx.x;
  if (c >= width || r >= rows) return;
  const size_t i = (size_t)r * stride + c;
  const float gv = ld(G + i), hv = ld(H + i);
  const float sig = 1.0f / (1.0f + expf(-gv));
  const float left = bf(bf(4.0f * bf(tanhf(bf(gv * 0.25f)))) * sig);
  const float right = bf(25.0f * bf(tanhf(bf(hv * (1.0f / 25.0f)))));
  O[(size_t)r * width + c] = __float2bfloat16_rn(left * right);
}

// tf_moss_gate_add without bf16(g y).
extern "C" __global__ void __launch_bounds__(256) tf_ctl_moss_gate_add_product(__nv_bfloat16* X, const __nv_bfloat16* Y,
                                                                               const __nv_bfloat16* G, int width,
                                                                               int rows) {
  using namespace tf_ctl_moss;
  const int c = blockIdx.y * 256 + threadIdx.x, r = blockIdx.x;
  if (c >= width || r >= rows) return;
  const size_t i = (size_t)r * width + c;
  X[i] = __float2bfloat16_rn(ld(X + i) + ld(G + c) * ld(Y + i));
}
// Gate 3: main's tf_exl3_plan_finish (PlanStage without alpha), the baseline, and two SwiGLU-OAI faults that go red.
#include "exl3_plan.cuh"

struct PlanStageMain {
    const int *items, *counts, *members;
    const __nv_bfloat16* x;
    const half *su0, *su1, *sv0, *sv1;
    half *x0, *x1;
    const float* z;
    void* out;
    int k, n, p, sk, slots, stride, skip, epi, act_mode;
    float limit;
};

extern "C" __global__ void tf_ctl_exl3_plan_finish_old(const PlanStageMain a) {
    const int it = blockIdx.x, blk = blockIdx.y, lane = threadIdx.x;
    if (it >= a.counts[0]) return;
    const int e = a.items[3 * it], first = a.items[3 * it + 1], count = a.items[3 * it + 2];
    if (e == a.skip) return;
    const int n = blk * 128 + 4 * lane;
    for (int j = 0; j < count; ++j) {
        const int p = a.members[first + j];
        float g[4] = {}, u[4] = {};
        for (int s = 0; s < a.sk; ++s)
            for (int q = 0; q < 4; ++q) {
                g[q] += a.z[((size_t)s * a.p + p) * a.n + n + q];
                if (a.epi == 2) u[q] += a.z[((size_t)(a.sk + s) * a.p + p) * a.n + n + q];
            }
        tf_exl3::fwht128(g, lane);
        if (a.epi == 2) tf_exl3::fwht128(u, lane);
        for (int q = 0; q < 4; ++q) {
            float v = g[q] * tf_exl3::HAD_SCALE * __half2float(a.sv0[(size_t)e * a.n + n + q]);
            if (a.epi == 2) {
                float up = u[q] * tf_exl3::HAD_SCALE * __half2float(a.sv1[(size_t)e * a.n + n + q]);
                if (a.act_mode == 0) { v = tf_exl3::bf16r(v); up = tf_exl3::bf16r(up); }
                if (a.limit > 0) { v = tf_exl3::upTo(v, a.limit); up = tf_exl3::within(up, a.limit); }
                v = v / (1.f + expf(-v));
                if (a.act_mode == 0) v = tf_exl3::bf16r(v);
                v *= up;
            }
            const size_t at = (size_t)p * a.n + n + q;
            if (a.epi == 0) reinterpret_cast<float*>(a.out)[at] = v;
            else reinterpret_cast<__nv_bfloat16*>(a.out)[at] = __float2bfloat16_rn(v);
        }
    }
}

extern "C" __global__ void tf_ctl_exl3_plan_finish_no_plus_one(const PlanStage a) { tf_exl3::plan_finish<tf_exl3::FINISH_NO_PLUS_ONE>(a); }
extern "C" __global__ void tf_ctl_exl3_plan_finish_alpha_one(const PlanStage a) { tf_exl3::plan_finish<tf_exl3::FINISH_ALPHA_ONE>(a); }

// Gate 3: main's grouped EXL3 loop, the baseline its load-ahead replaces, and two load-ahead faults that must go red.
#include "exl3_grouped.cuh"

#define TF_CTL_EXL3_GROUPED(NAME, CB, LOAD)                                                                       \
    extern "C" __global__ void NAME(tf_exl3::GroupedArgs a) {                                                  \
        tf_exl3::grouped_body<CB, 8, 4, LOAD>(a, blockIdx.x, blockIdx.y, blockIdx.z);                          \
    }
TF_CTL_EXL3_GROUPED(tf_ctl_exl3_grouped_old_0, 0, tf_exl3::PLAIN)
TF_CTL_EXL3_GROUPED(tf_ctl_exl3_grouped_old_1, 1, tf_exl3::PLAIN)
TF_CTL_EXL3_GROUPED(tf_ctl_exl3_grouped_old_2, 2, tf_exl3::PLAIN)
TF_CTL_EXL3_GROUPED(tf_ctl_exl3_grouped_skip, 2, tf_exl3::SKIP)
TF_CTL_EXL3_GROUPED(tf_ctl_exl3_grouped_stale, 2, tf_exl3::STALE)
#undef TF_CTL_EXL3_GROUPED

// Gate 3: Idol's glue (idol.cu) with one fp16 rounding dropped each; the intervals hold idol-sdk's rounding points.
namespace tf_ctl_idol {
__device__ __forceinline__ float h(float x) { return __half2float(__float2half_rn(x)); }
__device__ __forceinline__ float ld(const __half* p) { return __half2float(*p); }
__device__ __forceinline__ float block_sum(float v, float* part) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = v;
    __syncthreads();
    float total = 0.0f;
    for (int w = 0; w < 8; ++w) total += part[w];
    return total;
}
}  // namespace tf_ctl_idol

// tf_idol_norm modulating the unrounded LayerNorm: fp16(fp16(y s1) + sh) where idol-sdk rounds y first.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_idol_norm_late(const __half* X, const float* W, const float* B,
                                                                        const __half* S1, const __half* SH, __half* Y,
                                                                        int width, float eps, int mode) {
    using namespace tf_ctl_idol;
    __shared__ float part[8];
    const __half* x = X + (size_t)blockIdx.x * width;
    float a = 0.0f, q = 0.0f;
    for (int i = threadIdx.x; i < width; i += 256) a += ld(x + i);
    const float mean = block_sum(a, part) / (float)width;
    for (int i = threadIdx.x; i < width; i += 256) q += (ld(x + i) - mean) * (ld(x + i) - mean);
    const float inv = 1.0f / sqrtf(block_sum(q, part) / (float)width + eps);
    for (int i = threadIdx.x; i < width; i += 256) {
        float v = (ld(x + i) - mean) * inv;
        v = mode == 1 ? h(v * W[i] + B[i]) : h(h(v * ld(S1 + i)) + ld(SH + i));
        Y[(size_t)blockIdx.x * width + i] = __float2half_rn(v);
    }
}

// tf_idol_gate_add with g y left in fp32: fp16(x + g y) where idol-sdk rounds g y to fp16 first.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_idol_gate_add_once(__half* X, const __half* Y, const __half* G,
                                                                            int width, int rows) {
    using namespace tf_ctl_idol;
    const int c = blockIdx.y * 256 + threadIdx.x, r = blockIdx.x;
    if (c >= width || r >= rows) return;
    const size_t i = (size_t)r * width + c;
    X[i] = __float2half_rn(ld(X + i) + ld(G + c) * ld(Y + i));
}

// tf_idol_bias taking GELU of the fp32 sum: fp16(GELU(acc + b)) where idol-sdk rounds the linear's output first.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_idol_bias_late(const float* ACC, const __half* B, __half* Y,
                                                                        int width, int rows, int act) {
    using namespace tf_ctl_idol;
    const int c = blockIdx.y * 256 + threadIdx.x, r = blockIdx.x;
    if (c >= width || r >= rows) return;
    const size_t i = (size_t)r * width + c;
    float v = ACC[i] + ld(B + c);
    if (act) v = 0.5f * v * (1.0f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v)));
    Y[i] = __float2half_rn(act ? v : h(v));
}

// Gates 3 and 4: Idol's DINOv3 glue (idol_dino.cu), one bf16 rounding dropped or added, or its sums in fp16.
#include "idol_dino_controls.cuh"
// Idol's background remover's controls (fp16 rounding points or sums, a dropped NaN); none ship.
#include "idol_birefnet_controls.cuh"
