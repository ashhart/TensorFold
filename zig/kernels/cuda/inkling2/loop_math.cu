// Inkling 2 Mini: raw block residuals, two-bank messages, rank-64 gates and three-input QKV projection.
#include <cstdint>
#include <cuda_bf16.h>
#include <mma.h>

#include "common/qmm_frag.cuh"

namespace ink2l {

constexpr int kRank = 64;
constexpr int kMaxBlocks = 8;
constexpr int kMaxQueries = 3;

__device__ __forceinline__ float bf(uint16_t u) { return __uint_as_float(static_cast<uint32_t>(u) << 16); }
__device__ __forceinline__ uint16_t tobf(float f) { return __bfloat16_as_ushort(__float2bfloat16_rn(f)); }
__device__ __forceinline__ float bfr(float f) { return bf(tobf(f)); }

__device__ __forceinline__ float warpSum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// Every call has the same reduction tree: lanes' strided chains, each warp's butterfly, then warp 0 over partials.
template <int N>
__device__ __forceinline__ void blockSums(float (&v)[N], float* sh) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    for (int i = 0; i < N; ++i) v[i] = warpSum(v[i]);
    __syncthreads();
    if (lane == 0)
        for (int i = 0; i < N; ++i) sh[i * 32 + w] = v[i];
    __syncthreads();
    for (int i = 0; i < N; ++i) {
        const float t = lane < nw ? sh[i * 32 + lane] : 0.0f;
        v[i] = __shfl_sync(0xffffffffu, warpSum(t), 0);
    }
}

// A14(a): Mini's queried residual and message dot products scale after the dot, by binary32(1 / 48).
__device__ __forceinline__ float queryScale(int c) {
    return c == 2304 ? 1.0f / 48.0f : rsqrtf(static_cast<float>(c));
}

__device__ __forceinline__ void softmax2(float l0, float l1, float* w) {
    const float m = fmaxf(l0, l1);
    const float e0 = expf(l0 - m), e1 = expf(l1 - m);
    const float d = __fadd_rn(e0, e1);
    w[0] = __fdiv_rn(e0, d);
    w[1] = __fdiv_rn(e1, d);
}

struct ArArgs {
    const uint16_t* blocks[kMaxBlocks];  // bf16 [rows, C], one completed state a pointer
    const void* partial;                 // the final source, BF16 or FP32 according to partial_bf16
    const float* query;                  // fp32 [nq, C]
    float* out;                          // fp32 [nq, rows, C], deliberately neither normalized nor rounded
    int n_blocks, partial_bf16, nq, rows, C;
    float eps;
    int blocks_normed;                   // nonzero: `blocks` are fp32 rows tf_ink2_loop_prenorm's ar output holds, already normalized
};

// The AR half of Small's residual mix, without its fused RMSNorm. Needed before Mini's controller, not just after it.
__device__ void ar(const ArArgs a) {
    __shared__ float sh[32 * (1 + kMaxQueries)];
    __shared__ float wts[kMaxQueries][kMaxBlocks + 1];
    __shared__ float rr[kMaxBlocks + 1];
    const size_t row = blockIdx.x;
    const int n = a.n_blocks + 1;
    auto entry = [&](int j, int c) -> float {
        if (j < a.n_blocks)
            return a.blocks_normed ? reinterpret_cast<const float*>(a.blocks[j])[row * a.C + c] : bf(a.blocks[j][row * a.C + c]);
        return a.partial_bf16 ? bf(static_cast<const uint16_t*>(a.partial)[row * a.C + c])
                              : static_cast<const float*>(a.partial)[row * a.C + c];
    };
    for (int j = 0; j < n; ++j) {
        if (j < a.n_blocks && a.blocks_normed) {  // normalized once where the block completed: x * 1.0f is x
            if (threadIdx.x == 0) rr[j] = 1.0f;
            continue;
        }
        float ss[1] = {0.0f};
        for (int c = threadIdx.x; c < a.C; c += blockDim.x) {
            const float x = entry(j, c);
            ss[0] = __fmaf_rn(x, x, ss[0]);
        }
        blockSums<1>(ss, sh);
        if (threadIdx.x == 0) rr[j] = rsqrtf(__fadd_rn(__fdiv_rn(ss[0], static_cast<float>(a.C)), a.eps));
    }
    __syncthreads();
    for (int j = 0; j < n; ++j) {
        float dots[kMaxQueries] = {0.0f, 0.0f, 0.0f};
        for (int c = threadIdx.x; c < a.C; c += blockDim.x) {
            const float normalized = __fmul_rn(entry(j, c), rr[j]);
            for (int q = 0; q < a.nq; ++q)
                dots[q] = __fmaf_rn(a.query[static_cast<size_t>(q) * a.C + c], normalized, dots[q]);
        }
        blockSums<kMaxQueries>(dots, sh);
        if (threadIdx.x == 0)
            for (int q = 0; q < a.nq; ++q) wts[q][j] = __fmul_rn(dots[q], queryScale(a.C));
    }
    __syncthreads();
    if (threadIdx.x < a.nq) {
        float* l = wts[threadIdx.x];
        float m = -INFINITY, d = 0.0f;
        for (int j = 0; j < n; ++j) m = fmaxf(m, l[j]);
        for (int j = 0; j < n; ++j) d = __fadd_rn(d, expf(l[j] - m));
        for (int j = 0; j < n; ++j) l[j] = __fdiv_rn(expf(l[j] - m), d);
    }
    __syncthreads();
    for (int q = 0; q < a.nq; ++q) {
        float* out = a.out + (static_cast<size_t>(q) * a.rows + row) * a.C;
        for (int c = threadIdx.x; c < a.C; c += blockDim.x) {
            float h = 0.0f;
            for (int j = 0; j < n; ++j) {
                const float z = __fmul_rn(entry(j, c), rr[j]);
                h = __fmaf_rn(wts[q][j], z, h);
            }
            out[c] = h;
        }
    }
}

}  // namespace ink2l

extern "C" __global__ void __launch_bounds__(512) tf_ink2_loop_ar(const ink2l::ArArgs a) {
    ink2l::ar(a);
}

// Single-load register-cached AR for C <= 5 x 512 (one block a row, 512 threads): Pass 0 loads the
// partial row and every block row ONCE into registers (no SMEM staging); Passes 1-3 keep ar's exact
// per-column accumulation order, intrinsics, queryScale and blockSums trees, so output is 100%
// bit-exact (max_abs_diff == 0.0) vs tf_ink2_loop_ar. The preload loop is fully unrolled with
// predication (ok && j < n_blocks): no j >= n OOB reads, no runtime breaks.
namespace ink2ar {
constexpr int kFast = 5;
}  // namespace ink2ar

namespace ink2ar {
template <bool HN>
__device__ __forceinline__ void arReg(const ink2l::ArArgs& a, uint16_t* __restrict__ hn_out, float gate_eps) {
    __shared__ float sh[32 * (1 + ink2l::kMaxQueries)];
    __shared__ float wts[ink2l::kMaxQueries][ink2l::kMaxBlocks + 1];
    __shared__ float rr[ink2l::kMaxBlocks + 1];
    constexpr int kXs = ink2ar::kFast * 512;
    __shared__ float xs[HN ? ink2l::kMaxQueries * kXs : 1];  // HN: the output rows, for gate_norm's 256-thread chains
    const size_t row = blockIdx.x;
    const int C = a.C, n = a.n_blocks + 1;
    const float* prow_f = static_cast<const float*>(a.partial) + row * C;
    const uint16_t* prow_h = static_cast<const uint16_t*>(a.partial) + row * C;
    float px[ink2ar::kFast];
    float bx[ink2l::kMaxBlocks][ink2ar::kFast];
#pragma unroll
    for (int k = 0; k < ink2ar::kFast; ++k) {
        const int c = threadIdx.x + k * 512;
        const bool ok = c < C;
        px[k] = ok ? (a.partial_bf16 ? ink2l::bf(prow_h[c]) : prow_f[c]) : 0.0f;
#pragma unroll
        for (int j = 0; j < ink2l::kMaxBlocks; ++j) {
            const bool live = ok && j < a.n_blocks;
            bx[j][k] = live ? (a.blocks_normed ? reinterpret_cast<const float*>(a.blocks[j])[row * C + c]
                                               : ink2l::bf(a.blocks[j][row * C + c]))
                            : 0.0f;
        }
    }
    auto entry = [&](int j, int k) -> float {
        if (j < a.n_blocks) return bx[j][k];
        return px[k];
    };
#pragma unroll
    for (int j = 0; j < ink2l::kMaxBlocks + 1; ++j) {
        if (j >= n) break;
        if (j < a.n_blocks && a.blocks_normed) {  // normalized once where the block completed: x * 1.0f is x
            if (threadIdx.x == 0) rr[j] = 1.0f;
            continue;
        }
        float ss[1] = {0.0f};
#pragma unroll
        for (int k = 0; k < ink2ar::kFast; ++k) {
            const int c = threadIdx.x + k * 512;
            if (c >= C) continue;
            const float x = entry(j, k);
            ss[0] = __fmaf_rn(x, x, ss[0]);
        }
        ink2l::blockSums<1>(ss, sh);
        if (threadIdx.x == 0) rr[j] = rsqrtf(__fadd_rn(__fdiv_rn(ss[0], static_cast<float>(a.C)), a.eps));
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < ink2l::kMaxBlocks + 1; ++j) {
        if (j >= n) break;
        float dots[ink2l::kMaxQueries] = {0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int k = 0; k < ink2ar::kFast; ++k) {
            const int c = threadIdx.x + k * 512;
            if (c >= C) continue;
            const float normalized = __fmul_rn(entry(j, k), rr[j]);
            for (int q = 0; q < a.nq; ++q)
                dots[q] = __fmaf_rn(a.query[static_cast<size_t>(q) * a.C + c], normalized, dots[q]);
        }
        ink2l::blockSums<ink2l::kMaxQueries>(dots, sh);
        if (threadIdx.x == 0)
            for (int q = 0; q < a.nq; ++q) wts[q][j] = __fmul_rn(dots[q], ink2l::queryScale(a.C));
    }
    __syncthreads();
    if (threadIdx.x < a.nq) {
        float* l = wts[threadIdx.x];
        float m = -INFINITY, d = 0.0f;
        for (int j = 0; j < n; ++j) m = fmaxf(m, l[j]);
        for (int j = 0; j < n; ++j) d = __fadd_rn(d, expf(l[j] - m));
        for (int j = 0; j < n; ++j) l[j] = __fdiv_rn(expf(l[j] - m), d);
    }
    __syncthreads();
    for (int q = 0; q < a.nq; ++q) {
        float* out = a.out + (static_cast<size_t>(q) * a.rows + row) * a.C;
#pragma unroll
        for (int k = 0; k < ink2ar::kFast; ++k) {
            const int c = threadIdx.x + k * 512;
            if (c >= C) continue;
            float h = 0.0f;
#pragma unroll
            for (int j = 0; j < ink2l::kMaxBlocks + 1; ++j) {
                if (j < n) {
                    const float z = __fmul_rn(entry(j, k), rr[j]);
                    h = __fmaf_rn(wts[q][j], z, h);
                }
            }
            out[c] = h;
            if constexpr (HN) xs[q * kXs + c] = h;
        }
    }
    if constexpr (HN) {
        // tf_ink2_loop_gate_norm's state half of every output row: 256 lanes' chains i = t, t + 256, ... (threads past
        // 255 add zero partials: the same tree; blockSums sums each query's slot alone), its rsqrt (every thread holds the
        // broadcast sum: thread 0's bits) and its products, rounded to bf16 as tf_ink2_loop_gate_norm_wide stores them
        __syncthreads();
        float ss[ink2l::kMaxQueries] = {0.0f, 0.0f, 0.0f};
        if (threadIdx.x < 256)
            for (int i = threadIdx.x; i < C; i += 256)
#pragma unroll
                for (int q = 0; q < ink2l::kMaxQueries; ++q)
                    if (q < a.nq) ss[q] = __fmaf_rn(xs[q * kXs + i], xs[q * kXs + i], ss[q]);
        ink2l::blockSums<ink2l::kMaxQueries>(ss, sh);
#pragma unroll
        for (int q = 0; q < ink2l::kMaxQueries; ++q) {
            if (q >= a.nq) break;
            const float inv_h = rsqrtf(__fadd_rn(__fdiv_rn(ss[q], static_cast<float>(C)), gate_eps));
            uint16_t* hn = hn_out + (static_cast<size_t>(q) * a.rows + row) * C;
            for (int i = threadIdx.x; i < C; i += 512) hn[i] = ink2l::tobf(__fmul_rn(xs[q * kXs + i], inv_h));
        }
    }
}
}  // namespace ink2ar

extern "C" __global__ void __launch_bounds__(512) tf_ink2_loop_ar_reg(const ink2l::ArArgs a) {
    ink2ar::arReg<false>(a, nullptr, 0.0f);
}

// tf_ink2_loop_ar_reg's output, plus each output row's state half as the wide gate normalizes it: hn bf16
// [nq, rows, C] (gate_norm's chain, tree and products with gate_eps). Separate parameters, so ArArgs (every ar
// kernel's by-value parameter block) keeps its size.
extern "C" __global__ void __launch_bounds__(512)
    tf_ink2_loop_ar_reg_hn(const ink2l::ArArgs a, uint16_t* __restrict__ hn, float gate_eps) {
    ink2ar::arReg<true>(a, hn, gate_eps);
}

// A completed block's RMS normalization, once, where it completes: fp32 rows for ar() (fma sums, a.eps) and for
// tf_ink2_mix / tf_ink2_mix_rows (ops.cu: separate multiply and add, eps 1e-6f). Each flavor has its consumer's
// strided chain, reduction tree and roundings, so a consumer that reads it skips its own pass and keeps its bits.
extern "C" __global__ void __launch_bounds__(512) tf_ink2_loop_prenorm(const uint16_t* __restrict__ src, float* __restrict__ out_ar,
                                                                      float* __restrict__ out_mix, int C, float eps) {
    __shared__ float sh[32 * 2];
    const size_t row = blockIdx.x;
    float ss[2] = {0.0f, 0.0f};
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        const float x = ink2l::bf(src[row * C + c]);
        ss[0] = __fmaf_rn(x, x, ss[0]);
        ss[1] = __fadd_rn(ss[1], __fmul_rn(x, x));
    }
    ink2l::blockSums<2>(ss, sh);
    const float r_ar = rsqrtf(__fadd_rn(__fdiv_rn(ss[0], static_cast<float>(C)), eps));
    const float r_mix = rsqrtf(__fadd_rn(__fdiv_rn(ss[1], static_cast<float>(C)), 1e-6f));
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        const float x = ink2l::bf(src[row * C + c]);
        out_ar[row * C + c] = __fmul_rn(x, r_ar);
        out_mix[row * C + c] = __fmul_rn(x, r_mix);
    }
}

namespace ink2l {

// One block handles one token and role, sharing each bank's normalization across all 12 core queries.
__device__ void messages(const uint16_t* __restrict__ bank0, const uint16_t* __restrict__ bank1,
                          const float* __restrict__ qm, const float* __restrict__ qkv,
                          float* __restrict__ out, int core_layers, int c, float eps) {
    __shared__ float sh[2 * 32];
    __shared__ float inv[2], w[2];
    const size_t row = blockIdx.x;
    const int role = blockIdx.y;  // q, k, v, mlp
    const uint16_t* banks[2] = {bank0 + row * c, bank1 + row * c};
    const float* role_queries = role == 0 || role == 3 ? qm : qkv;
    const int odd = role == 3 || role == 2;
    float ss[2] = {0.0f, 0.0f};
    for (int i = threadIdx.x; i < c; i += blockDim.x) {
        const float x0 = bf(banks[0][i]), x1 = bf(banks[1][i]);
        ss[0] = __fmaf_rn(x0, x0, ss[0]);
        ss[1] = __fmaf_rn(x1, x1, ss[1]);
    }
    blockSums<2>(ss, sh);
    if (threadIdx.x == 0) {
        inv[0] = rsqrtf(__fadd_rn(__fdiv_rn(ss[0], static_cast<float>(c)), eps));
        inv[1] = rsqrtf(__fadd_rn(__fdiv_rn(ss[1], static_cast<float>(c)), eps));
    }
    __syncthreads();
    for (int j = 0; j < core_layers; ++j) {
        const float* query = role_queries + static_cast<size_t>(2 * j + odd) * c;
        float dots[2] = {0.0f, 0.0f};
        for (int i = threadIdx.x; i < c; i += blockDim.x) {
            const float q = query[i];
            const float b0 = __fmul_rn(bf(banks[0][i]), inv[0]);
            const float b1 = __fmul_rn(bf(banks[1][i]), inv[1]);
            dots[0] = __fmaf_rn(q, b0, dots[0]);
            dots[1] = __fmaf_rn(q, b1, dots[1]);
        }
        blockSums<2>(dots, sh);
        if (threadIdx.x == 0)
            softmax2(__fmul_rn(dots[0], queryScale(c)), __fmul_rn(dots[1], queryScale(c)), w);
        __syncthreads();
        float* dst = out + ((static_cast<size_t>(row * core_layers + j) * 4 + role) * static_cast<size_t>(c));
        for (int i = threadIdx.x; i < c; i += blockDim.x) {
            const float b0 = __fmul_rn(bf(banks[0][i]), inv[0]);
            const float b1 = __fmul_rn(bf(banks[1][i]), inv[1]);
            const float m0 = __fmul_rn(w[0], b0), m1 = __fmul_rn(w[1], b1);
            dst[i] = __fadd_rn(m0, m1);
        }
    }
}

// One gate a block. A shared BF16 zeta is the sole bottleneck output; forget/gain stay FP32 through tanh (A7(a)).
__device__ void gate(const float* __restrict__ h, const float* __restrict__ m,
                                            const uint16_t* __restrict__ down, const uint16_t* __restrict__ forget,
                                            const uint16_t* __restrict__ gain, float* __restrict__ out,
                                            int c, int m_stride, float eps) {
    __shared__ float sh[2 * 32];
    __shared__ float inv_h, inv_m;
    __shared__ uint16_t zeta[kRank];
    const size_t row = blockIdx.x;
    const float* hr = h + row * c;
    const float* mr = m + row * m_stride;
    float ss[2] = {0.0f, 0.0f};
    for (int i = threadIdx.x; i < c; i += blockDim.x) {
        ss[0] = __fmaf_rn(hr[i], hr[i], ss[0]);
        ss[1] = __fmaf_rn(mr[i], mr[i], ss[1]);
    }
    blockSums<2>(ss, sh);
    if (threadIdx.x == 0) {
        inv_h = rsqrtf(__fadd_rn(__fdiv_rn(ss[0], static_cast<float>(c)), eps));
        inv_m = rsqrtf(__fadd_rn(__fdiv_rn(ss[1], static_cast<float>(c)), eps));
    }
    __syncthreads();
    // Pair ranks to reuse each normalized input, but keep every dot's 256-thread chains and two-level reduction tree.
    for (int r = 0; r < kRank; r += 2) {
        float acc[2] = {0.0f, 0.0f};
        const uint16_t* d0 = down + static_cast<size_t>(r) * (2 * c);
        const uint16_t* d1 = d0 + 2 * c;
        for (int i = threadIdx.x; i < 2 * c; i += blockDim.x) {
            const bool message = i >= c;
            const float x = message ? mr[i - c] : hr[i];
            const float n = bfr(__fmul_rn(x, message ? inv_m : inv_h));
            acc[0] = __fmaf_rn(bf(d0[i]), n, acc[0]);
            acc[1] = __fmaf_rn(bf(d1[i]), n, acc[1]);
        }
        blockSums<2>(acc, sh);
        if (threadIdx.x == 0) {
            zeta[r] = tobf(acc[0]);
            zeta[r + 1] = tobf(acc[1]);
        }
    }
    __syncthreads();
    float* dst = out + row * c;
    for (int i = threadIdx.x; i < c; i += blockDim.x) {
        const uint16_t* f = forget + static_cast<size_t>(i) * kRank;
        const uint16_t* g = gain + static_cast<size_t>(i) * kRank;
        float sf = 0.0f, sg = 0.0f;
        for (int r = 0; r < kRank; ++r) {
            const float z = bf(zeta[r]);
            sf = __fmaf_rn(bf(f[r]), z, sf);
            sg = __fmaf_rn(bf(g[r]), z, sg);
        }
        const float fg = tanhf(sf), gg = tanhf(sg);
        // The reference's four FP32 rounding steps, kept explicit so contraction cannot erase any of them.
        const float keep = __fsub_rn(1.0f, fg);
        const float old = __fmul_rn(keep, hr[i]);
        const float added = __fmul_rn(gg, mr[i]);
        dst[i] = __fadd_rn(old, added);
    }
}

}  // namespace ink2l

// tf_ink2_loop_messages for one core layer (C <= 2560, 256 threads), plus each (row, role) message as the wide gate
// normalizes its message half: mn [rows, 4, C] bf16 beside out, from gate_norm's chain, tree, rsqrt and products.
extern "C" __global__ void __launch_bounds__(256)
    tf_ink2_loop_messages_norm(const uint16_t* __restrict__ bank0, const uint16_t* __restrict__ bank1,
                               const float* __restrict__ qm, const float* __restrict__ qkv, float* __restrict__ out,
                               uint16_t* __restrict__ mn, int rows, int c, float eps, float gate_eps) {
    using namespace ink2l;
    (void)rows;
    constexpr int kCache = 10;  // C <= 2560 over 256 threads
    __shared__ float sh[2 * 32];
    __shared__ float inv[2], w[2], inv_m;
    const size_t row = blockIdx.x;
    const int role = blockIdx.y;  // q, k, v, mlp
    const uint16_t* banks[2] = {bank0 + row * c, bank1 + row * c};
    const float* query = (role == 0 || role == 3 ? qm : qkv) + static_cast<size_t>(role == 3 || role == 2) * c;
    float ss[2] = {0.0f, 0.0f};
    for (int i = threadIdx.x; i < c; i += blockDim.x) {
        const float x0 = bf(banks[0][i]), x1 = bf(banks[1][i]);
        ss[0] = __fmaf_rn(x0, x0, ss[0]);
        ss[1] = __fmaf_rn(x1, x1, ss[1]);
    }
    blockSums<2>(ss, sh);
    if (threadIdx.x == 0) {
        inv[0] = rsqrtf(__fadd_rn(__fdiv_rn(ss[0], static_cast<float>(c)), eps));
        inv[1] = rsqrtf(__fadd_rn(__fdiv_rn(ss[1], static_cast<float>(c)), eps));
    }
    __syncthreads();
    float dots[2] = {0.0f, 0.0f};
    for (int i = threadIdx.x; i < c; i += blockDim.x) {
        const float q = query[i];
        const float b0 = __fmul_rn(bf(banks[0][i]), inv[0]);
        const float b1 = __fmul_rn(bf(banks[1][i]), inv[1]);
        dots[0] = __fmaf_rn(q, b0, dots[0]);
        dots[1] = __fmaf_rn(q, b1, dots[1]);
    }
    blockSums<2>(dots, sh);
    if (threadIdx.x == 0) softmax2(__fmul_rn(dots[0], queryScale(c)), __fmul_rn(dots[1], queryScale(c)), w);
    __syncthreads();
    float* dst = out + (row * 4 + role) * static_cast<size_t>(c);
    float held[kCache];
    float sm[1] = {0.0f};
#pragma unroll
    for (int k = 0; k < kCache; ++k) {
        const int i = threadIdx.x + k * 256;
        if (i >= c) continue;
        const float b0 = __fmul_rn(bf(banks[0][i]), inv[0]);
        const float b1 = __fmul_rn(bf(banks[1][i]), inv[1]);
        const float m0 = __fmul_rn(w[0], b0), m1 = __fmul_rn(w[1], b1);
        held[k] = __fadd_rn(m0, m1);
        dst[i] = held[k];
        sm[0] = __fmaf_rn(held[k], held[k], sm[0]);
    }
    blockSums<1>(sm, sh);
    if (threadIdx.x == 0) inv_m = rsqrtf(__fadd_rn(__fdiv_rn(sm[0], static_cast<float>(c)), gate_eps));
    __syncthreads();
    uint16_t* nd = mn + (row * 4 + role) * static_cast<size_t>(c);
#pragma unroll
    for (int k = 0; k < kCache; ++k) {
        const int i = threadIdx.x + k * 256;
        if (i < c) nd[i] = tobf(__fmul_rn(held[k], inv_m));
    }
}

extern "C" __global__ void __launch_bounds__(256)
    tf_ink2_loop_messages(const uint16_t* __restrict__ bank0, const uint16_t* __restrict__ bank1,
                          const float* __restrict__ qm, const float* __restrict__ qkv, float* __restrict__ out,
                          int rows, int core_layers, int c, float eps) {
    (void)rows;  // grid x is rows * core_layers; row derives from pair/core_layers, explicitly visit-local.
    ink2l::messages(bank0, bank1, qm, qkv, out, core_layers, c, eps);
}

extern "C" __global__ void __launch_bounds__(256)
    tf_ink2_loop_gate(const float* __restrict__ h, const float* __restrict__ m, const uint16_t* __restrict__ down,
                      const uint16_t* __restrict__ forget, const uint16_t* __restrict__ gain, float* __restrict__ out,
                      int rows, int c, float eps) {
    (void)rows;
    ink2l::gate(h, m, down, forget, gain, out, c, c, eps);
}

extern "C" __global__ void __launch_bounds__(256)
    tf_ink2_loop_gate_strided(const float* __restrict__ h, const float* __restrict__ m, const uint16_t* __restrict__ down,
                              const uint16_t* __restrict__ forget, const uint16_t* __restrict__ gain, float* __restrict__ out,
                              int rows, int c, int m_stride, float eps) {
    (void)rows;
    ink2l::gate(h, m, down, forget, gain, out, c, m_stride, eps);
}

// Fast multi-block gate, bit-identical to ink2l::gate: norm (inv + BF16 [h, m]) runs one block a row,
// down (rank pairs of zeta) one block a (row, rank pair), apply (channels) one block a (row, 256 channels).
// Splitting only parallelizes: every op, order and rounding matches the reference's, so every bit matches.
extern "C" __global__ void __launch_bounds__(256)
    tf_ink2_loop_gate_norm(const float* __restrict__ h, const float* __restrict__ m,
                            uint16_t* __restrict__ norm_hm, int rows, int c, int m_stride, float eps) {
    (void)rows;
    __shared__ float sh[2 * 32];
    __shared__ float inv_h, inv_m;
    const size_t row = blockIdx.x;
    const float* hr = h + row * c;
    const float* mr = m + row * m_stride;
    float ss[2] = {0.0f, 0.0f};
    for (int i = threadIdx.x; i < c; i += blockDim.x) {
        ss[0] = __fmaf_rn(hr[i], hr[i], ss[0]);
        ss[1] = __fmaf_rn(mr[i], mr[i], ss[1]);
    }
    ink2l::blockSums<2>(ss, sh);
    if (threadIdx.x == 0) {
        inv_h = rsqrtf(__fadd_rn(__fdiv_rn(ss[0], static_cast<float>(c)), eps));
        inv_m = rsqrtf(__fadd_rn(__fdiv_rn(ss[1], static_cast<float>(c)), eps));
    }
    __syncthreads();
    uint16_t* dst = norm_hm + row * 2 * c;
    for (int i = threadIdx.x; i < 2 * c; i += blockDim.x) {
        const bool message = i >= c;
        const float x = message ? mr[i - c] : hr[i];
        dst[i] = ink2l::tobf(__fmul_rn(x, message ? inv_m : inv_h));
    }
}

extern "C" __global__ void __launch_bounds__(256)
    tf_ink2_loop_gate_down(const uint16_t* __restrict__ norm_hm, const uint16_t* __restrict__ down,
                            uint16_t* __restrict__ zeta, int rows, int c) {
    (void)rows;
    __shared__ float sh[2 * 32];
    const size_t row = blockIdx.x;
    const int r = 2 * static_cast<int>(blockIdx.y);
    float acc[2] = {0.0f, 0.0f};
    const uint16_t* d0 = down + static_cast<size_t>(r) * (2 * c);
    const uint16_t* d1 = d0 + 2 * c;
    const uint16_t* nm = norm_hm + row * 2 * c;
    for (int i = threadIdx.x; i < 2 * c; i += blockDim.x) {
        const float n = ink2l::bf(nm[i]);
        acc[0] = __fmaf_rn(ink2l::bf(d0[i]), n, acc[0]);
        acc[1] = __fmaf_rn(ink2l::bf(d1[i]), n, acc[1]);
    }
    ink2l::blockSums<2>(acc, sh);
    if (threadIdx.x == 0) {
        zeta[row * ink2l::kRank + r] = ink2l::tobf(acc[0]);
        zeta[row * ink2l::kRank + r + 1] = ink2l::tobf(acc[1]);
    }
}

extern "C" __global__ void __launch_bounds__(256)
    tf_ink2_loop_gate_apply(const float* __restrict__ h, const float* __restrict__ m,
                            const uint16_t* __restrict__ zeta, const uint16_t* __restrict__ forget,
                            const uint16_t* __restrict__ gain, float* __restrict__ out,
                            int rows, int c, int m_stride) {
    (void)rows;
    __shared__ float z[ink2l::kRank];
    const size_t row = blockIdx.x;
    const int i = static_cast<int>(blockIdx.y) * 256 + threadIdx.x;
    for (int r = threadIdx.x; r < ink2l::kRank; r += blockDim.x) z[r] = ink2l::bf(zeta[row * ink2l::kRank + r]);
    __syncthreads();
    if (i >= c) return;
    const float* hr = h + row * c;
    const float* mr = m + row * m_stride;
    const uint16_t* f = forget + static_cast<size_t>(i) * ink2l::kRank;
    const uint16_t* g = gain + static_cast<size_t>(i) * ink2l::kRank;
    float sf = 0.0f, sg = 0.0f;
    for (int r = 0; r < ink2l::kRank; ++r) {
        sf = __fmaf_rn(ink2l::bf(f[r]), z[r], sf);
        sg = __fmaf_rn(ink2l::bf(g[r]), z[r], sg);
    }
    const float fg = tanhf(sf), gg = tanhf(sg);
    const float keep = __fsub_rn(1.0f, fg);
    const float old = __fmul_rn(keep, hr[i]);
    const float added = __fmul_rn(gg, mr[i]);
    out[row * c + i] = __fadd_rn(old, added);
}

// ---- Batched 3-role controller gate (M6a): bit-identical to 3x ink2l::gate ----
// One launch covers all three gated roles (q/k/v, mlp uses the single-role fast path): norm writes
// [role][row][2*C] BF16 pairs, down writes [role][row][64] zetas, apply writes [role][row][C] fp32
// gated rows, rmsnorm3 finishes with [role][row][C] bf16 rows (bit-identical to mix_reg n_blocks=0).
struct Gate3Args {
    const float* h[3];
    const float* m[3];
    const uint16_t* down[3];
    const uint16_t* forget[3];
    const uint16_t* gain[3];
    const uint16_t* norm[3];
    uint16_t* norm_hm;
    uint16_t* zeta;
    float* out_raw;
    uint16_t* out_bf16;
    int rows, c, m_stride;
    float eps;
};

extern "C" __global__ void __launch_bounds__(256)
    tf_ink2_loop_gate_norm3(Gate3Args a) {
    __shared__ float sh[2 * 32];
    __shared__ float inv_h, inv_m;
    const size_t row = blockIdx.x;
    const int role = static_cast<int>(blockIdx.y);
    const float* hr = a.h[role] + row * a.c;
    const float* mr = a.m[role] + row * a.m_stride;
    float ss[2] = {0.0f, 0.0f};
    for (int i = threadIdx.x; i < a.c; i += blockDim.x) {
        ss[0] = __fmaf_rn(hr[i], hr[i], ss[0]);
        ss[1] = __fmaf_rn(mr[i], mr[i], ss[1]);
    }
    ink2l::blockSums<2>(ss, sh);
    if (threadIdx.x == 0) {
        inv_h = rsqrtf(__fadd_rn(__fdiv_rn(ss[0], static_cast<float>(a.c)), a.eps));
        inv_m = rsqrtf(__fadd_rn(__fdiv_rn(ss[1], static_cast<float>(a.c)), a.eps));
    }
    __syncthreads();
    uint16_t* dst = a.norm_hm + (static_cast<size_t>(role) * a.rows + row) * 2 * a.c;
    for (int i = threadIdx.x; i < 2 * a.c; i += blockDim.x) {
        const bool message = i >= a.c;
        const float x = message ? mr[i - a.c] : hr[i];
        dst[i] = ink2l::tobf(__fmul_rn(x, message ? inv_m : inv_h));
    }
}

extern "C" __global__ void __launch_bounds__(256)
    tf_ink2_loop_gate_down3(Gate3Args a) {
    __shared__ float sh[2 * 32];
    const size_t row = blockIdx.x;
    const int r = 2 * static_cast<int>(blockIdx.y);
    const int role = static_cast<int>(blockIdx.z);
    float acc[2] = {0.0f, 0.0f};
    const uint16_t* d0 = a.down[role] + static_cast<size_t>(r) * (2 * a.c);
    const uint16_t* d1 = d0 + 2 * a.c;
    const uint16_t* nm = a.norm_hm + (static_cast<size_t>(role) * a.rows + row) * 2 * a.c;
    for (int i = threadIdx.x; i < 2 * a.c; i += blockDim.x) {
        const float n = ink2l::bf(nm[i]);
        acc[0] = __fmaf_rn(ink2l::bf(d0[i]), n, acc[0]);
        acc[1] = __fmaf_rn(ink2l::bf(d1[i]), n, acc[1]);
    }
    ink2l::blockSums<2>(acc, sh);
    if (threadIdx.x == 0) {
        a.zeta[(static_cast<size_t>(role) * a.rows + row) * ink2l::kRank + r] = ink2l::tobf(acc[0]);
        a.zeta[(static_cast<size_t>(role) * a.rows + row) * ink2l::kRank + r + 1] = ink2l::tobf(acc[1]);
    }
}

extern "C" __global__ void __launch_bounds__(256)
    tf_ink2_loop_gate_apply3(Gate3Args a) {
    __shared__ float z[ink2l::kRank];
    const size_t row = blockIdx.x;
    const int role = static_cast<int>(blockIdx.z);
    const int i = static_cast<int>(blockIdx.y) * 256 + threadIdx.x;
    const uint16_t* zrow = a.zeta + (static_cast<size_t>(role) * a.rows + row) * ink2l::kRank;
    for (int r = threadIdx.x; r < ink2l::kRank; r += blockDim.x) z[r] = ink2l::bf(zrow[r]);
    __syncthreads();
    if (i >= a.c) return;
    const float* hr = a.h[role] + row * a.c;
    const float* mr = a.m[role] + row * a.m_stride;
    const uint16_t* f = a.forget[role] + static_cast<size_t>(i) * ink2l::kRank;
    const uint16_t* g = a.gain[role] + static_cast<size_t>(i) * ink2l::kRank;
    float sf = 0.0f, sg = 0.0f;
    for (int r = 0; r < ink2l::kRank; ++r) {
        sf = __fmaf_rn(ink2l::bf(f[r]), z[r], sf);
        sg = __fmaf_rn(ink2l::bf(g[r]), z[r], sg);
    }
    const float fg = tanhf(sf), gg = tanhf(sg);
    const float keep = __fsub_rn(1.0f, fg);
    const float old = __fmul_rn(keep, hr[i]);
    const float added = __fmul_rn(gg, mr[i]);
    a.out_raw[(static_cast<size_t>(role) * a.rows + row) * a.c + i] = __fadd_rn(old, added);
}

// ---- M9.7b: the scalar gate's down and apply over row tiles (rows > 16: a batched call's) ----
// gate_down3 / gate_apply3 read each role's down / forget / gain rows once a row; these once a tile of rows. A
// (row, rank) still sums its own chain (i = t, t + blockDim, ...) through the same blockSums tree (blockSums reduces
// every slot alone), and a (row, channel) its own 64-step forget and gain chains in rank order: the same bits.
// roles = grid z. One role (z = 1) is gateFast's gate_down / gate_apply: norm_hm, zeta and out_raw in role 0's rows.
namespace ink2rt {
constexpr int kDownRows = 4;      // rows a down block
constexpr int kApplyRows = 8;     // rows an apply block
constexpr int kApplyThreads = 128;  // channels an apply block
}  // namespace ink2rt

extern "C" __global__ void __launch_bounds__(256) tf_ink2_loop_gate_down3_rows(Gate3Args a) {
    constexpr int R = ink2rt::kDownRows;
    __shared__ float sh[2 * R * 32];
    const int row0 = static_cast<int>(blockIdx.x) * R;
    const int n = min(R, a.rows - row0);
    const int r = 2 * static_cast<int>(blockIdx.y);
    const int role = static_cast<int>(blockIdx.z);
    float acc[2 * R];
#pragma unroll
    for (int j = 0; j < 2 * R; ++j) acc[j] = 0.0f;
    const uint16_t* d0 = a.down[role] + static_cast<size_t>(r) * (2 * a.c);
    const uint16_t* d1 = d0 + 2 * a.c;
    const uint16_t* nm = a.norm_hm + (static_cast<size_t>(role) * a.rows + row0) * 2 * a.c;
    for (int i = threadIdx.x; i < 2 * a.c; i += blockDim.x) {
        const float w0 = ink2l::bf(d0[i]), w1 = ink2l::bf(d1[i]);
#pragma unroll
        for (int j = 0; j < R; ++j) {
            if (j >= n) break;
            const float v = ink2l::bf(nm[static_cast<size_t>(j) * 2 * a.c + i]);
            acc[2 * j] = __fmaf_rn(w0, v, acc[2 * j]);
            acc[2 * j + 1] = __fmaf_rn(w1, v, acc[2 * j + 1]);
        }
    }
    ink2l::blockSums<2 * R>(acc, sh);
    if (threadIdx.x == 0)
        for (int j = 0; j < n; ++j) {
            uint16_t* z = a.zeta + (static_cast<size_t>(role) * a.rows + row0 + j) * ink2l::kRank + r;
            z[0] = ink2l::tobf(acc[2 * j]);
            z[1] = ink2l::tobf(acc[2 * j + 1]);
        }
}

extern "C" __global__ void __launch_bounds__(ink2rt::kApplyThreads) tf_ink2_loop_gate_apply3_rows(Gate3Args a) {
    constexpr int R = ink2rt::kApplyRows;
    __shared__ float z[R][ink2l::kRank];
    const int row0 = static_cast<int>(blockIdx.x) * R;
    const int n = min(R, a.rows - row0);
    const int role = static_cast<int>(blockIdx.z);
    const int i = static_cast<int>(blockIdx.y) * ink2rt::kApplyThreads + threadIdx.x;
    const uint16_t* zrows = a.zeta + (static_cast<size_t>(role) * a.rows + row0) * ink2l::kRank;
    for (int e = threadIdx.x; e < R * ink2l::kRank; e += blockDim.x)
        z[e / ink2l::kRank][e % ink2l::kRank] = e < n * ink2l::kRank ? ink2l::bf(zrows[e]) : 0.0f;
    __syncthreads();
    if (i >= a.c) return;
    const uint16_t* f = a.forget[role] + static_cast<size_t>(i) * ink2l::kRank;
    const uint16_t* g = a.gain[role] + static_cast<size_t>(i) * ink2l::kRank;
    float sf[R], sg[R];
#pragma unroll
    for (int j = 0; j < R; ++j) sf[j] = sg[j] = 0.0f;
    for (int r = 0; r < ink2l::kRank; ++r) {
        const float fr = ink2l::bf(f[r]), gr = ink2l::bf(g[r]);
#pragma unroll
        for (int j = 0; j < R; ++j) {
            sf[j] = __fmaf_rn(fr, z[j][r], sf[j]);
            sg[j] = __fmaf_rn(gr, z[j][r], sg[j]);
        }
    }
#pragma unroll
    for (int j = 0; j < R; ++j) {
        if (j >= n) break;
        const size_t row = static_cast<size_t>(row0 + j);
        const float fg = tanhf(sf[j]), gg = tanhf(sg[j]);
        const float keep = __fsub_rn(1.0f, fg);
        const float old = __fmul_rn(keep, a.h[role][row * a.c + i]);
        const float added = __fmul_rn(gg, a.m[role][row * a.m_stride + i]);
        a.out_raw[(static_cast<size_t>(role) * a.rows + row) * a.c + i] = __fadd_rn(old, added);
    }
}

namespace ink2l {

// Lightweight pure RMSNorm over an fp32 row: the exact c = tid + k*512 accumulation order,
// blockSums<1> tree and norm tail of tf_ink2_mix_reg's query==0 path (n_blocks=0), hence bit-exact.
// The 5-element register cache covers C <= 5 x 512, like the staged mixer's slices.
__device__ __forceinline__ void rms_row(const float* __restrict__ x, const uint16_t* __restrict__ norm,
                                        uint16_t* __restrict__ out, int C, float eps, float* sh) {
    float s = 0.0f, hc[5];
#pragma unroll
    for (int k = 0; k < 5; ++k) {
        const int c = threadIdx.x + k * 512;
        const float h = c < C ? x[c] : 0.0f;
        s = __fadd_rn(s, __fmul_rn(h, h));
        hc[k] = h;
    }
    float sv[1] = {s};
    blockSums<1>(sv, sh);
    const float r = rsqrtf(__fadd_rn(__fdiv_rn(sv[0], static_cast<float>(C)), eps));
#pragma unroll
    for (int k = 0; k < 5; ++k) {
        const int c = threadIdx.x + k * 512;
        if (c < C) out[c] = tobf(__fmul_rn(bf(norm[c]), __fmul_rn(hc[k], r)));
    }
}

}  // namespace ink2l

// Batched 3-role RMSNorm over gate3's fp32 rows (grid rows x 3, 512 threads a row).
extern "C" __global__ void __launch_bounds__(512)
    tf_ink2_loop_rmsnorm3(Gate3Args a) {
    __shared__ float sh[32];
    const size_t row = blockIdx.x;
    const int role = static_cast<int>(blockIdx.y);
    ink2l::rms_row(a.out_raw + (static_cast<size_t>(role) * a.rows + row) * a.c, a.norm[role],
                    a.out_bf16 + (static_cast<size_t>(role) * a.rows + row) * a.c, a.c, a.eps, sh);
}

// Single RMSNorm (grid rows, 512 threads a row): the MLP post-gate norm without mix overhead.
extern "C" __global__ void __launch_bounds__(512)
    tf_ink2_loop_rmsnorm1(const float* __restrict__ x, const uint16_t* __restrict__ norm,
                           uint16_t* __restrict__ out, int rows, int C, float eps) {
    (void)rows;
    __shared__ float sh[32];
    ink2l::rms_row(x + static_cast<size_t>(blockIdx.x) * C, norm, out + static_cast<size_t>(blockIdx.x) * C, C, eps, sh);
}

// ---- M9c: wide-row tensor-core gate v2 (prefill rows > 16; decode keeps gateFast) ----
// Same math as ink2l::gate with the two dots on mma.m16n8k16 (bf16 in, fp32 accumulate): matches the scalar gate
// within < 5e-4 max abs diff, as M8a's wmma path did. 64x64 tiles over 8 warps (4 down M, 2 across N: a warp
// 16 rows x 32 columns), 16-byte cp.async loads into rows of eight 16-byte pieces XOR-swizzled by row & 7, so the
// ldmatrix reads are conflict-free. norm_hm comes from tf_ink2_loop_gate_norm{,_wide} (exact). Needs C % 64 == 0.
namespace ink2tc {
constexpr int BM = 64, BN = 64, BK = 64;
constexpr int THREADS = 256;

// The half offset of piece c (halves 8c..8c+7) of row `row` in a [64][64] bf16 tile.
__device__ __forceinline__ int sw(int row, int c) { return row * BK + ((c ^ (row & 7)) << 3); }

// A warp's 16x32 slice: acc[j] += A[16 rows][64] . B[n8 tile j][64] over the tile's four k16 steps.
__device__ __forceinline__ void tile64(float (&acc)[4][4], const uint16_t* a_s, const uint16_t* b_s, int wm, int wn,
                                       int lane) {
#pragma unroll
    for (int kk = 0; kk < 4; kk += 2) {
        uint32_t a0[4], a1[4];
        qmm_frag::ldmatrix4(a0, a_s + sw(wm * 16 + (lane & 15), 2 * kk + (lane >> 4)));
        qmm_frag::ldmatrix4(a1, a_s + sw(wm * 16 + (lane & 15), 2 * kk + 2 + (lane >> 4)));
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            uint32_t b[4];
            qmm_frag::ldmatrix4(b, b_s + sw(wn * 32 + 8 * j + (lane & 7), 2 * kk + (lane >> 3)));
            qmm_frag::mma(acc[j], a0, b[0], b[1]);
            qmm_frag::mma(acc[j], a1, b[2], b[3]);
        }
    }
}

}  // namespace ink2tc

// gate_norm with each row staged once through shared memory by float4 loads, then tf_ink2_loop_gate_norm's own
// strided fma chains, blockSums tree and products (256 threads), written 8 bf16 (16 bytes) a store: the same bits.
// Dynamic shared: 8 * c bytes (the row's h, then its m). Needs c % 8 == 0 and 16-byte aligned h/m rows.
extern "C" __global__ void __launch_bounds__(256)
    tf_ink2_loop_gate_norm_wide(const float* __restrict__ h, const float* __restrict__ m,
                                uint16_t* __restrict__ norm_hm, int rows, int c, int m_stride, float eps) {
    (void)rows;
    extern __shared__ __align__(16) float xs[];
    __shared__ float sh[2 * 32];
    __shared__ float inv_h, inv_m;
    const size_t row = blockIdx.x;
    const float4* h4 = reinterpret_cast<const float4*>(h + row * c);
    const float4* m4 = reinterpret_cast<const float4*>(m + row * m_stride);
    float4* x4 = reinterpret_cast<float4*>(xs);
    for (int i = threadIdx.x; i < c / 4; i += blockDim.x) {
        x4[i] = h4[i];
        x4[c / 4 + i] = m4[i];
    }
    __syncthreads();
    float ss[2] = {0.0f, 0.0f};
    for (int i = threadIdx.x; i < c; i += blockDim.x) {
        ss[0] = __fmaf_rn(xs[i], xs[i], ss[0]);
        ss[1] = __fmaf_rn(xs[c + i], xs[c + i], ss[1]);
    }
    ink2l::blockSums<2>(ss, sh);
    if (threadIdx.x == 0) {
        inv_h = rsqrtf(__fadd_rn(__fdiv_rn(ss[0], static_cast<float>(c)), eps));
        inv_m = rsqrtf(__fadd_rn(__fdiv_rn(ss[1], static_cast<float>(c)), eps));
    }
    __syncthreads();
    uint4* dst = reinterpret_cast<uint4*>(norm_hm + row * 2 * c);
    for (int v = threadIdx.x; v < 2 * c / 8; v += blockDim.x) {
        const int i = 8 * v;
        const float inv = i >= c ? inv_m : inv_h;
        uint32_t w[4];
#pragma unroll
        for (int k = 0; k < 4; ++k)
            w[k] = static_cast<uint32_t>(ink2l::tobf(__fmul_rn(xs[i + 2 * k], inv))) |
                   (static_cast<uint32_t>(ink2l::tobf(__fmul_rn(xs[i + 2 * k + 1], inv))) << 16);
        dst[v] = make_uint4(w[0], w[1], w[2], w[3]);
    }
}

namespace ink2tc {

// zeta[rows, 64] = [lo | hi][rows, 2C] . down[64, 2C]^T: a row's first C inputs from `lo` (rows lo_ld apart), the rest
// from `hi` (hi_ld apart); BK = 64 K steps never straddle the two (C % 64 == 0). Two cp.async stages, registers across all.
__device__ __forceinline__ void downTile(const uint16_t* __restrict__ lo, int lo_ld, const uint16_t* __restrict__ hi, int hi_ld,
                                         const uint16_t* __restrict__ down, uint16_t* __restrict__ zeta, int rows, int C) {
    __shared__ __align__(128) uint16_t sa[2][BM * BK];
    __shared__ __align__(128) uint16_t sb[2][BN * BK];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp & 3, wn = warp >> 2;
    const int m0 = static_cast<int>(blockIdx.x) * BM;
    const int K = 2 * C, steps = K / BK;
    auto load = [&](int st, int step) {
        const size_t k0 = static_cast<size_t>(step) * BK;
        const bool upper = k0 >= static_cast<size_t>(C);
        const uint16_t* src = upper ? hi : lo;
        const size_t ld = upper ? hi_ld : lo_ld, kk = upper ? k0 - C : k0;
        for (int e = tid; e < BM * (BK / 8); e += THREADS) {
            const int r = e >> 3, c = e & 7;
            const bool in = m0 + r < rows;
            tf_common::cp16z(sa[st] + sw(r, c), src + static_cast<size_t>(in ? m0 + r : 0) * ld + kk + 8 * c, in);
            tf_common::cp16(sb[st] + sw(r, c), down + static_cast<size_t>(r) * K + k0 + 8 * c);
        }
        tf_common::cp_commit();
    };
    float acc[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f;
    load(0, 0);
    for (int step = 0; step < steps; ++step) {
        if (step + 1 < steps) load((step + 1) & 1, step + 1);
        else tf_common::cp_commit();
        tf_common::cp_wait<1>();
        __syncthreads();  // stage step & 1 is in for every thread
        tile64(acc, sa[step & 1], sb[step & 1], wm, wn, lane);
        __syncthreads();  // every warp is done with it before the next load takes it
    }
    const int g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int col = wn * 32 + 8 * j + 2 * t;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int r = m0 + wm * 16 + g + 8 * i;
            if (r >= rows) continue;
            const uint32_t v = static_cast<uint32_t>(ink2l::tobf(acc[j][2 * i])) |
                               (static_cast<uint32_t>(ink2l::tobf(acc[j][2 * i + 1])) << 16);
            *reinterpret_cast<uint32_t*>(zeta + static_cast<size_t>(r) * ink2l::kRank + col) = v;
        }
    }
}

// out[rows, C] = (1 - tanh(zeta . forget^T)) * h + tanh(zeta . gain^T) * m, a 64x64 tile a block: forget and
// gain ([C, 64] row-major, each tile's 64 rows whole) accumulate in registers side by side; float2 h/m/out.
// h and out may alias (each element is read, then written, by the one thread).
__device__ __forceinline__ void applyTile(const uint16_t* __restrict__ zeta, const uint16_t* __restrict__ w_f,
                                          const uint16_t* __restrict__ w_g, const float* h, const float* __restrict__ m,
                                          float* out, int rows, int C, int m_stride) {
    __shared__ __align__(128) uint16_t sz[BM * 64];
    __shared__ __align__(128) uint16_t sf[BN * 64];
    __shared__ __align__(128) uint16_t sg[BN * 64];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp & 3, wn = warp >> 2;
    const int n0 = static_cast<int>(blockIdx.y) * BN;
    const int m0 = static_cast<int>(blockIdx.x) * BM;
    for (int e = tid; e < BM * 8; e += THREADS) {
        const int r = e >> 3, c = e & 7;
        const bool in = m0 + r < rows;
        tf_common::cp16z(sz + sw(r, c), zeta + static_cast<size_t>(in ? m0 + r : 0) * ink2l::kRank + 8 * c, in);
        tf_common::cp16(sf + sw(r, c), w_f + static_cast<size_t>(n0 + r) * ink2l::kRank + 8 * c);
        tf_common::cp16(sg + sw(r, c), w_g + static_cast<size_t>(n0 + r) * ink2l::kRank + 8 * c);
    }
    tf_common::cp_commit();
    // this thread's h and m elements, in flight while the tiles land and the mma runs
    const int g = lane >> 2, t = lane & 3;
    float2 hv[2][4], mv[2][4];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int r = m0 + wm * 16 + g + 8 * i;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int col = n0 + wn * 32 + 8 * j + 2 * t;
            hv[i][j] = r < rows ? *reinterpret_cast<const float2*>(h + static_cast<size_t>(r) * C + col) : make_float2(0.0f, 0.0f);
            // read-only path: a kernel-level __restrict__ gave tf_ink2_loop_gate_apply_tc these loads, the struct-fed
            // tf_ink2_loop_gate_apply_tc3 needs them spelled out (m is never written here)
            mv[i][j] = r < rows ? __ldg(reinterpret_cast<const float2*>(m + static_cast<size_t>(r) * m_stride + col))
                                : make_float2(0.0f, 0.0f);
        }
    }
    tf_common::cp_wait<0>();
    __syncthreads();
    float af[4][4], ag[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
        af[j][0] = af[j][1] = af[j][2] = af[j][3] = ag[j][0] = ag[j][1] = ag[j][2] = ag[j][3] = 0.0f;
    tile64(af, sz, sf, wm, wn, lane);
    tile64(ag, sz, sg, wm, wn, lane);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int r = m0 + wm * 16 + g + 8 * i;
        if (r >= rows) continue;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int col = n0 + wn * 32 + 8 * j + 2 * t;
            const float f0 = tanhf(af[j][2 * i]), f1 = tanhf(af[j][2 * i + 1]);
            const float g0 = tanhf(ag[j][2 * i]), g1 = tanhf(ag[j][2 * i + 1]);
            // the reference's four FP32 rounding steps, explicit so contraction cannot erase any of them
            const float2 o = make_float2(__fadd_rn(__fmul_rn(__fsub_rn(1.0f, f0), hv[i][j].x), __fmul_rn(g0, mv[i][j].x)),
                                         __fadd_rn(__fmul_rn(__fsub_rn(1.0f, f1), hv[i][j].y), __fmul_rn(g1, mv[i][j].y)));
            *reinterpret_cast<float2*>(out + static_cast<size_t>(r) * C + col) = o;
        }
    }
}

}  // namespace ink2tc

// zeta[rows, 64] = norm_hm[rows, 2C] . down[64, 2C]^T: BK = 64 K steps, two cp.async stages, registers across all.
extern "C" __global__ void __launch_bounds__(256)
    tf_ink2_loop_gate_down_tc(const uint16_t* __restrict__ norm_hm, const uint16_t* __restrict__ down,
                              uint16_t* __restrict__ zeta, int rows, int C) {
    ink2tc::downTile(norm_hm, 2 * C, norm_hm + C, 2 * C, down, zeta, rows, C);
}

extern "C" __global__ void __launch_bounds__(256)
    tf_ink2_loop_gate_apply_tc(const uint16_t* __restrict__ zeta, const uint16_t* __restrict__ w_f,
                               const uint16_t* __restrict__ w_g, const float* h, const float* __restrict__ m,
                               float* out, int rows, int C, int m_stride) {
    ink2tc::applyTile(zeta, w_f, w_g, h, m, out, rows, C, m_stride);
}

// ---- M9.6 B1 + B2: the wide tensor-core gate with no norm pass, its roles in one launch a stage ----
// The state half `hn` comes from tf_ink2_loop_ar_reg (its output rows, normalized), the message half `mn` from
// tf_ink2_loop_messages_norm: the bits tf_ink2_loop_gate_norm_wide would have written. Role r is grid z = r; `out`
// must not alias another role's `h` (role 0's output takes its own buffer when the roles share one AR row).
struct GateTcArgs {
    const uint16_t* hn[3];  // bf16 [rows, C], hn_ld 0: C
    const uint16_t* mn[3];  // bf16 rows mn_ld apart
    const uint16_t* down[3];
    const uint16_t* forget[3];
    const uint16_t* gain[3];
    const float* h[3];      // fp32 [rows, C]
    const float* m[3];      // fp32 rows m_stride apart
    float* out[3];          // fp32 [rows, C]
    const uint16_t* norm[3];
    uint16_t* normed[3];    // bf16 [rows, C]: rms_row(out) (tf_ink2_loop_rmsnorm1's bits)
    uint16_t* zeta;         // bf16 [3][rows][64]
    int rows, C, mn_ld, m_stride;
    float eps;
};

// A role's field by constant-index parameter loads (a select, not an indexed copy of the 272-byte parameter block)
template <typename T>
__device__ __forceinline__ T role3(const T (&v)[3], int r) {
    return r == 0 ? v[0] : r == 1 ? v[1] : v[2];
}

extern "C" __global__ void __launch_bounds__(256) tf_ink2_loop_gate_down_tc3(const GateTcArgs a) {
    const int r = static_cast<int>(blockIdx.z);
    ink2tc::downTile(role3(a.hn, r), a.C, role3(a.mn, r), a.mn_ld, role3(a.down, r),
                     a.zeta + static_cast<size_t>(r) * a.rows * ink2l::kRank, a.rows, a.C);
}

extern "C" __global__ void __launch_bounds__(256) tf_ink2_loop_gate_apply_tc3(const GateTcArgs a) {
    const int r = static_cast<int>(blockIdx.z);
    ink2tc::applyTile(a.zeta + static_cast<size_t>(r) * a.rows * ink2l::kRank, role3(a.forget, r), role3(a.gain, r),
                      role3(a.h, r), role3(a.m, r), role3(a.out, r), a.rows, a.C, a.m_stride);
}

// Each role's gated rows normalized (grid rows x roles, 512 threads a row): tf_ink2_loop_rmsnorm1's bits.
extern "C" __global__ void __launch_bounds__(512) tf_ink2_loop_rmsnorm_tc3(const GateTcArgs a) {
    __shared__ float sh[32];
    const size_t row = blockIdx.x;
    const int r = static_cast<int>(blockIdx.y);
    ink2l::rms_row(role3(a.out, r) + row * a.C, role3(a.norm, r), role3(a.normed, r) + row * a.C, a.C, a.eps, sh);
}

// Mini QKV launch uses three independently normalized inputs; other kernels and cache/partial layouts are unchanged.
extern "C" __global__ void __launch_bounds__(256)
    tf_ink2_loop_project_qkv(const uint16_t* __restrict__ uq, const uint16_t* __restrict__ uk,
                             const uint16_t* __restrict__ uv, const uint16_t* __restrict__ wq,
                             const uint16_t* __restrict__ wk, const uint16_t* __restrict__ wv,
                             uint16_t* __restrict__ raw_q, uint16_t* __restrict__ raw_k,
                             uint16_t* __restrict__ raw_v, int rows, int c, int q_width, int kv_width) {
    constexpr int kRows = 16, kOutputs = 16, kSlice = 32;
    __shared__ uint16_t sx[kRows * kSlice], sw[kOutputs * kSlice];
    const int role = blockIdx.z;  // q=0, k=1, v=2
    const int width = role == 0 ? q_width : kv_width;
    const int row0 = static_cast<int>(blockIdx.x) * kRows;
    const int j0 = static_cast<int>(blockIdx.y) * kOutputs;
    if (j0 >= width) return;
    const uint16_t* x = role == 0 ? uq : role == 1 ? uk : uv;
    const uint16_t* w = role == 0 ? wq : role == 1 ? wk : wv;
    const int tr = threadIdx.x / kOutputs, tj = threadIdx.x % kOutputs;
    const int row = row0 + tr, j = j0 + tj;
    float sum = 0.0f;
    // A block reuses each normalized tile for 16 outputs and each weight tile for 16 prompt rows.
    for (int k0 = 0; k0 < c; k0 += kSlice) {
        for (int e = threadIdx.x; e < kRows * kSlice; e += blockDim.x) {
            const int r = e / kSlice, k = k0 + e % kSlice;
            sx[e] = row0 + r < rows && k < c ? x[static_cast<size_t>(row0 + r) * c + k] : 0;
            sw[e] = j0 + r < width && k < c ? w[static_cast<size_t>(j0 + r) * c + k] : 0;
        }
        __syncthreads();
        for (int k = 0; k < kSlice && k0 + k < c; ++k)
            sum = __fmaf_rn(ink2l::bf(sx[tr * kSlice + k]), ink2l::bf(sw[tj * kSlice + k]), sum);
        __syncthreads();
    }
    uint16_t* out = role == 0 ? raw_q : role == 1 ? raw_k : raw_v;
    if (row < rows && j < width) out[static_cast<size_t>(row) * width + j] = ink2l::tobf(sum);
}

// Global dynamic taps: dot in FP32, store as BF16, and widen to FP32 before the convolution.
extern "C" __global__ void __launch_bounds__(256)
    tf_ink2_loop_round_bf16(const float* __restrict__ in, float* __restrict__ out, long long n) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = ink2l::bf(ink2l::tobf(in[i]));
}

// Copy [rows, C] bf16 into taps[row][channel][C] (4 tap channels per row, matching Metal runtime.metal).
extern "C" __global__ void __launch_bounds__(256)
    tf_ink2_capture_tap(const uint16_t* __restrict__ src, uint16_t* __restrict__ dst,
                        int rows, int c, int channel) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(rows) * c;
    if (i >= total) return;
    const size_t row = i / c, d = i - row * c;
    dst[(row * 4 + channel) * c + d] = src[i];
}
