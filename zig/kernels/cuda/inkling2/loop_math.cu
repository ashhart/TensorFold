// Inkling 2 Mini: raw block residuals, two-bank messages, rank-64 gates and three-input QKV projection.
#include <cstdint>
#include <cuda_bf16.h>

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
};

// The AR half of Small's residual mix, without its fused RMSNorm. Needed before Mini's controller, not just after it.
__device__ void ar(const ArArgs a) {
    __shared__ float sh[32 * (1 + kMaxQueries)];
    __shared__ float wts[kMaxQueries][kMaxBlocks + 1];
    __shared__ float rr[kMaxBlocks + 1];
    const size_t row = blockIdx.x;
    const int n = a.n_blocks + 1;
    auto entry = [&](int j, int c) -> float {
        if (j < a.n_blocks) return bf(a.blocks[j][row * a.C + c]);
        return a.partial_bf16 ? bf(static_cast<const uint16_t*>(a.partial)[row * a.C + c])
                              : static_cast<const float*>(a.partial)[row * a.C + c];
    };
    for (int j = 0; j < n; ++j) {
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
                                            int c, float eps) {
    __shared__ float sh[2 * 32];
    __shared__ float inv_h, inv_m;
    __shared__ uint16_t zeta[kRank];
    const size_t row = blockIdx.x;
    const float* hr = h + row * c;
    const float* mr = m + row * c;
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
    ink2l::gate(h, m, down, forget, gain, out, c, eps);
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
