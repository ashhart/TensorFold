// Inkling 2 glue: embedding, block attention-residual mix, row dots, router top-k, short convs.
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include "exl3_decode.cuh"

namespace ink2o {

constexpr int kMaxBlocks = 8;
constexpr int kMaxQueries = 3;
constexpr int kConvTaps = 4;
constexpr int kMixCache = 10;  // tf_ink2_mix holds a row of up to kMixCache x blockDim columns in registers

__device__ __forceinline__ float bf(uint16_t u) { return __uint_as_float(static_cast<uint32_t>(u) << 16); }
__device__ __forceinline__ uint16_t tobf(float f) { return __bfloat16_as_ushort(__float2bfloat16_rn(f)); }
__device__ __forceinline__ float bfr(float f) { return bf(tobf(f)); }
__device__ __forceinline__ float warpSum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// Sums of N values over the block in a fixed order: warp shuffles, then warp 0 over the warps' partials.
template <int N>
__device__ __forceinline__ void blockSums(float (&v)[N], float* sh) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    for (int i = 0; i < N; ++i) v[i] = warpSum(v[i]);
    __syncthreads();
    if (lane == 0)
        for (int i = 0; i < N; ++i) sh[i * 32 + w] = v[i];
    __syncthreads();
    for (int i = 0; i < N; ++i) {
        float t = lane < nw ? sh[i * 32 + lane] : 0.0f;
        v[i] = __shfl_sync(0xffffffffu, warpSum(t), 0);
    }
}

}  // namespace ink2o

using namespace ink2o;

// Per row: the embedding row times rsqrt(mean square + eps) times the norm weight, one bf16 round (embed_norm).
extern "C" __global__ void __launch_bounds__(256) tf_ink2_embed(const int* __restrict__ ids, const uint16_t* __restrict__ table,
                                                               const uint16_t* __restrict__ w, uint16_t* __restrict__ out,
                                                               int C, float eps) {
    __shared__ float sh[32];
    const size_t row = blockIdx.x;
    const uint16_t* e = table + static_cast<size_t>(ids[row]) * C;
    float s[1] = {0.0f};
    for (int c = threadIdx.x; c < C; c += blockDim.x) s[0] += bf(e[c]) * bf(e[c]);
    blockSums<1>(s, sh);
    const float r = rsqrtf(s[0] / C + eps);
    for (int c = threadIdx.x; c < C; c += blockDim.x) out[row * C + c] = tobf(bf(w[c]) * (bf(e[c]) * r));
}

// The block attention-residual mix (Inkling2BlockAttentionResidual) and the norm after it, per row and query.
struct MixArgs {
    const uint16_t* blocks[kMaxBlocks];  // bf16 [rows, C] each: the completed blocks' inputs
    const void* partial;                 // [rows, C]: bf16 when partial_bf16, else fp32
    const float* query;                  // fp32 [nq, C]; null: no mix, the partial alone (layer 0's attention)
    const uint16_t* norm[kMaxQueries];   // bf16 [C] per query: the RMSNorm after the mix
    uint16_t* out;                       // bf16 [nq, rows, C]
    int n_blocks, partial_bf16, nq, rows, C;
    float eps;
};

constexpr int kMixStage = 4, kMixCols = kMixStage * 512;  // tf_ink2_mix: slices of 512 columns staged at a time

__device__ __forceinline__ void mixStage(const MixArgs& a, size_t row, int c0, uint16_t (*sb)[kMixCols], unsigned char* sp) {
    const int cols = min(kMixCols, a.C - c0);  // a multiple of 8: C % 128 == 0 (ops.zig)
    const int chunks = cols / 8;               // 16 bytes of bf16 a chunk
    for (int j = 0; j < a.n_blocks; ++j)
        for (int i = threadIdx.x; i < chunks; i += blockDim.x)
            __pipeline_memcpy_async(&sb[j][i * 8], a.blocks[j] + row * a.C + c0 + i * 8, 16);
    const int pe = a.partial_bf16 ? 2 : 4;
    const unsigned char* p = static_cast<const unsigned char*>(a.partial) + (row * a.C + c0) * pe;
    for (int i = threadIdx.x; i < cols * pe / 16; i += blockDim.x) __pipeline_memcpy_async(sp + i * 16, p + i * 16, 16);
    __pipeline_commit();  // one memory round trip a slice instead of one a block
    __pipeline_wait_prior(0);
    __syncthreads();
}

// Narrow calls (decode rows): staged slices; the same sums in the same order as tf_ink2_mix_rows.
extern "C" __global__ void __launch_bounds__(512) tf_ink2_mix(const MixArgs a) {
    constexpr int kV = (kMaxBlocks + 1) * (1 + kMaxQueries);
    __shared__ float sh[32 * kV];
    __shared__ float wts[kMaxQueries][kMaxBlocks + 1];
    __shared__ float rr[kMaxBlocks + 1];
    __shared__ __align__(16) uint16_t sb[kMaxBlocks][kMixCols];
    __shared__ __align__(16) unsigned char sp[kMixCols * 4];
    const size_t row = blockIdx.x;
    const int C = a.C, n = a.n_blocks + 1;
    // block j's column c, staged: thread t reads only the columns t + 512 k it adds, in the order the sums always took
    auto entry = [&](int j, int i) -> float {
        if (j < a.n_blocks) return bf(sb[j][i]);
        return a.partial_bf16 ? bf(reinterpret_cast<const uint16_t*>(sp)[i]) : reinterpret_cast<const float*>(sp)[i];
    };
    if (a.query) {
        // Normalize each source before its query dot; both passes retain the declared strided/shuffle reduction.
        float ss[kMaxBlocks + 1] = {};
        for (int c0 = 0; c0 < C; c0 += kMixCols) {
            mixStage(a, row, c0, sb, sp);
            for (int i = threadIdx.x; i < kMixCols && c0 + i < C; i += blockDim.x) {
#pragma unroll
                for (int j = 0; j < kMaxBlocks + 1; ++j)
                    if (j < n) { const float x = entry(j, i); ss[j] += x * x; }
            }
            __syncthreads();
        }
        blockSums<kMaxBlocks + 1>(ss, sh);
        if (threadIdx.x == 0) for (int j = 0; j < n; ++j) rr[j] = rsqrtf(ss[j] / C + 1e-6f); __syncthreads();
        float v[kV] = {};
        for (int c0 = 0; c0 < C; c0 += kMixCols) {
            mixStage(a, row, c0, sb, sp);
            for (int i = threadIdx.x; i < kMixCols && c0 + i < C; i += blockDim.x) {
                const int c = c0 + i;
                float qc[kMaxQueries];
#pragma unroll
                for (int q = 0; q < kMaxQueries; ++q) qc[q] = q < a.nq ? a.query[static_cast<size_t>(q) * C + c] : 0.0f;
#pragma unroll
                for (int j = 0; j < kMaxBlocks + 1; ++j) if (j < n) {
                    const float normalized = entry(j, i) * rr[j];
#pragma unroll
                    for (int q = 0; q < kMaxQueries; ++q)
                        if (q < a.nq) v[j * (1 + kMaxQueries) + 1 + q] += qc[q] * normalized;
                }
            }
            __syncthreads();
        }
        blockSums<kV>(v, sh);
        if (threadIdx.x == 0) for (int j = 0; j < n; ++j) for (int q = 0; q < a.nq; ++q) wts[q][j] =
            v[j * (1 + kMaxQueries) + 1 + q] * rsqrtf(static_cast<float>(C));
        __syncthreads();
        if (threadIdx.x < a.nq) {
            float* l = wts[threadIdx.x];
            float m = -INFINITY, t = 0.0f;
            for (int j = 0; j < n; ++j) m = fmaxf(m, l[j]);
            for (int j = 0; j < n; ++j) t += expf(l[j] - m);
            for (int j = 0; j < n; ++j) l[j] = expf(l[j] - m) / t;
        }
        __syncthreads();
    }
    // every query's mixed row kept in registers (C <= kMixCache x blockDim, ops.zig): read once, normed, written
    float s[kMaxQueries] = {0.0f, 0.0f, 0.0f};
    float hc[kMaxQueries][kMixCache];
#pragma unroll
    for (int k = 0; k < kMixCache; ++k) {
        const int c = threadIdx.x + k * 512;
        if (k % kMixStage == 0) {
            if (k > 0) __syncthreads();
            if (k * 512 < C) mixStage(a, row, k * 512, sb, sp);
        }
        const int i = threadIdx.x + (k % kMixStage) * 512;
        float h[kMaxQueries] = {0.0f, 0.0f, 0.0f};
        if (c < C) {
            if (a.query) {
#pragma unroll
                for (int j = 0; j < kMaxBlocks + 1; ++j)
                    if (j < n) {
                        const float x = entry(j, i);
#pragma unroll
                        for (int q = 0; q < kMaxQueries; ++q)
                            if (q < a.nq) h[q] += wts[q][j] * (x * rr[j]);
                    }
            } else {
                h[0] = entry(a.n_blocks, i);
            }
        }
#pragma unroll
        for (int q = 0; q < kMaxQueries; ++q) {
            s[q] += h[q] * h[q];  // rows past C add 0.0f * 0.0f: the sums keep their bits
            hc[q][k] = h[q];
        }
    }
    blockSums<kMaxQueries>(s, sh);
#pragma unroll
    for (int q = 0; q < kMaxQueries; ++q) {
        if (q >= a.nq) break;
        const float r = rsqrtf(s[q] / C + a.eps);
        uint16_t* o = a.out + (static_cast<size_t>(q) * a.rows + row) * C;
#pragma unroll
        for (int k = 0; k < kMixCache; ++k) {
            const int c = threadIdx.x + k * 512;
            if (c < C) o[c] = tobf(bf(a.norm[q][c]) * (hc[q][k] * r));
        }
    }
}

// Wide calls (prompt chunks, wide verifies): the block-by-block mix, which hides its loads behind other rows.
extern "C" __global__ void __launch_bounds__(512) tf_ink2_mix_rows(const MixArgs a) {
    __shared__ float sh[32 * (1 + kMaxQueries)];
    __shared__ float wts[kMaxQueries][kMaxBlocks + 1];
    __shared__ float rr[kMaxBlocks + 1];
    const size_t row = blockIdx.x;
    const int C = a.C, n = a.n_blocks + 1;
    auto entry = [&](int j, int c) -> float {
        if (j < a.n_blocks) return bf(a.blocks[j][row * C + c]);
        return a.partial_bf16 ? bf(static_cast<const uint16_t*>(a.partial)[row * C + c])
                              : static_cast<const float*>(a.partial)[row * C + c];
    };
    if (a.query) {
        for (int j = 0; j < n; ++j) {
            float ss[1] = {0.0f};
            for (int c = threadIdx.x; c < C; c += blockDim.x) { const float x = entry(j, c); ss[0] += x * x; }
            blockSums<1>(ss, sh); if (threadIdx.x == 0) rr[j] = rsqrtf(ss[0] / C + 1e-6f);
        }
        __syncthreads();
        for (int j = 0; j < n; ++j) {
            float dots[kMaxQueries] = {0.0f, 0.0f, 0.0f};
            for (int c = threadIdx.x; c < C; c += blockDim.x) {
                const float normalized = entry(j, c) * rr[j]; for (int q = 0; q < a.nq; ++q)
                    dots[q] += a.query[static_cast<size_t>(q) * C + c] * normalized;
            }
            blockSums<kMaxQueries>(dots, sh);
            if (threadIdx.x == 0) for (int q = 0; q < a.nq; ++q) wts[q][j] = dots[q] * rsqrtf(static_cast<float>(C));
        }
        __syncthreads();
        if (threadIdx.x < a.nq) {
            float* l = wts[threadIdx.x];
            float m = -INFINITY, t = 0.0f;
            for (int j = 0; j < n; ++j) m = fmaxf(m, l[j]);
            for (int j = 0; j < n; ++j) t += expf(l[j] - m);
            for (int j = 0; j < n; ++j) l[j] = expf(l[j] - m) / t;
        }
        __syncthreads();
    }
    for (int q = 0; q < a.nq; ++q) {
        float s[1] = {0.0f};
        for (int c = threadIdx.x; c < C; c += blockDim.x) {
            float h = 0.0f;
            if (a.query)
                for (int j = 0; j < n; ++j) h += wts[q][j] * (entry(j, c) * rr[j]);
            else
                h = entry(a.n_blocks, c);
            s[0] += h * h;
        }
        blockSums<1>(s, sh);
        const float r = rsqrtf(s[0] / C + a.eps);
        uint16_t* o = a.out + (static_cast<size_t>(q) * a.rows + row) * C;
        for (int c = threadIdx.x; c < C; c += blockDim.x) {
            float h = 0.0f;
            if (a.query)
                for (int j = 0; j < n; ++j) h += wts[q][j] * (entry(j, c) * rr[j]);
            else
                h = entry(a.n_blocks, c);
            o[c] = tobf(bf(a.norm[q][c]) * (h * r));
        }
    }
}

// out[row, j] = x[row, :] . W[j, :] in fp32 (bf16 inputs), a warp an output; act 1 takes the sigmoid; bf16 or fp32 out.
extern "C" __global__ void __launch_bounds__(256) tf_ink2_rowdot(const uint16_t* __restrict__ x, const uint16_t* __restrict__ W,
                                                                void* __restrict__ out, int J, int C, int out_bf16, int act) {
    const int lane = threadIdx.x & 31, j = blockIdx.y * 8 + (threadIdx.x >> 5);
    if (j >= J) return;
    const size_t row = blockIdx.x;
    const uint16_t* xr = x + row * C;
    const uint16_t* wr = W + static_cast<size_t>(j) * C;
    float s = 0.0f;
    for (int c = lane; c < C; c += 32) s += bf(xr[c]) * bf(wr[c]);
    s = warpSum(s);
    if (lane != 0) return;
    if (act == 1) s = 1.0f / (1.0f + expf(-s));
    if (out_bf16)
        static_cast<uint16_t*>(out)[row * J + j] = tobf(s);
    else
        static_cast<float*>(out)[row * J + j] = s;
}

// Router picks per row: top-K of score + bias (ties to the lower id), weights score / sum(scores) * scale (fp32).
extern "C" __global__ void __launch_bounds__(512) tf_ink2_topk(const float* __restrict__ scores, const float* __restrict__ bias,
                                                              int* __restrict__ pick, float* __restrict__ wts, int E, int K,
                                                              float scale) {
    __shared__ float bv[16];
    __shared__ int bi[16];
    __shared__ float chosen[32];
    __shared__ int chosen_id[32];
    const size_t row = blockIdx.x;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    float sel[8];
    const int per = (E + blockDim.x - 1) / blockDim.x;
    for (int i = 0; i < per && i < 8; ++i) {
        const int e = threadIdx.x + i * blockDim.x;
        sel[i] = e < E ? scores[row * E + e] + bias[e] : -INFINITY;
    }
    for (int k = 0; k < K; ++k) {
        float v = -INFINITY;
        int id = 0x7fffffff;
        for (int i = 0; i < per && i < 8; ++i) {
            const int e = threadIdx.x + i * blockDim.x;
            if (e < E && (sel[i] > v || (sel[i] == v && e < id))) { v = sel[i]; id = e; }
        }
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, v, o);
            const int oi = __shfl_xor_sync(0xffffffffu, id, o);
            if (ov > v || (ov == v && oi < id)) { v = ov; id = oi; }
        }
        if (lane == 0) { bv[w] = v; bi[w] = id; }
        __syncthreads();
        if (threadIdx.x == 0) {
            float m = bv[0];
            int mi = bi[0];
            for (int i = 1; i < nw; ++i)
                if (bv[i] > m || (bv[i] == m && bi[i] < mi)) { m = bv[i]; mi = bi[i]; }
            chosen_id[k] = mi;
            chosen[k] = scores[row * E + mi];
        }
        __syncthreads();
        const int got = chosen_id[k];
        if (got % blockDim.x == static_cast<int>(threadIdx.x)) sel[got / blockDim.x] = -INFINITY;
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float t = 0.0f;
        for (int k = 0; k < K; ++k) t += chosen[k];
        for (int k = 0; k < K; ++k) {
            pick[row * K + k] = chosen_id[k];
            wts[row * K + k] = chosen[k] / (t + 1e-20f) * scale;
        }
    }
}

// Short causal conv over a stream's rows (4 taps, last 3 inputs in `state`): static [C][4] or dynamic [4][C] + dyn.
struct ConvArgs {
    const void* x;          // [rows, C]: fp32 (`parts` rank partials, part_stride apart, added in rank order) or bf16
    float* state;           // fp32 [3, C], oldest first
    const uint16_t* w;      // bf16 taps
    const float* dyn;       // fp32 [rows, heads * 4] or null
    const uint16_t* base;   // bf16 [rows, C] (mode 1, null: zero)
    float* resid;           // fp32 [rows, C] (modes 1 and 2)
    uint16_t* out;          // bf16 [rows, C] (modes 0 and 2)
    const int4* segs;       // per stream (blockIdx.y): row0, rows, state slot; null: one stream, all rows
    float* hist;            // fp32 [rows, C]: each row's input as the state keeps it (null: none)
    long long part_stride;
    long long state_stride;  // elements between the streams' states
    int parts, x_bf16, round_in, heads, C, rows, mode;  // mode 0: out = y; 1: resid = base + bf16(y); 2: resid + bf16(y)
};

// Rows in blocks of 32 (z), each from its three prior inputs recomputed (same bits); block 0 owns the state.
extern "C" __global__ void __launch_bounds__(128) tf_ink2_conv(const ConvArgs a) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int C = a.C, r0 = blockIdx.z * 32, hd = a.dyn ? c / (C / a.heads) : 0;
    const int4 sg = a.segs ? a.segs[blockIdx.y] : make_int4(0, a.rows, 0, 0);
    if (c >= C || r0 >= sg.y) return;
    float* state = a.state + sg.z * a.state_stride;
    const auto input = [&](int t) {
        const size_t at = static_cast<size_t>(t) * C + c;
        float x = a.x_bf16 ? bf(static_cast<const uint16_t*>(a.x)[at]) : static_cast<const float*>(a.x)[at];
        for (int r = 1; r < a.parts; ++r) x += static_cast<const float*>(a.x)[at + r * a.part_stride];
        return a.round_in ? bfr(x) : x;
    };
    const int t0 = sg.x + r0, t1 = sg.x + min(sg.y, r0 + 32), end = sg.x + sg.y;
    float h0 = r0 == 0 ? state[c] : input(t0 - 3), h1 = r0 == 0 ? state[C + c] : input(t0 - 2);
    float h2 = r0 == 0 ? state[2 * C + c] : input(t0 - 1);
    for (int t = t0; t < t1; ++t) {
        const size_t at = static_cast<size_t>(t) * C + c;
        const float x = input(t);
        float y;
        if (a.hist) a.hist[at] = x;
        if (a.dyn) {
            const float* d = a.dyn + (static_cast<size_t>(t) * a.heads + hd) * kConvTaps;
            y = (bf(a.w[c]) + d[0]) * x + (bf(a.w[C + c]) + d[1]) * h2;
            y += (bf(a.w[2 * C + c]) + d[2]) * h1;
            y += (bf(a.w[3 * C + c]) + d[3]) * h0;
            y = (y + x) / 3.0f;
        } else {
            const uint16_t* w = a.w + static_cast<size_t>(c) * kConvTaps;
            y = bf(w[0]) * h0 + bf(w[1]) * h1;
            y += bf(w[2]) * h2;
            y += bf(w[3]) * x;
            y = (y + x) / 2.2360679774997896f;
        }
        if (a.mode == 0) a.out[at] = tobf(y);
        else if (a.mode == 1) a.resid[at] = (a.base ? bf(a.base[at]) : 0.0f) + bfr(y);
        else a.out[at] = tobf(a.resid[at] + bfr(y));
        h0 = h1, h1 = h2, h2 = x;
    }
    if (r0 != 0) return;
    const bool last = t1 == end;  // else the stream's last three inputs, recomputed
    state[c] = last ? h0 : input(end - 3);
    state[C + c] = last ? h1 : input(end - 2);
    state[2 * C + c] = last ? h2 : input(end - 1);
}

// A conv site of every slot: slot 0's state, its C, and byte offsets of its rows in a hist and in a slot's backup.
struct RewindSite {
    float* state;
    long long hist, backup;
    int C, pad;
};

// A window keeps its first n rows: each site's state becomes the last 3 of (backup's 3 inputs, rows' inputs from hist).
extern "C" __global__ void __launch_bounds__(128) tf_ink2_conv_rewind(const RewindSite* __restrict__ sites, const char* backup,
                                                                      const char* hist, int slot, int row0, int n) {
    const RewindSite s = sites[blockIdx.y];
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= s.C) return;
    const int C = s.C;
    const float* b = reinterpret_cast<const float*>(backup + s.backup);
    const float* x = reinterpret_cast<const float*>(hist + s.hist) + static_cast<size_t>(row0) * C;
    float* st = s.state + static_cast<size_t>(slot) * 3 * C;
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        const int i = n - 3 + k;
        st[k * C + c] = i < 0 ? b[(3 + i) * C + c] : x[static_cast<size_t>(i) * C + c];
    }
}

// The final logit softcap in place: x = tanh(x / cap) * cap.
extern "C" __global__ void tf_ink2_softcap(float* __restrict__ x, long long n, float cap) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) x[i] = tanhf(x[i] / cap) * cap;
}

namespace ink2o {

__device__ __forceinline__ bool better(float v, int i, float bv, int bi) { return v > bv || (v == bv && i < bi); }

}  // namespace ink2o

// Each row's greedy pick over its first n fp32 columns: the largest value, the lowest id among equals, plus base.
extern "C" __global__ void __launch_bounds__(1024) tf_ink2_top1(const float* __restrict__ x, long long stride, int n,
                                                                int base, float* __restrict__ val, int* __restrict__ id) {
    using namespace ink2o;
    __shared__ float sv[32];
    __shared__ int si[32];
    const float* r = x + static_cast<size_t>(blockIdx.x) * stride;
    float bv = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (better(r[i], i, bv, bi)) {
            bv = r[i];
            bi = i;
        }
    }
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (better(ov, oi, bv, bi)) {
            bv = ov;
            bi = oi;
        }
    }
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (lane == 0) {
        sv[warp] = bv;
        si[warp] = bi;
    }
    __syncthreads();
    if (warp != 0) return;
    const int nw = blockDim.x >> 5;
    bv = lane < nw ? sv[lane] : -INFINITY;
    bi = lane < nw ? si[lane] : 0x7fffffff;
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (better(ov, oi, bv, bi)) {
            bv = ov;
            bi = oi;
        }
    }
    if (lane == 0) {
        val[blockIdx.x] = bv;
        id[blockIdx.x] = (bi == 0x7fffffff ? 0 : bi) + base;
    }
}

namespace ink2r {

// sample.cu's keys, unchanged: a draft raced with them lands on the token a keyed draw at that position would take
__device__ __forceinline__ unsigned long long mix(unsigned long long x) {
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

__device__ __forceinline__ float race_uniform(unsigned long long seed, unsigned pos, unsigned id) {
    unsigned long long x = mix(seed + 0x9E3779B97F4A7C15ull);
    x = mix(x ^ (static_cast<unsigned long long>(pos) * 0xD1B54A32D192ED03ull));
    x = mix(x ^ static_cast<unsigned long long>(id));
    const float u = (static_cast<float>(static_cast<unsigned>(x >> 40)) + 0.5f) * (1.0f / 16777216.0f);
    return fminf(u, __uint_as_float(0x3F7FFFFFu));
}

}  // namespace ink2r

// One fp32 row's Gumbel race at `pos` under sample.cu's keys (ids base + i): the best score and its id.
extern "C" __global__ void __launch_bounds__(1024) tf_ink2_race1(const float* __restrict__ x, int n, int base,
                                                                 unsigned long long seed, float inv_t, unsigned pos,
                                                                 float* __restrict__ val, int* __restrict__ id) {
    using namespace ink2o;
    __shared__ float sv[32];
    __shared__ int si[32];
    float bv = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = bfr(x[i]) * inv_t - logf(-logf(ink2r::race_uniform(seed, pos, static_cast<unsigned>(base + i))));
        if (better(v, i, bv, bi)) {
            bv = v;
            bi = i;
        }
    }
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (better(ov, oi, bv, bi)) {
            bv = ov;
            bi = oi;
        }
    }
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (lane == 0) {
        sv[warp] = bv;
        si[warp] = bi;
    }
    __syncthreads();
    if (warp != 0) return;
    bv = lane < static_cast<int>(blockDim.x >> 5) ? sv[lane] : -INFINITY;
    bi = lane < static_cast<int>(blockDim.x >> 5) ? si[lane] : 0x7fffffff;
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (better(ov, oi, bv, bi)) {
            bv = ov;
            bi = oi;
        }
    }
    if (lane == 0) {
        val[0] = bv;
        id[0] = (bi == 0x7fffffff ? 0 : bi) + base;
    }
}

// fp32 logits rows into bf16 whole rows at column col0: the first n columns rounded, the rest of `width` -inf.
extern "C" __global__ void tf_ink2_tobf(const float* __restrict__ x, long long stride, int n, uint16_t* __restrict__ out,
                                        long long out_stride, long long col0, int width) {
    using namespace ink2o;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width) return;
    const size_t row = blockIdx.y;
    out[row * out_stride + col0 + i] = i < n ? tobf(x[row * stride + i]) : static_cast<uint16_t>(0xff80u);
}

// A link step's words (header and ids) written to the device buffer every rank broadcasts from.
struct Ink2Words {
    unsigned w[96];
};

extern "C" __global__ void tf_ink2_put_words(unsigned* __restrict__ dst, int n, const Ink2Words words) {
    const int i = threadIdx.x;
    if (i < n) dst[i] = words.w[i];
}
