// Nemotron's router: fp32 logits in K slices (one fma chain a slice), then each row's top-k and the shared slots.

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

#include "glue_math.cuh"

using namespace tf_glue;

// Block (16-row tile, 16-expert block, K slice s): PART[s][row][e] = x[row] . w[e] over the slice, in index order.
extern "C" __global__ void __launch_bounds__(256) tf_nemo_router(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ W,
                                                                   float* __restrict__ PART, int R, int D, int E, int SK) {
    extern __shared__ uint32_t tile[];  // 16 input rows, then 16 expert rows, each per / 2 words plus one of padding
    const int per = D / SK, words = per / 2, stride = words + 1;
    const int row0 = blockIdx.x * 16, e0 = blockIdx.y * 16, k0 = blockIdx.z * per;
    uint32_t* xs = tile;
    uint32_t* ws = tile + 16 * stride;
    for (int c = threadIdx.x; c < 16 * words; c += 256) {
        const int r = c / words, w = c % words;
        const uint32_t* xr = reinterpret_cast<const uint32_t*>(X + static_cast<int64_t>(row0 + r) * D + k0);
        const uint32_t* wr = reinterpret_cast<const uint32_t*>(W + static_cast<int64_t>(e0 + r) * D + k0);
        xs[r * stride + w] = row0 + r < R ? xr[w] : 0u;
        ws[r * stride + w] = e0 + r < E ? wr[w] : 0u;
    }
    __syncthreads();
    const int rr = threadIdx.x / 16, ee = threadIdx.x % 16;
    const uint32_t* xp = xs + rr * stride;
    const uint32_t* wp = ws + ee * stride;
    float acc = 0.0f;
    for (int w = 0; w < words; ++w) {
        const uint32_t a = xp[w], b = wp[w];
        acc = __fmaf_rn(__uint_as_float(a << 16), __uint_as_float(b << 16), acc);
        acc = __fmaf_rn(__uint_as_float(a & 0xFFFF0000u), __uint_as_float(b & 0xFFFF0000u), acc);
    }
    if (row0 + rr < R && e0 + ee < E) PART[(static_cast<int64_t>(blockIdx.z) * R + row0 + rr) * E + e0 + ee] = acc;
}

namespace {

constexpr int TT = 128;    // rows and experts a prompt tile holds
constexpr int TW = 16;     // words of a K step a stage holds
constexpr int TS = TW + 4;  // a staged row's words: 16-byte copies, eight rows in a quarter warp on distinct banks

__device__ __forceinline__ void cp16(uint32_t* dst, const void* src, bool read) {
    const uint32_t at = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(at), "l"(src), "r"(read ? 16 : 0));
}

__device__ __forceinline__ uint32_t word(uint4 v, int w) { return w == 0 ? v.x : w == 1 ? v.y : w == 2 ? v.z : v.w; }

// A word's low (half 0) or high bf16 as a float.
__device__ __forceinline__ float unpack(uint32_t v, int half) {
    return __uint_as_float(half ? v & 0xFFFF0000u : v << 16);
}

}  // namespace

// tf_nemo_router's chains, a block (128 rows, 128 experts, slice) and a thread 8 x 8; rows and router 16-byte aligned.
extern "C" __global__ void __launch_bounds__(256)
tf_nemo_router_tiles(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ W, float* __restrict__ PART,
                     int R, int D, int E, int SK) {
    extern __shared__ __align__(16) uint32_t staged[];  // [2][x, w][TT][TS]
    const int per = D / SK, steps = per / (2 * TW);
    const int row0 = blockIdx.x * TT, e0 = blockIdx.y * TT, k0 = blockIdx.z * per;
    const int tx = threadIdx.x % 16, ty = threadIdx.x / 16;
    auto load = [&](int step) {
        uint32_t* st = staged + (step & 1) * 2 * TT * TS;
#pragma unroll
        for (int j = 0; j < 4; ++j) {  // 2 x 128 rows of four 16-byte copies
            const int c = threadIdx.x + 256 * j, wr = c >= 4 * TT, r = (c / 4) % TT, q = c % 4;
            const int at = (wr ? e0 : row0) + r;
            const bool live = at < (wr ? E : R);
            const __nv_bfloat16* src =
                (wr ? W : X) + static_cast<int64_t>(live ? at : 0) * D + k0 + step * 2 * TW + q * 8;
            cp16(st + (wr * TT + r) * TS + q * 4, src, live);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };
    float acc[8][8] = {};
    load(0);
    for (int step = 0; step < steps; ++step) {
        if (step + 1 < steps) {
            load(step + 1);
            asm volatile("cp.async.wait_group 1;\n" ::);
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();
        const uint32_t* xs = staged + (step & 1) * 2 * TT * TS;
        const uint32_t* ws = xs + TT * TS;
#pragma unroll
        for (int q = 0; q < TW / 4; ++q) {
            uint4 xv[8], wv[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                xv[i] = *reinterpret_cast<const uint4*>(xs + (ty + 16 * i) * TS + 4 * q);
                wv[i] = *reinterpret_cast<const uint4*>(ws + (tx + 16 * i) * TS + 4 * q);
            }
#pragma unroll
            for (int w = 0; w < 4; ++w)
#pragma unroll
                for (int half = 0; half < 2; ++half) {  // each word's low input, then its high one
                    float xf[8], wf[8];
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        xf[i] = unpack(word(xv[i], w), half);
                        wf[i] = unpack(word(wv[i], w), half);
                    }
#pragma unroll
                    for (int i = 0; i < 8; ++i)
#pragma unroll
                        for (int j = 0; j < 8; ++j) acc[i][j] = __fmaf_rn(xf[i], wf[j], acc[i][j]);
                }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = row0 + ty + 16 * i;
        if (row >= R) continue;
#pragma unroll
        for (int j = 0; j < 8; ++j)
            if (e0 + tx + 16 * j < E)
                PART[(static_cast<int64_t>(blockIdx.z) * R + row) * E + e0 + tx + 16 * j] = acc[i][j];
    }
}

namespace {

constexpr int EB = 16;  // experts a decode block stages
constexpr int QB = 8;  // 16-byte loads a decode thread has in flight

// acc's fma chain over 8 bf16 pairs, low half then high half of each word, in index order (tf_nemo_router's).
__device__ __forceinline__ float chain8(float acc, uint4 a, uint4 b) {
    const uint32_t as[4] = {a.x, a.y, a.z, a.w}, bs[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        acc = __fmaf_rn(__uint_as_float(as[i] << 16), __uint_as_float(bs[i] << 16), acc);
        acc = __fmaf_rn(__uint_as_float(as[i] & 0xFFFF0000u), __uint_as_float(bs[i] & 0xFFFF0000u), acc);
    }
    return acc;
}

}  // namespace

// Decode windows: block (EB experts, slice) stages it at once, thread (row, expert) runs tf_nemo_router's chain.
extern "C" __global__ void __launch_bounds__(256)
tf_nemo_router_rows(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ W, float* __restrict__ PART,
                    int R, int D, int E, int SK) {
    extern __shared__ uint4 rows_q[];  // R input rows, then EB expert rows, each quads + 1 (a 16-byte padding step)
    const int s = blockIdx.y, e0 = blockIdx.x * EB, per = D / SK, k0 = s * per, quads = per / 8, stride = quads + 1;
    const int n = (R + EB) * quads;
    uint4 v[QB];
#pragma unroll
    for (int j = 0; j < QB; ++j) {
        const int c = threadIdx.x + blockDim.x * j, r = c / quads, q = c % quads;
        const bool wr = r >= R, live = c < n && (!wr || e0 + r - R < E);
        const __nv_bfloat16* src = wr ? W + static_cast<int64_t>(e0 + r - R) * D : X + static_cast<int64_t>(r) * D;
        v[j] = live ? reinterpret_cast<const uint4*>(src + k0)[q] : make_uint4(0u, 0u, 0u, 0u);
    }
#pragma unroll
    for (int j = 0; j < QB; ++j) {
        const int c = threadIdx.x + blockDim.x * j;
        if (c < n) rows_q[(c / quads) * stride + c % quads] = v[j];
    }
    for (int c = threadIdx.x + blockDim.x * QB; c < n; c += blockDim.x) {  // past QB loads a thread: the rest in order
        const int r = c / quads, q = c % quads;
        const bool wr = r >= R, live = !wr || e0 + r - R < E;
        const __nv_bfloat16* src = wr ? W + static_cast<int64_t>(e0 + r - R) * D : X + static_cast<int64_t>(r) * D;
        rows_q[r * stride + q] = live ? reinterpret_cast<const uint4*>(src + k0)[q] : make_uint4(0u, 0u, 0u, 0u);
    }
    __syncthreads();
    const int rr = threadIdx.x / EB, ee = threadIdx.x % EB;
    if (rr >= R || e0 + ee >= E) return;
    const uint4* xp = rows_q + rr * stride;
    const uint4* wp = rows_q + (R + ee) * stride;
    float acc = 0.0f;
    uint4 xa = xp[0], wa = wp[0];
    for (int q = 0; q < quads; ++q) {  // the next pair's loads go out before this pair's chain
        const uint4 xn = xp[q + 1 < quads ? q + 1 : q], wn = wp[q + 1 < quads ? q + 1 : q];
        acc = chain8(acc, xa, wa);
        xa = xn, wa = wn;
    }
    PART[(static_cast<int64_t>(s) * R + rr) * E + e0 + ee] = acc;
}

namespace {

constexpr int MAXSK = 8;  // router slices topk sums (routerShape takes at most 6)

// The larger of two (value, index) picks: a higher value, or the lower index on a tie (values are never NaN).
__device__ __forceinline__ bool beats(float v, int i, float bv, int bi) { return v > bv || (v == bv && i < bi); }

}  // namespace

// Sum logits' slices in order; rank sigmoid scores plus bias before adding the shared slots. MAXC: a lane's experts.
template <int MAXC>
__device__ __forceinline__ void nemo_topk(const float* __restrict__ PART, const float* __restrict__ BIAS, int* __restrict__ IDX,
                                          float* __restrict__ WT, int R, float scaling, int E, int SK, int TOPK, int NS, int norm) {
    const int64_t r = blockIdx.x;
    const int lane = threadIdx.x;
    float score[MAXC], sel[MAXC];
    float part[MAXC][MAXSK], bias[MAXC];  // every slice's and bias's load issued before the sums
#pragma unroll
    for (int c = 0; c < MAXC; ++c) {
        const int e = lane + 32 * c;
        bias[c] = e < E ? BIAS[e] : 0.0f;
#pragma unroll
        for (int s = 0; s < MAXSK; ++s) part[c][s] = e < E && s < SK ? PART[(s * R + r) * E + e] : 0.0f;
    }
#pragma unroll
    for (int c = 0; c < MAXC; ++c) {
        const int e = lane + 32 * c;
        score[c] = 0.0f;
        sel[c] = -INFINITY;
        if (e >= E) continue;
        float logit = 0.0f;
#pragma unroll
        for (int s = 0; s < MAXSK; ++s)
            if (s < SK) logit = __fadd_rn(logit, part[c][s]);
        score[c] = tf_sigmoid(logit);
        const float v = __fadd_rn(score[c], bias[c]);
        sel[c] = v != v ? -INFINITY : v;  // a NaN logit is never picked ahead of a number
    }
    float total = 0.0f;
    float probs[MAXC];
    int ids[MAXC];
    for (int k = 0; k < TOPK; ++k) {
        float bv = -INFINITY;
        int bi = 0x7fffffff;
#pragma unroll
        for (int c = 0; c < MAXC; ++c)
            if (lane + 32 * c < E && beats(sel[c], lane + 32 * c, bv, bi)) bv = sel[c], bi = lane + 32 * c;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, off);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, off);
            if (beats(ov, oi, bv, bi)) bv = ov, bi = oi;
        }
        float p = 0.0f;
#pragma unroll
        for (int c = 0; c < MAXC; ++c)
            if (bi == lane + 32 * c) p = score[c], sel[c] = -INFINITY;
        p = __shfl_sync(0xffffffffu, p, bi % 32);
        if (k < MAXC) probs[k] = p, ids[k] = bi;
        total = __fadd_rn(total, p);
    }
    if (lane != 0) return;
    for (int k = 0; k < NS; ++k) {
        float w = 1.0f;
        int id = E + (k - TOPK);
        if (k < TOPK) {
            id = ids[k];
            w = norm ? __fmul_rn(__fdiv_rn(probs[k], __fadd_rn(total, 1e-20f)), scaling) : __fmul_rn(probs[k], scaling);
        }
        IDX[r * NS + k] = id;
        WT[r * NS + k] = w;
    }
}

// Up to 256 experts and 8 picks a row (Lightning, Super): the released kernel.
extern "C" __global__ void __launch_bounds__(32) tf_nemo_topk(const float* __restrict__ PART, const float* __restrict__ BIAS,
                                                               int* __restrict__ IDX, float* __restrict__ WT, int R, float scaling, int E,
                                                               int SK, int TOPK, int NS, int norm) {
    nemo_topk<8>(PART, BIAS, IDX, WT, R, scaling, E, SK, TOPK, NS, norm);
}

// Up to 512 experts and 16 picks a row (Bolt's 496 experts, 9 picks).
extern "C" __global__ void __launch_bounds__(32) tf_nemo_topk_wide(const float* __restrict__ PART, const float* __restrict__ BIAS,
                                                                    int* __restrict__ IDX, float* __restrict__ WT, int R, float scaling,
                                                                    int E, int SK, int TOPK, int NS, int norm) {
    nemo_topk<16>(PART, BIAS, IDX, WT, R, scaling, E, SK, TOPK, NS, norm);
}
