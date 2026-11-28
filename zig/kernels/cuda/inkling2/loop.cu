// Inkling 2 Mini's loop: cross-pass message attention and the rank-64 message gates, in fp32 as transformers runs them.
#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

#include "common/cp_async.cuh"
#include "common/qmm_frag.cuh"
#include "common/reduce.cuh"
#include "common/vec.cuh"

namespace ink2_loop {

constexpr int kWidth = 2304;                  // hidden size
constexpr int kRank = 64;                     // gate_rank
constexpr int kPair = 2 * kWidth;             // the down projection's input: [norm(h); norm(msg)]
constexpr int kMaxSources = 4;                // bank states a message attends over (Mini: 2)
constexpr int kMsgThreads = 256;              // message kernel: one block a (row, gate)
constexpr int kPer = kWidth / kMsgThreads;    // values of a vector each message thread holds
constexpr int kWarps = kMsgThreads / 32;
constexpr int kSplit = 9;                     // fixed K slices of the down projection, summed in slice order
constexpr int kSlice = kPair / kSplit;        // 512 inputs a slice
constexpr int kWarpK = kSlice / 4;            // 128 inputs a warp (8 mma k-steps), the four added in warp order
constexpr int kBlendCols = 32;                // blend block: 32 outputs, 8 lanes each
constexpr int kBlendRows = 32;                // most rows a blend block walks
static_assert(kWidth % kMsgThreads == 0 && kWarpK % 32 == 0 && kWidth % kBlendCols == 0, "shape");

// The four gates' bf16 weights of one core layer, in q, k, v, mlp order (down [R][2W], forget and gain [W][R]).
struct Gates {
    const __nv_bfloat16* down[4];
    const __nv_bfloat16* forget[4];
    const __nv_bfloat16* gain[4];
};

// One gate's pointer by selects, so a runtime gate index never copies the parameter struct to local memory.
__device__ __forceinline__ const __nv_bfloat16* pick(const __nv_bfloat16* const (&a)[4], int i) {
    return i == 0 ? a[0] : i == 1 ? a[1] : i == 2 ? a[2] : a[3];
}

// N block totals: each warp's butterfly, then the warp sums added in warp order (bits independent of the grid).
template <int N>
__device__ __forceinline__ void totals(float (&v)[N], float* scratch) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
#pragma unroll
    for (int n = 0; n < N; ++n) {
        const float s = tf_common::warp_sum(v[n]);
        if (lane == 0) scratch[n * kWarps + w] = s;
    }
    __syncthreads();
#pragma unroll
    for (int n = 0; n < N; ++n) {
        float t = 0.0f;
#pragma unroll
        for (int k = 0; k < kWarps; ++k) t = __fadd_rn(t, scratch[n * kWarps + k]);
        v[n] = t;
    }
    __syncthreads();
}

// Squares summed in index order over a thread's values (the reference squares, then sums).
__device__ __forceinline__ float squares(const float (&x)[kPer]) {
    float s = 0.0f;
#pragma unroll
    for (int j = 0; j < kPer; ++j) s = __fadd_rn(s, __fmul_rn(x[j], x[j]));
    return s;
}

// Pull the launch's gate weights toward L2, spread over the grid: the down and blend kernels read them next.
__device__ __forceinline__ void warm(const Gates& w, int gates) {
    constexpr int kDown = kRank * kPair * 2 / 128, kSide = kWidth * kRank * 2 / 128;
    const int first = blockIdx.x * blockDim.x + threadIdx.x, step = gridDim.x * blockDim.x;
    for (int gi = 0; gi < gates; ++gi)
        for (int part = 0; part < 3; ++part) {
            const __nv_bfloat16* p = part == 0 ? pick(w.down, gi) : part == 1 ? pick(w.forget, gi) : pick(w.gain, gi);
            const char* base = reinterpret_cast<const char*>(p);
            for (int l = first; l < (part == 0 ? kDown : kSide); l += step)
                asm volatile("prefetch.global.L2 [%0];" ::"l"(base + (size_t)l * 128));
        }
}

}  // namespace ink2_loop

using namespace ink2_loop;

// Messages of a visit's streams and the gates' bf16 input [norm(h); norm(msg)]: a block per (row, gate).
extern "C" __global__ void __launch_bounds__(kMsgThreads) tf_ink2_loop_msg(
    const float* __restrict__ bank, int sources, const float* __restrict__ queries, const float* __restrict__ h,
    long long h_row, long long h_gate, int gates, float scale, float eps, Gates w, float* __restrict__ msg,
    __nv_bfloat16* __restrict__ pair) {
    __shared__ float scratch[(kMaxSources + 1) * kWarps];
    warm(w, gates);
    // bank [rows][sources][W]: the previous pass's blocks after its first, then its output; queries [gates][W] raw
    const int row = blockIdx.x / gates, gate = blockIdx.x % gates, tid = threadIdx.x;
    const float* b = bank + (size_t)row * sources * kWidth;
    const float* q = queries + (size_t)gate * kWidth;
    const float* x = h + row * h_row + gate * h_gate;
    float e[kMaxSources][kPer], hv[kPer], qv[kPer];
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
        const int i = tid + j * kMsgThreads;
        hv[j] = x[i];
        qv[j] = q[i];
#pragma unroll
        for (int s = 0; s < kMaxSources; ++s) e[s][j] = s < sources ? b[s * kWidth + i] : 0.0f;
    }
    // unweighted RMS norms of the bank states (the message's keys and values) and of h
    float ss[kMaxSources + 1];
#pragma unroll
    for (int s = 0; s < kMaxSources; ++s) ss[s] = squares(e[s]);
    ss[kMaxSources] = squares(hv);
    totals(ss, scratch);
    const float rh = tf_common::rms_scale(ss[kMaxSources], kWidth, eps);
    float d[kMaxSources];
#pragma unroll
    for (int s = 0; s < kMaxSources; ++s) {
        const float r = tf_common::rms_scale(ss[s], kWidth, eps);
        d[s] = 0.0f;
#pragma unroll
        for (int j = 0; j < kPer; ++j) {
            e[s][j] = __fmul_rn(e[s][j], r);
            d[s] = __fmaf_rn(qv[j], e[s][j], d[s]);
        }
    }
    totals(d, scratch);
    // softmax over the sources of (q . k) * W^-0.5, in fp32 as the reference's eager attention
    float m = -INFINITY;
#pragma unroll
    for (int s = 0; s < kMaxSources; ++s) {
        d[s] = __fmul_rn(d[s], scale);
        if (s < sources) m = fmaxf(m, d[s]);
    }
    float den = 0.0f;
#pragma unroll
    for (int s = 0; s < kMaxSources; ++s) {
        d[s] = s < sources ? expf(__fsub_rn(d[s], m)) : 0.0f;
        den = __fadd_rn(den, d[s]);
    }
#pragma unroll
    for (int s = 0; s < kMaxSources; ++s) d[s] = __fdiv_rn(d[s], den);
    float mv[kPer];
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
        float acc = 0.0f;
#pragma unroll
        for (int s = 0; s < kMaxSources; ++s)
            if (s < sources) acc = __fmaf_rn(d[s], e[s][j], acc);
        mv[j] = acc;
    }
    float sm[1] = {squares(mv)};
    totals(sm, scratch);
    const float rm = tf_common::rms_scale(sm[0], kWidth, eps);
    const size_t o = (size_t)row * gates + gate;
    float* mo = msg + o * kWidth;
    __nv_bfloat16* po = pair + o * kPair;
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
        const int i = tid + j * kMsgThreads;
        mo[i] = mv[j];
        po[i] = __float2bfloat16_rn(__fmul_rn(hv[j], rh));
        po[kWidth + i] = __float2bfloat16_rn(__fmul_rn(mv[j], rm));
    }
}

// g = down . pair on bf16 mma: K cut in fixed slices and warp quarters, so no row count or launch shape moves bits.
template <int NT>
__device__ __forceinline__ void down_rows(const __nv_bfloat16* __restrict__ pair, const Gates& w, int rows, int gates,
                                          float* __restrict__ part, unsigned* __restrict__ count,
                                          __nv_bfloat16* __restrict__ g) {
    // grid (row tiles of 16, kSplit, gates x 8 / NT output groups of NT * 8); count starts zeroed and is left zeroed
    constexpr int kOut = NT * 8, kGroups = kRank / kOut;
    __shared__ float fold[4][16 * kOut];
    __shared__ unsigned last;
    const int tile = blockIdx.x, slice = blockIdx.y, gate = blockIdx.z / kGroups, n0 = blockIdx.z % kGroups * kOut;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
    const int ra = tile * 16 + gid, rb = ra + 8, k0 = slice * kSlice + warp * kWarpK;
    const bool va = ra < rows, vb = rb < rows;
    // K order inside a 32-input chunk: lane tig's 16-byte word (inputs 8 tig..8 tig+7) feeds two mma k-steps
    const uint4* xa = reinterpret_cast<const uint4*>(pair + ((size_t)(va ? ra : 0) * gates + gate) * kPair + k0) + tig;
    const uint4* xb = reinterpret_cast<const uint4*>(pair + ((size_t)(vb ? rb : 0) * gates + gate) * kPair + k0) + tig;
    const uint4* wt = reinterpret_cast<const uint4*>(pick(w.down, gate) + (size_t)(n0 + gid) * kPair + k0) + tig;
    float acc[NT][4] = {};
#pragma unroll
    for (int c = 0; c < kWarpK / 32; ++c) {
        const uint4 z = make_uint4(0u, 0u, 0u, 0u), pa = va ? xa[4 * c] : z, pb = vb ? xb[4 * c] : z;
        uint4 wb[NT];
#pragma unroll
        for (int n = 0; n < NT; ++n) wb[n] = tf_common::ld_nc(wt + (size_t)n * 8 * (kPair / 8) + 4 * c);
        const uint32_t a0[4] = {pa.x, pb.x, pa.y, pb.y}, a1[4] = {pa.z, pb.z, pa.w, pb.w};
#pragma unroll
        for (int n = 0; n < NT; ++n) {
            qmm_frag::mma(acc[n], a0, wb[n].x, wb[n].y);
            qmm_frag::mma(acc[n], a1, wb[n].z, wb[n].w);
        }
    }
#pragma unroll
    for (int n = 0; n < NT; ++n) {
        const int c = n * 8 + tig * 2;
        fold[warp][gid * kOut + c] = acc[n][0];
        fold[warp][gid * kOut + c + 1] = acc[n][1];
        fold[warp][(gid + 8) * kOut + c] = acc[n][2];
        fold[warp][(gid + 8) * kOut + c + 1] = acc[n][3];
    }
    __syncthreads();
    // the slice's sum: its four warp quarters added in warp order
    for (int e = threadIdx.x; e < 16 * kOut; e += blockDim.x) {
        const int r = tile * 16 + e / kOut;
        if (r >= rows) continue;
        const float s = __fadd_rn(__fadd_rn(__fadd_rn(fold[0][e], fold[1][e]), fold[2][e]), fold[3][e]);
        part[(((size_t)r * gates + gate) * kSplit + slice) * kRank + n0 + e % kOut] = s;
    }
    __threadfence();
    __syncthreads();
    unsigned* c = count + ((size_t)tile * gates + gate) * kGroups + blockIdx.z % kGroups;
    if (threadIdx.x == 0) last = atomicAdd(c, 1u) == kSplit - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    // the tile's last slice block: slices added in slice order, then rounded to bf16 as the bf16 down_proj output
    for (int e = threadIdx.x; e < 16 * kOut; e += blockDim.x) {
        const int r = tile * 16 + e / kOut;
        if (r >= rows) continue;
        const size_t o = (size_t)r * gates + gate;
        const float* p = part + o * kSplit * kRank + n0 + e % kOut;
        float s = 0.0f;
#pragma unroll
        for (int k = 0; k < kSplit; ++k) s = __fadd_rn(s, __ldcg(p + k * kRank));
        g[o * kRank + n0 + e % kOut] = __float2bfloat16_rn(s);
    }
    if (threadIdx.x == 0) *c = 0;
}

// The same sums with 8, 16 or 64 outputs a block: more blocks for decode rows, fewer A reloads for prompt chunks.
extern "C" __global__ void __launch_bounds__(128) tf_ink2_loop_down8(
    const __nv_bfloat16* __restrict__ pair, Gates w, int rows, int gates, float* __restrict__ part,
    unsigned* __restrict__ count, __nv_bfloat16* __restrict__ g) {
    down_rows<1>(pair, w, rows, gates, part, count, g);
}

extern "C" __global__ void __launch_bounds__(128) tf_ink2_loop_down16(
    const __nv_bfloat16* __restrict__ pair, Gates w, int rows, int gates, float* __restrict__ part,
    unsigned* __restrict__ count, __nv_bfloat16* __restrict__ g) {
    down_rows<2>(pair, w, rows, gates, part, count, g);
}

extern "C" __global__ void __launch_bounds__(128) tf_ink2_loop_down64(
    const __nv_bfloat16* __restrict__ pair, Gates w, int rows, int gates, float* __restrict__ part,
    unsigned* __restrict__ count, __nv_bfloat16* __restrict__ g) {
    down_rows<8>(pair, w, rows, gates, part, count, g);
}

// out = (1 - tanh(F g)) * h + tanh(M g) * msg in fp32; F g one fma chain over r = 0..63, passed lane to lane.
extern "C" __global__ void __launch_bounds__(256) tf_ink2_loop_blend(
    const float* __restrict__ h, long long h_row, long long h_gate, const float* __restrict__ msg,
    const __nv_bfloat16* __restrict__ g, Gates w, int rows, int gates, int per, float* __restrict__ out) {
    // grid (W / 32, gates, row groups of `per`); lane j of an output's eight holds its weights 8 j..8 j+7
    __shared__ float4 gs[kBlendRows][kRank / 4];
    __shared__ float2 dots[kBlendRows][kBlendCols];
    const int j = threadIdx.x & 7, c = threadIdx.x >> 3, col = blockIdx.x * kBlendCols + c, gate = blockIdx.y;
    const int r0 = blockIdx.z * per, n = min(per, rows - r0);
    float* gf = reinterpret_cast<float*>(gs);
    for (int e = threadIdx.x; e < n * kRank; e += blockDim.x)
        gf[e] = __bfloat162float(g[((size_t)(r0 + e / kRank) * gates + gate) * kRank + e % kRank]);
    float f8[8], m8[8];
    const uint4* fp = reinterpret_cast<const uint4*>(pick(w.forget, gate) + (size_t)col * kRank) + j;
    const uint4* mp = reinterpret_cast<const uint4*>(pick(w.gain, gate) + (size_t)col * kRank) + j;
    tf_common::unpack8(tf_common::ld_nc(fp), f8);
    tf_common::unpack8(tf_common::ld_nc(mp), m8);
    __syncthreads();
    // step k: lane j continues row k - j's chains from lane j - 1 over its 8 terms; lane 7 ends row k - 7
    float cf = 0.0f, cm = 0.0f;
    for (int k = 0; k < n + 7; ++k) {
        float fd = __shfl_up_sync(0xffffffffu, cf, 1, 8), md = __shfl_up_sync(0xffffffffu, cm, 1, 8);
        if (j == 0) fd = 0.0f, md = 0.0f;
        const int t = min(max(k - j, 0), n - 1);
        const float4 lo = gs[t][2 * j], hi = gs[t][2 * j + 1];
        const float g8[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            fd = __fmaf_rn(f8[i], g8[i], fd);
            md = __fmaf_rn(m8[i], g8[i], md);
        }
        cf = fd, cm = md;
        if (j == 7 && k >= 7) dots[k - 7][c] = make_float2(fd, md);
    }
    __syncthreads();
    for (int e = threadIdx.x; e < n * kBlendCols; e += blockDim.x) {
        const int r = r0 + e / kBlendCols, x = blockIdx.x * kBlendCols + e % kBlendCols;
        const float2 d = dots[e / kBlendCols][e % kBlendCols];
        const size_t o = (size_t)r * gates + gate;
        const float hv = h[r * h_row + gate * h_gate + x];
        const float keep = __fsub_rn(1.0f, tanhf(d.x));
        out[o * kWidth + x] = __fadd_rn(__fmul_rn(keep, hv), __fmul_rn(tanhf(d.y), msg[o * kWidth + x]));
    }
}

namespace ink2_loop {

constexpr int kMaxBlocks = 8, kMaxQueries = 3, kRowThreads = 512;

__device__ __forceinline__ float bf(uint16_t u) { return __uint_as_float(static_cast<uint32_t>(u) << 16); }
__device__ __forceinline__ uint16_t tobf(float f) { return __bfloat16_as_ushort(__float2bfloat16_rn(f)); }

// A message visit's residual mix (ops.cu's MixArgs without the norm): fp32 [nq][rows][C] out.
struct RawMix {
    const uint16_t* blocks[kMaxBlocks];  // bf16 [rows, C] each: the completed blocks' inputs
    const void* partial;                 // [rows, C]: bf16 when partial_bf16, else fp32
    const float* query;                  // fp32 [nq, C]
    float* out;
    int n_blocks, partial_bf16, nq, rows, C;
};

// The weighted RMSNorm after each gate (Inkling2RMSNorm): a weight per gate, bf16 [C] each.
struct Norms {
    const uint16_t* w[kMaxQueries];
};

}  // namespace ink2_loop

// The block attention-residual mix of tf_ink2_mix_rows, same sums in the same order, left unnormed: a block a row.
extern "C" __global__ void __launch_bounds__(kRowThreads) tf_ink2_loop_mix(const RawMix a) {
    __shared__ float sh[32];
    __shared__ float wts[kMaxQueries][kMaxBlocks + 1];
    __shared__ float rr[kMaxBlocks + 1];
    const size_t row = blockIdx.x;
    const int C = a.C, n = a.n_blocks + 1, groups = blockDim.x >> 5;
    auto entry = [&](int j, int c) -> float {
        if (j < a.n_blocks) return bf(a.blocks[j][row * C + c]);
        return a.partial_bf16 ? bf(static_cast<const uint16_t*>(a.partial)[row * C + c])
                              : static_cast<const float*>(a.partial)[row * C + c];
    };
    // each source's RMS first, then the query dots of the normalized source, as tf_ink2_mix_rows sums them
    for (int j = 0; j < n; ++j) {
        float s = 0.0f;
        for (int c = threadIdx.x; c < C; c += blockDim.x) {
            const float x = entry(j, c);
            s += x * x;
        }
        s = tf_common::block_sum_tree(s, sh, groups);
        if (threadIdx.x == 0) rr[j] = rsqrtf(s / C + 1e-6f);
    }
    __syncthreads();
    for (int j = 0; j < n; ++j) {
        float v[kMaxQueries] = {0.0f, 0.0f, 0.0f};
        for (int c = threadIdx.x; c < C; c += blockDim.x) {
            const float normalized = entry(j, c) * rr[j];
            for (int q = 0; q < a.nq; ++q) v[q] += a.query[static_cast<size_t>(q) * C + c] * normalized;
        }
        for (int q = 0; q < kMaxQueries; ++q) v[q] = tf_common::block_sum_tree(v[q], sh, groups);
        if (threadIdx.x == 0)
            for (int q = 0; q < a.nq; ++q) wts[q][j] = v[q] * rsqrtf(static_cast<float>(C));
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
    for (int q = 0; q < a.nq; ++q) {
        float* o = a.out + (static_cast<size_t>(q) * a.rows + row) * C;
        for (int c = threadIdx.x; c < C; c += blockDim.x) {
            float h = 0.0f;
            for (int j = 0; j < n; ++j) h += wts[q][j] * (entry(j, c) * rr[j]);
            o[c] = h;
        }
    }
}

// Each gate's fp32 output [rows][gates][C] normed into the bf16 projection inputs [gates][rows][C]: grid (rows, gates).
extern "C" __global__ void __launch_bounds__(kRowThreads) tf_ink2_loop_norm(const float* __restrict__ x, Norms w,
                                                                           uint16_t* __restrict__ out, int rows, int C,
                                                                           float eps) {
    __shared__ float sh[32];
    const int row = blockIdx.x, gate = blockIdx.y, gates = gridDim.y;
    const float* xr = x + (static_cast<size_t>(row) * gates + gate) * C;
    const uint16_t* wg = gate == 0 ? w.w[0] : gate == 1 ? w.w[1] : w.w[2];
    float s = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) s += xr[c] * xr[c];
    s = tf_common::block_sum_tree(s, sh, blockDim.x >> 5);
    const float r = rsqrtf(s / C + eps);
    uint16_t* o = out + (static_cast<size_t>(gate) * rows + row) * C;
    for (int c = threadIdx.x; c < C; c += blockDim.x) o[c] = tobf(bf(wg[c]) * (xr[c] * r));
}

// bf16 rows into source `s` of the fp32 bank [rows][sources][C] (exact): a block a row.
extern "C" __global__ void __launch_bounds__(256) tf_ink2_loop_bank(const uint16_t* __restrict__ x,
                                                                   float* __restrict__ bank, int sources, int s, int C) {
    const size_t row = blockIdx.x;
    for (int c = threadIdx.x; c < C; c += blockDim.x) bank[(row * sources + s) * C + c] = bf(x[row * C + c]);
}
