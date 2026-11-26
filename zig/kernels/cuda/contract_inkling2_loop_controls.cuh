// Contract-only twins of inkling2/loop.cu's tf_ink2_loop_blend: main's ascending chain, and the old 8-chunk tree.

// Included by contract_controls_set.cu after contract_controls.cu.
#include <cuda_bf16.h>
#include <stdint.h>

#include "common/cp_async.cuh"
#include "common/vec.cuh"

namespace ink2_ctl {

constexpr int kWidth = 2304, kRank = 64, kBlendRows = 32, kBlendCols = 32;

// loop.cu's Gates: one core layer's q, k, v and mlp gate weights (down [R][2W], forget and gain [W][R]).
struct Gates {
    const __nv_bfloat16* down[4];
    const __nv_bfloat16* forget[4];
    const __nv_bfloat16* gain[4];
};

__device__ __forceinline__ const __nv_bfloat16* pick(const __nv_bfloat16* const (&a)[4], int i) {
    return i == 0 ? a[0] : i == 1 ? a[1] : i == 2 ? a[2] : a[3];
}

__device__ __forceinline__ float bf(uint16_t u) { return __uint_as_float(static_cast<uint32_t>(u) << 16); }

// Main's gateUp (inkling2/loop_math.cu) on the blend's arguments: a thread a column, grid (W / 64, gates, rows).
extern "C" __global__ void __launch_bounds__(64) tf_ctl_ink2_blend_chain(
    const float* __restrict__ h, long long h_row, long long h_gate, const float* __restrict__ m,
    const uint16_t* __restrict__ zeta, Gates w, int rows, int gates, int per, float* __restrict__ out) {
    (void)rows, (void)per;
    __shared__ float z[kRank];
    const size_t row = blockIdx.z, o = row * gates + blockIdx.y;
    const int gate = blockIdx.y;
    if (threadIdx.x < kRank) z[threadIdx.x] = bf(zeta[o * kRank + threadIdx.x]);
    __syncthreads();
    const int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint4* f = reinterpret_cast<const uint4*>(pick(w.forget, gate) + static_cast<size_t>(i) * kRank);
    const uint4* g = reinterpret_cast<const uint4*>(pick(w.gain, gate) + static_cast<size_t>(i) * kRank);
    float sf = 0.0f, sg = 0.0f;
#pragma unroll
    for (int q = 0; q < kRank / 8; ++q) {
        const uint4 fq = __ldg(f + q), gq = __ldg(g + q);
        const uint32_t fw[4] = {fq.x, fq.y, fq.z, fq.w}, gw[4] = {gq.x, gq.y, gq.z, gq.w};
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            const float zr = z[8 * q + t];
            const uint16_t fb = static_cast<uint16_t>(fw[t >> 1] >> (16 * (t & 1)));
            const uint16_t gb = static_cast<uint16_t>(gw[t >> 1] >> (16 * (t & 1)));
            sf = __fmaf_rn(bf(fb), zr, sf);
            sg = __fmaf_rn(bf(gb), zr, sg);
        }
    }
    const float fg = tanhf(sf), gg = tanhf(sg);
    const float keep = __fsub_rn(1.0f, fg);
    const float old = __fmul_rn(keep, h[row * h_row + gate * h_gate + i]);
    const float added = __fmul_rn(gg, m[o * kWidth + i]);
    out[o * kWidth + i] = __fadd_rn(old, added);
}

// The blend before main's order, verbatim: lane j chains terms 8 j..8 j+7, then a xor tree over 4, 2, 1 lanes.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_ink2_blend_tree(
    const float* __restrict__ h, long long h_row, long long h_gate, const float* __restrict__ msg,
    const __nv_bfloat16* __restrict__ g, Gates w, int rows, int gates, int per, float* __restrict__ out) {
    // grid (W / 32, gates, row groups of `per`); lane j of an output's eight holds its weights 8 j..8 j+7
    __shared__ float4 gs[kBlendRows][kRank / 4];
    const int j = threadIdx.x & 7, col = blockIdx.x * kBlendCols + (threadIdx.x >> 3), gate = blockIdx.y;
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
    for (int rb = 0; rb < n; rb += 8) {
        float mine_f = 0.0f, mine_m = 0.0f;
        for (int t = 0; t < 8 && rb + t < n; ++t) {
            const float4 lo = gs[rb + t][2 * j], hi = gs[rb + t][2 * j + 1];
            const float g8[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
            float fd = 0.0f, md = 0.0f;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                fd = __fmaf_rn(f8[i], g8[i], fd);
                md = __fmaf_rn(m8[i], g8[i], md);
            }
#pragma unroll
            for (int o = 4; o > 0; o >>= 1) {
                fd = __fadd_rn(fd, __shfl_xor_sync(0xffffffffu, fd, o));
                md = __fadd_rn(md, __shfl_xor_sync(0xffffffffu, md, o));
            }
            if (j == t) mine_f = fd, mine_m = md;
        }
        // lane j finishes row rb + j of its output
        const int r = r0 + rb + j;
        if (rb + j < n) {
            const size_t o = (size_t)r * gates + gate;
            const float hv = h[r * h_row + gate * h_gate + col];
            const float keep = __fsub_rn(1.0f, tanhf(mine_f));
            out[o * kWidth + col] = __fadd_rn(__fmul_rn(keep, hv), __fmul_rn(tanhf(mine_m), msg[o * kWidth + col]));
        }
    }
}

}  // namespace ink2_ctl
