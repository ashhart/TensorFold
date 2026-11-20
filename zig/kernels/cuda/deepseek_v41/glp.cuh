// GLPv3 paired additive steering for DeepSeek-V4.1 (EXL3, CUDA): the three edits the paired site needs.
//
// Credits. The GLP format and the DeepSeek paired-site concept (hc_fold_pre_engram) are Matt Suiche's
// (MattSuiche/weightless, spec b181a07d3980fc48abe39d95ddb044459dd0b868). The rank-one directional idea is
// Arditi, Obeso, Syed, Paleka, Panickssery, Gurnee and Nanda's (NeurIPS 2024), background only: these kernels
// implement the additive edit alone (no projection, no ablation) and inherit no efficacy claim from that work.
// The additive control-vector convention (h += alpha * d) is llama.cpp's GGUF one. The reader, writer and
// startup convention is TensorFold's own #1835 (c5947a8d2e9543e6c6a11bd30dc4c43c32f3b0fb). No external serving
// engine or plugin code, and no checkpoint or trained-vector tensor, is vendored here.
//
// The arithmetic, in the order the parent's reference models document: every sum and product is FP32 with no
// contraction (__fmul_rn then __fadd_rn), every store is one BF16 round to nearest even through the canonical
// tf_common::pack8. Alpha 0 (either sign) is an exact bypass: not a byte is read or written.
//   stream: the four copies [R, 4 width] in place, x' = x + alpha d, and the tap [R, width] (null: none) the mean
//           of the four steered copies, each scaled by 1/4 before the FP32 add so four finite copies cannot overflow
//   writer: the gathered attention partials [world, R, width], summed in rank order and rounded to BF16 as the
//           mHC post reads them, then edited; rank 0's slot takes the result and the others -0.0, so the post's
//           rank-ordered sum returns it bit for bit
//   final:  mhc_cuda's collapse by the carried pre-mix, ordered 128-column square chains, RMS scale and norm
//           weight, on the steered streams, so the head reads the edited last fold
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "common/reduce.cuh"
#include "common/vec.cuh"

namespace dsv41_glp {

constexpr int NT = 320;        // threads a CTA: a row's groups of eight columns, two a thread at width 5120
constexpr int MAXWIDTH = 5120; // the served width: the widest row the launch checks admit
constexpr int MAXW = 8;        // ranks a writer reduces (the launch checks admit no more)
constexpr int CB = 128;        // columns a block of the final's square chain (the mHC final's NB x 128)

// Planted faults for the contract's controls; the production image instantiates NONE only.
enum Control : int { NONE = 0, SUM = 1, HALF = 2, DROP_TAIL = 3, DROP_NONFINITE = 4, RANK_ORDER = 5, TRUNCATE = 6 };

struct StreamArgs {
    uint16_t* x; long long xs;        // streams [R, 4 width] bf16, row stride xs: edited in place
    uint16_t* tap; long long ts;      // the DSpark tap [R, width] bf16, row stride ts (null: none)
    const float* dir;                 // [width]
    float alpha; int R; int width;
};
struct WriterArgs {
    uint16_t* g; long long gr; long long gs;   // gathered partials [world, R, width] bf16: rank stride, row stride
    const float* dir;
    float alpha; int R; int width; int world;
};
struct FinalArgs {
    const uint16_t* x; long long xs;  // streams [R, 4 width] bf16
    const float* pre;                 // [R, 4] the carried pre-mix
    const void* nw; uint16_t* out;    // norm weight [width] (bf16 or fp32), the normed input [R, width] bf16
    float eps; int nwbf16; int R; int width;
};
// The by-value launch arguments are glp_ops.zig's extern structs (and the contract's mirrors) byte for byte.
static_assert(sizeof(StreamArgs) == 56 && offsetof(StreamArgs, dir) == 32 && offsetof(StreamArgs, alpha) == 40 &&
              offsetof(StreamArgs, R) == 44 && offsetof(StreamArgs, width) == 48, "StreamArgs differs from glp_ops.zig");
static_assert(sizeof(WriterArgs) == 48 && offsetof(WriterArgs, dir) == 24 && offsetof(WriterArgs, alpha) == 32 &&
              offsetof(WriterArgs, R) == 36 && offsetof(WriterArgs, world) == 44, "WriterArgs differs from glp_ops.zig");
static_assert(sizeof(FinalArgs) == 56 && offsetof(FinalArgs, eps) == 40 && offsetof(FinalArgs, R) == 48 &&
              offsetof(FinalArgs, width) == 52, "FinalArgs differs from glp_ops.zig");

__device__ __forceinline__ uint4 ld16(const void* p) { return *reinterpret_cast<const uint4*>(p); }
__device__ __forceinline__ void widen8(const uint4 v, float (&f)[8]) {     // 8 bf16, column ascending -> fp32
    f[0] = __uint_as_float(v.x << 16); f[1] = __uint_as_float(v.x & 0xffff0000u);
    f[2] = __uint_as_float(v.y << 16); f[3] = __uint_as_float(v.y & 0xffff0000u);
    f[4] = __uint_as_float(v.z << 16); f[5] = __uint_as_float(v.z & 0xffff0000u);
    f[6] = __uint_as_float(v.w << 16); f[7] = __uint_as_float(v.w & 0xffff0000u);
}
__device__ __forceinline__ void load8(const float* p, float (&d)[8]) {
    const float4 a = __ldg(reinterpret_cast<const float4*>(p)), b = __ldg(reinterpret_cast<const float4*>(p) + 1);
    d[0] = a.x; d[1] = a.y; d[2] = a.z; d[3] = a.w; d[4] = b.x; d[5] = b.y; d[6] = b.z; d[7] = b.w;
}

// The one BF16 store: round to nearest even through the canonical packer. TRUNCATE is the planted rounding fault.
template <int CTL> __device__ __forceinline__ uint4 pack_controlled8(const float (&f)[8]) {
    if constexpr (CTL == TRUNCATE) {
        const uint32_t h0 = __float_as_uint(f[0]) >> 16, h1 = __float_as_uint(f[1]) >> 16, h2 = __float_as_uint(f[2]) >> 16,
                       h3 = __float_as_uint(f[3]) >> 16, h4 = __float_as_uint(f[4]) >> 16, h5 = __float_as_uint(f[5]) >> 16,
                       h6 = __float_as_uint(f[6]) >> 16, h7 = __float_as_uint(f[7]) >> 16;
        return make_uint4(h0 | (h1 << 16), h2 | (h3 << 16), h4 | (h5 << 16), h6 | (h7 << 16));
    } else {
        return tf_common::pack8(f);
    }
}

// Scale each BF16 copy by 1/4 before the FP32 add to avoid overflow; SUM deliberately omits that scale.
template <int CTL> __device__ __forceinline__ float mean4(float a, float b, float c, float d) {
    if constexpr (CTL == SUM) {
        return __fadd_rn(__fadd_rn(__fadd_rn(a, b), c), d);
    } else {
        return __fadd_rn(__fadd_rn(__fadd_rn(__fmul_rn(a, 0.25f), __fmul_rn(b, 0.25f)), __fmul_rn(c, 0.25f)),
                         __fmul_rn(d, 0.25f));
    }
}

// The edit of one element, x + alpha * d in FP32 (the caller rounds to BF16 once). HALF: the planted FP16 sum;
// DROP_NONFINITE: a nonfinite input flushed to zero.
template <int CTL> __device__ __forceinline__ float edit(float x, float alpha, float d) {
    const float p = __fmul_rn(alpha, d);
    if constexpr (CTL == HALF) {
        return __half2float(__hadd(__float2half(x), __float2half(p)));
    } else if constexpr (CTL == DROP_NONFINITE) {
        return __fadd_rn(isfinite(x) ? x : 0.0f, p);
    } else {
        return __fadd_rn(x, p);
    }
}

// ---- fold: the four copies in place, the tap of the steered copies ----
template <int CTL> __device__ __forceinline__ void stream_body(const StreamArgs& a) {
    if (a.alpha == 0.0f) return;
    const int r = blockIdx.x, ng = a.width >> 3, w = a.width;
    uint16_t* xr = a.x + static_cast<long long>(r) * a.xs;
    uint16_t* tr = a.tap != nullptr ? a.tap + static_cast<long long>(r) * a.ts : nullptr;
    for (int g = threadIdx.x; g < ng; g += NT) {
        float d[8], y[4][8];
        load8(a.dir + g * 8, d);
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            float x[8];
            widen8(ld16(xr + c * w + g * 8), x);
#pragma unroll
            for (int k = 0; k < 8; ++k) y[c][k] = edit<CTL>(x[k], a.alpha, d[k]);
            if constexpr (CTL == DROP_TAIL) {
                if (g == ng - 1) y[c][7] = x[7];
            }
            const uint4 stored = pack_controlled8<CTL>(y[c]);
            *reinterpret_cast<uint4*>(xr + c * w + g * 8) = stored;
            widen8(stored, y[c]);   // the steered copy as stored: the tap's mean reads these
        }
        if (tr != nullptr) {
            float t[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) t[k] = mean4<CTL>(y[0][k], y[1][k], y[2][k], y[3][k]);
            *reinterpret_cast<uint4*>(tr + g * 8) = pack_controlled8<CTL>(t);
        }
    }
}

// Store the edited writer in rank 0 and -0 elsewhere so hc_post's rank-ordered sum preserves its bits.
template <int CTL> __device__ __forceinline__ void writer_body(const WriterArgs& a) {
    if (a.alpha == 0.0f) return;
    const int r = blockIdx.x, ng = a.width >> 3, world = a.world;
    uint16_t* row = a.g + static_cast<long long>(r) * a.gs;
    const uint4 negzero = make_uint4(0x80008000u, 0x80008000u, 0x80008000u, 0x80008000u);
    for (int g = threadIdx.x; g < ng; g += NT) {
        float s[8];
        if constexpr (CTL == RANK_ORDER) {   // the planted fault: the ranks summed last to first
            widen8(ld16(row + (world - 1) * a.gr + g * 8), s);
            for (int rk = world - 2; rk >= 0; --rk) {
                float p[8];
                widen8(ld16(row + rk * a.gr + g * 8), p);
#pragma unroll
                for (int k = 0; k < 8; ++k) s[k] = __fadd_rn(s[k], p[k]);
            }
        } else {
            widen8(ld16(row + g * 8), s);
            for (int rk = 1; rk < world; ++rk) {
                float p[8];
                widen8(ld16(row + rk * a.gr + g * 8), p);
#pragma unroll
                for (int k = 0; k < 8; ++k) s[k] = __fadd_rn(s[k], p[k]);
            }
        }
        widen8(tf_common::pack8(s), s);      // the post reads the BF16-rounded partial sum
        float d[8], y[8];
        load8(a.dir + g * 8, d);
#pragma unroll
        for (int k = 0; k < 8; ++k) y[k] = edit<CTL>(s[k], a.alpha, d[k]);
        if constexpr (CTL == DROP_TAIL) {
            if (g == ng - 1) y[7] = s[7];
        }
        *reinterpret_cast<uint4*>(row + g * 8) = pack_controlled8<CTL>(y);
        for (int rk = 1; rk < world; ++rk) *reinterpret_cast<uint4*>(row + rk * a.gr + g * 8) = negzero;
    }
}

// Match mhc_cuda's final collapse/RMS order on the edited streams.
__device__ __forceinline__ void final_body(const FinalArgs& a) {
    extern __shared__ __align__(16) unsigned char dyn[];
    uint16_t* cs = reinterpret_cast<uint16_t*>(dyn);
    __shared__ float part[MAXWIDTH / CB];
    __shared__ float rn;
    const int r = blockIdx.x, t = threadIdx.x, ng = a.width >> 3, w = a.width;
    const uint16_t* xr = a.x + static_cast<long long>(r) * a.xs;
    const float q0 = __ldg(a.pre + r * 4), q1 = __ldg(a.pre + r * 4 + 1), q2 = __ldg(a.pre + r * 4 + 2), q3 = __ldg(a.pre + r * 4 + 3);
    for (int g = t; g < ng; g += NT) {
        float x[4][8], c[8];
#pragma unroll
        for (int s = 0; s < 4; ++s) widen8(ld16(xr + s * w + g * 8), x[s]);
#pragma unroll
        for (int k = 0; k < 8; ++k)
            c[k] = __fadd_rn(__fadd_rn(__fadd_rn(__fmul_rn(q0, x[0][k]), __fmul_rn(q1, x[1][k])),
                                       __fmul_rn(q2, x[2][k])), __fmul_rn(q3, x[3][k]));
        *reinterpret_cast<uint4*>(cs + g * 8) = tf_common::pack8(c);   // the collapsed row, BF16-rounded once
    }
    __syncthreads();
    const int nb = w / CB;
    if (t < nb) {
        float acc = 0.0f;
        for (int q = 0; q < CB / 8; ++q) {
            float v[8];
            widen8(*reinterpret_cast<const uint4*>(cs + t * CB + q * 8), v);
#pragma unroll
            for (int k = 0; k < 8; ++k) acc = __fmaf_rn(v[k], v[k], acc);
        }
        part[t] = acc;
    }
    __syncthreads();
    if (t == 0) {
        float total = 0.0f;
        for (int e = 0; e < nb; ++e) total = __fadd_rn(total, part[e]);
        rn = tf_common::rms_scale(total, w, a.eps);
    }
    __syncthreads();
    for (int g = t; g < ng; g += NT) {
        float c[8], wv[8], o[8];
        widen8(*reinterpret_cast<const uint4*>(cs + g * 8), c);
        if (a.nwbf16) {
            widen8(ld16(static_cast<const char*>(a.nw) + g * 16), wv);
        } else {
            load8(static_cast<const float*>(a.nw) + g * 8, wv);
        }
#pragma unroll
        for (int k = 0; k < 8; ++k) o[k] = __fmul_rn(__fmul_rn(c[k], rn), wv[k]);
        *reinterpret_cast<uint4*>(a.out + static_cast<long long>(r) * w + g * 8) = tf_common::pack8(o);
    }
}

}  // namespace dsv41_glp
