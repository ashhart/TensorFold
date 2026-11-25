// Inkling 2 Mini's loop kernels on HIP, each under its named HIP baseline (docs/kernel-contract.md).
#include "inkling2/zmath.hpp"

namespace ink2l {

using ink2z::bf;
using ink2z::zexp;

constexpr int kHidden = 2304;                      // Mini's hidden width: every bank, query and message row
constexpr int kLanes = 32;                         // a W32 sum's lanes: element e goes to lane (e / 8) % 32
constexpr int kRowChunks = kHidden / 8 / kLanes;   // chunks of 8 a lane holds of a 2,304 row
constexpr int kMsgThreads = 64;                    // a message block: wave 0 reduces bank B_12, wave 1 P_18
constexpr int kMsgVecs = kHidden / 4 / kMsgThreads;  // float4 of a message each thread writes
static_assert(kHidden % (8 * kLanes) == 0 && kHidden % (4 * kMsgThreads) == 0);

// What a control plants in the arithmetic; kServed is the named baselines'.
enum Plant { kServed, kRsqrt, kFma, kSignedZero, kSerialRho, kButterflyDesc };

// The two bf16 values of a 32-bit word, the lower one first.
__device__ __forceinline__ float lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }

__device__ __forceinline__ float4 ld4(const float* p) { return *reinterpret_cast<const float4*>(p); }

// The baselines' 1 / sqrt: a correctly rounded square root, then a correctly rounded division (not rsqrt).
__device__ __forceinline__ float invSqrt(float v) { return 1.0f / sqrtf(v); }

// rho's inverse from a row's sum of squares: the mean, plus eps, then 1 / sqrt, each an fp32 step.
template <int P>
__device__ __forceinline__ float inverse(float squares, float eps) {
    const float mean = squares / static_cast<float>(kHidden);
    return P == kRsqrt ? rsqrtf(mean + eps) : invSqrt(mean + eps);
}

// The W32 butterfly: xor 1, 2, 4, 8, 16, a balanced tree in lane order; addition commutes, so every lane agrees.
template <int P>
__device__ __forceinline__ float butterfly(float v) {
    if (P == kButterflyDesc) {
#pragma unroll
        for (int o = kLanes / 2; o >= 1; o >>= 1) v = v + __shfl_xor(v, o);
    } else {
#pragma unroll
        for (int o = 1; o < kLanes; o <<= 1) v = v + __shfl_xor(v, o);
    }
    return v;
}

// A bf16 row's chunks lane + 32 i (elements 8 (lane + 32 i) + j at x[i][j]), widened to f32.
__device__ __forceinline__ void loadBank(const uint16_t* row, int lane, float (&x)[kRowChunks][8]) {
    const uint4* p = reinterpret_cast<const uint4*>(row);
#pragma unroll
    for (int i = 0; i < kRowChunks; ++i) {
        const uint4 w = p[lane + kLanes * i];
        x[i][0] = lo(w.x), x[i][1] = hi(w.x), x[i][2] = lo(w.y), x[i][3] = hi(w.y);
        x[i][4] = lo(w.z), x[i][5] = hi(w.z), x[i][6] = lo(w.w), x[i][7] = hi(w.w);
    }
}

// HIP-RMS-W32's sum of squares: each lane's chain ascending from +0 by fmaf, then the butterfly.
template <int P>
__device__ __forceinline__ float squares(const float (&x)[kRowChunks][8]) {
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < kRowChunks; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) s = fmaf(x[i][j], x[i][j], s);
    return butterfly<P>(s);
}

// The serial-rho control's sum: lane 0 adds the rounded squares in input order from +0, every lane gets it.
__device__ __forceinline__ float serialSquares(const uint16_t* row, int lane) {
    float s = 0.0f;
    if (lane == 0)
        for (int e = 0; e < kHidden; ++e) {
            const float v = bf(row[e]);
            s = s + v * v;
        }
    return __shfl(s, 0);
}

// Two logits' softmax: the max from -inf, exp of each difference, the sum from +0, each share.
__device__ __forceinline__ void softmax2(float l0, float l1, float& w0, float& w1) {
    const float top = fmaxf(fmaxf(-__builtin_inff(), l0), l1);
    const float e0 = zexp(l0 - top), e1 = zexp(l1 - top);
    float den = 0.0f;
    den = den + e0;
    den = den + e1;
    w0 = e0 / den;
    w1 = e1 / den;
}

// One output: the two shares summed from +0, so two -0 shares give +0 (controls drop the +0 or fuse a share).
template <int P>
__device__ __forceinline__ float share(float w0, float b0, float w1, float b1) {
    const float m0 = w0 * b0;
    if (P == kSignedZero) return m0 + w1 * b1;
    if (P == kFma) return fmaf(w1, b1, 0.0f + m0);
    return (0.0f + m0) + w1 * b1;
}

// A message block's LDS: both normalized banks, then the two logits.
struct MsgStage {
    float b[2][kHidden];
    float logit[2];
};

// Block (row, message `core * 4 + role`): wave k normalizes bank k (HIP-RMS-W32), dots it with the query
// (HIP-RETRIEVE-W32) and keeps it in LDS; then every thread writes 36 of the message's columns.
template <int P>
__device__ void messages(const uint16_t* bank0, const uint16_t* bank1, const float* qm, const float* kv, float* out,
                         int rows, float scale, float eps, MsgStage& st) {
    const size_t row = blockIdx.x;
    const int msg = blockIdx.y, role = msg & 3, core = msg >> 2;
    const int t = threadIdx.x, k = t / kLanes, lane = t % kLanes;
    // q and mlp read q_mlp, k and v read kv; q and k take row 2j, v and mlp row 2j + 1
    const float* query = (role == 0 || role == 3 ? qm : kv) + static_cast<size_t>(2 * core + (role >= 2)) * kHidden;
    const uint16_t* bank = (k == 0 ? bank0 : bank1) + row * kHidden;
    float x[kRowChunks][8];
    loadBank(bank, lane, x);
    const float inv = inverse<P>(P == kSerialRho ? serialSquares(bank, lane) : squares<P>(x), eps);
    float d = 0.0f;
#pragma unroll
    for (int i = 0; i < kRowChunks; ++i) {
        const int e = 8 * (lane + kLanes * i);
        const float4 q0 = ld4(query + e), q1 = ld4(query + e + 4);
        const float4 b0 = make_float4(x[i][0] * inv, x[i][1] * inv, x[i][2] * inv, x[i][3] * inv);
        const float4 b1 = make_float4(x[i][4] * inv, x[i][5] * inv, x[i][6] * inv, x[i][7] * inv);
        d = fmaf(q0.x, b0.x, d);
        d = fmaf(q0.y, b0.y, d);
        d = fmaf(q0.z, b0.z, d);
        d = fmaf(q0.w, b0.w, d);
        d = fmaf(q1.x, b1.x, d);
        d = fmaf(q1.y, b1.y, d);
        d = fmaf(q1.z, b1.z, d);
        d = fmaf(q1.w, b1.w, d);
        *reinterpret_cast<float4*>(&st.b[k][e]) = b0;
        *reinterpret_cast<float4*>(&st.b[k][e + 4]) = b1;
    }
    d = butterfly<P>(d);
    if (lane == 0) st.logit[k] = d * scale;
    __syncthreads();
    float w0, w1;
    softmax2(st.logit[0], st.logit[1], w0, w1);
    float4* dst = reinterpret_cast<float4*>(out + (static_cast<size_t>(msg) * rows + row) * kHidden);
#pragma unroll
    for (int v = 0; v < kMsgVecs; ++v) {
        const int c = t + kMsgThreads * v;
        const float4 a = ld4(&st.b[0][4 * c]), b = ld4(&st.b[1][4 * c]);
        dst[c] = make_float4(share<P>(w0, a.x, w1, b.x), share<P>(w0, a.y, w1, b.y), share<P>(w0, a.z, w1, b.z),
                             share<P>(w0, a.w, w1, b.w));
    }
}

}  // namespace ink2l

// Messages: grid (rows, count * 4 messages); banks bf16 [rows][2,304], queries f32 from the first core's rows.
#define INK2_MESSAGES(name, plant)                                                                                  \
    extern "C" __global__ void __launch_bounds__(ink2l::kMsgThreads)                                               \
        name(const uint16_t* __restrict__ bank0, const uint16_t* __restrict__ bank1, const float* __restrict__ qm, \
             const float* __restrict__ kv, float* __restrict__ out, int rows, float scale, float eps) {            \
        __shared__ ink2l::MsgStage st;                                                                             \
        ink2l::messages<plant>(bank0, bank1, qm, kv, out, rows, scale, eps, st);                                   \
    }

INK2_MESSAGES(tf_ink2_loop_messages, ink2l::kServed)
// The contract's controls: rsqrt, the second share fused, no +0 start, rho's squares serial, the butterfly reversed.
INK2_MESSAGES(tf_ink2_loop_ctl_rsqrt, ink2l::kRsqrt)
INK2_MESSAGES(tf_ink2_loop_ctl_fma, ink2l::kFma)
INK2_MESSAGES(tf_ink2_loop_ctl_signed_zero, ink2l::kSignedZero)
INK2_MESSAGES(tf_ink2_loop_ctl_serial_rho, ink2l::kSerialRho)
INK2_MESSAGES(tf_ink2_loop_ctl_butterfly_desc, ink2l::kButterflyDesc)

// The contract's proof of the f32 math: op 0 tanh, 1 exp, 2 1 / sqrt as the kernels run them; 3-5 the device's own.
extern "C" __global__ void __launch_bounds__(256)
    tf_ink2_loop_math(const float* __restrict__ in, float* __restrict__ out, int count, int op) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= count) return;
    const float x = in[i];
    float y;
    switch (op) {
        case 0: y = ink2z::ztanh(x); break;
        case 1: y = ink2z::zexp(x); break;
        case 2: y = ink2l::invSqrt(x); break;
        case 3: y = tanhf(x); break;
        case 4: y = expf(x); break;
        default: y = rsqrtf(x); break;
    }
    out[i] = y;
}
