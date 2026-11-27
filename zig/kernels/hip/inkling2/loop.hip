// Inkling 2's loop kernels on HIP at two hidden widths, each under its named HIP baseline (docs/kernel-contract.md).
#include "inkling2/zmath.hpp"

namespace ink2l {

using ink2z::bf;
using ink2z::tobf;
using ink2z::zexp;
using ink2z::ztanh;

constexpr int kLanes = 32;                         // a W32 sum's lanes: element e goes to lane (e / 8) % 32
constexpr int kMsgThreads = 64;                    // a message block: wave 0 reduces bank B_12, wave 1 P_18
constexpr int kRank = 64;                          // the controller's bottleneck
constexpr int kDownThreads = 256;                  // a down block: 8 waves, one zeta rank each
constexpr int kZ = kDownThreads / kLanes;          // ranks a down block
constexpr int kUpCols = 64;                        // columns an up block
constexpr int kUpRows = 4;                         // rows an up block, one a thread row
constexpr int kGates = 3;                          // gates a launch: q, k and v share one
// What a hidden width W sizes (Mini's 2,304, the tiny checkpoint's 256): every bank, query, message and gate row.
template <int W> constexpr int kRowChunks = W / 8 / kLanes;              // chunks of 8 a lane holds of a row
template <int W> constexpr int kMsgVecs = W / 4 / kMsgThreads;           // float4 of a message each thread writes
template <int W> constexpr int kJoined = 2 * W;                          // the joined norm: h's half, then m's
template <int W> constexpr int kJoinedChunks = kJoined<W> / 8 / kLanes;  // chunks of 8 a lane holds of a joined row
template <int W> constexpr int kPer = W / kDownThreads;                  // columns each down thread rounds a row
template <int W> constexpr bool kFits = W % (8 * kLanes) == 0 && W % (4 * kMsgThreads) == 0 && W % kDownThreads == 0;
static_assert(kFits<2304> && kFits<256> && kRank % kZ == 0);

// What a control plants in the arithmetic; kServed is the named baselines'.
enum Plant { kServed, kRsqrt, kFma, kSignedZero, kSerialRho, kButterflyDesc, kUnfusedSq, kTanhf };

// The two bf16 values of a 32-bit word, the lower one first.
__device__ __forceinline__ float lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }

__device__ __forceinline__ float4 ld4(const float* p) { return *reinterpret_cast<const float4*>(p); }

// The baselines' 1 / sqrt: a correctly rounded square root, then a correctly rounded division (not rsqrt).
__device__ __forceinline__ float invSqrt(float v) { return 1.0f / sqrtf(v); }

// rho's inverse from a row's sum of squares: the mean, plus eps, then 1 / sqrt, each an fp32 step.
template <int W, int P>
__device__ __forceinline__ float inverse(float squares, float eps) {
    const float mean = squares / static_cast<float>(W);
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
template <int W>
__device__ __forceinline__ void loadBank(const uint16_t* row, int lane, float (&x)[kRowChunks<W>][8]) {
    const uint4* p = reinterpret_cast<const uint4*>(row);
#pragma unroll
    for (int i = 0; i < kRowChunks<W>; ++i) {
        const uint4 w = p[lane + kLanes * i];
        x[i][0] = lo(w.x), x[i][1] = hi(w.x), x[i][2] = lo(w.y), x[i][3] = hi(w.y);
        x[i][4] = lo(w.z), x[i][5] = hi(w.z), x[i][6] = lo(w.w), x[i][7] = hi(w.w);
    }
}

// HIP-RMS-W32's sum of squares: each lane's chain ascending from +0 by fmaf, then the butterfly.
template <int W, int P>
__device__ __forceinline__ float squares(const float (&x)[kRowChunks<W>][8]) {
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < kRowChunks<W>; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) s = fmaf(x[i][j], x[i][j], s);
    return butterfly<P>(s);
}

// The serial-rho control's sum: lane 0 adds the rounded squares in input order from +0, every lane gets it.
template <int W>
__device__ __forceinline__ float serialSquares(const uint16_t* row, int lane) {
    float s = 0.0f;
    if (lane == 0)
        for (int e = 0; e < W; ++e) {
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
template <int W>
struct MsgStage {
    float b[2][W];
    float logit[2];
};

// Block (row, message core * 4 + role): wave k normalizes and dots bank k, then every thread writes W / 64 columns.
template <int W, int P>
__device__ void messages(const uint16_t* bank0, const uint16_t* bank1, const float* qm, const float* kv, float* out,
                         int rows, float scale, float eps, MsgStage<W>& st) {
    const size_t row = blockIdx.x;
    const int msg = blockIdx.y, role = msg & 3, core = msg >> 2;
    const int t = threadIdx.x, k = t / kLanes, lane = t % kLanes;
    // q and mlp read q_mlp, k and v read kv; q and k take row 2j, v and mlp row 2j + 1
    const float* query = (role == 0 || role == 3 ? qm : kv) + static_cast<size_t>(2 * core + (role >= 2)) * W;
    const uint16_t* bank = (k == 0 ? bank0 : bank1) + row * W;
    float x[kRowChunks<W>][8];
    loadBank<W>(bank, lane, x);
    const float inv = inverse<W, P>(P == kSerialRho ? serialSquares<W>(bank, lane) : squares<W, P>(x), eps);
    float d = 0.0f;
#pragma unroll
    for (int i = 0; i < kRowChunks<W>; ++i) {
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
    float4* dst = reinterpret_cast<float4*>(out + (static_cast<size_t>(msg) * rows + row) * W);
#pragma unroll
    for (int v = 0; v < kMsgVecs<W>; ++v) {
        const int c = t + kMsgThreads * v;
        const float4 a = ld4(&st.b[0][4 * c]), b = ld4(&st.b[1][4 * c]);
        dst[c] = make_float4(share<P>(w0, a.x, w1, b.x), share<P>(w0, a.y, w1, b.y), share<P>(w0, a.z, w1, b.z),
                             share<P>(w0, a.w, w1, b.w));
    }
}

// A gate: h, m, out f32 [rows][W] (out may be h); bf16 down [64][2 W], forget, gain [W][64], zeta [rows][64].
struct Gate {
    const float* h;
    const float* m;
    const uint16_t* down;
    const uint16_t* forget;
    const uint16_t* gain;
    uint16_t* zeta;
    float* out;
};
struct Gates {
    Gate g[kGates];
};

// A down block's LDS: the joined norm (bf16) and the two inverses.
template <int W>
struct DownStage {
    uint16_t joined[kJoined<W>];
    float inv[2];
};

// HIP-RMS-W32 of an f32 row (fmaf squares; a control rounds each square first), or the serial-rho control's chain.
template <int W, int P>
__device__ __forceinline__ float rowSquares(const float* row, int lane) {
    if (P == kSerialRho) {
        float s = 0.0f;
        if (lane == 0)
            for (int e = 0; e < W; ++e) s = s + row[e] * row[e];
        return __shfl(s, 0);
    }
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < kRowChunks<W>; ++i) {
        const float4 a = ld4(row + 8 * (lane + kLanes * i)), b = ld4(row + 8 * (lane + kLanes * i) + 4);
        const float v[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
        for (int j = 0; j < 8; ++j) s = P == kUnfusedSq ? s + v[j] * v[j] : fmaf(v[j], v[j], s);
    }
    return butterfly<P>(s);
}

// Block (row, rank group, gate): waves 0-1 find rho, all round the joined norm, wave w runs zeta rank 8 group + w.
template <int W, int P>
__device__ void gateDown(const Gate& g, float eps, DownStage<W>& st) {
    const size_t row = blockIdx.x;
    const int t = threadIdx.x, wave = t / kLanes, lane = t % kLanes;
    const int rank = blockIdx.y * kZ + wave;
    const uint4* wr = reinterpret_cast<const uint4*>(g.down + static_cast<size_t>(rank) * kJoined<W>);
    uint4 wv[kJoinedChunks<W>];
#pragma unroll
    for (int i = 0; i < kJoinedChunks<W>; ++i) wv[i] = wr[lane + kLanes * i];
    if (wave < 2) {
        const float s = rowSquares<W, P>((wave == 0 ? g.h : g.m) + row * W, lane);
        if (lane == 0) st.inv[wave] = inverse<W, kServed>(s, eps);
    }
    __syncthreads();
    const float ih = st.inv[0], im = st.inv[1];
#pragma unroll
    for (int k = 0; k < kPer<W>; ++k) {
        const int c = t + kDownThreads * k;
        st.joined[c] = tobf(g.h[row * W + c] * ih);
        st.joined[W + c] = tobf(g.m[row * W + c] * im);
    }
    __syncthreads();
    const uint4* jv = reinterpret_cast<const uint4*>(st.joined);
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < kJoinedChunks<W>; ++i) {
        const uint4 w = wv[i], j = jv[lane + kLanes * i];
        acc = fmaf(lo(w.x), lo(j.x), acc);
        acc = fmaf(hi(w.x), hi(j.x), acc);
        acc = fmaf(lo(w.y), lo(j.y), acc);
        acc = fmaf(hi(w.y), hi(j.y), acc);
        acc = fmaf(lo(w.z), lo(j.z), acc);
        acc = fmaf(hi(w.z), hi(j.z), acc);
        acc = fmaf(lo(w.w), lo(j.w), acc);
        acc = fmaf(hi(w.w), hi(j.w), acc);
    }
    acc = butterfly<P>(acc);
    if (lane == 0) g.zeta[row * kRank + rank] = tobf(acc);
}

// Block (4 rows, 64 columns, gate), HIP-GATE-UP-R64: 64-term chains in rank order, tanh, then (1 - F) h + G m.
template <int W, int P>
__device__ void gateUp(const Gate& g, int rows) {
    __shared__ float z[kUpRows][kRank];
    const int t = blockIdx.x * kUpRows + threadIdx.y, col = blockIdx.y * kUpCols + threadIdx.x;
    if (t < rows) z[threadIdx.y][threadIdx.x] = bf(g.zeta[static_cast<size_t>(t) * kRank + threadIdx.x]);
    uint4 f[kRank / 8], u[kRank / 8];
#pragma unroll
    for (int q = 0; q < kRank / 8; ++q) {
        f[q] = reinterpret_cast<const uint4*>(g.forget + static_cast<size_t>(col) * kRank)[q];
        u[q] = reinterpret_cast<const uint4*>(g.gain + static_cast<size_t>(col) * kRank)[q];
    }
    __syncthreads();
    if (t >= rows) return;
    const float* zt = z[threadIdx.y];
    float sf = 0.0f, sg = 0.0f;
#pragma unroll
    for (int q = 0; q < kRank / 8; ++q) {
        const uint32_t fw[4] = {f[q].x, f[q].y, f[q].z, f[q].w}, uw[4] = {u[q].x, u[q].y, u[q].z, u[q].w};
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float z0 = zt[8 * q + 2 * e], z1 = zt[8 * q + 2 * e + 1];
            sf = sf + lo(fw[e]) * z0;
            sg = sg + lo(uw[e]) * z0;
            sf = sf + hi(fw[e]) * z1;
            sg = sg + hi(uw[e]) * z1;
        }
    }
    const float forget = P == kTanhf ? tanhf(sf) : ztanh(sf), gain = P == kTanhf ? tanhf(sg) : ztanh(sg);
    const size_t at = static_cast<size_t>(t) * W + col;
    const float keep = 1.0f - forget;
    const float right = gain * g.m[at];
    // the baseline's four fp32 steps; a control fuses the last two
    g.out[at] = P == kFma ? fmaf(keep, g.h[at], right) : keep * g.h[at] + right;
}

}  // namespace ink2l

// Messages: grid (rows, count * 4 messages); banks bf16 [rows][W], queries f32 from the first core's rows.
#define INK2_MESSAGES(name, width, plant)                                                                           \
    extern "C" __global__ void __launch_bounds__(ink2l::kMsgThreads)                                               \
        name(const uint16_t* __restrict__ bank0, const uint16_t* __restrict__ bank1, const float* __restrict__ qm, \
             const float* __restrict__ kv, float* __restrict__ out, int rows, float scale, float eps) {            \
        __shared__ ink2l::MsgStage<width> st;                                                                      \
        ink2l::messages<width, plant>(bank0, bank1, qm, kv, out, rows, scale, eps, st);                            \
    }

// Gate down: grid (rows, 8 rank groups, gates); each block writes 8 of its row's 64 zeta.
#define INK2_GATE_DOWN(name, width, plant)                                                                          \
    extern "C" __global__ void __launch_bounds__(ink2l::kDownThreads) name(ink2l::Gates gs, float eps) {            \
        __shared__ ink2l::DownStage<width> st;                                                                     \
        ink2l::gateDown<width, plant>(gs.g[blockIdx.z], eps, st);                                                  \
    }

// Gate up: grid (row groups of 4, W / 64 column blocks, gates), blocks of 64 x 4; out may be h, read before written.
#define INK2_GATE_UP(name, width, plant)                                                                            \
    extern "C" __global__ void __launch_bounds__(ink2l::kUpCols * ink2l::kUpRows) name(ink2l::Gates gs, int rows) { \
        ink2l::gateUp<width, plant>(gs.g[blockIdx.z], rows);                                                       \
    }

// One width's kernels, each named with its suffix: messages, gate down and gate up, each served, then its controls.
#define INK2_LOOP(width, sfx)                                                               \
    INK2_MESSAGES(tf_ink2_loop_messages##sfx, width, ink2l::kServed)                        \
    INK2_MESSAGES(tf_ink2_loop_ctl_rsqrt##sfx, width, ink2l::kRsqrt)                        \
    INK2_MESSAGES(tf_ink2_loop_ctl_fma##sfx, width, ink2l::kFma)                            \
    INK2_MESSAGES(tf_ink2_loop_ctl_signed_zero##sfx, width, ink2l::kSignedZero)             \
    INK2_MESSAGES(tf_ink2_loop_ctl_serial_rho##sfx, width, ink2l::kSerialRho)               \
    INK2_MESSAGES(tf_ink2_loop_ctl_butterfly_desc##sfx, width, ink2l::kButterflyDesc)       \
    INK2_GATE_DOWN(tf_ink2_loop_gate_down##sfx, width, ink2l::kServed)                      \
    INK2_GATE_DOWN(tf_ink2_loop_ctl_down_serial_rho##sfx, width, ink2l::kSerialRho)         \
    INK2_GATE_DOWN(tf_ink2_loop_ctl_down_butterfly_desc##sfx, width, ink2l::kButterflyDesc) \
    INK2_GATE_DOWN(tf_ink2_loop_ctl_unfused_sq##sfx, width, ink2l::kUnfusedSq)              \
    INK2_GATE_UP(tf_ink2_loop_gate_up##sfx, width, ink2l::kServed)                          \
    INK2_GATE_UP(tf_ink2_loop_ctl_tanhf##sfx, width, ink2l::kTanhf)                         \
    INK2_GATE_UP(tf_ink2_loop_ctl_gate_fma##sfx, width, ink2l::kFma)

// Inkling 2 Mini's hidden 2,304 (no suffix) and tiny-inkling2-mini's 256 (_w256); hip_loop.zig refuses every other.
INK2_LOOP(2304, )
INK2_LOOP(256, _w256)

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
