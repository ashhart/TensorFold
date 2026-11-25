// Inkling 2 Mini's routed experts on HIP: w13 gate/up with the softcapped SwiGLU, w2 down, the ascending-id combine.
#include "inkling2/experts.hpp"

namespace ink2e {

// OCP E2M1 (FP4): sign, 2 exponent bits, 1 mantissa bit; code 8 is -0.
__device__ __forceinline__ float e2m1(uint32_t n) {
    const uint32_t s = (n & 8u) ? 0x80000000u : 0u;
    const uint32_t e = (n >> 1) & 3u, m = n & 1u;
    if (e == 0u) return f32(s | (m ? 0x3f000000u : 0u));
    return f32(s | ((e + 126u) << 23) | (m << 22));
}

// OCP E4M3FN, its subnormals exact; 0x7f and 0xff are NaN.
__device__ __forceinline__ float e4m3(uint32_t n) {
    const uint32_t s = (n & 0x80u) ? 0x80000000u : 0u;
    const uint32_t e = (n >> 3) & 15u, m = n & 7u;
    if (e == 15u && m == 7u) return f32(0x7fc00000u);
    if (e == 0u) return f32(s | __float_as_uint(static_cast<float>(m) * 0x1.0p-9f));
    return f32(s | ((e + 120u) << 23) | (m << 20));
}

// NVFP4's outer factor: bf16(amax) / bf16(denominator), one fp32 division.
__device__ __forceinline__ float phiOf(const float* amax, const float* den, int expert) {
    return roundbf(amax[expert]) / roundbf(den[0]);
}

constexpr int kWide = 256;        // threads a gate/up or down block: its 64 outputs, four threads an output
constexpr int kTile = 64;         // inputs a K tile
constexpr int kRun = 32;          // inputs one thread decodes a tile: 16 code bytes (NVFP4) or 32 (MXFP8)
constexpr int kBatch = 8;         // routes of one expert that one pass over its rows serves
constexpr int kRow = kTile + 2;   // bf16 an LDS row of decoded weights: 33 dwords, so lanes read distinct banks

// A block's LDS: the tile's decoded bf16 weights (gate/up: gate rows, then up), decode tables, x tiles, routes.
template <int R>
struct Stage {
    uint16_t w[R][kRow];
    uint16_t tab[256 * 8];  // NVFP4: bf16(E2M1 magnitude * E4M3 scale * phi), by scale code and magnitude code
    float e4[256];          // MXFP8: each E4M3 code's value
    float xs[kBatch][kTile];
    int list[kWide];
    int count;
    int seen;
};

// One thread's share of a tile in registers: its run's code words and scale bytes, and up to two x values.
struct Raw {
    uint32_t q[8];
    uint32_t sc;
    uint16_t x[2];
};

// The gate's and (Two) the up's chains over a decoded tile, interleaved: each acc = acc + w * x in input order.
template <bool Two>
__device__ __forceinline__ void chains(float& a, float& u, const uint16_t* wa, const uint16_t* wu, const float* xs,
                                       int cols) {
    if (cols == kTile) {
        const uint32_t* pa = reinterpret_cast<const uint32_t*>(wa);
        const uint32_t* pu = reinterpret_cast<const uint32_t*>(wu);
        const float2* px = reinterpret_cast<const float2*>(xs);
#pragma unroll
        for (int c = 0; c < kTile / 2; ++c) {
            const uint32_t va = pa[c], vu = Two ? pu[c] : 0u;  // low half: the even input
            const float2 xv = px[c];
            a = a + __uint_as_float(va << 16) * xv.x;
            if (Two) u = u + __uint_as_float(vu << 16) * xv.x;
            a = a + __uint_as_float(va & 0xffff0000u) * xv.y;
            if (Two) u = u + __uint_as_float(vu & 0xffff0000u) * xv.y;
        }
        return;
    }
    for (int c = 0; c < cols; ++c) {
        a = a + bf(wa[c]) * xs[c];
        if (Two) u = u + bf(wu[c]) * xs[c];
    }
}

// Block (x, route): 64 outputs for its expert's routes from this one on; tiles decode once for kBatch routes.
template <bool Nv, bool Gu>
__device__ __forceinline__ void project(const uint16_t* __restrict__ x, const uint8_t* __restrict__ w,
                                        const uint8_t* __restrict__ s, const float* __restrict__ amax,
                                        const float* __restrict__ den, const int* __restrict__ ids,
                                        uint16_t* __restrict__ y, int k, int n, int topk, int experts) {
    constexpr int per = Nv ? 16 : 32;  // inputs a scale
    constexpr int R = Gu ? 2 * kThreads : kThreads;
    __shared__ Stage<R> st;
    const int r = blockIdx.y, routes = gridDim.y, t = threadIdx.x;
    const int n0 = blockIdx.x * kThreads, o = t % kThreads, col = n0 + o, slot0 = t / kThreads;
    const int e = ids[r];
    if (e < 0 || e >= experts) {
        if (t < kThreads && col < n) y[static_cast<size_t>(r) * n + col] = kNan;
        return;
    }
    if (t == 0) st.seen = 0;
    __syncthreads();
    for (int i = t; i < r; i += kWide)
        if (ids[i] == e) st.seen = 1;
    __syncthreads();
    if (st.seen) return;  // an earlier route's block serves this expert
    const size_t row0 = static_cast<size_t>(e) * (Gu ? 2 * n : n) + static_cast<size_t>(Gu ? 2 * n0 : n0);
    const int held = min(R, (Gu ? 2 : 1) * (n - n0));  // rows this block decodes
    const int rb = Nv ? k / 2 : k, rs = k / per;       // global code and scale bytes a row
    const bool wide = (reinterpret_cast<uintptr_t>(w) & 15) == 0 && rb % 16 == 0;
    const float phi = Nv ? phiOf(amax, den, e) : 1.0f;
    // each weight's decode is a function of its code and scale (and the expert's phi): tabulated once a block
    if constexpr (Nv) {
        for (int i = t; i < 256 * 8; i += kWide) st.tab[i] = tobf(e2m1(i % 8) * e4m3(i / 8) * phi);
    } else {
        st.e4[t] = e4m3(t);
    }
    __syncthreads();
    const int lr = t / (kTile / kRun), q = t % (kTile / kRun);  // this thread's run: row lr, inputs 32q..
    const int wslot = Gu ? (lr & 1) * kThreads + (lr >> 1) : lr;
    for (int base = r; base < routes; base += kWide) {
        if (t == 0) st.count = 0;
        __syncthreads();
        if (base + t < routes && ids[base + t] == e) st.list[atomicAdd(&st.count, 1)] = base + t;
        __syncthreads();
        const int count = st.count;
        for (int b0 = 0; b0 < count; b0 += kBatch) {
            const int m = min(kBatch, count - b0);
            // the tile at c0 into registers: this thread's run and scales, and its two x values
            auto load = [&](int c0, Raw& raw) {
                const int cols = min(kTile, k - c0), len = min(kRun, cols - kRun * q);
                if (lr < held && len > 0) {
                    const uint8_t* src = w + (row0 + lr) * rb + (Nv ? (c0 + kRun * q) / 2 : c0 + kRun * q);
                    const int bytes = Nv ? len / 2 : len;
#pragma unroll
                    for (int u = 0; u < 2; ++u) {  // constant indexes into q[], so it stays in registers
                        if (16 * u + 16 > bytes) continue;
                        const uint4 v = wide ? *reinterpret_cast<const uint4*>(src + 16 * u)
                                             : make_uint4(*reinterpret_cast<const uint32_t*>(src + 16 * u),
                                                          *reinterpret_cast<const uint32_t*>(src + 16 * u + 4),
                                                          *reinterpret_cast<const uint32_t*>(src + 16 * u + 8),
                                                          *reinterpret_cast<const uint32_t*>(src + 16 * u + 12));
                        raw.q[4 * u] = v.x;
                        raw.q[4 * u + 1] = v.y;
                        raw.q[4 * u + 2] = v.z;
                        raw.q[4 * u + 3] = v.w;
                    }
                    if (bytes == 8) {  // an NVFP4 run of 16 inputs: a reduction that is not whole 32-input runs
                        const uint2 v = *reinterpret_cast<const uint2*>(src);
                        raw.q[0] = v.x;
                        raw.q[1] = v.y;
                    }
                    const uint8_t* ss = s + (row0 + lr) * rs + (c0 + kRun * q) / per;
                    raw.sc = Nv ? ss[0] | (len > 16 ? static_cast<uint32_t>(ss[1]) << 8 : 0u) : ss[0];
                }
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int b = slot0 + 4 * h;
                    if (b < m && o < cols) {
                        const int route = st.list[b0 + b];
                        raw.x[h] = x[static_cast<size_t>(Gu ? route / topk : route) * k + c0 + o];
                    }
                }
            };
            // the run's decoded bf16 weights and the x values into LDS, as today's rounding points give them
            auto put = [&](int c0, const Raw& raw) {
                const int cols = min(kTile, k - c0), len = min(kRun, cols - kRun * q);
                if (lr < held && len > 0) {
                    uint32_t* d = reinterpret_cast<uint32_t*>(&st.w[wslot][kRun * q]);
#pragma unroll
                    for (int j = 0; j < kRun; j += 2) {
                        if (j >= len) continue;  // a short run: the loop stays unrolled, so q[] indexes are constants
                        uint16_t h2[2];
#pragma unroll
                        for (int i = 0; i < 2; ++i) {
                            const int jj = j + i;
                            if constexpr (Nv) {
                                const uint32_t code = (raw.q[jj / 8] >> (4 * (jj % 8))) & 15u;  // low nibble: even
                                const uint32_t sc = (raw.sc >> (8 * (jj / 16))) & 255u;
                                // (-a) * b * phi is -(a * b * phi) exactly, so the sign only flips the bf16 bits
                                h2[i] = st.tab[sc * 8 + (code & 7u)] ^ static_cast<uint16_t>((code & 8u) << 12);
                            } else {
                                const uint32_t code = (raw.q[jj / 4] >> (8 * (jj % 4))) & 255u;
                                h2[i] = tobf(ldexpf(st.e4[code], static_cast<int>(raw.sc) - 127));
                            }
                        }
                        d[j / 2] = h2[0] | static_cast<uint32_t>(h2[1]) << 16;
                    }
                }
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int b = slot0 + 4 * h;
                    if (b < m && o < cols) st.xs[b][o] = bf(raw.x[h]);
                }
            };
            float acc[2] = {0.0f, 0.0f}, acc_u[2] = {0.0f, 0.0f};
            Raw cur{}, next{};
            load(0, cur);
            for (int c0 = 0; c0 < k; c0 += kTile) {
                const int cols = min(kTile, k - c0);
                put(c0, cur);
                if (c0 + kTile < k) load(c0 + kTile, next);
                __syncthreads();
                if (col < n)
                    #pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int b = slot0 + 4 * h;
                        if (b >= m) continue;
                        chains<Gu>(acc[h], acc_u[h], st.w[o], st.w[Gu ? kThreads + o : o], st.xs[b], cols);
                    }
                __syncthreads();
                cur = next;
            }
            if (col < n)
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int b = slot0 + 4 * h;
                    if (b >= m) continue;
                    uint16_t* out = y + static_cast<size_t>(st.list[b0 + b]) * n + col;
                    *out = Gu ? activation(roundbf(acc[h]), roundbf(acc_u[h])) : tobf(acc[h]);
                }
        }
        __syncthreads();
    }
}

}  // namespace ink2e

using namespace ink2e;

// Gate/up: block (x, route) holds 64 outputs n; w13 row 2n is the gate, 2n + 1 the up; an id past `experts` is NaN.
extern "C" __global__ void __launch_bounds__(kWide)
    tf_ink2_nvfp4_gu(const uint16_t* __restrict__ x, const uint8_t* __restrict__ w, const uint8_t* __restrict__ s,
                     const float* __restrict__ amax, const float* __restrict__ den, const int* __restrict__ ids,
                     uint16_t* __restrict__ y, int k, int n, int topk, int experts) {
    project<true, true>(x, w, s, amax, den, ids, y, k, n, topk, experts);
}

extern "C" __global__ void __launch_bounds__(kWide)
    tf_ink2_mxfp8_gu(const uint16_t* __restrict__ x, const uint8_t* __restrict__ w, const uint8_t* __restrict__ s,
                     const int* __restrict__ ids, uint16_t* __restrict__ y, int k, int n, int topk, int experts) {
    project<false, true>(x, w, s, nullptr, nullptr, ids, y, k, n, topk, experts);
}

// Down: block (x, route) holds 64 outputs of a route's own activation row; one bf16 round each.
extern "C" __global__ void __launch_bounds__(kWide)
    tf_ink2_nvfp4_down(const uint16_t* __restrict__ x, const uint8_t* __restrict__ w, const uint8_t* __restrict__ s,
                       const float* __restrict__ amax, const float* __restrict__ den, const int* __restrict__ ids,
                       uint16_t* __restrict__ y, int k, int n, int experts) {
    project<true, false>(x, w, s, amax, den, ids, y, k, n, 1, experts);
}

extern "C" __global__ void __launch_bounds__(kWide)
    tf_ink2_mxfp8_down(const uint16_t* __restrict__ x, const uint8_t* __restrict__ w, const uint8_t* __restrict__ s,
                       const int* __restrict__ ids, uint16_t* __restrict__ y, int k, int n, int experts) {
    project<false, false>(x, w, s, nullptr, nullptr, ids, y, k, n, 1, experts);
}

// Combine: block (x, token); from fp32 zero, sum + weight * down in ascending (id, slot), ids past `experts` skipped.
extern "C" __global__ void __launch_bounds__(kThreads)
    tf_ink2_combine(const uint16_t* __restrict__ down, const int* __restrict__ ids, const float* __restrict__ weights,
                    uint16_t* __restrict__ y, int n, int topk, int experts) {
    const int token = blockIdx.y, col = blockIdx.x * kThreads + threadIdx.x;
    if (col >= n) return;
    const int* id = ids + static_cast<size_t>(token) * topk;
    float sum = 0.0f;
    int last_id = -1, last_slot = -1;
    for (int step = 0; step < topk; ++step) {
        int best_id = experts, best_slot = -1;
        for (int slot = 0; slot < topk; ++slot) {
            const int e = id[slot];
            if (e < 0 || e >= experts) continue;
            const bool after = e > last_id || (e == last_id && slot > last_slot);
            if (after && (e < best_id || (e == best_id && slot < best_slot))) {
                best_id = e;
                best_slot = slot;
            }
        }
        if (best_slot < 0) break;
        const size_t r = static_cast<size_t>(token) * topk + best_slot;
        sum = sum + weights[r] * bf(down[r * n + col]);
        last_id = best_id;
        last_slot = best_slot;
    }
    y[static_cast<size_t>(token) * n + col] = tobf(sum);
}

// The contract's proof: both activation factors at each given bf16 input; `native` uses the device's tanhf and expf.
extern "C" __global__ void __launch_bounds__(256)
    tf_ink2_act_factors(const uint16_t* __restrict__ in, float* __restrict__ gate, float* __restrict__ up, int count,
                        int native) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= count) return;
    const float v = bf(in[i]);
    if (native) {
        gate[i] = 4.0f * tanhf(v / 4.0f) * (1.0f / (1.0f + expf(-v)));
        up[i] = 25.0f * tanhf(v / 25.0f);
    } else {
        gate[i] = gateFactor(v);
        up[i] = upFactor(v);
    }
}
