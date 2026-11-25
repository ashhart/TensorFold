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

// One thread's share of a tile in registers: its run's code words, each scale byte and x value in its own word.
struct Raw {
    uint32_t q[8];
    uint32_t s0, s1;
    uint32_t x[2];
};

// A workgroup barrier for LDS alone: the global loads of the next tile stay in flight across it.
__device__ __forceinline__ void ldsBarrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

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
template <bool Nv, bool Gu, bool Wide>
__device__ __forceinline__ void project(const uint16_t* __restrict__ x, const uint8_t* __restrict__ w,
                                        const uint8_t* __restrict__ s, const float* __restrict__ amax,
                                        const float* __restrict__ den, const int* __restrict__ ids,
                                        uint16_t* __restrict__ y, int k, int n, int topk, int experts,
                                        Stage<Gu ? 2 * kThreads : kThreads>& st) {
    constexpr int per = Nv ? 16 : 32;  // inputs a scale
    constexpr int R = Gu ? 2 * kThreads : kThreads;
    const int r = blockIdx.y, routes = gridDim.y, t = threadIdx.x;
    const int n0 = blockIdx.x * kThreads, o = t % kThreads, col = n0 + o, slot0 = t / kThreads;
    const int e = ids[r];
    if (e < 0 || e >= experts) {
        if (t < kThreads && col < n) y[static_cast<size_t>(r) * n + col] = kNan;
        return;
    }
    if (t == 0) st.seen = 0;
    ldsBarrier();
    for (int i = t; i < r; i += kWide)
        if (ids[i] == e) st.seen = 1;
    ldsBarrier();
    if (st.seen) return;  // an earlier route's block serves this expert
    const size_t row0 = static_cast<size_t>(e) * (Gu ? 2 * n : n) + static_cast<size_t>(Gu ? 2 * n0 : n0);
    const int held = min(R, (Gu ? 2 : 1) * (n - n0));  // rows this block decodes
    const int rb = Nv ? k / 2 : k, rs = k / per;       // global code and scale bytes a row
    const float phi = Nv ? phiOf(amax, den, e) : 1.0f;
    // each weight's decode is a function of its code and scale (and the expert's phi): tabulated once a block
    if constexpr (Nv) {
        for (int i = t; i < 256 * 8; i += kWide) st.tab[i] = tobf(e2m1(i % 8) * e4m3(i / 8) * phi);
    } else {
        st.e4[t] = e4m3(t);
    }
    ldsBarrier();
    const int lr = t / (kTile / kRun), q = t % (kTile / kRun);  // this thread's run: row lr, inputs 32q..
    const int wslot = Gu ? (lr & 1) * kThreads + (lr >> 1) : lr;
    for (int base = r; base < routes; base += kWide) {
        if (t == 0) st.count = 0;
        ldsBarrier();
        if (base + t < routes && ids[base + t] == e) st.list[atomicAdd(&st.count, 1)] = base + t;
        ldsBarrier();
        const int count = st.count;
        for (int b0 = 0; b0 < count; b0 += kBatch) {
            const int m = min(kBatch, count - b0);
            // the tile at c0 into registers: this thread's run and scales, and its two x values
            auto load = [&](int c0, Raw& raw) {
                // every load is unconditional, from an address that is always valid, so none waits before its use
                const int cols = min(kTile, k - c0), len = min(kRun, cols - kRun * q);
                const int in0 = len > 0 ? c0 + kRun * q : 0;   // a run past the tile reads its row's start
                const size_t row = row0 + min(lr, held - 1);  // a row past the block's reads its last one
                const uint8_t* src = w + row * rb + (Nv ? in0 / 2 : in0);
                if constexpr (Nv && !Wide) {  // rows 8-byte aligned: a 16-input run reads its 8 bytes twice
                    const uint2 a = *reinterpret_cast<const uint2*>(src);
                    const uint2 b = *reinterpret_cast<const uint2*>(len > 16 ? src + 8 : src);
                    raw.q[0] = a.x;
                    raw.q[1] = a.y;
                    raw.q[2] = b.x;
                    raw.q[3] = b.y;
                } else {
#pragma unroll
                    for (int u = 0; u < (Nv ? 1 : 2); ++u) {
                        const uint4 v = *reinterpret_cast<const uint4*>(src + 16 * u);
                        raw.q[4 * u] = v.x;
                        raw.q[4 * u + 1] = v.y;
                        raw.q[4 * u + 2] = v.z;
                        raw.q[4 * u + 3] = v.w;
                    }
                }
                const uint8_t* ss = s + row * rs + in0 / per;
                raw.s0 = ss[0];
                raw.s1 = Nv ? ss[len > 16 ? 1 : 0] : 0u;
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int b = slot0 + 4 * h;
                    const bool live = b < m && o < cols;
                    const int route = st.list[b0 + (live ? b : 0)];
                    raw.x[h] = x[static_cast<size_t>(Gu ? route / topk : route) * k + (live ? c0 + o : 0)];
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
                                const uint32_t sc = jj < 16 ? raw.s0 : raw.s1;
                                // (-a) * b * phi is -(a * b * phi) exactly, so the sign only flips the bf16 bits
                                h2[i] = st.tab[sc * 8 + (code & 7u)] ^ static_cast<uint16_t>((code & 8u) << 12);
                            } else {
                                const uint32_t code = (raw.q[jj / 4] >> (8 * (jj % 4))) & 255u;
                                h2[i] = tobf(ldexpf(st.e4[code], static_cast<int>(raw.s0) - 127));
                            }
                        }
                        d[j / 2] = h2[0] | static_cast<uint32_t>(h2[1]) << 16;
                    }
                }
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int b = slot0 + 4 * h;
                    if (b < m && o < cols) st.xs[b][o] = bf(static_cast<uint16_t>(raw.x[h]));
                }
            };
            float acc[2] = {0.0f, 0.0f}, acc_u[2] = {0.0f, 0.0f};
            // two tiles in flight: tile t + 2 loads while tile t is decoded and chained; past the end, the last again
            const int last = (k - 1) / kTile * kTile;
            // tile c0's runs decode, then the same registers load tile `ahead`: two tiles stay in flight
            auto step = [&](int c0, Raw& raw, int ahead) {
                put(c0, raw);
                load(ahead, raw);
                ldsBarrier();
                if (col < n)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int b = slot0 + 4 * h;
                        if (b >= m) continue;
                        const uint16_t* wu = st.w[Gu ? kThreads + o : o];
                        chains<Gu>(acc[h], acc_u[h], st.w[o], wu, st.xs[b], min(kTile, k - c0));
                    }
                ldsBarrier();
            };
            Raw ra{}, rb{};
            load(0, ra);
            load(min(kTile, last), rb);
            for (int c0 = 0; c0 < k; c0 += 2 * kTile) {
                step(c0, ra, min(c0 + 2 * kTile, last));
                if (c0 + kTile >= k) break;
                step(c0 + kTile, rb, min(c0 + 3 * kTile, last));
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
        ldsBarrier();
    }
}

// NVFP4 rows load 16 bytes at a time when every row starts 16-byte aligned (MXFP8's always do; the launcher checks).
__device__ __forceinline__ bool wide16(const uint8_t* w, int k) {
    return (reinterpret_cast<uintptr_t>(w) & 15) == 0 && (k / 2) % 16 == 0;
}

}  // namespace ink2e

using namespace ink2e;

// Gate/up: block (x, route) holds 64 outputs n; w13 row 2n is the gate, 2n + 1 the up; an id past `experts` is NaN.
extern "C" __global__ void __launch_bounds__(kWide)
    tf_ink2_nvfp4_gu(const uint16_t* __restrict__ x, const uint8_t* __restrict__ w, const uint8_t* __restrict__ s,
                     const float* __restrict__ amax, const float* __restrict__ den, const int* __restrict__ ids,
                     uint16_t* __restrict__ y, int k, int n, int topk, int experts) {
    __shared__ Stage<2 * kThreads> st;  // one LDS block for both load widths
    if (wide16(w, k)) project<true, true, true>(x, w, s, amax, den, ids, y, k, n, topk, experts, st);
    else project<true, true, false>(x, w, s, amax, den, ids, y, k, n, topk, experts, st);
}

extern "C" __global__ void __launch_bounds__(kWide)
    tf_ink2_mxfp8_gu(const uint16_t* __restrict__ x, const uint8_t* __restrict__ w, const uint8_t* __restrict__ s,
                     const int* __restrict__ ids, uint16_t* __restrict__ y, int k, int n, int topk, int experts) {
    __shared__ Stage<2 * kThreads> st;
    project<false, true, true>(x, w, s, nullptr, nullptr, ids, y, k, n, topk, experts, st);
}

// Down: block (x, route) holds 64 outputs of a route's own activation row; one bf16 round each.
extern "C" __global__ void __launch_bounds__(kWide)
    tf_ink2_nvfp4_down(const uint16_t* __restrict__ x, const uint8_t* __restrict__ w, const uint8_t* __restrict__ s,
                       const float* __restrict__ amax, const float* __restrict__ den, const int* __restrict__ ids,
                       uint16_t* __restrict__ y, int k, int n, int experts) {
    __shared__ Stage<kThreads> st;  // one LDS block for both load widths
    if (wide16(w, k)) project<true, false, true>(x, w, s, amax, den, ids, y, k, n, 1, experts, st);
    else project<true, false, false>(x, w, s, amax, den, ids, y, k, n, 1, experts, st);
}

extern "C" __global__ void __launch_bounds__(kWide)
    tf_ink2_mxfp8_down(const uint16_t* __restrict__ x, const uint8_t* __restrict__ w, const uint8_t* __restrict__ s,
                       const int* __restrict__ ids, uint16_t* __restrict__ y, int k, int n, int experts) {
    __shared__ Stage<kThreads> st;
    project<false, false, true>(x, w, s, nullptr, nullptr, ids, y, k, n, 1, experts, st);
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
