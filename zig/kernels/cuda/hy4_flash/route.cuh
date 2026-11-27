// LatentMoE top-16 sigmoid + bias router (`512` experts, bit-equal to `tf_moe_topk_sigmoid_score_warp`).
#pragma once

#include <stdint.h>

namespace tf_hy4_route {

constexpr unsigned kFull = 0xffffffffu;
constexpr int kExperts = 512;
constexpr int kTop = 16;

// Planted contract faults (hy4_route_controls.cuh); production is kNone.
enum Fault { kNone, kHighTies, kDistinct };

// Branch-free `tf_glue::tf_exp` on `x` clamped to `[-104, 104]`, bit-identical to `tf_exp`.
__device__ __forceinline__ float exp_bf(float x) {
    const float xc = fminf(fmaxf(x, -104.0f), 104.0f);
    const float j = __fsub_rn(__fmaf_rn(xc, 0x1.715476p+0f, 0x1.8p+23f), 0x1.8p+23f);
    float f = __fmaf_rn(j, -0x1.62e4p-1f, xc);
    f = __fmaf_rn(j, -0x1.7f7d1cp-20f, f);
    float p = 0x1.a01a02p-13f;
    p = __fmaf_rn(p, f, 0x1.6c16c2p-10f);
    p = __fmaf_rn(p, f, 0x1.111112p-7f);
    p = __fmaf_rn(p, f, 0x1.555556p-5f);
    p = __fmaf_rn(p, f, 0x1.555556p-3f);
    p = __fmaf_rn(p, f, 0.5f);
    p = __fmaf_rn(p, f, 1.0f);
    p = __fmaf_rn(p, f, 1.0f);
    const int i = static_cast<int>(j);
    const float a = __int_as_float(i > 0 ? 0x7f000000 : 0x02000000);
    const float b = __int_as_float(i > 0 ? i << 23 : (i + 250) << 23);
    float r = __fmul_rn(__fmul_rn(p, a), b);
    r = x < -104.0f ? 0.0f : r;
    r = x > 104.0f ? __int_as_float(0x7f800000) : r;
    return x != x ? x : r;
}

// Order-keeping uint32 of a choice: NaN (1) below -inf (0x007fffff), -0 tied with +0 as the float compare ties them.
__device__ __forceinline__ uint32_t okey(float ch) {
    if (ch != ch) return 1u;
    uint32_t b = __float_as_uint(ch);
    if ((b << 1) == 0u) b = 0u;
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
constexpr uint32_t kNegInfKey = 0x007fffffu;  // okey(-inf)

typedef unsigned long long u64;
// (key << 32 | ~id): the u64 order is (choice desc, id asc); kHighTies drops the complement (ties to the higher id).
template <Fault F>
__device__ __forceinline__ u64 pack_key(float ch, int id) {
    const uint32_t lo = F == kHighTies ? static_cast<uint32_t>(id) : static_cast<uint32_t>(~id);
    return (static_cast<u64>(okey(ch)) << 32) | lo;
}
__device__ __forceinline__ void cas(u64& a, u64& b) {  // a <- max, b <- min
    const bool s = a > b;
    const u64 h = s ? a : b, l = s ? b : a;
    a = h, b = l;
}
// 16 registers sorted descending (bitonic network, 80 compare-exchanges, every index a compile-time constant).
__device__ __forceinline__ void sort16(u64 (&c)[16]) {
#pragma unroll
    for (int k = 2; k <= 16; k <<= 1)
#pragma unroll
        for (int j = k >> 1; j > 0; j >>= 1)
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                const int l = i ^ j;
                if (l > i) {
                    if ((i & k) == 0) cas(c[i], c[l]);
                    else cas(c[l], c[i]);
                }
            }
}
// The 16 largest of this lane's and lane ^ m's sorted lists, sorted: max(c[i], o[15 - i]) is bitonic and holds them.
__device__ __forceinline__ void merge16(u64 (&c)[16], int m) {
    u64 o[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) o[i] = __shfl_xor_sync(kFull, c[i], m);
#pragma unroll
    for (int i = 0; i < 16; ++i) c[i] = c[i] > o[15 - i] ? c[i] : o[15 - i];
#pragma unroll
    for (int j = 8; j > 0; j >>= 1)
#pragma unroll
        for (int i = 0; i < 16; ++i)
            if ((i & j) == 0) cas(c[i], c[i + j]);
}
// Pick `lane` (lanes < 16) of the warp's top 16 of 512 packed keys; past the choices above -inf, expert 0.
template <Fault F, bool HIGH>
__device__ __forceinline__ int warp_top16(u64 (&c)[16], int lane) {
    sort16(c);
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) merge16(c, m);
    u64 p = 0;
#pragma unroll
    for (int j = 0; j < 16; ++j)
        if (j == lane) p = c[j];
    const uint32_t lo = static_cast<uint32_t>(p);
    const int id = static_cast<int>(HIGH ? lo : ~lo);
    if (F == kDistinct) return id;
    return static_cast<uint32_t>(p >> 32) > kNegInfKey ? id : 0;
}

// The whole row: block r of <<<R, 128>>>; IDS int32 [R, 16] and W fp32 [R, 16] in pick order.
template <Fault F>
__device__ __forceinline__ void route_m4(const float* __restrict__ LOG, const float* __restrict__ BIAS, int R,
                                         float scale, int* __restrict__ IDS, float* __restrict__ W) {
    __shared__ float scs[kExperts];
    __shared__ u64 cs[kExperts];
    const int t = threadIdx.x, r = blockIdx.x;
    if (r >= R) return;  // block-uniform
    const float* lg = LOG + static_cast<size_t>(r) * kExperts;
    float x[4], b[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) x[j] = __ldg(lg + t + 128 * j), b[j] = __ldg(BIAS + t + 128 * j);
#pragma unroll
    for (int j = 0; j < 4; ++j) x[j] = __fadd_rn(1.0f, exp_bf(-x[j]));  // all exps issue before the divides
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float sc = __fdiv_rn(1.0f, x[j]);
        scs[t + 128 * j] = sc;
        cs[t + 128 * j] = pack_key<F>(__fadd_rn(sc, b[j]), t + 128 * j);
    }
    __syncthreads();
    if (t >= 32) return;
    u64 c[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) c[j] = cs[t + 32 * j];
    const int id = warp_top16<F, F == kHighTies>(c, t);
    const float s = t < kTop ? scs[id] : 0.f;
    float sum = 0.f;
#pragma unroll
    for (int k = 0; k < kTop; ++k) sum = __fadd_rn(sum, __shfl_sync(kFull, s, k));
    const float den = __fadd_rn(sum, 1e-20f);
    if (t < kTop) {
        IDS[static_cast<size_t>(r) * kTop + t] = id;
        W[static_cast<size_t>(r) * kTop + t] = __fmul_rn(__fdiv_rn(s, den), scale);
    }
}

// Single-row fused top-16 + grouped MoE plan (bit-equal to `tf_hy4_route_m4` + `plan_kernel` at `P=16, E=512, T=16`).
__device__ __forceinline__ void route_m4_plan1(const float* __restrict__ LOG, const float* __restrict__ BIAS, float scale,
                                               int* __restrict__ IDS, float* __restrict__ W, int* __restrict__ MEMBERS,
                                               int* __restrict__ ITEMS, int* __restrict__ COUNTS) {
    __shared__ float scs[kExperts];
    __shared__ u64 cs[kExperts];
    const int t = threadIdx.x;
    float x[4], b[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) x[j] = __ldg(LOG + t + 128 * j), b[j] = __ldg(BIAS + t + 128 * j);
#pragma unroll
    for (int j = 0; j < 4; ++j) x[j] = __fadd_rn(1.0f, exp_bf(-x[j]));
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float sc = __fdiv_rn(1.0f, x[j]);
        scs[t + 128 * j] = sc;
        cs[t + 128 * j] = pack_key<kNone>(__fadd_rn(sc, b[j]), t + 128 * j);
    }
    __syncthreads();
    if (t >= 32) return;
    u64 c[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) c[j] = cs[t + 32 * j];
    const int id = warp_top16<kNone, false>(c, t);
    const float s = t < kTop ? scs[id] : 0.f;
    float sum = 0.f;
#pragma unroll
    for (int k = 0; k < kTop; ++k) sum = __fadd_rn(sum, __shfl_sync(kFull, s, k));
    const float den = __fadd_rn(sum, 1e-20f);
    if (t < kTop) {
        IDS[t] = id;
        W[t] = __fmul_rn(__fdiv_rn(s, den), scale);
        unsigned kv = (static_cast<unsigned>(id) << 4) | static_cast<unsigned>(t);
#pragma unroll
        for (int k = 2; k <= 16; k <<= 1) {
#pragma unroll
            for (int j = k >> 1; j > 0; j >>= 1) {
                const unsigned other = __shfl_xor_sync(0xFFFFu, kv, j);
                const bool asc = ((t & k) == 0);
                const bool take_hi = ((t & j) != 0) ? asc : !asc;
                kv = take_hi ? max(kv, other) : min(kv, other);
            }
        }
        const int my_id = static_cast<int>(kv >> 4);
        MEMBERS[t] = static_cast<int>(kv & 15u);
        const int prev_id = __shfl_up_sync(0xFFFFu, my_id, 1);
        const bool is_start = (t == 0) || (my_id != prev_id);
        const unsigned mask = __ballot_sync(0xFFFFu, is_start);
        const int n_items = __popc(mask);
        if (is_start) {
            const int idx = __popc(mask & ((1u << t) - 1u));
            const unsigned rest = mask >> (t + 1);
            const int next_t = rest ? (t + 1 + __ffs(rest) - 1) : 16;
            ITEMS[3 * idx + 0] = my_id;
            ITEMS[3 * idx + 1] = t;
            ITEMS[3 * idx + 2] = next_t - t;
        }
        if (t == 0) {
            COUNTS[0] = n_items;
            COUNTS[1] = n_items;
        }
    }
}

}  // namespace tf_hy4_route
