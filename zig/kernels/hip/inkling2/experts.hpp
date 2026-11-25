#pragma once

// Inkling 2 Mini's routed experts on HIP: weights decoded in registers, fp32 sums in input order, Zig's f32 math.

#include <hip/hip_runtime.h>
#include <cstdint>

namespace ink2e {

constexpr int kThreads = 64;      // outputs a block: one thread an output
constexpr int kMaxK = 2304;       // the longest reduction (Mini's hidden width), staged in LDS
constexpr int kMaxTopk = 16;      // route slots a token the combine orders
constexpr uint16_t kNan = 0x7fc0;  // every NaN output

__device__ __forceinline__ float bf(uint16_t u) { return __uint_as_float(static_cast<uint32_t>(u) << 16); }

// Round to nearest even, as bf16_round.bits; NaN to kNan.
__device__ __forceinline__ uint16_t tobf(float f) {
    if (f != f) return kNan;
    const uint32_t u = __float_as_uint(f);
    return static_cast<uint16_t>((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

__device__ __forceinline__ float roundbf(float f) { return bf(tobf(f)); }

__device__ __forceinline__ float f32(uint32_t bits) { return __uint_as_float(bits); }

// Zig 0.17 std.math.ldexp for f32, integer step for integer step (its subnormal rounding included).
__device__ __forceinline__ float zldexp(float x, int n) {
    const uint32_t repr = __float_as_uint(x);
    const uint32_t sign = repr & 0x80000000u;
    if (x != x || (repr & 0x7fffffffu) == 0x7f800000u) return x;
    int exponent = static_cast<int>((repr << 1) >> 24);
    if (exponent == 0) exponent += 8 - __clz(static_cast<int>(repr << 1));
    if (n >= 0) {
        if (n > 254 - exponent) return f32(0x7f800000u | sign);
        if (exponent + n <= 0) return f32((repr << n) | sign);
        if (exponent <= 0) {
            uint32_t result = static_cast<uint32_t>(n + exponent) << 23;
            result |= (repr << (1 - exponent)) & 0x7fffffu;
            return f32(result | sign);
        }
        return f32(repr + (static_cast<uint32_t>(n) << 23));
    }
    if (n <= -exponent) {
        if (n < -(23 + exponent)) return f32(sign);
        const int shift = min(-n, -(exponent + n) + 1);
        const bool exact_tie = __builtin_ctz(repr) == shift - 1;
        uint32_t result = (repr & 0x7fffffu) | (static_cast<uint32_t>(exponent > 0) << 23);
        result >>= shift - 1;
        result = ((result + 1u) >> 1) & ~static_cast<uint32_t>(exact_tie);
        return f32(result | sign);
    }
    return f32(repr - (static_cast<uint32_t>(-n) << 23));
}

// Zig 0.17's f32 @exp: compiler-rt's exp_f32 (musl's expf), operation for operation.
__device__ __forceinline__ float zexp(float x) {
    const float half[2] = {0.5f, -0.5f};
    const float ln2hi = f32(0x3f317200u), ln2lo = f32(0x35bfbe8eu), invln2 = f32(0x3fb8aa3bu);
    const float p1 = f32(0x3e2aaa8fu), p2 = f32(0xbb355215u);
    uint32_t hx = __float_as_uint(x);
    const int sign = static_cast<int>(hx >> 31);
    hx &= 0x7fffffffu;
    if (x != x) return x;
    if (hx >= 0x42aeac50u) {
        if (hx > 0x7f800000u) return x;
        if (hx >= 0x42b17218u && sign == 0) return x * 0x1.0p127f;
        if (sign != 0 && hx >= 0x42cff1b5u) return 0.0f;
    }
    int k;
    float hi, lo;
    if (hx > 0x3eb17218u) {
        k = hx > 0x3f851592u ? static_cast<int>(invln2 * x + half[sign]) : 1 - sign - sign;
        const float fk = static_cast<float>(k);
        hi = x - fk * ln2hi;
        lo = fk * ln2lo;
        x = hi - lo;
    } else if (hx > 0x39000000u) {
        k = 0;
        hi = x;
        lo = 0.0f;
    } else {
        return 1.0f + x;
    }
    const float xx = x * x;
    const float c = x - xx * (p1 + xx * p2);
    const float y = 1.0f + (x * c / (2.0f - c) - lo + hi);
    return k == 0 ? y : zldexp(y, k);
}

// Zig 0.17 std.math.expm1 for f32 (musl's expm1f), operation for operation.
__device__ __forceinline__ float zexpm1(float x) {
    const float o_threshold = f32(0x42b17180u), ln2_hi = f32(0x3f317180u), ln2_lo = f32(0x3717f7d1u);
    const float invln2 = f32(0x3fb8aa3bu), q1 = f32(0xbd088868u), q2 = f32(0x3acf3010u);
    if (x != x) return f32(0x7fc00000u);
    const uint32_t ux = __float_as_uint(x);
    const uint32_t hx = ux & 0x7fffffffu;
    const uint32_t sign = ux >> 31;
    if (ux == 0xff800000u) return -1.0f;
    if (hx >= 0x4195b844u) {
        if (hx > 0x7f800000u) return x;
        if (sign != 0) return -1.0f;
        if (x > o_threshold) return x * 0x1.0p127f;
    }
    float hi, lo, c = 0.0f;
    int k;
    if (hx > 0x3eb17218u) {
        if (hx < 0x3f851592u) {
            hi = sign == 0 ? x - ln2_hi : x + ln2_hi;
            lo = sign == 0 ? ln2_lo : -ln2_lo;
            k = sign == 0 ? 1 : -1;
        } else {
            float kf = invln2 * x;
            kf = sign != 0 ? kf - 0.5f : kf + 0.5f;
            k = static_cast<int>(kf);
            const float t = static_cast<float>(k);
            hi = x - t * ln2_hi;
            lo = t * ln2_lo;
        }
        x = hi - lo;
        c = (hi - x) - lo;
    } else if (hx < 0x33000000u) {
        return x;
    } else {
        k = 0;
    }
    const float hfx = 0.5f * x;
    const float hxs = x * hfx;
    const float r1 = 1.0f + hxs * (q1 + hxs * q2);
    const float t = 3.0f - r1 * hfx;
    float e = hxs * ((r1 - t) / (6.0f - x * t));
    if (k == 0) return x - (x * e - hxs);
    e = x * (e - c) - c;
    e -= hxs;
    if (k == -1) return 0.5f * (x - e) - 0.5f;
    if (k == 1) return x < -0.25f ? -2.0f * (e - (x + 0.5f)) : 1.0f + 2.0f * (x - e);
    const float twopk = f32(static_cast<uint32_t>((0x7f + k) << 23));
    if (k < 0 || k > 56) {
        float y = x - e + 1.0f;
        y = k == 128 ? y * 2.0f * 0x1.0p127f : y * twopk;
        return y - 1.0f;
    }
    const float uf = f32(static_cast<uint32_t>(0x7f - k) << 23);
    return k < 23 ? (x - e + (1.0f - uf)) * twopk : (x - (e + uf) + 1.0f) * twopk;
}

// Zig 0.17 std.math.tanh for f32 (musl's tanhf), operation for operation.
__device__ __forceinline__ float ztanh(float x) {
    const uint32_t u = __float_as_uint(x);
    const uint32_t ux = u & 0x7fffffffu;
    const float ax = f32(ux);
    float t;
    if (ux > 0x3f0c9f54u) {
        if (ux > 0x41200000u) {
            t = 1.0f + 0.0f / x;
        } else {
            t = zexpm1(2.0f * ax);
            t = 1.0f - 2.0f / (t + 2.0f);
        }
    } else if (ux > 0x3e82c578u) {
        t = zexpm1(2.0f * ax);
        t = t / (t + 2.0f);
    } else if (ux >= 0x00800000u) {
        t = zexpm1(-2.0f * ax);
        t = -t / (t + 2.0f);
    } else {
        t = ax;
    }
    return (u >> 31) != 0 ? -t : t;
}

// The activation's gate factor softcap4(g) * sigmoid(g), as loop_model.expertActivation's first product.
__device__ __forceinline__ float gateFactor(float g) {
    const float sigmoid = 1.0f / (1.0f + zexp(-g));
    return 4.0f * ztanh(g / 4.0f) * sigmoid;
}

// The activation's up factor softcap25(u).
__device__ __forceinline__ float upFactor(float u) { return 25.0f * ztanh(u / 25.0f); }

// bf16(softcap4(g) * sigmoid(g) * softcap25(u)) on the bf16-rounded projections, left to right in f32.
__device__ __forceinline__ uint16_t activation(float g, float u) { return tobf(gateFactor(g) * upFactor(u)); }

}  // namespace ink2e
