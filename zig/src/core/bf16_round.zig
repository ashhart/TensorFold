//! bf16 rounding of float64 values, for float64 references of kernels that round to bf16 between their steps.
const std = @import("std");

/// v to the nearest bf16, ties to even, in one rounding (not through fp32); NaN and infinities pass, overflow is Inf.
pub fn bf16(v: f64) f64 {
    if (!std.math.isFinite(v) or v == 0) return v;
    const e = @max(std.math.frexp(v).exponent, -125); // |v| in [2^(e-1), 2^e); below 2^-126 the subnormal step
    const step = std.math.ldexp(@as(f64, 1), e - 8);
    const q = v / step; // a power of two apart: exact
    var t = @floor(q);
    const d = q - t;
    if (d > 0.5 or (d == 0.5 and @mod(t, 2) != 0)) t += 1;
    const out = t * step;
    return if (@abs(out) >= 0x1p128) std.math.copysign(std.math.inf(f64), v) else out;
}

/// fp32 to bf16 bits, ties to even, as __float2bfloat16_rn rounds (NaN stays NaN).
pub fn bits(v: f32) u16 {
    const u: u32 = @bitCast(v);
    if (std.math.isNan(v)) return @truncate((u >> 16) | 0x40);
    return @truncate((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}

/// The fp32 value of bf16 bits.
pub fn value(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

/// fp32 rounded to bf16 and back: a kernel's `__bfloat162float(__float2bfloat16_rn(v))`.
pub fn round32(v: f32) f32 {
    return value(bits(v));
}

test "ties to even, the subnormal step, overflow, and fp32 values rounded once as the fp32 path rounds them" {
    try std.testing.expectEqual(@as(f64, 1), bf16(1 + 0x1p-8)); // a tie, to the even 1
    try std.testing.expectEqual(@as(f64, 1 + 0x1p-6), bf16(1 + 3 * 0x1p-8)); // a tie, to the even 1 + 2^-6
    try std.testing.expectEqual(@as(f64, 1 + 0x1p-7), bf16(1 + 0x1p-8 + 0x1p-40)); // just past the tie
    try std.testing.expectEqual(@as(f64, 0x1p-133), bf16(0x1p-133 * 1.4));
    try std.testing.expectEqual(@as(f64, 0), bf16(0x1p-135));
    try std.testing.expect(std.math.isInf(bf16(0x1p128 * 0.9999)));
    try std.testing.expectEqual(@as(f64, -255 * 0x1p120), bf16(-255 * 0x1p120 - 0x1p111)); // the largest, under half a step past it
    try std.testing.expect(std.math.isNan(bf16(std.math.nan(f64))));
    var prng = std.Random.DefaultPrng.init(5);
    for (0..100000) |_| {
        const u = prng.random().int(u32);
        const v: f32 = @bitCast(u);
        if (!std.math.isFinite(v)) continue;
        const want: f64 = round32(v);
        try std.testing.expectEqual(want, bf16(v));
    }
}
