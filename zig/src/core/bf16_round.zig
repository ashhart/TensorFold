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
    const out = std.math.copysign(t * step, v); // an underflow to zero keeps v's sign, as bits() does
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
    // the largest, under half a step past it
    try std.testing.expectEqual(@as(f64, -255 * 0x1p120), bf16(-255 * 0x1p120 - 0x1p111));
    try std.testing.expect(std.math.isNan(bf16(std.math.nan(f64))));
    var prng = std.Random.DefaultPrng.init(5);
    for (0..100000) |_| {
        const u = prng.random().int(u32);
        const v: f32 = @bitCast(u);
        if (!std.math.isFinite(v)) continue;
        try std.testing.expectEqual(bits(v), bitsOf(bf16(v))); // bits: a zero's sign counts
    }
}

/// bf16(v)'s bits, for tests that must tell -0 from +0 (float equality does not).
fn bitsOf(v: f64) u16 {
    return bits(@floatCast(v));
}

test "raw bits at the edges: signed zeros from underflow, subnormal ties, the normal boundary, overflow" {
    const cases = [_]struct { f64, u16 }{
        .{ -0x1p-135, 0x8000 }, // underflow to zero keeps the sign
        .{ 0x1p-135, 0x0000 },
        .{ -0.0, 0x8000 },
        .{ -0x1p-134, 0x8000 }, // half the least subnormal: a tie, to the even zero
        .{ -0x1p-134 - 0x1p-160, 0x8001 }, // just past the tie
        .{ 0x1p-133, 0x0001 }, // the least subnormal
        .{ -3 * 0x1p-134, 0x8002 }, // a subnormal tie, to the even 2
        .{ 0x1p-126 - 0x1p-134, 0x0080 }, // the largest subnormal's tie, up to the least normal
        .{ -(1 + 0x1p-8), 0xbf80 }, // a tie, to the even -1
        .{ -(1 + 3 * 0x1p-8), 0xbf82 },
        .{ 255 * 0x1p120, 0x7f7f }, // the largest finite
        .{ -(255 * 0x1p120 + 0x1p119), 0xff80 }, // its tie with 2^128: to the even -Inf
        .{ -(255 * 0x1p120 + 0x1p119 - 0x1p90), 0xff7f },
        .{ 0x1p300, 0x7f80 },
        .{ -std.math.inf(f64), 0xff80 },
    };
    for (cases) |cs| try std.testing.expectEqual(cs[1], bitsOf(bf16(cs[0])));
    try std.testing.expect(std.math.isNan(value(bitsOf(bf16(-std.math.nan(f64))))));
    try std.testing.expectEqual(@as(u16, 0x8000), bits(-0x1p-135)); // fp32's path agrees
}
