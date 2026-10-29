//! Independent FP64 scalar references for the quantized Metal operator gate, never used for inference.
const std = @import("std");

pub fn fp8(code: u8) f64 {
    const negative = code >= 128;
    const bits = code % 128;
    const exponent = bits / 8;
    const fraction = bits % 8;
    if (bits == 127) return std.math.nan(f64);
    var value: f64 = @floatFromInt(if (exponent == 0) fraction else 8 + @as(u16, fraction));
    var shift: i32 = if (exponent == 0) -9 else @as(i32, exponent) - 10;
    while (shift < 0) : (shift += 1) value /= 2;
    while (shift > 0) : (shift -= 1) value *= 2;
    return if (negative) -value else value;
}

pub fn fp4(code: u8) f64 {
    const bits = code % 8;
    const value: f64 = switch (bits) {
        0 => 0,
        1 => 0.5,
        2 => 1,
        3 => 1.5,
        4 => 2,
        5 => 3,
        6 => 4,
        7 => 6,
        else => unreachable,
    };
    return if (code >= 8) -value else value;
}

pub fn fromBf16(bits: u16) f64 {
    return @as(f32, @bitCast(@as(u32, bits) << 16));
}

/// Nearest BF16 by neighboring representable values, independent of the production integer rounding trick.
pub fn roundBf16(x: f64) f64 {
    const f: f32 = @floatCast(x);
    const bits: u32 = @bitCast(f);
    const low: u16 = @truncate(bits >> 16);
    const hi = low +% 1;
    const a = fromBf16(low);
    const b = fromBf16(hi);
    const da = @abs(@as(f64, f) - a);
    const db = @abs(@as(f64, f) - b);
    return if (da < db or (da == db and low % 2 == 0)) a else b;
}

pub fn projection(codes: []const u8, scales: []const u8, global: f32, n: usize, k: usize, x: []align(1) const u16) f64 {
    var sum: f64 = 0;
    for (0..k) |i| {
        const byte = codes[n * (k / 2) + i / 2];
        const code = if (i % 2 == 0) byte % 16 else byte / 16;
        sum += fp4(code) * fp8(scales[n * (k / 16) + i / 16]) * fromBf16(x[i]);
    }
    return sum * global;
}

pub fn swiglu(gate: f64, up: f64) f64 {
    const g = roundBf16(gate);
    return roundBf16(g / (1 + @exp(-g)) * roundBf16(up));
}
