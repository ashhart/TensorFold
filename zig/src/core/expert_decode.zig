//! Stored routed-expert weights decoded on the host (Inkling 2 Mini): NVFP4 and MXFP8 as stored, no requantizing.
const std = @import("std");

/// An E2M1 code (low 4 bits; bit 3 the sign): 0, 0.5, 1, 1.5, 2, 3, 4, 6.
pub fn e2m1(code: u8) f64 {
    const magnitude = [8]f64{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };
    const v = magnitude[code & 7];
    return if (code & 8 != 0) -v else v;
}

/// An E4M3 (fn) code; 0x7F and 0xFF are NaN.
pub fn e4m3(code: u8) f64 {
    const e: i32 = (code >> 3) & 15;
    const m: f64 = @floatFromInt(code & 7);
    const v: f64 = if (e == 15 and code & 7 == 7) std.math.nan(f64) else if (e == 0) m * 0x1p-9 else (8 + m) * std.math.ldexp(@as(f64, 1), e - 10);
    return if (code & 0x80 != 0) -v else v;
}

/// An E8M0 scale: 2^(code - 127); 255 is NaN.
pub fn e8m0(code: u8) f64 {
    if (code == 255) return std.math.nan(f64);
    return std.math.ldexp(@as(f64, 1), @as(i32, code) - 127);
}

/// Input i's E2M1 code of a row of packed bytes, low nibble first (the order the CUDA packers assume).
pub fn nibble(row: []const u8, i: usize) u8 {
    const b = row[i / 2];
    return if (i % 2 == 0) b & 15 else b >> 4;
}

/// NVFP4 weight (row, i) of one expert's [n, k/2] codes and [n, k/16] E4M3 scales, times its factor: exact in f64.
pub fn nvfp4(codes: []const u8, scales: []const u8, k: usize, row: usize, i: usize, factor: f64) f64 {
    return e2m1(nibble(codes[row * (k / 2) ..][0 .. k / 2], i)) * e4m3(scales[row * (k / 16) + i / 16]) * factor;
}

/// The same weight as the decode kernels form it in fp32: code x block scale (exact), then one rounding by the factor.
pub fn nvfp4F32(codes: []const u8, scales: []const u8, k: usize, row: usize, i: usize, factor: f32) f32 {
    const cs: f32 = @floatCast(e2m1(nibble(codes[row * (k / 2) ..][0 .. k / 2], i)) * e4m3(scales[row * (k / 16) + i / 16]));
    return cs * factor;
}

/// MXFP8 weight (row, i) of one expert's [n, k] E4M3 codes and [n, k/32] E8M0 scales: exact in f64 and in f32.
pub fn mxfp8(codes: []const u8, scales: []const u8, k: usize, row: usize, i: usize) f64 {
    return e4m3(codes[row * k + i]) * e8m0(scales[row * (k / 32) + i / 32]);
}

test "E2M1, E4M3 and E8M0 decode their extremes, signs and NaN codes" {
    const t = std.testing;
    try t.expectEqual(@as(f64, 6), e2m1(7));
    try t.expectEqual(@as(f64, -0.5), e2m1(9));
    try t.expect(std.math.signbit(e2m1(8)));
    try t.expectEqual(@as(f64, 448), e4m3(0x7E));
    try t.expectEqual(@as(f64, -0x1p-9), e4m3(0x81)); // the least subnormal
    try t.expectEqual(@as(f64, 1), e4m3(0x38));
    try t.expect(std.math.isNan(e4m3(0x7F)) and std.math.isNan(e4m3(0xFF)));
    try t.expectEqual(@as(f64, 1), e8m0(127));
    try t.expectEqual(@as(f64, 0x1p-127), e8m0(0));
    try t.expect(std.math.isNan(e8m0(255)));
}

test "NVFP4 and MXFP8 weights take input i's nibble, its 16- or 32-input scale and the factor" {
    const t = std.testing;
    const codes = [_]u8{ 0x7A, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x16 };
    const scales = [_]u8{ 0x40, 0x38 }; // 2, then 1
    try t.expectEqual(@as(f64, -1 * 2 * 0.25), nvfp4(&codes, &scales, 32, 0, 0, 0.25)); // low nibble 0xA = -1
    try t.expectEqual(@as(f64, 6 * 2 * 0.25), nvfp4(&codes, &scales, 32, 0, 1, 0.25));
    try t.expectEqual(@as(f64, 4 * 1 * 0.25), nvfp4(&codes, &scales, 32, 0, 30, 0.25)); // byte 15 low: 6 = 4
    try t.expectEqual(@as(f32, 4 * 0.1), nvfp4F32(&codes, &scales, 32, 0, 30, 0.1));
    var mx: [64]u8 = @splat(0x38);
    mx[33] = 0xC0; // -2
    const ms = [_]u8{ 127, 120 };
    try t.expectEqual(@as(f64, 1), mxfp8(&mx, &ms, 64, 0, 31));
    try t.expectEqual(@as(f64, -2 * 0x1p-7), mxfp8(&mx, &ms, 64, 0, 33));
}
