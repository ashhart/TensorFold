//! ModelOpt's MXFP8 and NVFP4 vision linears decode on the host to bf16, fp32 products rounded once, nearest even.
const std = @import("std");
const core = @import("core");
const convert = core.quant.convert;
const fp8 = core.quant.fp8;
const nvfp4_format = core.quant.nvfp4;

pub const Kind = enum { mxfp8, nvfp4 };

/// Inputs sharing one scale: an e8m0 power of two (MXFP8) or an e4m3 value (NVFP4).
pub fn group(kind: Kind) usize {
    return switch (kind) {
        .mxfp8 => 32,
        .nvfp4 => nvfp4_format.group,
    };
}

/// Every e4m3fn code as fp32; 0x7f and 0xff are NaN, as torch's float8_e4m3fn reads them.
fn e4m3Values() [256]f32 {
    var t: [256]f32 = undefined;
    for (&t, 0..) |*v, b| {
        const exp: i32 = @intCast((b >> 3) & 0xF);
        const man: f32 = @floatFromInt(b & 7);
        const mag: f32 = if (b & 0x7f == 0x7f) std.math.nan(f32) else if (exp == 0) std.math.ldexp(man, -9) else std.math.ldexp(8 + man, exp - 10);
        v.* = if (b & 0x80 != 0) -mag else mag;
    }
    return t;
}

/// Every e8m0 byte as the fp32 power of two 2^(e - 127); 255 is NaN.
fn e8m0Values() [256]f32 {
    var t: [256]f32 = undefined;
    for (&t, 0..) |*v, e| v.* = if (e == 255) std.math.nan(f32) else std.math.ldexp(@as(f32, 1), @as(i32, @intCast(e)) - 127);
    return t;
}

/// bf16 [n, k] of an NVFP4 linear: e2m1 code * e4m3 scale per 16 inputs * `global`, low nibble first, as fp32.
pub fn nvfp4(out: []u16, codes: []const u8, scales: []const u8, global: f32, n: usize, k: usize) void {
    const grid = [16]f32{ 0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6 };
    const e4m3 = e4m3Values();
    const groups = k / 16;
    for (0..n) |r| for (0..groups) |g| {
        const scale = e4m3[scales[r * groups + g]];
        const bytes = codes[r * (k / 2) + g * 8 ..][0..8];
        const dst = out[r * k + g * 16 ..][0..16];
        for (bytes, 0..) |byte, j| {
            dst[2 * j] = convert.bf16(grid[byte & 0xF] * scale * global);
            dst[2 * j + 1] = convert.bf16(grid[byte >> 4] * scale * global);
        }
    };
}

/// bf16 [n, k] of an MXFP8 linear: e4m3 code * 2^(e8m0 - 127) per 32 inputs, as fp32.
pub fn mxfp8(out: []u16, codes: []const u8, scales: []const u8, n: usize, k: usize) void {
    const e4m3 = e4m3Values();
    const powers = e8m0Values();
    const groups = k / 32;
    for (0..n) |r| for (0..groups) |g| {
        const scale = powers[scales[r * groups + g]];
        const src = codes[r * k + g * 32 ..][0..32];
        const dst = out[r * k + g * 32 ..][0..32];
        for (src, dst) |code, *v| v.* = convert.bf16(e4m3[code] * scale);
    };
}

fn bf16Value(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

test "Qwen tower weights: every e4m3 and e8m0 byte decodes as core's references" {
    const e4m3 = e4m3Values();
    const powers = e8m0Values();
    for (0..256) |b| {
        const want = fp8.e4m3(@intCast(b));
        if (std.math.isNan(want)) {
            try std.testing.expect(std.math.isNan(e4m3[b]));
        } else try std.testing.expectEqual(@as(f32, @floatCast(want)), e4m3[b]);
        if (b == 255) {
            try std.testing.expect(std.math.isNan(powers[b]));
        } else try std.testing.expectEqual(std.math.pow(f64, 2, @as(f64, @floatFromInt(b)) - 127), @as(f64, powers[b]));
    }
    try std.testing.expectEqual(@as(u32, 0x00400000), @as(u32, @bitCast(powers[0]))); // 2^-127, an fp32 subnormal
}

test "Qwen tower weights: NVFP4 against hand-decoded values" {
    // one row of 32: bytes 0 and 8 open the two scale groups; 0xF2 holds code 2 (1.0) low and code 15 (-6) high
    var codes: [16]u8 = @splat(0);
    codes[0] = 0xF2;
    codes[1] = 0x97; // code 7 (6), code 9 (-0.5)
    codes[8] = 0x35; // column 16: code 5 (3), column 17: code 3 (1.5)
    codes[15] = 0x80; // column 30: code 0, column 31: code 8 (negative zero)
    const scales = [_]u8{ 0x38, 0x40 }; // 1.0 and 2.0
    var out: [32]u16 = undefined;
    nvfp4(&out, &codes, &scales, 0.25, 1, 32);
    const want = [_]struct { at: usize, value: f32 }{
        .{ .at = 0, .value = 0.25 },
        .{ .at = 1, .value = -1.5 },
        .{ .at = 2, .value = 1.5 },
        .{ .at = 3, .value = -0.125 },
        .{ .at = 16, .value = 1.5 },
        .{ .at = 17, .value = 0.75 },
        .{ .at = 5, .value = 0 },
    };
    for (want) |w| try std.testing.expectEqual(w.value, bf16Value(out[w.at]));
    try std.testing.expectEqual(@as(u16, 0x0000), out[30]);
    try std.testing.expectEqual(@as(u16, 0x8000), out[31]); // -0 stays -0
    // a global scale that is no power of two: the fp32 product rounds to bf16 once, nearest even
    nvfp4(out[0..16], codes[0..8], scales[0..1], 0.0123, 1, 16);
    try std.testing.expectEqual(convert.bf16(1.0 * 1.0 * 0.0123), out[0]);
    try std.testing.expectEqual(convert.bf16(-6.0 * 1.0 * 0.0123), out[1]);
}

test "Qwen tower weights: NVFP4 NaN scales stay NaN and a zero scale zeroes its group" {
    const codes: [16]u8 = @splat(0x22);
    const scales = [_]u8{ 0x7f, 0x00 };
    var out: [32]u16 = undefined;
    nvfp4(&out, &codes, &scales, 1, 1, 32);
    for (out[0..16]) |v| try std.testing.expect(std.math.isNan(bf16Value(v)));
    for (out[16..]) |v| try std.testing.expectEqual(@as(f32, 0), bf16Value(v));
}

test "Qwen tower weights: MXFP8 against hand-decoded values" {
    // two rows of 32: e4m3 0x38 is 1.0, 0xC0 is -2, 0x7E is 448, 0x01 is 2^-9
    var codes: [64]u8 = @splat(0x38);
    codes[1] = 0xC0;
    codes[2] = 0x7E;
    codes[3] = 0x01;
    codes[32] = 0x00;
    const scales = [_]u8{ 127 + 3, 127 - 2 }; // 8 and 0.25
    var out: [64]u16 = undefined;
    mxfp8(&out, &codes, &scales, 2, 32);
    try std.testing.expectEqual(@as(f32, 8), bf16Value(out[0]));
    try std.testing.expectEqual(@as(f32, -16), bf16Value(out[1]));
    try std.testing.expectEqual(@as(f32, 3584), bf16Value(out[2]));
    try std.testing.expectEqual(@as(f32, 0.015625), bf16Value(out[3]));
    try std.testing.expectEqual(@as(f32, 0), bf16Value(out[32]));
    try std.testing.expectEqual(@as(f32, 0.25), bf16Value(out[33]));
    // e8m0 255 is NaN and 0 is 2^-127 (the bf16 subnormal 0x0040 for 1.0)
    mxfp8(&out, &codes, &.{ 255, 0 }, 2, 32);
    try std.testing.expect(std.math.isNan(bf16Value(out[0])));
    try std.testing.expectEqual(@as(u16, 0x0040), out[33]);
}
