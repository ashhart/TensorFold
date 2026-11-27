//! Rows as NVFP4 (e2m1 codes, an e4m3 scale per 16, an optional fp32 global scale), quantized on the host.
const std = @import("std");
const nvfp4 = @import("quant/nvfp4.zig");
const fp8 = @import("quant/fp8.zig");
const e4m3FromF32 = @import("exlr8_codec.zig").e4m3FromF32;

pub const group = nvfp4.group;
pub const max_code = 6.0; // e2m1's largest magnitude

/// The e2m1 code nearest x (ties to the even code), saturating at +-6.
pub fn e2m1Code(x: f64) u8 {
    const mags = [8]f64{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };
    const a = @abs(x);
    if (a >= max_code) return 7 | @as(u8, if (x < 0) 8 else 0); // infinities too
    var best: u8 = 0;
    for (mags, 0..) |m, i| {
        const d = @abs(a - m);
        const bd = @abs(a - mags[best]);
        if (d < bd or (d == bd and i % 2 == 0 and best % 2 == 1)) best = @intCast(i);
    }
    return best | @as(u8, if (x < 0 and best != 0) 8 else 0);
}

/// What a NaN in a row does: refuse the row, or store 0 in its place.
pub const NanRule = enum { reject, zero };

/// e4m3's largest scale: a value past 6 * this * global saturates there (infinities included).
pub const max_scale = 448.0;

/// A group's scale: max(amax / 6 / global, min_scale) rounded to e4m3; values clamp to +-6 * 448 * global first.
pub const Format = struct { global: ?f32 = null, min_scale: f32 = 0, nan: NanRule = .reject };

/// One row of `x` (a multiple of 16 long) into codes (two a byte, even index low) and e4m3 scales.
pub fn quantize(f: Format, x: []const f32, codes: []u8, scales: []u8) error{NaN}!void {
    std.debug.assert(x.len % group == 0 and codes.len == x.len / 2 and scales.len == x.len / group);
    const g: f64 = if (f.global) |v| v else 1;
    const top = max_code * max_scale * g;
    for (0..x.len / group) |b| {
        var v: [group]f64 = undefined;
        for (&v, x[b * group ..][0..group]) |*o, e| {
            if (std.math.isNan(e) and f.nan == .reject) return error.NaN;
            o.* = if (std.math.isNan(e)) 0 else std.math.clamp(@as(f64, e), -top, top);
        }
        var amax: f64 = 0;
        for (v) |e| amax = @max(amax, @abs(e));
        const code = e4m3FromF32(@floatCast(@max(amax / max_code / g, f.min_scale)));
        scales[b] = code;
        const s = fp8.e4m3(code) * g;
        for (v, 0..) |e, j| {
            const n = if (s == 0) 0 else e2m1Code(e / s);
            const at = b * group + j;
            if (at % 2 == 0) codes[at / 2] = n else codes[at / 2] |= n << 4;
        }
    }
}

/// The stored row's values: code * scale (* global).
pub fn dequantize(f: Format, codes: []const u8, scales: []const u8, out: []f64) void {
    const g: f64 = if (f.global) |v| v else 1;
    for (out, 0..) |*o, i| {
        const byte = codes[i / 2];
        const n: u8 = if (i % 2 == 0) byte & 0xF else byte >> 4;
        o.* = nvfp4.e2m1(n) * fp8.e4m3(scales[i / group]) * g;
    }
}

test "nvfp4 store: e2m1 rounds to the nearest code, ties to even, and saturates" {
    const cases = [_]struct { f64, f64 }{ .{ 0.2, 0 }, .{ 0.25, 0 }, .{ 0.26, 0.5 }, .{ 0.75, 1 }, .{ 1.25, 1 }, .{ 1.75, 2 }, .{ 2.5, 2 }, .{ 3.5, 4 }, .{ 5, 4 }, .{ 5.1, 6 }, .{ 9, 6 }, .{ -2.6, -3 }, .{ -0.1, 0 } };
    for (cases) |c| try std.testing.expectEqual(c[1], nvfp4.e2m1(e2m1Code(c[0])));
    // planted: truncation toward zero would give 0.5 for 0.75 and 4 for 5.1
    try std.testing.expect(nvfp4.e2m1(e2m1Code(0.75)) != 0.5 and nvfp4.e2m1(e2m1Code(5.1)) != 4);
}

test "nvfp4 store: a stored row's error stays within half a step of its group's scale, with and without a global scale" {
    var prng = std.Random.DefaultPrng.init(11);
    var x: [128]f32 = undefined;
    for (&x) |*v| v.* = @floatCast(prng.random().floatNorm(f64) * 3);
    x[5] = 0;
    x[17] = -40; // one loud value: its group's scale grows, the others' do not
    // the global scale puts the row's amax at e4m3's top (40 / 6 / global = 448)
    for ([_]Format{ .{}, .{ .global = 40.0 / (6.0 * 448.0) } }) |f| {
        var codes: [64]u8 = undefined;
        var scales: [8]u8 = undefined;
        var back: [128]f64 = undefined;
        try quantize(f, &x, &codes, &scales);
        dequantize(f, &codes, &scales, &back);
        for (0..8) |b| {
            const s = fp8.e4m3(scales[b]) * (if (f.global) |g| g else 1);
            for (b * group..(b + 1) * group) |i| {
                const err = @abs(back[i] - x[i]);
                // half a step is at most 1 scale (steps 0.5, 1, 2); e4m3 rounding of the scale adds up to 6/16
                try std.testing.expect(err <= s * (1 + 6.0 / 16.0) + 1e-12);
            }
        }
        try std.testing.expectEqual(@as(f64, 0), back[5]);
        try std.testing.expect(@abs(back[17] + 40) <= 40.0 / 6.0);
    }
}

test "nvfp4 store: a global scale too small for a group saturates its e4m3 scale (planted: the bound breaks)" {
    var x: [16]f32 = @splat(1);
    x[0] = 40;
    var codes: [8]u8 = undefined;
    var scales: [1]u8 = undefined;
    var back: [16]f64 = undefined;
    const f: Format = .{ .global = 0.01 }; // 40 / 6 / 0.01 = 667 > 448: the scale clips
    try quantize(f, &x, &codes, &scales);
    dequantize(f, &codes, &scales, &back);
    try std.testing.expect(@abs(back[0] - 40) > 448 * 0.01); // far past half a step of the clipped scale
}

test "nvfp4 store: infinities saturate at 6 * 448 (planted: before, inf / scale encoded as 0)" {
    var x: [16]f32 = @splat(1);
    x[0] = std.math.inf(f32);
    x[1] = -std.math.inf(f32);
    var codes: [8]u8 = undefined;
    var scales: [1]u8 = undefined;
    var back: [16]f64 = undefined;
    try quantize(.{}, &x, &codes, &scales);
    dequantize(.{}, &codes, &scales, &back);
    try std.testing.expectEqual(@as(f64, 6 * 448), back[0]);
    try std.testing.expectEqual(@as(f64, -6 * 448), back[1]);
    try std.testing.expectEqual(@as(f64, 0), back[2]); // 1 is under half a step of the saturated scale
    try std.testing.expectEqual(@as(u8, 7), e2m1Code(std.math.inf(f64)));
    try std.testing.expectEqual(@as(u8, 15), e2m1Code(-std.math.inf(f64)));
}

test "nvfp4 store: a NaN refuses the row, or stores 0 when the format says so" {
    var x: [16]f32 = @splat(2);
    x[3] = std.math.nan(f32);
    var codes: [8]u8 = undefined;
    var scales: [1]u8 = undefined;
    var back: [16]f64 = undefined;
    try std.testing.expectError(error.NaN, quantize(.{}, &x, &codes, &scales));
    try quantize(.{ .nan = .zero }, &x, &codes, &scales);
    dequantize(.{}, &codes, &scales, &back);
    try std.testing.expectEqual(@as(f64, 0), back[3]);
    try std.testing.expect(@abs(back[4] - 2) <= 0.125); // 2 at its group's e4m3 scale
}

test "nvfp4 store: a scale floor keeps a quiet group's scale at the floor (its values round to 0)" {
    var x: [16]f32 = @splat(0.01);
    var codes: [8]u8 = undefined;
    var scales: [1]u8 = undefined;
    var back: [16]f64 = undefined;
    try quantize(.{}, &x, &codes, &scales);
    dequantize(.{}, &codes, &scales, &back);
    try std.testing.expect(back[0] > 0); // no floor: the scale follows the group down
    try quantize(.{ .min_scale = 0.1 }, &x, &codes, &scales);
    dequantize(.{}, &codes, &scales, &back);
    try std.testing.expectEqual(e4m3FromF32(0.1), scales[0]);
    try std.testing.expectEqual(@as(f64, 0), back[0]);
}
