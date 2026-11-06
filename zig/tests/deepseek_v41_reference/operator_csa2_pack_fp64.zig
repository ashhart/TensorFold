//! CSA2 packed-cache code tables for the FP64 ideal-reference stages (DeepSeek-V4.1-Flash inference, MIT).
const std = @import("std");
const math = @import("math.zig");
const h = @import("operator_csa2_fp64.zig");

/// Packed-cache tables: generated codes give both sides bit-identical BF16 keys, so only attention arithmetic differs.
fn e4(code: u8) f32 {
    const magnitude = code & 127;
    if (magnitude == 127) return std.math.nan(f32);
    const exponent: i32 = @intCast(magnitude >> 3);
    const mantissa = magnitude & 7;
    const base: f32 = if (exponent == 0) @as(f32, @floatFromInt(mantissa)) / 512 else std.math.ldexp(1 + @as(f32, @floatFromInt(mantissa)) / 8, exponent - 7);
    return if ((code & 128) != 0) -base else base;
}

pub fn e2(code: u8) f32 {
    const levels = [_]f32{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };
    const v = levels[code & 7];
    return if ((code & 8) != 0) -v else v;
}

/// One ring row: 512 FP8 codes plus the FP32 scale of each 32-column block, exactly as dsv41_unpack reads it.
pub fn decodeWindowRow(bytes: []const u8) [h.width]u16 {
    var row: [h.width]u16 = undefined;
    for (0..h.width) |d| {
        const scale: f32 = @bitCast(@as(u32, bytes[h.width + d / 32]) << 23);
        row[d] = math.b(e4(bytes[d]) * scale);
    }
    return row;
}

/// One compressed row: 256 FP4 nibble bytes plus the FP8 scale of each 16-column block.
pub fn decodeMainRow(bytes: []const u8) [h.width]u16 {
    var row: [h.width]u16 = undefined;
    for (0..h.width) |d| {
        const nibble: u8 = if ((d & 1) != 0) bytes[d / 2] >> 4 else bytes[d / 2] & 15;
        row[d] = math.b(e2(nibble) * e4(bytes[h.width / 2 + d / 16]));
    }
    return row;
}

/// FP8 code under the scale the pack kernel picks for the block: Metal rint rounds ties to even.
fn roundHalfEven(x: f32) f32 {
    const toward: f32 = @trunc(x);
    const magnitude: f32 = if (@abs(x - toward) > 0.5) toward + std.math.sign(x) else toward;
    if (@abs(x - magnitude) != 0.5) return magnitude;
    const as_int: i64 = @intFromFloat(magnitude);
    if (@mod(as_int, 2) == 0) return magnitude;
    return if (x < 0) magnitude + 1 else magnitude - 1;
}

fn e4Code(x: f32) u8 {
    const a = @min(@abs(x), 448.0);
    if (a == 0) return if (std.math.signbit(x)) 128 else 0;
    const bits: u32 = @bitCast(a);
    const unbiased: i32 = @as(i32, @intCast((bits >> 23) & 255)) - 127;
    const e = @max(unbiased, -6);
    const q: i32 = @intFromFloat(roundHalfEven(std.math.ldexp(a, 3 - @as(i32, @intCast(e)))));
    const code: u32 = if (e == -6 and q < 8) @intCast(q) else @intCast((e + 7) * 8 + q - 8);
    return @as(u8, @intCast(@min(code, 126))) | @as(u8, if (std.math.signbit(x)) 128 else 0);
}

fn e4Scale(x: f32) u8 {
    const bits: u32 = @bitCast(x);
    const unbiased: i32 = @as(i32, @intCast((bits >> 23) & 255)) - 127 + @intFromBool((bits & 0x7fffff) != 0);
    return @intCast(unbiased + 127);
}

pub fn e2Code(x: f32) u8 {
    const a = @min(@abs(x), 6.0);
    const pick: u8 = if (a <= 0.25) 0 else if (a < 0.75) 1 else if (a <= 1.25) 2 else if (a < 1.75) 3 else if (a <= 2.5) 4 else if (a < 3.5) 5 else if (a <= 5.0) 6 else 7;
    return pick | @as(u8, if (std.math.signbit(x)) 8 else 0);
}

/// Pack a window row as dsv41_pack kind 0: per 32-column block, a power-of-two FP8 scale and FP8 codes.
pub fn packWindowRow(values: []const f32, out: []u8) void {
    @memset(out[0 .. h.width + h.width / 32], 0);
    for (0..h.width / 32) |block| {
        var a: f32 = 1e-4;
        for (values[block * 32 ..][0..32]) |v| a = @max(a, @abs(v));
        const z = a / 448;
        const zbits: u32 = @bitCast(z);
        const scale = std.math.ldexp(@as(f32, 1.0), @as(i32, @intCast((zbits >> 23) & 255)) - 127 + @intFromBool((zbits & 0x7fffff) != 0));
        out[h.width + block] = e4Scale(scale);
        for (0..32) |j| out[block * 32 + j] = e4Code(values[block * 32 + j] / scale);
    }
}

/// Pack a compressed row as dsv41_pack kind 1: per 16-column block, an FP8 scale and FP4 codes.
pub fn packMainRow(values: []const f32, out: []u8) void {
    @memset(out[0 .. h.width / 2 + h.width / 16], 0);
    for (0..h.width / 16) |block| {
        var a: f32 = 6.0 * 0x1p-9;
        for (values[block * 16 ..][0..16]) |v| a = @max(a, @abs(v));
        const scale = e4(e4Code(@max(a, 6.0 * 0x1p-9) / 6.0));
        out[h.width / 2 + block] = e4Code(@max(a, 6.0 * 0x1p-9) / 6.0);
        for (0..16) |j| {
            const code = e2Code(values[block * 16 + j] / scale);
            const at = block * 8 + j / 2;
            if ((j & 1) != 0) out[at] |= code << 4 else out[at] |= code;
        }
    }
}
