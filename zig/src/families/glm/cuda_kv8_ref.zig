//! CPU format reference for synthetic device gates, never an inference implementation.
const std = @import("std");
const fp8 = @import("core").quant.fp8;
pub fn bf(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}
pub fn bfBits(x: f32) u16 {
    const b: u32 = @bitCast(x);
    return @truncate((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}
pub fn code(x: f32) u8 {
    const sign: u8 = @intCast((@as(u32, @bitCast(x)) >> 24) & 0x80);
    const v: f32 = @floatCast(@abs(fp8.toE4m3(x)));
    const b: u32 = @bitCast(v);
    return sign | if (v < 0x1p-6) @as(u8, @intFromFloat(v * 512)) else @as(u8, @intCast((((b >> 23) - 120) << 3) | ((b >> 20) & 7)));
}
pub fn decoded(bits: u8, scale: f32) u16 {
    const value: f32 = @floatCast(fp8.e4m3(bits));
    return bfBits(value * scale);
}
pub fn pack(row: []const u16, out: []u8) !f32 {
    if (row.len == 0 or row.len != out.len) return error.BadRow;
    var amax: f32 = 0;
    for (row) |x| {
        const v = bf(x);
        if (!std.math.isFinite(v)) return error.NonFiniteRow;
        amax = @max(amax, @abs(v));
    }
    var inv: f32 = if (amax > 0) 448 / amax else 1;
    var scale: f32 = if (amax > 0) amax / 448 else 1;
    if (scale < 0x1p-126) {
        scale = 0x1p-126;
        inv = 1 / scale;
    }
    for (row, out) |x, *slot| slot.* = code(bf(x) * inv);
    return scale;
}

test "GLM FP8 KV: reference handles every finite BF16 magnitude and round-even ties" {
    try std.testing.expectEqual(@as(u8, 0x38), code(1.0625));
    try std.testing.expectEqual(@as(u8, 0x3a), code(1.1875));
    try std.testing.expectEqual(@as(u8, 0x80), code(-0.0));
    for (0..65536) |raw| {
        const bits: u16 = @intCast(raw);
        if (!std.math.isFinite(bf(bits))) continue;
        var q: [1]u8 = undefined;
        const scale = try pack(&.{bits}, &q);
        try std.testing.expect(std.math.isFinite(scale) and scale > 0);
        try std.testing.expect(std.math.isFinite(bf(decoded(q[0], scale))));
    }
    var q: [1]u8 = undefined;
    try std.testing.expectError(error.NonFiniteRow, pack(&.{0x7fc0}, &q));
}
