//! Token rows in float64 at their rounding points: MLX's affine dequantize and a bf16 scale (cuda/embed_rows.zig).
const std = @import("std");
const bf = @import("bf16_round.zig");

/// Code i of an MLX row packed `bits` wide in little-endian u32 words, low code first (a code may cross a word).
pub fn code(row: []const u32, i: usize, bits: u4) u32 {
    const bit = i * bits;
    const sh: u6 = @intCast(bit % 32);
    var v: u64 = row[bit / 32];
    if (sh + bits > 32) v |= @as(u64, row[bit / 32 + 1]) << 32;
    return @intCast((v >> sh) & ((@as(u64, 1) << bits) - 1));
}

/// MLX's dequantize, q s + b in fp32 then bf16, times the bf16 `scale` (null: none) in fp32 then bf16 again.
pub fn affine(q: u32, s: f64, b: f64, scale: ?f64) f64 {
    const e: f32 = @mulAdd(f32, @floatFromInt(q), @floatCast(s), @floatCast(b)); // q s is exact in fp32
    const v = bf.bf16(e);
    const x = scale orelse return v;
    const p: f32 = @floatCast(v * x); // exact in float64: two bf16 values
    return bf.bf16(p);
}

test "MLX's packing: codes in order across word edges at every width" {
    var prng = std.Random.DefaultPrng.init(11);
    for ([_]u4{ 2, 3, 4, 5, 6, 8 }) |bits| {
        var want: [128]u32 = undefined;
        var row: [33]u32 = @splat(0);
        for (&want, 0..) |*q, i| {
            q.* = prng.random().int(u32) & ((@as(u32, 1) << bits) - 1);
            const bit = i * bits;
            row[bit / 32] |= q.* << @intCast(bit % 32);
            if (bit % 32 + bits > 32) row[bit / 32 + 1] |= q.* >> @intCast(32 - bit % 32);
        }
        for (want, 0..) |q, i| try std.testing.expectEqual(q, code(&row, i, bits));
    }
    try std.testing.expectEqual(@as(u32, 8), code(&.{0x8765_4321}, 7, 4)); // the top nibble is code 7
}

test "the rounding points: the bf16 value before the scale, and the scale's product rounded once" {
    try std.testing.expectEqual(@as(f64, 3 * 0.5 - 1), affine(3, 0.5, -1, null));
    try std.testing.expectEqual(@as(f64, 31), affine(15, 2, 1, 1)); // exact throughout
    // 1 + 2^-8 rounds to 1 (a tie to even) before the scale: times 3 gives 3, where the unrounded value gives 3 + 2^-6
    try std.testing.expectEqual(@as(f64, 3), affine(1, 1, 0x1p-8, 3));
    try std.testing.expectEqual(@as(f64, 3 + 0x1p-6), bf.bf16((1 + 0x1p-8) * 3));
    try std.testing.expect(std.math.isNan(affine(1, std.math.nan(f64), 0, 62)));
    try std.testing.expect(std.math.isInf(affine(255, 0x1p120, 0, 0x1p10))); // past fp32's range: Inf, as the fp32 product
}
