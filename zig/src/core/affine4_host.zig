//! MLX's affine 4-bit layout at group 64 on the host: 8 codes a u32 word, low nibble first, bf16 scale and bias.
const std = @import("std");

pub const group = 64;

pub fn f32of(b: u16) f32 {
    return @bitCast(@as(u32, b) << 16);
}

/// The nearest bf16, ties to even; NaN stays NaN.
pub fn bf16of(x: f32) u16 {
    const u: u32 = @bitCast(x);
    if (std.math.isNan(x)) return @intCast((u >> 16) | 0x40);
    return @intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

/// Row-major values from codes, scales and biases: scale * q + bias in f32 (the rows' width a multiple of 64).
pub fn dequantize(words: []const u32, scales: []const u16, biases: []const u16, out: []f32) void {
    for (out, 0..) |*o, i| {
        const q: u32 = (words[i / 8] >> @intCast(4 * (i % 8))) & 0xf;
        o.* = f32of(scales[i / group]) * @as(f32, @floatFromInt(q)) + f32of(biases[i / group]);
    }
}

/// Codes, scales and biases for row-major values: each group's minimum is its bias, its range over 15 its scale.
pub fn quantize(values: []const f32, words: []u32, scales: []u16, biases: []u16) void {
    @memset(words, 0);
    for (0..values.len / group) |g| {
        const v = values[g * group ..][0..group];
        const lo = std.mem.min(f32, v);
        const hi = std.mem.max(f32, v);
        scales[g] = bf16of((hi - lo) / 15);
        biases[g] = bf16of(lo);
        const s = f32of(scales[g]);
        const b = f32of(biases[g]);
        for (v, g * group..) |x, i| {
            const q: u32 = if (s == 0) 0 else @intFromFloat(std.math.clamp(@round((x - b) / s), 0, 15));
            words[i / 8] |= q << @intCast(4 * (i % 8));
        }
    }
}

test "dequantized codes land within half a step, a flat group exactly on its bias; bf16 ties go to even" {
    var values: [2 * group]f32 = undefined;
    for (&values, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) * 0.37) * 0.05;
    @memset(values[group..], 0.02);
    var words: [2 * group / 8]u32 = undefined;
    var scales: [2]u16 = undefined;
    var biases: [2]u16 = undefined;
    quantize(&values, &words, &scales, &biases);
    var back: [2 * group]f32 = undefined;
    dequantize(&words, &scales, &biases, &back);
    for (values, back, 0..) |v, b, i| try std.testing.expect(@abs(v - b) <= f32of(scales[i / group]) * 0.51 + 1e-3);
    try std.testing.expectEqual(@as(u16, 0x3f80), bf16of(1.0));
    try std.testing.expectEqual(@as(u16, 0x3f80), bf16of(@bitCast(@as(u32, 0x3f808000))));
    try std.testing.expectEqual(@as(u16, 0x3f82), bf16of(@bitCast(@as(u32, 0x3f818000))));
    try std.testing.expect(std.math.isNan(f32of(bf16of(std.math.nan(f32)))));
}
