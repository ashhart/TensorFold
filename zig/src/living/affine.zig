//! MLX's affine layout on the host at any group and code width; at 4 bits in groups of 64 it is affine4_host's.
const std = @import("std");
const Quant = @import("family.zig").Quant;

pub fn f32of(b: u16) f32 {
    return @bitCast(@as(u32, b) << 16);
}

/// The nearest bf16, ties to even; NaN stays NaN.
pub fn bf16of(x: f32) u16 {
    const u: u32 = @bitCast(x);
    if (std.math.isNan(x)) return @intCast((u >> 16) | 0x40);
    return @intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

/// u32 words `n` values take at `q`'s width.
pub fn wordsOf(comptime q: Quant, n: usize) usize {
    return n * q.bits / 32;
}

/// Row-major values from codes, scales and biases: scale * code + bias in f32 (the rows a whole number of groups).
pub fn dequantize(comptime q: Quant, words: []const u32, scales: []const u16, biases: []const u16, out: []f32) void {
    const per = 32 / q.bits;
    const mask: u32 = (1 << q.bits) - 1;
    for (out, 0..) |*o, i| {
        const c: u32 = (words[i / per] >> @intCast(q.bits * (i % per))) & mask;
        o.* = f32of(scales[i / q.group]) * @as(f32, @floatFromInt(c)) + f32of(biases[i / q.group]);
    }
}

/// Codes, scales and biases for row-major values: a group's minimum is its bias, its range over the top code its scale.
pub fn quantize(comptime q: Quant, values: []const f32, words: []u32, scales: []u16, biases: []u16) void {
    comptime std.debug.assert(32 % q.bits == 0);
    const per = 32 / q.bits;
    const top: f32 = @floatFromInt((1 << q.bits) - 1);
    @memset(words, 0);
    for (0..values.len / q.group) |g| {
        const v = values[g * q.group ..][0..q.group];
        const lo = std.mem.min(f32, v);
        const hi = std.mem.max(f32, v);
        scales[g] = bf16of((hi - lo) / top);
        biases[g] = bf16of(lo);
        const s = f32of(scales[g]);
        const b = f32of(biases[g]);
        for (v, g * q.group..) |x, i| {
            const c: u32 = if (s == 0) 0 else @intFromFloat(std.math.clamp(@round((x - b) / s), 0, top));
            words[i / per] |= c << @intCast(q.bits * (i % per));
        }
    }
}

test "a row at 4 bits in groups of 32 lands within half a step of each value, a flat group exactly on its bias" {
    const q: Quant = .{ .bits = 4, .group = 32 };
    var values: [3 * 32]f32 = undefined;
    for (&values, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) * 0.41) * 0.07;
    @memset(values[64..], -0.03);
    var words: [wordsOf(q, 3 * 32)]u32 = undefined;
    var scales: [3]u16 = undefined;
    var biases: [3]u16 = undefined;
    quantize(q, &values, &words, &scales, &biases);
    var back: [3 * 32]f32 = undefined;
    dequantize(q, &words, &scales, &biases, &back);
    for (values, back, 0..) |v, b, i| try std.testing.expect(@abs(v - b) <= f32of(scales[i / 32]) * 0.51 + 1e-3);
    for (back[64..]) |b| try std.testing.expectEqual(f32of(bf16of(-0.03)), b);
    // written again from what it reads back, a row is the same codes: a save's values are a fixed point
    var again: [wordsOf(q, 3 * 32)]u32 = undefined;
    var s2: [3]u16 = undefined;
    var b2: [3]u16 = undefined;
    quantize(q, &back, &again, &s2, &b2);
    var back2: [3 * 32]f32 = undefined;
    dequantize(q, &again, &s2, &b2, &back2);
    for (back, back2) |x, y| try std.testing.expectApproxEqAbs(x, y, 1e-6);
}

test "8-bit codes pack four a word" {
    const q: Quant = .{ .bits = 8, .group = 32 };
    var values: [32]f32 = undefined;
    for (&values, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) / 31;
    var words: [8]u32 = undefined;
    var s: [1]u16 = undefined;
    var b: [1]u16 = undefined;
    quantize(q, &values, &words, &s, &b);
    var back: [32]f32 = undefined;
    dequantize(q, &words, &s, &b, &back);
    for (values, back) |v, x| try std.testing.expectApproxEqAbs(v, x, f32of(s[0]) * 0.51 + 1e-3);
}
