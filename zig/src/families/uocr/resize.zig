//! The encoder's tables for a 640 tile, resized as the publisher's torch code resizes them: fp32 math, one bf16 rounding.
const std = @import("std");

pub fn bf16(v: f32) u16 {
    const u: u32 = @bitCast(v);
    if (std.math.isNan(v)) return 0x7fc0;
    return @truncate((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}

pub fn widen(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

/// torch's antialiased bicubic filter (a = -0.5) as its CPU build computes it: float, multiply-adds fused.
fn cubic(x0: f32) f32 {
    const a: f32 = -0.5;
    const x = @abs(x0);
    if (x < 1.0) return @mulAdd(f32, @mulAdd(f32, a + 2.0, x, -(a + 3.0)) * x, x, 1.0);
    if (x < 2.0) return @mulAdd(f32, @mulAdd(f32, @mulAdd(f32, a, x, -(5.0 * a)), x, 8.0 * a), x, -(4.0 * a));
    return 0.0;
}

/// One output position's taps along a dimension of `in` resized to `out` (antialiased, align_corners=False), with
/// torch's mixed widths: its `+ 0.5` literals are double, so the centre and the filter's argument round from double.
const Taps = struct {
    first: usize,
    count: usize,
    w: [64]f32,

    fn of(in: usize, out: usize, i: usize) Taps {
        const scale: f32 = @as(f32, @floatFromInt(in)) / @as(f32, @floatFromInt(out));
        const support: f32 = if (scale >= 1.0) @floatCast(2.0 * @as(f64, scale)) else 2.0;
        const invscale: f32 = if (scale >= 1.0) @floatCast(1.0 / @as(f64, scale)) else 1.0;
        const center: f32 = @floatCast(@as(f64, scale) * (@as(f64, @floatFromInt(i)) + 0.5));
        const lo: usize = @intFromFloat(@max(@trunc(@as(f64, center) - support + 0.5), 0.0));
        const hi: usize = @min(@as(usize, @intFromFloat(@max(@trunc(@as(f64, center) + support + 0.5), 0.0))), in);
        var t: Taps = .{ .first = lo, .count = hi - lo, .w = undefined };
        var total: f32 = 0.0;
        for (0..t.count) |j| {
            const off: f32 = @as(f32, @floatFromInt(j + lo)) - center;
            const v = cubic(@floatCast((@as(f64, off) + 0.5) * invscale));
            t.w[j] = v;
            total += v;
        }
        if (total != 0.0) for (t.w[0..t.count]) |*v| {
            v.* /= total;
        };
        return t;
    }
};

/// A [side, side, C] bf16 grid (positions by rows, channels last) to [to, to, C], bicubic with antialias, in fp32:
/// the width first, then the height, each tap a fused multiply-add as torch's arm64 build does, then one rounding to bf16.
pub fn bicubicGrid(a: std.mem.Allocator, src: []const u16, side: usize, channels: usize, to: usize, out: []u16) !void {
    std.debug.assert(src.len == side * side * channels and out.len == to * to * channels);
    const mid = try a.alloc(f32, side * to * channels);
    defer a.free(mid);
    for (0..to) |x| {
        const t = Taps.of(side, to, x);
        for (0..side) |y| for (0..channels) |c| {
            var s: f32 = 0.0;
            for (0..t.count) |j| s = @mulAdd(f32, t.w[j], widen(src[(y * side + t.first + j) * channels + c]), s);
            mid[(y * to + x) * channels + c] = s;
        };
    }
    for (0..to) |y| {
        const t = Taps.of(side, to, y);
        for (0..to) |x| for (0..channels) |c| {
            var s: f32 = 0.0;
            for (0..t.count) |j| s = @mulAdd(f32, t.w[j], mid[((t.first + j) * to + x) * channels + c], s);
            out[(y * to + x) * channels + c] = bf16(s);
        };
    }
}

/// A relative-position table [len, C] to [to, C] by linear interpolation (align_corners=False), fp32, then bf16.
pub fn linearTable(src: []const u16, len: usize, channels: usize, to: usize, out: []u16) void {
    std.debug.assert(src.len == len * channels and out.len == to * channels);
    const scale: f32 = @as(f32, @floatFromInt(len)) / @as(f32, @floatFromInt(to));
    for (0..to) |i| {
        const at = @max(scale * (@as(f32, @floatFromInt(i)) + 0.5) - 0.5, 0.0);
        const lo: usize = @intFromFloat(@floor(at));
        const hi = @min(lo + 1, len - 1);
        const l1 = at - @as(f32, @floatFromInt(lo));
        const l0 = 1.0 - l1;
        for (0..channels) |c| out[i * channels + c] = bf16(l0 * widen(src[lo * channels + c]) + l1 * widen(src[hi * channels + c]));
    }
}

test "uocr: a same-size resize keeps every value, and a constant grid stays constant when it shrinks" {
    const a = std.testing.allocator;
    var grid: [4 * 4 * 2]u16 = undefined;
    for (&grid, 0..) |*g, i| g.* = bf16(@floatFromInt(i));
    var same: [4 * 4 * 2]u16 = undefined;
    try bicubicGrid(a, &grid, 4, 2, 4, &same);
    try std.testing.expectEqualSlices(u16, &grid, &same);
    var flat: [8 * 8]u16 = @splat(bf16(0.75));
    var small: [5 * 5]u16 = undefined;
    try bicubicGrid(a, &flat, 8, 1, 5, &small);
    for (small) |v| try std.testing.expectEqual(bf16(0.75), v);
    var tab: [7 * 1]u16 = undefined;
    for (&tab, 0..) |*t, i| t.* = bf16(@floatFromInt(i));
    var tab2: [7]u16 = undefined;
    linearTable(&tab, 7, 1, 7, &tab2);
    try std.testing.expectEqualSlices(u16, &tab, &tab2);
}
