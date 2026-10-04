//! The cluster's exact reductions: a fixed binary tree over S slices, with any contiguous split of the slices over ranks.
const std = @import("std");

/// Slices [begin, end) of a reduction (heads, intermediate columns, expert groups).
pub const Range = struct {
    begin: u32 = 0,
    end: u32 = 0,

    pub fn len(r: Range) u32 {
        return r.end - r.begin;
    }
};

/// Split `units` over ranks in proportion to `weights`: largest remainder, ties to the lower rank, contiguous in rank order.
pub fn partition(units: u32, weights: []const u64, out: []Range) void {
    std.debug.assert(out.len == weights.len and weights.len <= 256);
    var total: u128 = 0;
    for (weights) |w| total += w;
    var share: [256]u32 = @splat(0);
    var rest: [256]u128 = @splat(0);
    var given: u32 = 0;
    for (weights, 0..) |w, i| {
        if (total == 0) break;
        const exact = @as(u128, units) * w;
        share[i] = @intCast(exact / total);
        rest[i] = exact % total;
        given += share[i];
    }
    while (given < units) : (given += 1) {
        var best: usize = 0;
        for (0..weights.len) |i| {
            if (total == 0 or rest[i] > rest[best] or (rest[best] == 0 and weights[best] == 0 and weights[i] > 0)) best = i;
        }
        share[best] += 1;
        rest[best] = 0;
    }
    var at: u32 = 0;
    for (out, 0..) |*r, i| {
        r.* = .{ .begin = at, .end = at + share[i] };
        at += share[i];
    }
}

/// The maximal aligned dyadic intervals that cover `r` exactly, smallest offset first.
pub fn cover(r: Range, out: []Range) usize {
    var n: usize = 0;
    var at = r.begin;
    while (at < r.end) {
        var size: u32 = std.math.floorPowerOfTwo(u32, r.end - at);
        if (at != 0) size = @min(size, @as(u32, 1) << @intCast(@ctz(at)));
        out[n] = .{ .begin = at, .end = at + size };
        n += 1;
        at += size;
    }
    return n;
}

/// One aligned interval's partial sum: what a rank sends for the slices it owns.
pub const Part = struct { range: Range, data: []const f32 };

/// Element `i` of the tree over [lo, hi): a part that is exactly this interval, else left half + right half.
fn eval(parts: []const Part, lo: u32, hi: u32, i: usize) f32 {
    for (parts) |p| if (p.range.begin == lo and p.range.end == hi) return p.data[i];
    const mid = lo + (hi - lo) / 2;
    std.debug.assert(hi - lo > 1);
    return eval(parts, lo, mid, i) + eval(parts, mid, hi, i);
}

/// The root of the tree over [0, units) from parts that tile it with aligned intervals; units is a power of two.
pub fn reduce(units: u32, parts: []const Part, out: []f32) void {
    std.debug.assert(std.math.isPowerOfTwo(units));
    for (out, 0..) |*o, i| o.* = eval(parts, 0, units, i);
}

/// Element `i` of aligned interval [lo, hi) from single slices (slices[k] is slice base + k), summed in the tree's order.
pub fn subtree(slices: []const []const f32, base: u32, lo: u32, hi: u32, i: usize) f32 {
    if (hi - lo == 1) return slices[lo - base][i];
    const mid = lo + (hi - lo) / 2;
    return subtree(slices, base, lo, mid, i) + subtree(slices, base, mid, hi, i);
}

/// A rank's parts from its own slice partials (`slices[k]` is slice r.begin + k): each covering interval in tree order.
pub fn local(r: Range, slices: []const []const f32, scratch: []f32, out: []Part) usize {
    var ranges: [32]Range = undefined;
    const n = cover(r, &ranges);
    const width = if (slices.len == 0) 0 else slices[0].len;
    for (ranges[0..n], 0..) |c, k| {
        const dst = scratch[k * width ..][0..width];
        for (dst, 0..) |*d, i| d.* = subtree(slices, r.begin, c.begin, c.end, i);
        out[k] = .{ .range = c, .data = dst };
    }
    return n;
}

/// Which rank owns slice `s` under `ranges`.
pub fn owner(ranges: []const Range, s: u32) usize {
    for (ranges, 0..) |r, i| if (s >= r.begin and s < r.end) return i;
    unreachable;
}

test "partitions follow the weights, stay contiguous and cover every slice" {
    var r: [4]Range = undefined;
    partition(8, &.{ 1, 1, 1, 1 }, &r);
    for (r, 0..) |x, i| try std.testing.expectEqual(Range{ .begin = @intCast(2 * i), .end = @intCast(2 * i + 2) }, x);
    partition(8, &.{ 2, 1, 1, 0 }, &r);
    try std.testing.expectEqual(@as(u32, 4), r[0].len());
    try std.testing.expectEqual(@as(u32, 0), r[3].len());
    try std.testing.expectEqual(@as(u32, 8), r[3].end);
    var three: [3]Range = undefined;
    partition(8, &.{ 1, 1, 1 }, &three);
    try std.testing.expectEqual(@as(u32, 3), three[0].len());
    try std.testing.expectEqual(@as(u32, 3), three[1].len());
    try std.testing.expectEqual(@as(u32, 2), three[2].len());
    var many: [16]Range = undefined;
    partition(8, &(@as([16]u64, @splat(5))), &many);
    var total: u32 = 0;
    for (many) |x| total += x.len();
    try std.testing.expectEqual(@as(u32, 8), total);
}

test "covers are maximal aligned intervals" {
    var c: [8]Range = undefined;
    try std.testing.expectEqual(@as(usize, 1), cover(.{ .begin = 0, .end = 8 }, &c));
    try std.testing.expectEqual(@as(usize, 2), cover(.{ .begin = 2, .end = 5 }, &c));
    try std.testing.expectEqual(Range{ .begin = 2, .end = 4 }, c[0]);
    try std.testing.expectEqual(Range{ .begin = 4, .end = 5 }, c[1]);
    try std.testing.expectEqual(@as(usize, 2), cover(.{ .begin = 3, .end = 6 }, &c));
    try std.testing.expectEqual(Range{ .begin = 4, .end = 6 }, c[1]);
    try std.testing.expectEqual(@as(usize, 0), cover(.{ .begin = 5, .end = 5 }, &c));
}

test "every split of the slices over any number of ranks gives the same bits; a plain left-to-right sum does not" {
    const units = 8;
    const width = 64;
    var prng: std.Random.DefaultPrng = .init(11);
    var slices: [units][width]f32 = undefined;
    for (&slices) |*s| for (s) |*v| {
        v.* = (prng.random().float(f32) - 0.5) * std.math.pow(f32, 10, @floatFromInt(prng.random().intRangeAtMost(i32, -3, 8)));
    };
    var want: [width]f32 = undefined;
    var differs = false;
    const splits = [_][]const u64{ &.{1}, &.{ 1, 1 }, &.{ 1, 1, 1 }, &.{ 1, 1, 1, 1 }, &.{ 5, 1, 2 }, &.{ 1, 1, 1, 1, 1, 1, 1, 1 }, &.{ 0, 3, 0, 1 } };
    for (splits, 0..) |weights, w| {
        var ranges: [8]Range = undefined;
        partition(units, weights, ranges[0..weights.len]);
        var parts: [32]Part = undefined;
        var np: usize = 0;
        var scratch: [8][4 * width]f32 = undefined;
        for (ranges[0..weights.len], 0..) |r, k| {
            var mine: [units][]const f32 = undefined;
            for (r.begin..r.end, 0..) |s, j| mine[j] = &slices[s];
            np += local(r, mine[0..r.len()], &scratch[k], parts[np..]);
        }
        var got: [width]f32 = undefined;
        reduce(units, parts[0..np], &got);
        if (w == 0) want = got;
        try std.testing.expectEqualSlices(u32, @ptrCast(&want), @ptrCast(&got));
    }
    for (0..width) |i| {
        var naive: f32 = 0;
        for (slices) |s| naive += s[i];
        differs = differs or @as(u32, @bitCast(naive)) != @as(u32, @bitCast(want[i]));
    }
    try std.testing.expect(differs);
}
