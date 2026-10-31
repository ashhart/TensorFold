//! A drafter's hold pass priced from its timed passes: by the blocks it holds and their depth.
const std = @import("std");

/// A drafter's hold pass of `streams` blocks, each `depth` drafts deep, in ms (timed at load).
pub const HoldCost = struct { streams: u32, depth: u32, ms: f64 };

/// A hold pass's ms: linear in depth at each timed stream count, then in streams (past the widest, by its slope).
pub fn holdMs(costs: []const HoldCost, streams: usize, depth: i64) f64 {
    var below: ?u32 = null;
    var above: ?u32 = null;
    for (costs) |c| {
        if (c.streams <= streams and (below == null or c.streams > below.?)) below = c.streams;
        if (c.streams >= streams and (above == null or c.streams < above.?)) above = c.streams;
    }
    const lo = below orelse above.?;
    const hi = above orelse lo;
    const at_lo = atDepth(costs, lo, depth);
    if (hi == lo) {
        if (above != null) return at_lo;
        // past the widest timed count: the slope from the count below it
        var under: ?u32 = null;
        for (costs) |c| if (c.streams < lo and (under == null or c.streams > under.?)) {
            under = c.streams;
        };
        const u = under orelse return at_lo * @as(f64, @floatFromInt(streams)) / @as(f64, @floatFromInt(lo));
        const slope = (at_lo - atDepth(costs, u, depth)) / @as(f64, @floatFromInt(lo - u));
        return at_lo + slope * @as(f64, @floatFromInt(streams - lo));
    }
    const t = @as(f64, @floatFromInt(streams - lo)) / @as(f64, @floatFromInt(hi - lo));
    return at_lo + t * (atDepth(costs, hi, depth) - at_lo);
}

/// Interpolate neighboring depths, or extrapolate the nearest end segment (one alone: as timed).
fn atDepth(costs: []const HoldCost, streams: u32, depth: i64) f64 {
    var a: ?HoldCost = null;
    var b: ?HoldCost = null;
    for (costs) |c| {
        if (c.streams != streams) continue;
        if (c.depth == depth) return c.ms;
        if (c.depth < depth and (a == null or c.depth > a.?.depth)) a = c;
        if (c.depth > depth and (b == null or c.depth < b.?.depth)) b = c;
    }
    if (a == null) {
        a = b.?;
        b = null;
        for (costs) |c| if (c.streams == streams and c.depth > a.?.depth and (b == null or c.depth < b.?.depth)) {
            b = c;
        };
        if (b == null) return a.?.ms;
    } else if (b == null) {
        b = a.?;
        a = null;
        for (costs) |c| if (c.streams == streams and c.depth < b.?.depth and (a == null or c.depth > a.?.depth)) {
            a = c;
        };
        if (a == null) return b.?.ms;
    }
    const x = a.?;
    const y = b.?;
    const t = (@as(f64, @floatFromInt(depth)) - @as(f64, @floatFromInt(x.depth))) / @as(f64, @floatFromInt(y.depth - x.depth));
    return x.ms + t * (y.ms - x.ms);
}

test "hold prices retain measured kernel transitions and use the adjacent depth segment" {
    const t = [_]HoldCost{
        .{ .streams = 8, .depth = 4, .ms = 18 },
        .{ .streams = 8, .depth = 1, .ms = 10 },
        .{ .streams = 8, .depth = 2, .ms = 17 },
    };
    try std.testing.expectEqual(@as(f64, 17), holdMs(&t, 8, 2));
    try std.testing.expectEqual(@as(f64, 17.5), holdMs(&t, 8, 3));
    try std.testing.expectEqual(@as(f64, 18.5), holdMs(&t, 8, 5));
    try std.testing.expectEqual(@as(f64, 3), holdMs(&t, 8, 0));
}

test "a hold's ms between and past the timed passes" {
    const t = [_]HoldCost{ .{ .streams = 1, .depth = 1, .ms = 1.0 }, .{ .streams = 1, .depth = 4, .ms = 2.5 }, .{ .streams = 4, .depth = 1, .ms = 3.0 }, .{ .streams = 4, .depth = 4, .ms = 4.5 }, .{ .streams = 8, .depth = 1, .ms = 4.0 } };
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), holdMs(&t, 1, 1), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), holdMs(&t, 1, 3), 1e-12); // linear in depth
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), holdMs(&t, 1, 5), 1e-12); // and past the deepest
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 + 2.0 / 3.0), holdMs(&t, 2, 1), 1e-12); // linear in streams
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), holdMs(&t, 8, 3), 1e-12); // one depth timed: as timed
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), holdMs(&t, 12, 1), 1e-12); // past the widest: the slope below it
}
