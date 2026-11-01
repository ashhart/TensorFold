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

/// A hold of `view.len()` streams' blocks in order, priced as the drafter runs it: `rows` (0: unbounded) bounds a pass's
/// rows, each block as long as the pass's deepest. A pass takes streams in order while its blocks still fit, the next
/// then starts fresh; each pass costs its own `holdMs`. `view.at(i)` is stream i's depth (zero or less: no block).
pub fn rowsMs(costs: []const HoldCost, rows: u32, view: anytype) f64 {
    const n = view.len();
    var ms: f64 = 0;
    var i: usize = 0;
    while (true) {
        while (i < n and view.at(i) <= 0) i += 1;
        if (i >= n) return ms;
        var blocks: usize = 0;
        var deepest: i64 = 0;
        while (i < n) : (i += 1) {
            const d = view.at(i);
            if (d <= 0) continue;
            const wide = @max(deepest, d);
            if (rows != 0 and blocks > 0 and @as(i64, @intCast(blocks + 1)) * wide > rows) break;
            blocks += 1;
            deepest = wide;
        }
        ms += holdMs(costs, blocks, deepest);
    }
}

/// The padded pass's price for these depths in stream order; zero-depth lanes take no block.
pub fn passMs(costs: []const HoldCost, rows: u32, depths: []const i64) f64 {
    const View = struct {
        d: []const i64,
        fn len(v: @This()) usize {
            return v.d.len;
        }
        fn at(v: @This(), i: usize) i64 {
            return v.d[i];
        }
    };
    return rowsMs(costs, rows, View{ .d = depths });
}

/// The same for allocated node `counts`: a charged stream's block is its count past `free` nodes (an empty `charged`
/// charges every stream; the others, and counts within `free`, take no block).
pub fn countsMs(costs: []const HoldCost, rows: u32, counts: []const u32, charged: []const bool, free: u32) f64 {
    const View = struct {
        counts: []const u32,
        charged: []const bool,
        free: u32,
        fn len(v: @This()) usize {
            return v.counts.len;
        }
        fn at(v: @This(), i: usize) i64 {
            if (v.charged.len != 0 and !v.charged[i]) return 0;
            return @as(i64, v.counts[i]) - v.free;
        }
    };
    return rowsMs(costs, rows, View{ .counts = counts, .charged = charged, .free = free });
}

/// Ignore work settled outside this round; the clock measures only what follows its start.
pub fn startRound(engine: anytype) void {
    if (engine.backend.draft_timing) |t| _ = t.take(t.ptr);
    engine.clock.start();
}

/// A deferred hold belongs to the round that settles it, not the round that just launches another hold.
pub fn observedHeadMs(backend: anytype, rule: anytype, depths: []const i64) f64 {
    if (backend.draft_timing) |t| return t.take(t.ptr);
    return rule.headMs(depths);
}

/// The head's holds are priced passes (by blocks and depth) that the depth rule and shared allocation weigh.
pub fn pricesPasses(engine: anytype) bool {
    const c = engine.cfg;
    return c.family_streams and c.family_mtp and c.allocate_shared_drafts and (c.granted_depth or c.limit_shared_drafts) and c.k.hold_costs.len > 0;
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

// SYNTHETIC grids: `flat` costs 10 ms whatever the pass, `product` a block-row (blocks x depth) ms, each up to 32 blocks.
const flat = [_]HoldCost{ .{ .streams = 1, .depth = 1, .ms = 10 }, .{ .streams = 1, .depth = 9, .ms = 10 }, .{ .streams = 32, .depth = 1, .ms = 10 }, .{ .streams = 32, .depth = 9, .ms = 10 } };
const product = [_]HoldCost{ .{ .streams = 1, .depth = 1, .ms = 1 }, .{ .streams = 1, .depth = 9, .ms = 9 }, .{ .streams = 32, .depth = 1, .ms = 32 }, .{ .streams = 32, .depth = 9, .ms = 288 } };

test "a hold past the pass's rows is priced as the passes the drafter makes, in stream order" {
    const near = std.testing.expectApproxEqAbs;
    const deep_first = [_]i64{ 9, 1, 1, 1, 1, 1, 1, 1 }; // three blocks of nine rows fit 32, the other five go in a pass of their own
    try near(@as(f64, 10), passMs(&flat, 0, &deep_first), 1e-9); // an unbounded pass is one
    try near(@as(f64, 20), passMs(&flat, 32, &deep_first), 1e-9);
    try near(@as(f64, 27 + 5), passMs(&product, 32, &deep_first), 1e-9);
    // the same depths in another order make other passes: seven blocks of one, then the nine alone
    try near(@as(f64, 7 + 9), passMs(&product, 32, &.{ 1, 1, 1, 1, 1, 1, 1, 9 }), 1e-9);
    try near(@as(f64, 3 + 27 + 2), passMs(&product, 32, &.{ 1, 1, 1, 9, 1, 1, 1, 1 }), 1e-9);
    // zero depths take no block and do not split a pass
    try near(@as(f64, 27), passMs(&product, 32, &.{ 0, 9, 0, 1, 1, 0 }), 1e-9);
    try near(@as(f64, 0), passMs(&product, 32, &.{ 0, 0 }), 1e-12);
    try near(@as(f64, 0), passMs(&product, 32, &.{}), 1e-12);
}

test "the row limit's boundary: blocks x deepest equal to the rows is one pass, one row more is two" {
    var depths: [9]i64 = @splat(4);
    try std.testing.expectApproxEqAbs(@as(f64, 10), passMs(&flat, 32, depths[0..8]), 1e-9); // 8 x 4 = 32
    try std.testing.expectApproxEqAbs(@as(f64, 20), passMs(&flat, 32, &depths), 1e-9); // 9 x 4 = 36: eight, then one
    depths[8] = 3;
    try std.testing.expectApproxEqAbs(@as(f64, 20), passMs(&flat, 32, &depths), 1e-9); // the ninth block needs a row more than 32 either way
    try std.testing.expectApproxEqAbs(@as(f64, 10), passMs(&flat, 36, &depths), 1e-9); // 9 x 4 fits a pass of 36
}

test "allocated counts price the same passes: charged streams only, past the free node" {
    const near = std.testing.expectApproxEqAbs;
    try near(@as(f64, 27), countsMs(&product, 32, &.{ 10, 2, 2 }, &.{}, 1), 1e-9); // blocks 9, 1, 1
    try near(@as(f64, 27), countsMs(&product, 32, &.{ 9, 1, 50, 1 }, &.{ true, true, false, true }, 0), 1e-9); // a copy's 50 takes no block
    try near(@as(f64, 0), countsMs(&product, 32, &.{ 1, 1, 1 }, &.{}, 1), 1e-12); // nothing past the free first
    try near(@as(f64, 20), countsMs(&flat, 32, &.{ 9, 1, 1, 1, 1, 1, 1, 1 }, &.{}, 0), 1e-9);
    try near(@as(f64, 10), countsMs(&flat, 0, &.{ 9, 1, 1, 1, 1, 1, 1, 1 }, &.{}, 0), 1e-9);
}
