//! A drafter's hold pass priced from its timed passes: by the blocks it holds and their depth.
const std = @import("std");
const Table = @import("table.zig").Table;

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

/// Price the executor's greedy, ordered groups; nonpositive depths take no block and `rows=0` is unbounded.
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

/// The physical passes' price for these depths in stream order; zero-depth lanes take no block.
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

/// Allocated counts past the free first nodes; an empty charged mask charges every stream.
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

/// The observed prefix's expected drafts, optimistic about conditionals not tried yet.
pub fn observedGain(probabilities: []const f64, tried: usize, count: usize) f64 {
    const n = @min(count, @min(tried, probabilities.len));
    var survival: f64 = 1;
    var gain: f64 = 0;
    for (probabilities[0..n]) |chance| {
        survival *= chance;
        gain += survival;
    }
    return gain + survival * @as(f64, @floatFromInt(count - n));
}

/// Confidence-head holds use the same reserved-row controller at admission and during decoding.
pub fn priced(engine: anytype) bool {
    return pricesPasses(engine) and engine.cfg.granted_depth;
}

/// The round's committed work before any hold: live rows and the tokens they are expected to yield.
pub const Base = struct {
    streams: usize,
    rows: u32,
    tokens: f64,

    /// Every stream awaits one row and keeps one token (a round no reservation has touched).
    pub fn pending(streams: usize) Base {
        return .{ .streams = streams, .rows = @intCast(streams), .tokens = @floatFromInt(streams) };
    }
};

/// A hold offer: `depth` is the hold's priced depth, `rows` the reserved rows it keeps, `gain` its expected drafts.
pub const Prefix = struct {
    depth: i64,
    gain: f64,
    rows: f64,
    lowered: bool, // full observed gain was below full raw confidence gain
    shortened: bool,
};

/// The cheapest per-extra-row price at which a perfect-acceptance pass could exist, for one cost shape.
pub const ProbeCache = struct {
    rows: u32,
    heads: u32,
    depth: u32,
    stamp: u64,
    marginal: f64,
};

/// A full requested hold stays when it beats plain rounds alone; otherwise the best-scoring nonempty prefix
/// (the full offer on ties) is offered. `probabilities` are cumulative joint chances per reserved draft;
/// `observed` conditionals are initialized only for min(tried, len). Null: no pricing input.
pub fn prefix(costs: *const Table, holds: []const HoldCost, base: Base, fixed: f64, probabilities: []const f64, observed: []const f64, tried: usize, wanted: i64) ?Prefix {
    if (holds.len == 0 or wanted <= 0 or probabilities.len == 0) return null;
    const full: usize = @min(probabilities.len, @as(usize, @intCast(wanted)));
    const plain = costs.get(base.rows) orelse return null;
    const plain_score = base.tokens / (plain + fixed);
    const seen = @min(tried, observed.len);
    var survival: f64 = 1;
    var observed_gain: f64 = 0;
    var raw: f64 = 0;
    var best: ?Prefix = null;
    var best_score: f64 = undefined;
    var lowered = false;
    for (probabilities[0..full], 1..) |chance, n| {
        raw += chance;
        if (n <= seen) {
            survival *= observed[n - 1];
            observed_gain += survival;
        }
        const seen_gain = if (n <= seen) observed_gain else observed_gain + survival * @as(f64, @floatFromInt(n - seen));
        const gain = @min(raw, seen_gain);
        const last = n == full;
        if (last) lowered = seen_gain < raw;
        const depth: i64 = if (last) wanted else @intCast(n);
        const verify = costs.get(@as(i64, base.rows) + @as(i64, @intCast(n))) orelse continue;
        const score = (base.tokens + gain) / (verify + fixed + holdMs(holds, 1, depth));
        if (last and score > plain_score) {
            best = .{ .depth = depth, .gain = gain, .rows = @floatFromInt(n), .lowered = false, .shortened = false };
            break;
        }
        if (best == null or score >= best_score) {
            best_score = score;
            best = .{ .depth = depth, .gain = gain, .rows = @floatFromInt(n), .lowered = false, .shortened = !last };
        }
    }
    var chosen = best orelse return null;
    chosen.lowered = lowered;
    return chosen;
}

/// Whether a perfect-acceptance exploration pass could beat plain rounds under the price model (unknown costs: it could).
/// Caches the cheapest marginal ms a draft row on `rule.explore_price`, keyed by the cost values it read.
pub fn canExplore(rule: anytype, whos: anytype, base: Base, fixed: f64) bool {
    const costs = rule.shared_costs;
    const holds = rule.k.hold_costs;
    const plain = costs.get(base.rows) orelse return true;
    if (holds.len == 0) return true;
    var heads: u32 = 0;
    var depth: u32 = 0;
    for (whos) |w| {
        if (w.finished or w.forced or w.draft_room <= 1) continue;
        heads += 1;
        const room: i64 = @min(@as(i64, rule.most_drafts), w.draft_room - 1);
        depth = @max(depth, @as(u32, @intCast(room)));
    }
    const room_rows = if (rule.batch_rows > base.rows) rule.batch_rows - base.rows else 0;
    if (heads == 0 or depth == 0 or room_rows == 0) return false;
    // A smaller physical cap can split a reserved cohort; the single-pass bound cannot exclude those shapes.
    if (rule.k.hold_rows != 0 and rule.batch_rows > rule.k.hold_rows) return true;
    var hasher = std.hash.Wyhash.init(0);
    inline for (.{ base.rows, heads, depth, rule.batch_rows }) |v| {
        const x: u64 = v;
        hasher.update(std.mem.asBytes(&x));
    }
    for (holds) |c| {
        const w = [3]u64{ c.streams, c.depth, @as(u64, @bitCast(c.ms)) };
        hasher.update(std.mem.asBytes(&w));
    }
    var rows: u32 = base.rows;
    while (rows <= rule.batch_rows) : (rows += 1) {
        const ms = costs.get(rows);
        const w = [2]u64{ @intFromBool(ms != null), if (ms) |v| @as(u64, @bitCast(v)) else 0 };
        hasher.update(std.mem.asBytes(&w));
    }
    const stamp = hasher.final();
    const marginal = if (rule.explore_price) |c|
        if (c.rows == base.rows and c.heads == heads and c.depth == depth and c.stamp == stamp) c.marginal else null
    else
        null;
    const cheapest = marginal orelse m: {
        var lowest = std.math.inf(f64);
        var blocks: u32 = 1;
        while (blocks <= @min(heads, rule.batch_rows)) : (blocks += 1) {
            // Rows R need padded depth at least ceil(R/blocks); any feasible deeper padding may be cheaper, so price R at the
            // lowest hold over its class depth and every deeper feasible one (descending depths keep that minimum).
            var padded: u32 = @min(depth, rule.batch_rows / blocks);
            var hold = std.math.inf(f64);
            while (padded >= 1) : (padded -= 1) {
                hold = @min(hold, holdMs(holds, blocks, padded));
                var extra: u32 = @max(blocks, (padded - 1) * blocks + 1);
                const last = @min(room_rows, padded * blocks);
                while (extra <= last) : (extra += 1) {
                    const verify = costs.get(@as(i64, base.rows) + extra) orelse return true;
                    lowest = @min(lowest, (verify - plain + hold) / @as(f64, @floatFromInt(extra)));
                }
            }
        }
        rule.explore_price = .{ .rows = base.rows, .heads = heads, .depth = depth, .stamp = stamp, .marginal = lowest };
        break :m lowest;
    };
    if (!std.math.isFinite(cheapest)) return false;
    return base.tokens * cheapest < plain + fixed;
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

/// Head-step work past each first draft, priced by budgets rather than charged again as fixed overhead.
pub fn extraStepMs(depths: []const i64, step_ms: f64, batched: bool) f64 {
    if (step_ms == 0) return 0;
    var steps: i64 = 0;
    for (depths) |d| steps = if (batched) @max(steps, d - 1) else steps + @max(0, d - 1);
    return step_ms * @as(f64, @floatFromInt(steps));
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
