//! A lone stream's lanes each round: the head tree, or the head's chain with grafted copies, that lands the most tokens a round time.
const std = @import("std");
const Allocator = std.mem.Allocator;
const shape = @import("shape.zig");
const Table = @import("table.zig").Table;

/// Lane counts the planner weighs for a head tree (each at most the cap).
pub const sizes = [_]u32{ 1, 2, 3, 4, 6, 8, 12, 16, 24, 32 };

/// The pick that keeps the head's chain and grafts copies onto it.
pub const chain_grafts = sizes.len;

/// A stream's graft prospects this round: rows its suffix matches would graft, and the share of grafted rows it keeps.
pub const Grafts = struct { rows: u32, hit: f64 };

pub const Pick = struct {
    choice: usize, // an index into `sizes`, or `chain_grafts`
    tree: ?shape.Shape = null, // the head tree to draft
    depth: u32 = 0, // the head's chain depth (chain_grafts)
    grafts: u32 = 0, // grafted rows the round's window keeps
    expected: f64 = 0, // tokens the round is expected to land, the bonus token included
};

/// A pick's measured round times beyond its table price: the last few, their median the pick's own correction.
pub const Measured = struct {
    ms: [9]f64 = @splat(0),
    n: usize = 0,

    pub fn add(m: *Measured, x: f64) void {
        m.ms[m.n % m.ms.len] = x;
        m.n += 1;
    }

    pub fn median(m: *const Measured) f64 {
        const k = @min(m.n, m.ms.len);
        if (k == 0) return 0;
        var xs = m.ms;
        std.mem.sort(f64, xs[0..k], {}, std.sort.asc(f64));
        return if (k % 2 == 1) xs[k / 2] else (xs[k / 2 - 1] + xs[k / 2]) / 2;
    }
};

/// A round's price: the window by rows, a head level, and the stream's time beyond both (chains and trees apart).
pub const Costs = struct {
    window: *const Table,
    head_ms: f64,
    over_ms: f64,
    tree_over_ms: ?f64 = null,
    measured: ?*const [sizes.len + 1]Measured = null, // each pick's own correction (none measured: the table's price)

    fn own(c: Costs, choice: usize) f64 {
        const m = c.measured orelse return 0;
        return m[choice].median();
    }

    pub fn round(c: Costs, rows: usize, steps: usize) ?f64 {
        const w = c.window.get(@intCast(rows)) orelse return null;
        return w + c.head_ms * @as(f64, @floatFromInt(steps)) + c.over_ms;
    }

    pub fn treeRound(c: Costs, rows: usize, steps: usize) ?f64 {
        const w = c.window.get(@intCast(rows)) orelse return null;
        return w + c.head_ms * @as(f64, @floatFromInt(steps)) + (c.tree_over_ms orelse c.over_ms);
    }
};

fn levels(s: shape.Shape) usize {
    var most: usize = 0;
    for (s.depths) |d| most = @max(most, d);
    return if (s.depths.len > 0) most + 1 else 0;
}

/// The head's chain of `depth` drafts: tokens it is expected to land (the bonus token not counted).
fn chainExpected(o: shape.Odds, depth: usize) f64 {
    var total: f64 = 0;
    var reach: f64 = 1;
    for (0..depth) |d| {
        reach *= o.chance(@min(d, shape.max_depth - 1), 0);
        total += reach;
    }
    return total;
}

/// A pick's table price (the window, its head levels, the stream's time beyond them), for measuring its own correction.
pub fn tablePrice(c: Costs, p: Pick) ?f64 {
    if (p.tree) |t| return if (t.isChain()) c.round(1 + t.parents.len, levels(t)) else c.treeRound(1 + t.parents.len, levels(t));
    return c.round(1 + p.depth + p.grafts, p.depth);
}

/// The most expected tokens a round time: a head tree of each size to `cap` lanes, or the 1-4 draft chain with grafts.
pub fn pick(gpa: Allocator, o: shape.Odds, c: Costs, cap: u32, depth_cap: usize, grafts: ?Grafts, wider: bool) !Pick {
    var best: Pick = .{ .choice = 0 };
    var best_rate: f64 = -1;
    errdefer if (best.tree) |t| t.deinit(gpa);
    for (sizes, 0..) |n, i| {
        if (n > cap) break;
        const t = try shape.best(gpa, o, n, depth_cap);
        const priced = if (t.isChain()) c.round(1 + t.parents.len, levels(t)) else c.treeRound(1 + t.parents.len, levels(t));
        const cost = (priced orelse {
            t.deinit(gpa);
            continue;
        }) + c.own(i);
        const rate = (1 + t.expected) / cost;
        if (rate > best_rate) {
            if (best.tree) |old| old.deinit(gpa);
            best = .{ .choice = i, .tree = t, .expected = 1 + t.expected };
            best_rate = rate;
        } else t.deinit(gpa);
    }
    if (grafts) |g| if (g.rows > 0) {
        for (1..@min(4, @as(usize, cap)) + 1) |d| {
            const rows: u32 = @intCast(@min(@as(usize, g.rows), @as(usize, cap) -| d));
            const cost = (c.round(1 + d + rows, d) orelse continue) + c.own(chain_grafts);
            const expected = 1 + chainExpected(o, @min(d, depth_cap)) + g.hit * @as(f64, @floatFromInt(rows));
            if (expected / cost > best_rate) {
                if (best.tree) |old| old.deinit(gpa);
                best = .{ .choice = chain_grafts, .depth = @intCast(d), .grafts = rows, .expected = expected };
                best_rate = expected / cost;
            }
        }
    };
    if (wider and best.choice + 1 < sizes.len and sizes[best.choice + 1] <= cap) {
        const t = try shape.best(gpa, o, sizes[best.choice + 1], depth_cap);
        if (best.tree) |old| old.deinit(gpa);
        best = .{ .choice = best.choice + 1, .tree = t, .expected = 1 + t.expected };
    }
    return best;
}

test "a cheap wide window picks the widest tree, a costly one a short chain" {
    const gpa = std.testing.allocator;
    var flat: Table = .{};
    defer flat.deinit(gpa);
    var steep: Table = .{};
    defer steep.deinit(gpa);
    for (1..34) |r| {
        try flat.put(gpa, @intCast(r), 10.0);
        try steep.put(gpa, @intCast(r), 5.0 + 2.0 * @as(f64, @floatFromInt(r)));
    }
    const o = shape.Odds.init(.{ 0.8, 0.1, 0.03, 0.01 }, 10);
    const wide = try pick(gpa, o, .{ .window = &flat, .head_ms = 0, .over_ms = 0 }, 32, 16, null, false);
    defer if (wide.tree) |t| t.deinit(gpa);
    try std.testing.expectEqual(@as(usize, sizes.len - 1), wide.choice);
    const narrow = try pick(gpa, o, .{ .window = &steep, .head_ms = 0.5, .over_ms = 1 }, 32, 16, null, false);
    defer if (narrow.tree) |t| t.deinit(gpa);
    try std.testing.expect(sizes[narrow.choice] <= 4);
}

test "a pick whose own rounds run past its table price loses to one that does not" {
    const gpa = std.testing.allocator;
    var flat: Table = .{};
    defer flat.deinit(gpa);
    for (1..34) |r| try flat.put(gpa, @intCast(r), 10.0);
    const o = shape.Odds.init(.{ 0.8, 0.1, 0.03, 0.01 }, 10);
    var measured: [sizes.len + 1]Measured = @splat(.{});
    for (0..5) |_| measured[sizes.len - 1].add(30.0);
    const p = try pick(gpa, o, .{ .window = &flat, .head_ms = 0, .over_ms = 0, .measured = &measured }, 32, 16, null, false);
    defer if (p.tree) |t| t.deinit(gpa);
    try std.testing.expect(p.choice < sizes.len - 1);
    try std.testing.expectEqual(@as(f64, 30.0), measured[sizes.len - 1].median());
}

test "grafts that keep landing win the round when they add more rows than the head would" {
    const gpa = std.testing.allocator;
    var costs: Table = .{};
    defer costs.deinit(gpa);
    for (1..34) |r| try costs.put(gpa, @intCast(r), 5.0 + 0.5 * @as(f64, @floatFromInt(r)));
    const o = shape.Odds.init(.{ 0.6, 0.1, 0.03, 0.01 }, 10);
    const p = try pick(gpa, o, .{ .window = &costs, .head_ms = 0.5, .over_ms = 1 }, 32, 16, .{ .rows = 20, .hit = 0.95 }, false);
    defer if (p.tree) |t| t.deinit(gpa);
    try std.testing.expectEqual(chain_grafts, p.choice);
    try std.testing.expect(p.grafts == 20 and p.depth >= 1);
}
