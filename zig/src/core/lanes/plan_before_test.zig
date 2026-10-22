//! Frozen successful-path planner and traversal from the preceding release prefix, for CPU equivalence only.
const std = @import("std");
const shape = @import("shape.zig");
const planner = @import("plan_lanes.zig");
const Allocator = std.mem.Allocator;
const Shape = shape.Shape;
const Odds = shape.Odds;
const max_depth = shape.max_depth;
const ranks = shape.ranks;
const sizes = planner.sizes;
const chain_grafts = planner.chain_grafts;
const Costs = planner.Costs;
const Grafts = planner.Grafts;
const Pick = planner.Pick;
fn own(c: Costs, choice: usize) f64 {
    const m = c.measured orelse return 0;
    return m[choice].median();
}
const Candidate = struct { parent: i32, depth: u8, rank: u8, reach: f64, value: f64 };

fn higher(_: void, a: Candidate, b: Candidate) std.math.Order {
    return std.math.order(b.value, a.value);
}

/// The prior best-first traversal, ordered by each lane's chance of being reached and taken.
pub fn bestShape(gpa: Allocator, o: Odds, lanes: usize, depth_cap: usize) !Shape {
    var parents: std.ArrayList(i32) = .empty;
    errdefer parents.deinit(gpa);
    var rs: std.ArrayList(u8) = .empty;
    errdefer rs.deinit(gpa);
    var ds: std.ArrayList(u8) = .empty;
    errdefer ds.deinit(gpa);
    var queue: std.PriorityQueue(Candidate, void, higher) = .empty;
    defer queue.deinit(gpa);
    const cap = @min(depth_cap, max_depth);
    var expected: f64 = 0;
    if (lanes > 0 and cap > 0) try queue.push(gpa, .{ .parent = -1, .depth = 0, .rank = 0, .reach = 1, .value = o.chance(0, 0) });
    while (parents.items.len < lanes) {
        const c = queue.pop() orelse break;
        const lane: i32 = @intCast(parents.items.len);
        try parents.append(gpa, c.parent);
        try rs.append(gpa, c.rank);
        try ds.append(gpa, c.depth);
        expected += c.value;
        if (c.rank + 1 < ranks) try queue.push(gpa, .{ .parent = c.parent, .depth = c.depth, .rank = c.rank + 1, .reach = c.reach, .value = c.reach * o.chance(c.depth, c.rank + 1) });
        if (c.depth + 1 < cap) try queue.push(gpa, .{ .parent = lane, .depth = c.depth + 1, .rank = 0, .reach = c.value, .value = c.value * o.chance(c.depth + 1, 0) });
    }
    return .{ .parents = try parents.toOwnedSlice(gpa), .ranks = try rs.toOwnedSlice(gpa), .depths = try ds.toOwnedSlice(gpa), .expected = expected };
}

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

pub fn pick(gpa: Allocator, o: shape.Odds, c: Costs, cap: u32, depth_cap: usize, grafts: ?Grafts, wider: bool) !Pick {
    var best: Pick = .{ .choice = 0 };
    var best_rate: f64 = -1;
    errdefer if (best.tree) |t| t.deinit(gpa);
    for (sizes, 0..) |n, i| {
        if (n > cap) break;
        const t = try bestShape(gpa, o, n, depth_cap);
        const priced = if (t.isChain()) c.round(1 + t.parents.len, levels(t)) else c.treeRound(1 + t.parents.len, levels(t));
        const cost = (priced orelse {
            t.deinit(gpa);
            continue;
        }) + own(c, i);
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
            const cost = (c.round(1 + d + rows, d) orelse continue) + own(c, chain_grafts);
            const expected = 1 + chainExpected(o, @min(d, depth_cap)) + g.hit * @as(f64, @floatFromInt(rows));
            if (expected / cost > best_rate) {
                if (best.tree) |old| old.deinit(gpa);
                best = .{ .choice = chain_grafts, .depth = @intCast(d), .grafts = rows, .expected = expected };
                best_rate = expected / cost;
            }
        }
    };
    if (wider and best.choice + 1 < sizes.len and sizes[best.choice + 1] <= cap) {
        const t = try bestShape(gpa, o, sizes[best.choice + 1], depth_cap);
        if (best.tree) |old| old.deinit(gpa);
        best = .{ .choice = best.choice + 1, .tree = t, .expected = 1 + t.expected };
    }
    return best;
}
