//! A head tree's shape for a stream's lanes: which of the head's ranked tokens each lane takes, from the stream's odds.
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const max_depth = 16;
pub const ranks = 4;

/// How often the target took the head's rank-r token at each depth when that lane was verified (with a prior).
pub const Odds = struct {
    took: [max_depth][ranks]f64 = @splat(@splat(0)),
    seen: [max_depth][ranks]f64 = @splat(@splat(0)),

    /// Odds that start from `prior` (the chance for each rank, the same at every depth) worth `weight` lanes.
    pub fn init(prior: [ranks]f64, weight: f64) Odds {
        var o: Odds = .{};
        for (&o.took, &o.seen) |*t, *s| for (t, s, prior) |*tr, *sr, p| {
            tr.* = p * weight;
            sr.* = weight;
        };
        return o;
    }

    /// A verified lane at `depth` and `rank`: whether the target took it.
    pub fn observe(o: *Odds, depth: usize, rank: usize, took: bool) void {
        if (depth >= max_depth or rank >= ranks) return;
        o.seen[depth][rank] += 1;
        if (took) o.took[depth][rank] += 1;
    }

    pub fn chance(o: Odds, depth: usize, rank: usize) f64 {
        return o.took[depth][rank] / o.seen[depth][rank];
    }
};

/// Draft-tree lanes: parent (-1 for the pending row), head rank and depth (0 for the first level).
pub const Shape = struct {
    parents: []i32,
    ranks: []u8,
    depths: []u8,
    expected: f64, // tokens the lanes are expected to land (the bonus token not counted)

    pub fn deinit(s: Shape, gpa: Allocator) void {
        gpa.free(s.parents);
        gpa.free(s.ranks);
        gpa.free(s.depths);
    }

    /// Levels the tree spans: its deepest lane's depth plus one.
    pub fn levels(s: Shape) u32 {
        var most: u32 = 0;
        for (s.depths) |d| most = @max(most, d);
        return most + 1;
    }

    pub fn isChain(s: Shape) bool {
        for (s.ranks) |r| if (r != 0) return false;
        return true;
    }
};

/// The odds a stream starts from: the head's chance by rank (any depth), worth a few lanes until its own come in.
pub const start_odds = [ranks]f64{ 0.75, 0.08, 0.02, 0.01 };
pub const start_weight = 4;

/// The lanes of the shape's chain (rank 0 from the pending row down), in depth order; their count.
pub fn chainLanes(s: Shape, out: []usize) usize {
    var n: usize = 0;
    var at: i32 = -1;
    while (n < out.len) {
        const next = for (s.parents, s.ranks, 0..) |p, r, i| {
            if (p == at and r == 0) break i;
        } else return n;
        out[n] = next;
        at = @intCast(next);
        n += 1;
    }
    return n;
}

const Candidate = struct { parent: i32, depth: u8, rank: u8, reach: f64, value: f64 };

fn higher(_: void, a: Candidate, b: Candidate) std.math.Order {
    return std.math.order(b.value, a.value);
}

/// One best-first traversal, with parent indices and expected sums independent of the requested prefix width.
pub const Cursor = struct {
    odds: Odds,
    cap: usize,
    queue: std.PriorityQueue(Candidate, void, higher) = .empty,
    count: usize = 0,
    expected: f64 = 0,

    pub const Lane = struct { parent: i32, rank: u8, depth: u8 };

    pub fn init(gpa: Allocator, odds: Odds, depth_cap: usize) !Cursor {
        var out: Cursor = .{ .odds = odds, .cap = @min(depth_cap, max_depth) };
        if (out.cap > 0) try out.queue.push(gpa, .{ .parent = -1, .depth = 0, .rank = 0, .reach = 1, .value = odds.chance(0, 0) });
        return out;
    }

    pub fn deinit(c: *Cursor, gpa: Allocator) void {
        c.queue.deinit(gpa);
    }

    pub fn next(c: *Cursor, gpa: Allocator) !?Lane {
        const node = c.queue.pop() orelse return null;
        const lane: i32 = @intCast(c.count);
        c.count += 1;
        c.expected += node.value;
        if (node.rank + 1 < ranks) try c.queue.push(gpa, .{ .parent = node.parent, .depth = node.depth, .rank = node.rank + 1, .reach = node.reach, .value = node.reach * c.odds.chance(node.depth, node.rank + 1) });
        if (node.depth + 1 < c.cap) try c.queue.push(gpa, .{ .parent = lane, .depth = node.depth + 1, .rank = 0, .reach = node.value, .value = node.value * c.odds.chance(node.depth + 1, 0) });
        return .{ .parent = node.parent, .rank = node.rank, .depth = node.depth };
    }
};

/// The best-first draft lanes, ordered by their chance of being reached and taken.
pub fn best(gpa: Allocator, o: Odds, lanes: usize, depth_cap: usize) !Shape {
    var parents: std.ArrayList(i32) = .empty;
    errdefer parents.deinit(gpa);
    var rs: std.ArrayList(u8) = .empty;
    errdefer rs.deinit(gpa);
    var ds: std.ArrayList(u8) = .empty;
    errdefer ds.deinit(gpa);
    var cursor = try Cursor.init(gpa, o, if (lanes > 0) depth_cap else 0);
    defer cursor.deinit(gpa);
    while (parents.items.len < lanes) {
        const lane = try cursor.next(gpa) orelse break;
        try parents.append(gpa, lane.parent);
        try rs.append(gpa, lane.rank);
        try ds.append(gpa, lane.depth);
    }
    const out_parents = try parents.toOwnedSlice(gpa);
    errdefer gpa.free(out_parents);
    const out_ranks = try rs.toOwnedSlice(gpa);
    errdefer gpa.free(out_ranks);
    return .{ .parents = out_parents, .ranks = out_ranks, .depths = try ds.toOwnedSlice(gpa), .expected = cursor.expected };
}

test "odds that only the head's first choice lands give a chain" {
    const gpa = std.testing.allocator;
    const s = try best(gpa, Odds.init(.{ 0.7, 0, 0, 0 }, 10), 5, max_depth);
    defer s.deinit(gpa);
    try std.testing.expect(s.isChain());
    try std.testing.expectEqualSlices(i32, &.{ -1, 0, 1, 2, 3 }, s.parents);
    try std.testing.expectApproxEqAbs(0.7 + 0.49 + 0.343 + 0.2401 + 0.16807, s.expected, 1e-9);
}

test "a second choice joins once it beats the chain's next lane, and lanes follow their parents" {
    const gpa = std.testing.allocator;
    const s = try best(gpa, Odds.init(.{ 0.6, 0.3, 0, 0 }, 10), 4, max_depth);
    defer s.deinit(gpa);
    // values: d0r0 .6, d1r0 .36, d0r1 .3, d1r1 under the first .18, d2r0 .216, d1r0 under the second .18
    try std.testing.expectEqualSlices(i32, &.{ -1, 0, -1, 1 }, s.parents);
    try std.testing.expectEqualSlices(u8, &.{ 0, 0, 1, 0 }, s.ranks);
    try std.testing.expectEqualSlices(u8, &.{ 0, 1, 0, 2 }, s.depths);
    for (s.parents, 0..) |p, i| try std.testing.expect(p < @as(i32, @intCast(i)));
}

test "observed lanes move the odds from the prior" {
    var o = Odds.init(.{ 0.5, 0.1, 0, 0 }, 2);
    for (0..8) |_| o.observe(0, 0, true);
    o.observe(0, 1, false);
    try std.testing.expectApproxEqAbs((1.0 + 8.0) / (2.0 + 8.0), o.chance(0, 0), 1e-12);
    try std.testing.expectApproxEqAbs(0.2 / 3.0, o.chance(0, 1), 1e-12);
    try std.testing.expectEqual(@as(f64, 0.5), o.chance(1, 0));
}
