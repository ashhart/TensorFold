//! A hold's drafts from its rows' candidates: a chain, the planner's lanes, or a best-first tree as the 27B engine's.
const std = @import("std");
const platform = @import("sys");
const lanes = @import("lanes");
const engine = @import("cuda_engine.zig");

/// Children a node, the selector's weight, the softmax scale, and the nodes a tree takes by default.
const branch = 4;
const edge_weight = 0.6;
const scale = 1.5;
const default_nodes = 15;
const lp_top = 20;
const lp_words = 3 + 2 * lp_top;

/// A hold's rows: each row's logprob words (ids, then logits at lp_top), its selector projection, the codebooks.
pub const Candidates = struct {
    words: []const u8,
    projected: []const u8,
    topk: usize,
    rank: usize,
    predecessor: []const u16, // (vocab, rank) bf16
    successor: []const u16,
};

/// A grown tree's nodes at most: 0 when the drafter drafts chains and lane trees (TF_NO_TREES, TF_DRAFT_NODES=0).
pub fn nodes() usize {
    if (platform.c.getenv("TF_NO_TREES") != null) return 0;
    const n = if (platform.c.getenv("TF_DRAFT_NODES")) |v| std.fmt.parseInt(usize, std.mem.span(v), 10) catch default_nodes else default_nodes;
    return @min(n, engine.most_window - 1);
}

/// Offered tree nodes (parent row kept) and landed ones by own log-chance in half-nat bins, greedy then every round's.
pub const Landing = struct {
    counts: [2][16][2]f64 = @splat(@splat(.{ 0, 0 })),

    fn bin(own: f64) usize {
        return @min(15, @as(usize, @intFromFloat(@max(0, @floor(-own / 0.5)))));
    }

    pub fn observe(l: *Landing, sampled: bool, own: f64, landed: bool) void {
        for (l.counts[@intFromBool(sampled)..]) |*table| {
            const x = &table[bin(own)];
            x[0] += 1;
            if (landed) x[1] += 1;
        }
    }

    pub fn chance(l: *const Landing, sampled: bool, own: f64) f64 {
        const x = l.counts[@intFromBool(sampled)][bin(own)];
        return (x[1] + 2 * @exp(own)) / (x[0] + 2);
    }

    /// A verified window's offered nodes (the first `offered`, node i at row i + 1) against the rows it kept.
    pub fn observeTree(l: *Landing, sampled: bool, parents: []const i32, own: []const f64, offered: usize, kept: []const u32) void {
        for (parents[0..@min(offered, parents.len)], own[0..@min(offered, own.len)], 0..) |p, o, i| {
            const parent_row: u32 = if (p < 0) 0 else @intCast(p + 1);
            if (std.mem.indexOfScalar(u32, kept, parent_row) == null) continue;
            l.observe(sampled, o, std.mem.indexOfScalar(u32, kept, @intCast(i + 1)) != null);
        }
    }
};

fn bf16(h: u16) f64 {
    return @as(f32, @bitCast(@as(u32, h) << 16));
}

fn bf32(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

/// Row `r`'s `rank`th candidate after `pred`: unary logit + pred . projected . successor, ties to the earlier one.
fn choice(c: Candidates, r: usize, pred: u32, rank: usize) u32 {
    const w = ids(c, r);
    const hp = std.mem.bytesAsSlice(u16, c.projected[r * c.rank * 2 ..][0 .. c.rank * 2]);
    var scores: [lp_top]f32 = undefined;
    for (0..c.topk) |j| {
        var edge: f32 = 0;
        for (0..c.rank) |k| edge += bf32(c.predecessor[pred * c.rank + k]) * bf32(hp[k]) * bf32(c.successor[w[2 + j] * c.rank + k]);
        scores[j] = @as(f32, @bitCast(w[2 + lp_top + j])) + edge;
    }
    var taken: [lp_top]bool = @splat(false);
    var pick: usize = 0;
    for (0..rank + 1) |_| {
        var best: f32 = -std.math.inf(f32);
        pick = 0;
        for (0..c.topk) |j| if (!taken[j] and scores[j] > best) {
            best = scores[j];
            pick = j;
        };
        taken[pick] = true;
    }
    return w[2 + pick];
}

/// The path through the rows' candidates: each row's best after the one before.
pub fn chain(c: Candidates, pending: u32, out: []u32) void {
    var pred = pending;
    for (out, 0..) |*o, r| {
        o.* = choice(c, r, pred, 0);
        pred = o.*;
    }
}

/// The planner's lanes in order: lane i the rank-r candidate at its depth's row after its parent lane's token.
pub fn lanesOf(c: Candidates, pending: u32, rows: usize, l: *const lanes.shape.Shape, out: []u32) !void {
    for (l.parents, l.depths, l.ranks, 0..) |par, depth, rank, i| {
        if (depth >= rows or rank >= c.topk or par >= @as(i32, @intCast(i))) return error.BadLanes;
        out[i] = choice(c, depth, if (par < 0) pending else out[@intCast(par)], rank);
    }
}

fn ids(c: Candidates, r: usize) []align(1) const u32 {
    return std.mem.bytesAsSlice(u32, c.words[r * lp_words * 4 ..][0 .. lp_words * 4]);
}

/// Row `r`'s candidates after `pred` as log-chances: softmax of (unary + edge x selector) / scale.
fn chancesAfter(c: Candidates, r: usize, pred: u32, out: *[lp_top]f64) void {
    const w = ids(c, r);
    const hp = std.mem.bytesAsSlice(u16, c.projected[r * c.rank * 2 ..][0 .. c.rank * 2]);
    var most: f64 = -std.math.inf(f64);
    for (0..c.topk) |j| {
        var edge: f64 = 0;
        for (0..c.rank) |k| edge += bf16(c.predecessor[pred * c.rank + k]) * bf16(hp[k]) * bf16(c.successor[w[2 + j] * c.rank + k]);
        out[j] = (@as(f64, @as(f32, @bitCast(w[2 + lp_top + j]))) + edge_weight * edge) / scale;
        most = @max(most, out[j]);
    }
    var total: f64 = 0;
    for (out[0..c.topk]) |v| total += @exp(v - most);
    const log_total = most + @log(total);
    for (out[0..c.topk]) |*v| v.* -= log_total;
}

/// Up to `most` nodes best first: the open node of least path cost next (ties by parent, token, row); the count grown.
pub fn grow(c: Candidates, pending: u32, rows: usize, most: usize, tokens: []u32, parents: []i32, own: []f64) usize {
    const Open = struct { cost: f64, parent: i32, token: u32, row: usize, own: f64 };
    var open: [branch * (engine.most_window + 1)]Open = undefined;
    var n_open: usize = 0;
    var count: usize = 0;
    var pred = pending;
    var parent: i32 = -1;
    var row: usize = 0;
    var path: f64 = 0;
    while (true) {
        // the node just taken (or the pending token) opens its best children on the next row
        if (row < rows) {
            var lc: [lp_top]f64 = undefined;
            chancesAfter(c, row, pred, &lc);
            var order: [lp_top]usize = undefined;
            for (order[0..c.topk], 0..) |*o, i| o.* = i;
            std.mem.sort(usize, order[0..c.topk], &lc, struct {
                fn more(v: *const [lp_top]f64, a: usize, b: usize) bool {
                    return v[a] > v[b] or (v[a] == v[b] and a < b);
                }
            }.more);
            const w = ids(c, row);
            for (order[0..@min(branch, c.topk)]) |j| {
                if (!std.math.isFinite(lc[j]) or n_open == open.len) break;
                open[n_open] = .{ .cost = path - lc[j], .parent = parent, .token = w[2 + j], .row = row, .own = lc[j] };
                n_open += 1;
            }
        }
        if (count == @min(most, tokens.len) or n_open == 0) break;
        var best: usize = 0;
        for (open[1..n_open], 1..) |o, i| {
            const b = open[best];
            if (o.cost < b.cost or (o.cost == b.cost and (o.parent < b.parent or (o.parent == b.parent and (o.token < b.token or (o.token == b.token and o.row < b.row)))))) best = i;
        }
        const node = open[best];
        open[best] = open[n_open - 1];
        n_open -= 1;
        tokens[count] = node.token;
        parents[count] = node.parent;
        own[count] = node.own;
        pred = node.token;
        parent = @intCast(count);
        row = node.row + 1;
        path = node.cost;
        count += 1;
    }
    return count;
}

test "landing: a node is offered when its parent's row is kept, and landed when its own is" {
    var l: Landing = .{};
    // nodes 0 and 1 under the pending row, node 2 under node 0; the window kept rows 0, 1 (node 0) and 3 (node 2)
    l.observeTree(false, &.{ -1, -1, 0 }, &.{ -0.1, -1.2, -0.3 }, 3, &.{ 0, 1, 3 });
    try std.testing.expectEqual(@as(f64, 2), l.counts[0][0][0]); // nodes 0 and 2, both landed
    try std.testing.expectEqual(@as(f64, 2), l.counts[0][0][1]);
    try std.testing.expectEqual(@as(f64, 1), l.counts[0][2][0]); // node 1 offered, not landed
    try std.testing.expectEqual(@as(f64, 0), l.counts[0][2][1]);
    try std.testing.expectEqual(l.counts[0], l.counts[1]); // greedy rounds count for sampled streams too
    l.observeTree(true, &.{-1}, &.{-0.1}, 1, &.{0});
    try std.testing.expect(l.chance(true, -0.1) < l.chance(false, -0.1));
    try std.testing.expectApproxEqAbs((2 + 2 * @exp(@as(f64, -0.1))) / 4, l.chance(false, -0.1), 1e-12);
}

test "best first: a node's children follow it, the cheapest open node first, chances that sum to one a row" {
    // two rows, four candidates each, no selector (rank 1, zero codebooks): unary logits alone order the tree
    const rank = 1;
    var words: [2 * lp_words * 4]u8 = @splat(0);
    const w = std.mem.bytesAsSlice(u32, &words);
    const unary = [2][4]f32{ .{ 3, 1, 0, -1 }, .{ 2, 2, -5, -5 } };
    for (0..2) |r| for (0..4) |j| {
        w[r * lp_words + 2 + j] = @intCast(10 * r + j);
        w[r * lp_words + 2 + lp_top + j] = @bitCast(unary[r][j]);
    };
    const zeros: [32]u16 = @splat(0);
    const c: Candidates = .{ .words = &words, .projected = std.mem.sliceAsBytes(zeros[0..2]), .topk = 4, .rank = rank, .predecessor = &zeros, .successor = &zeros };
    var lc: [lp_top]f64 = undefined;
    chancesAfter(c, 0, 0, &lc);
    var total: f64 = 0;
    for (lc[0..4]) |v| total += @exp(v);
    try std.testing.expectApproxEqAbs(@as(f64, 1), total, 1e-12);
    var tokens: [6]u32 = undefined;
    var parents: [6]i32 = undefined;
    var own: [6]f64 = undefined;
    const n = grow(c, 5, 2, 6, &tokens, &parents, &own);
    try std.testing.expectEqual(@as(usize, 6), n);
    try std.testing.expectEqual(@as(u32, 0), tokens[0]); // row 0's best first
    try std.testing.expectEqual(@as(i32, -1), parents[0]);
    for (parents[0..n], 0..) |p, i| try std.testing.expect(p < @as(i32, @intCast(i)));
    for (tokens[0..n], parents[0..n]) |t, p| try std.testing.expectEqual(p >= 0, t >= 10); // row 1's ids under row 0's
}
