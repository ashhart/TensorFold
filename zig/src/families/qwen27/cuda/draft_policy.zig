//! draft_tree.best_first on the host: which of DFlash2's candidates a round verifies, and where each hangs.

const std = @import("std");
const lanes = @import("lanes");

pub const branch = 4;
const edge_weight = 0.6;
const noise_weight = 0.7;
const scale = 1.5;
pub const rank = 256; // the selector's codebook width
pub const top = 16; // candidates a depth

/// A drafted block's candidates: each depth's 16 token ids, their logits (unary) and the selector's projection.
pub const Candidates = struct { ids: []const u32, unary: []const f64, projected: []const f64, depths: usize };

/// The tree: nodes in pop order (parents first), each node's parent (-1: hangs from the anchor), path scores.
pub const Tree = struct { tokens: std.ArrayList(u32) = .empty, parents: std.ArrayList(i32) = .empty, scores: std.ArrayList(f64) = .empty };

const Item = struct { score: f64, parent: i32, token: u32, depth: usize };

fn before(_: void, a: Item, b: Item) std.math.Order {
    // heapq's tuple order: score, then parent, then token, then depth
    if (a.score != b.score) return std.math.order(a.score, b.score);
    if (a.parent != b.parent) return std.math.order(a.parent, b.parent);
    if (a.token != b.token) return std.math.order(a.token, b.token);
    return std.math.order(a.depth, b.depth);
}

/// best_first: expand the anchor's candidates, then pop the best path score until `max_nodes` are placed.
pub fn bestFirst(gpa: std.mem.Allocator, c: Candidates, pred: []const f32, succ: []const f32, anchor: u32, max_nodes: usize, sampling: ?lanes.Sampling, first_position: u64, out: *Tree) !void {
    out.tokens.clearRetainingCapacity();
    out.parents.clearRetainingCapacity();
    out.scores.clearRetainingCapacity();
    const temp: f64 = if (sampling) |s| @max(s.temperature, 1e-6) else 1.0;
    var heap: std.PriorityQueue(Item, void, before) = .empty;
    defer heap.deinit(gpa);
    const Ctx = struct {
        gpa: std.mem.Allocator,
        c: Candidates,
        pred: []const f32,
        succ: []const f32,
        temp: f64,
        sampling: ?lanes.Sampling,
        first_position: u64,
        heap: *std.PriorityQueue(Item, void, before),

        fn expand(x: @This(), token: u32, depth: usize, parent: i32, path: f64) !void {
            var values: [top]f64 = undefined;
            const p = x.pred[@as(usize, token) * rank ..][0..rank];
            const proj = x.c.projected[depth * rank ..][0..rank];
            var weighted: [rank]f64 = undefined;
            for (&weighted, p, proj) |*wgt, a, b| wgt.* = @as(f64, a) * b;
            for (&values, 0..) |*v, i| {
                const cand = x.c.ids[depth * top + i];
                const s = x.succ[@as(usize, cand) * rank ..][0..rank];
                var edge: f64 = 0;
                for (s, weighted) |a, b| edge += @as(f64, a) * b;
                v.* = (x.c.unary[depth * top + i] + edge_weight * edge) / x.temp;
                if (x.sampling) |smp| v.* += noise_weight * -@log(-@log(lanes.sampling.uniform(smp.seed, x.first_position + depth, cand)));
                v.* /= scale;
            }
            var most = values[0];
            for (values[1..]) |v| most = @max(most, v);
            var total: f64 = 0;
            for (&values) |*v| {
                v.* -= most;
                total += @exp(v.*);
            }
            const log_total = @log(total);
            var order: [top]usize = undefined;
            for (&order, 0..) |*o, i| o.* = i;
            std.mem.sort(usize, &order, &values, struct {
                fn more(vals: *const [top]f64, a: usize, b: usize) bool {
                    return vals[a] > vals[b] or (vals[a] == vals[b] and a < b);
                }
            }.more);
            for (order[0..branch]) |i| {
                const logp = values[i] - log_total;
                if (!std.math.isFinite(logp)) break;
                try x.heap.push(x.gpa, .{ .score = path - logp, .parent = parent, .token = x.c.ids[depth * top + i], .depth = depth });
            }
        }
    };
    const ctx: Ctx = .{ .gpa = gpa, .c = c, .pred = pred, .succ = succ, .temp = temp, .sampling = sampling, .first_position = first_position, .heap = &heap };
    try ctx.expand(anchor, 0, -1, 0.0);
    while (heap.count() > 0 and out.tokens.items.len < max_nodes) {
        const it = heap.pop().?;
        const me: i32 = @intCast(out.tokens.items.len);
        try out.tokens.append(gpa, it.token);
        try out.parents.append(gpa, it.parent);
        try out.scores.append(gpa, it.score);
        if (it.depth + 1 < c.depths) try ctx.expand(it.token, it.depth + 1, me, it.score);
    }
}

test "best_first pops by path score, parents before children, ties by parent" {
    const gpa = std.testing.allocator;
    const vocab = 32;
    const pred = try gpa.alloc(f32, vocab * rank);
    defer gpa.free(pred);
    const succ = try gpa.alloc(f32, vocab * rank);
    defer gpa.free(succ);
    @memset(pred, 0);
    @memset(succ, 0);
    var ids: [2 * top]u32 = undefined;
    var unary: [2 * top]f64 = undefined;
    for (0..2) |d| for (0..top) |i| {
        ids[d * top + i] = @intCast(i);
        unary[d * top + i] = -@as(f64, @floatFromInt(i));
    };
    const projected: [2 * rank]f64 = @splat(0);
    var tree: Tree = .{};
    defer {
        tree.tokens.deinit(gpa);
        tree.parents.deinit(gpa);
        tree.scores.deinit(gpa);
    }
    try bestFirst(gpa, .{ .ids = &ids, .unary = &unary, .projected = &projected, .depths = 2 }, pred, succ, 3, 5, null, 100, &tree);
    // logp_i = -i / 1.5 - ln(sum): the anchor's 0 and 1 (score 0.72, 1.39) come before 0's child 0 (1.44)
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 0, 2, 1 }, tree.tokens.items);
    try std.testing.expectEqualSlices(i32, &.{ -1, -1, 0, -1, 0 }, tree.parents.items);
}
