//! Batched MTP predictions correct a DFlash prefix, whose remaining marginal rows can then continue after it.
const std = @import("std");
const tree = @import("cuda_draft_tree.zig");
const coop = @import("cuda_draft_cooperative.zig");
const frame = @import("cuda_draft_frame.zig");
const batch = @import("cuda_mtp_batch.zig");
const engine = @import("cuda_engine.zig");
const lanes = @import("lanes");
const most = engine.most_window;
const lp_words = 43;
const branch = 4;

pub fn enabled(raw: ?[]const u8) !bool {
    const value = raw orelse return false;
    if (std.mem.eql(u8, value, "0")) return false;
    if (std.mem.eql(u8, value, "1")) return true;
    return error.BadGraft;
}

pub fn depths(base: *const coop.Merged, out: []usize) !void {
    if (base.count >= most or out.len < base.count) return error.BadGraft;
    for (base.parents[0..base.count], 0..) |parent, i| {
        if (parent < -1 or parent >= @as(i32, @intCast(i))) return error.BadGraft;
        out[i] = if (parent < 0) 1 else out[@intCast(parent)] + 1;
    }
}

fn original(base: *const coop.Merged, parent: i32, token: u32) i32 {
    if (parent < -1) return -2;
    for (base.tokens[0..base.count], base.parents[0..base.count], 0..) |t, p, i|
        if (t == token and p == parent) return @intCast(i);
    return -2;
}

const Choices = struct { ids: [40]u32 = undefined, probability: [40]f64 = @splat(0), count: usize = 0 };

fn choices(c: tree.Candidates, row: usize, pending: u32, mtp: ?*const batch.LogitsRow, weight: f64) !Choices {
    const rows = c.words.len / (lp_words * 4);
    const has_df = row < rows;
    var result: Choices = .{};
    if (has_df) {
        var lc: [20]f64 = undefined;
        tree.chancesAfter(c, row, pending, &lc);
        const words = tree.ids(c, row);
        for (0..c.topk) |j| {
            if (!std.math.isFinite(lc[j])) return error.BadGraftScores;
            result.ids[j] = words[2 + j];
            result.probability[j] = @exp(lc[j]) * (if (mtp != null) 1 - weight else @as(f64, 1));
        }
        result.count = c.topk;
    }
    if (mtp) |m| {
        var scores: [16]f64 = undefined;
        var high: f64 = -std.math.inf(f64);
        for (m.ids, m.logits, 0..) |token, logit, j| {
            if (token >= c.predecessor.len / c.rank or !std.math.isFinite(logit)) return error.BadGraftScores;
            scores[j] = tree.drawn(c, row, token, logit) / tree.scale;
            high = @max(high, scores[j]);
        }
        var total: f64 = 0;
        for (scores) |s| total += @exp(s - high);
        const normal = high + @log(total);
        const mix: f64 = if (has_df) weight else if (weight > 0) 1 else 0;
        for (m.ids, scores) |token, score| {
            const found = std.mem.indexOfScalar(u32, result.ids[0..result.count], token);
            const at = found orelse blk: {
                result.ids[result.count] = token;
                result.count += 1;
                break :blk result.count - 1;
            };
            result.probability[at] += mix * @exp(score - normal);
        }
    }
    return result;
}

/// Original-prefix indices select MTP predictions; a repaired prefix continues with DFlash until another query exists.
pub fn grow(c: tree.Candidates, pending: u32, base: *const coop.Merged, mtp: []const batch.LogitsRow, weight: f64, budget: usize) !coop.Merged {
    if (!std.math.isFinite(weight) or weight < 0 or weight > 1 or budget >= most or
        c.rank == 0 or c.topk == 0 or c.topk > 20 or mtp.len != base.count + 1 or
        c.words.len % (lp_words * 4) != 0) return error.BadGraft;
    var base_depth: [most]usize = undefined;
    try depths(base, &base_depth);
    const rows = c.words.len / (lp_words * 4);
    if (c.projected.len != rows * c.rank * 2 or pending >= c.predecessor.len / c.rank) return error.BadGraft;
    var out: coop.Merged = .{};
    if (weight == 0) {
        out.count = tree.grow(c, pending, rows, budget, &out.tokens, &out.parents, &out.own);
        return out;
    }
    const Open = struct { cost: f64, parent: i32, token: u32, row: usize, own: f64, original: i32 };
    var open: [branch * (most + 1)]Open = undefined;
    var n_open: usize = 0;
    var parent: i32 = -1;
    var pred = pending;
    var row: usize = 0;
    var path: f64 = 0;
    var orig: i32 = -1;
    while (true) {
        const model = if (orig == -1) &mtp[base.count] else if (orig >= 0) &mtp[@intCast(orig)] else null;
        if (row < rows or model != null) {
            const next = try choices(c, row, pred, model, weight);
            var order: [40]usize = undefined;
            for (order[0..next.count], 0..) |*v, i| v.* = i;
            std.mem.sort(usize, order[0..next.count], &next, struct {
                fn more(v: *const Choices, a: usize, b: usize) bool {
                    return v.probability[a] > v.probability[b] or
                        (v.probability[a] == v.probability[b] and v.ids[a] < v.ids[b]);
                }
            }.more);
            for (order[0..@min(branch, next.count)]) |j| {
                if (next.probability[j] <= 0 or n_open == open.len) continue;
                const own = @log(@min(1, next.probability[j]));
                open[n_open] = .{ .cost = path - own, .parent = parent, .token = next.ids[j], .row = row, .own = own, .original = original(base, orig, next.ids[j]) };
                n_open += 1;
            }
        }
        if (out.count == budget or n_open == 0) break;
        var best: usize = 0;
        for (open[1..n_open], 1..) |o, i| {
            const b = open[best];
            if (o.cost < b.cost or (o.cost == b.cost and (o.parent < b.parent or
                (o.parent == b.parent and (o.token < b.token or (o.token == b.token and o.row < b.row)))))) best = i;
        }
        const node = open[best];
        open[best] = open[n_open - 1];
        n_open -= 1;
        out.tokens[out.count] = node.token;
        out.parents[out.count] = node.parent;
        out.own[out.count] = node.own;
        parent = @intCast(out.count);
        pred = node.token;
        row = node.row + 1;
        path = node.cost;
        orig = node.original;
        out.count += 1;
    }
    return out;
}

fn candidates(x: anytype, item: lanes.drafter.Hold, index: usize) tree.Candidates {
    const d = x.df;
    const rows = d.c.block - 1;
    const base = index * rows;
    return .{ .words = d.staged.bytes[frame.words_at + base * lp_words * 4 ..][0 .. rows * lp_words * 4], .projected = d.staged.bytes[frame.hp_at + base * d.c.rank * 2 ..][0 .. rows * d.c.rank * 2], .rank = d.c.rank, .topk = d.c.topk, .predecessor = d.w.predecessor, .successor = d.w.successor, .sampling = item.sampling, .position = item.position };
}

pub fn forward(x: anytype, items: []const lanes.drafter.Hold) !void {
    var first: usize = 0;
    while (first < items.len) {
        const group_size = @min(4, most / (x.query_nodes + 1));
        if (group_size == 0) return error.BadGraft;
        const n = @min(group_size, items.len - first);
        var queries: [most]batch.Branch = undefined;
        var roots: [4]*@import("cuda_mtp.zig").Ctx = undefined;
        var bases: [4]coop.Merged = undefined;
        var starts: [4]usize = undefined;
        var root_indices: [4]usize = undefined;
        var n_queries: usize = 0;
        var n_roots: usize = 0;
        for (items[first..][0..n], 0..) |item, local| {
            const df = x.df.streams.get(item.stream) orelse return error.UnknownStream;
            const mt = x.mtp.streams.get(item.stream) orelse return error.UnknownStream;
            if (df.grown > x.query_nodes) return error.BadGraft;
            const base = &bases[local];
            base.count = if (item.depth == 0) 0 else df.grown;
            @memcpy(base.tokens[0..base.count], df.held[0..base.count]);
            @memcpy(base.parents[0..base.count], df.parents[0..base.count]);
            @memcpy(base.own[0..base.count], df.own[0..base.count]);
            starts[local] = n_queries;
            root_indices[local] = n_roots;
            if (base.count == 0) continue;
            roots[n_roots] = mt;
            n_roots += 1;
            var depth: [most]usize = undefined;
            try depths(base, &depth);
            for (base.tokens[0..base.count], depth[0..base.count], 0..) |token, dep, node| {
                if (dep > x.df.c.block - 1) return error.BadGraft;
                const hidden = x.df.x + ((first + local) * (x.df.c.block - 1) + dep - 1) * x.e.g.hidden * 2;
                queries[n_queries] = .{ .ctx = mt, .token = token, .hidden = if (dep == 1) mt.last.ptr else hidden, .position = try std.math.add(u64, mt.end, dep - 1) };
                queries[n_queries].parent = if (base.parents[node] < 0) -1 else @intCast(starts[local] + @as(usize, @intCast(base.parents[node])));
                n_queries += 1;
            }
        }
        if (n_roots > 0) {
            var predictions: [most]batch.LogitsRow = undefined;
            if (x.prefix) try batch.prefixTopk(x.mtp, queries[0..n_queries], roots[0..n_roots], predictions[0 .. n_queries + n_roots]) else try batch.branchTopk(x.mtp, queries[0..n_queries], roots[0..n_roots], predictions[0 .. n_queries + n_roots]);
            for (items[first..][0..n], 0..) |item, local| {
                const base = &bases[local];
                if (base.count == 0) continue;
                var model: [most]batch.LogitsRow = undefined;
                @memcpy(model[0..base.count], predictions[starts[local]..][0..base.count]);
                model[base.count] = predictions[n_queries + root_indices[local]];
                const merged = try grow(candidates(x, item, first + local), item.pending, base, model[0 .. base.count + 1], x.weight, x.df.tree_nodes);
                const df = x.df.streams.get(item.stream).?;
                @memcpy(df.held[0..merged.count], merged.tokens[0..merged.count]);
                @memcpy(df.parents[0..merged.count], merged.parents[0..merged.count]);
                @memcpy(df.own[0..merged.count], merged.own[0..merged.count]);
                df.held_n = merged.count;
                df.grown = merged.count;
            }
        }
        first += n;
    }
}

pub fn warm(x: anytype) !f64 {
    if (x.prefix) try @import("cuda_mtp_check.zig").prefix(x.mtp);
    const ctx = x.mtp.pool.items[0];
    var queries: [most]batch.Branch = undefined;
    for (queries[0..x.query_nodes], 0..) |*q, i| q.* = .{ .ctx = ctx, .token = 0, .hidden = x.df.x + (i % (x.df.c.block - 1)) * x.e.g.hidden * 2, .position = ctx.end + i % (x.df.c.block - 1), .parent = if (i % (x.df.c.block - 1) == 0) -1 else @as(i32, @intCast(i)) - 1 };
    var output: [most]batch.LogitsRow = undefined;
    var quickest: f64 = std.math.inf(f64);
    for (0..4) |_| {
        const t0 = std.Io.Clock.awake.now(x.df.io);
        if (x.prefix) try batch.prefixTopk(x.mtp, queries[0..x.query_nodes], &.{ctx}, output[0 .. x.query_nodes + 1]) else try batch.branchTopk(x.mtp, queries[0..x.query_nodes], &.{ctx}, output[0 .. x.query_nodes + 1]);
        const ns = t0.durationTo(std.Io.Clock.awake.now(x.df.io)).toNanoseconds();
        quickest = @min(quickest, @as(f64, @floatFromInt(ns)) / 1e6);
    }
    return quickest;
}

fn peak(token: u32) batch.LogitsRow {
    var row: batch.LogitsRow = undefined;
    for (&row.ids, &row.logits, 0..) |*id, *logit, i| {
        id.* = @intCast(i);
        logit.* = if (i == token) 20 else -50;
    }
    return row;
}

test "graft repairs an omitted token then continues DFlash without borrowing another prefix's MTP state" {
    var words: [3 * lp_words * 4]u8 = @splat(0);
    const w = std.mem.bytesAsSlice(u32, &words);
    for (0..3) |row| for (0..2) |j| {
        w[row * lp_words + 2 + j] = @intCast(1 + row * 2 + j);
        w[row * lp_words + 22 + j] = @bitCast(@as(f32, if (j == 0) 5 else -5));
    };
    const zeros: [80]u16 = @splat(0);
    const c: tree.Candidates = .{ .words = &words, .projected = std.mem.sliceAsBytes(zeros[0..3]), .topk = 2, .rank = 1, .predecessor = &zeros, .successor = &zeros };
    var base: coop.Merged = .{ .count = 3 };
    @memcpy(base.tokens[0..3], &[_]u32{ 1, 3, 5 });
    @memcpy(base.parents[0..3], &[_]i32{ -1, 0, 1 });
    @memset(base.own[0..3], -0.01);
    const model = [_]batch.LogitsRow{ peak(9), peak(5), peak(7), peak(1) };
    const corrected = try grow(c, 0, &base, &model, 1, 3);
    try std.testing.expectEqualSlices(u32, &.{ 1, 9, 5 }, corrected.tokens[0..corrected.count]);
    try std.testing.expectEqualSlices(i32, &.{ -1, 0, 1 }, corrected.parents[0..corrected.count]);
    const wider = try grow(c, 0, &base, &model, 1, 10);
    for (wider.tokens[0..wider.count], wider.parents[0..wider.count]) |token, parent|
        try std.testing.expect(!(parent == 2 and token == 7));
    var bad = model;
    bad[3].logits[0] = std.math.nan(f32);
    try std.testing.expectError(error.BadGraftScores, grow(c, 0, &base, &bad, 1, 3));
    base.parents[1] = 1;
    try std.testing.expectError(error.BadGraft, grow(c, 0, &base, &model, 1, 3));
}

test "graft extends the original tree past the diffusion block while zero weight keeps the old builder" {
    var words: [3 * lp_words * 4]u8 = @splat(0);
    const w = std.mem.bytesAsSlice(u32, &words);
    for (0..3) |row| for (0..2) |j| {
        w[row * lp_words + 2 + j] = @intCast(1 + row * 2 + j);
        w[row * lp_words + 22 + j] = @bitCast(@as(f32, if (j == 0) 5 else -5));
    };
    const zeros: [80]u16 = @splat(0);
    const c: tree.Candidates = .{ .words = &words, .projected = std.mem.sliceAsBytes(zeros[0..3]), .topk = 2, .rank = 1, .predecessor = &zeros, .successor = &zeros };
    var base: coop.Merged = .{ .count = 3 };
    @memcpy(base.tokens[0..3], &[_]u32{ 1, 3, 5 });
    @memcpy(base.parents[0..3], &[_]i32{ -1, 0, 1 });
    @memset(base.own[0..3], -0.01);
    const model = [_]batch.LogitsRow{ peak(3), peak(5), peak(7), peak(1) };
    const extended = try grow(c, 0, &base, &model, 1, 4);
    try std.testing.expectEqualSlices(u32, &.{ 1, 3, 5, 7 }, extended.tokens[0..extended.count]);
    const control = try grow(c, 0, &base, &model, 0, 4);
    var expected: coop.Merged = .{};
    expected.count = tree.grow(c, 0, 3, 4, &expected.tokens, &expected.parents, &expected.own);
    try std.testing.expectEqualSlices(u32, expected.tokens[0..expected.count], control.tokens[0..control.count]);
    try std.testing.expectEqualSlices(i32, expected.parents[0..expected.count], control.parents[0..control.count]);
    try std.testing.expectEqualSlices(f64, expected.own[0..expected.count], control.own[0..control.count]);
}
