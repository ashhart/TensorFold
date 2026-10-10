//! decode.draft_decode for one stream: copies from the context, else a DFlash2 tree; verify, walk, commit, absorb.

const std = @import("std");
const lanes = @import("lanes");
const c = @import("shape.zig");
const st = @import("state.zig");
const tree = @import("tree.zig");
const kern = @import("kernels.zig");
const dr = @import("draft.zig");
const dl = @import("draft_load.zig");
const Engine = @import("engine.zig").Engine;
const Forward = @import("forward.zig").Forward;
const Part = @import("forward.zig").Part;
const Lanes = @import("lanes.zig").Cuda;

const bf = 2;
const taps_width = dl.taps.len * dl.hidden;

/// decode.CopyIndex: earlier 8-token matches of the context's suffix, the longest continuation proposed.
pub const CopyIndex = struct {
    pub const n = 8;
    const Key = [n]u32;
    gpa: std.mem.Allocator,
    positions: std.AutoHashMapUnmanaged(Key, std.ArrayList(usize)) = .empty,
    indexed: usize = 0,

    pub fn deinit(x: *CopyIndex) void {
        var it = x.positions.valueIterator();
        while (it.next()) |l| l.deinit(x.gpa);
        x.positions.deinit(x.gpa);
    }

    fn update(x: *CopyIndex, context: []const u32) !void {
        if (context.len < n) return;
        const last = context.len - n; // the trailing query itself stays out of the index
        var start = x.indexed;
        while (start < last) : (start += 1) {
            const gop = try x.positions.getOrPut(x.gpa, context[start..][0..n].*);
            if (!gop.found_existing) gop.value_ptr.* = .empty;
            try gop.value_ptr.append(x.gpa, start);
        }
        x.indexed = @max(x.indexed, last);
    }

    /// The longest continuation after an earlier occurrence of the last 8 tokens (latest first), at least 8 long.
    pub fn propose(x: *CopyIndex, context: []const u32, max_nodes: usize) ![]const u32 {
        if (context.len < 2 * n) return &.{};
        try x.update(context);
        const starts = x.positions.get(context[context.len - n ..][0..n].*) orelse return &.{};
        var best: []const u32 = &.{};
        var i = starts.items.len;
        while (i > 0) {
            i -= 1;
            const from = starts.items[i] + n;
            const cont = context[from..@min(context.len, from + max_nodes)];
            if (cont.len > best.len) {
                best = cont;
                if (best.len == max_nodes) break;
            }
        }
        return if (best.len >= n) best else &.{};
    }
};

/// decode.next_copy_rows: a copy's next window, twice as wide after a whole copy, half after a broken one.
pub fn nextCopyRows(rows: usize, whole: bool, tree_rows: usize, max_rows: usize) usize {
    const first = @min(max_rows, @max(tree_rows, 16));
    return if (whole) @min(max_rows, @max(rows, first) * 2) else @max(first, @min(rows, max_rows) / 2);
}

pub const Stats = struct { rounds: usize = 0, drafted: usize = 0, accepted: usize = 0 };

/// prefill_state with a drafter: the prompt in even chunks, its last `window` rows' taps into the context.
pub fn prefill(e: *Engine, seq: *st.Seq, ctx: *dr.Context, d: *dr.DFlash2, prompt: []const u32) !void {
    try seq.reset(e.ops(), e.w.g);
    ctx.len = 0;
    ctx.end = 0;
    const tap_from = prompt.len -| dl.window;
    ctx.skip(tap_from);
    var f = e.forward();
    f.taps = e.scratch.taps;
    const rows = @import("engine.zig").prompt_rows;
    const chunks = (prompt.len + rows - 1) / rows;
    for (0..chunks) |j| {
        const b = @import("engine.zig").chunkBounds(0, prompt.len, rows, j);
        try f.chunk(prompt[b[0]..b[1]], seq, j == chunks - 1);
        if (b[1] <= tap_from) continue;
        const skip = tap_from -| b[0];
        try d.absorb(ctx, e.scratch.taps + skip * taps_width * bf, b[1] - b[0] - skip);
    }
}

/// draft_decode, greedy: rounds until `count` tokens or an end token; the first token is `pending`.
pub fn decode(e: *Engine, seq: *st.Seq, ctx: *dr.Context, d: *dr.DFlash2, prompt: []const u32, pending: u32, count: usize, max_rows: usize, out: *std.ArrayList(u32), stats: *Stats) !void {
    const gpa = e.gpa;
    var context: std.ArrayList(u32) = .empty;
    defer context.deinit(gpa);
    try context.appendSlice(gpa, prompt);
    try out.append(gpa, pending);
    try context.append(gpa, pending);
    var copies: CopyIndex = .{ .gpa = gpa };
    defer copies.deinit();
    const tree_rows = max_rows;
    var copy_rows = nextCopyRows(tree_rows, false, tree_rows, max_rows);
    var tokens: [st.max_rows]u32 = undefined;
    var parents: [st.max_rows]i32 = undefined;
    var sampled: [st.max_rows]u32 = undefined;
    var path: [st.max_rows]u32 = undefined;
    var f = e.forward();
    f.taps = e.scratch.taps;
    while (out.items.len < count and !e.w.isEos(out.items[out.items.len - 1])) {
        const last = out.items[out.items.len - 1];
        tokens[0] = last;
        parents[0] = -1;
        var rows: usize = 1;
        const copied = try copies.propose(context.items, copy_rows - 1);
        if (copied.len > 0) {
            for (copied, 1..) |t, r| {
                tokens[r] = t;
                parents[r] = @as(i32, @intCast(r)) - 1;
            }
            rows += copied.len;
        } else if (ctx.len > 0) {
            const p = try d.propose(ctx, last, context.items.len, tree_rows - 1, null);
            for (p.tokens, p.parents, 1..) |t, pr, r| {
                tokens[r] = t;
                parents[r] = if (pr < 0) 0 else pr + 1;
            }
            rows += p.tokens.len;
        }
        const part = [_]Part{.{ .seq = seq, .tokens = tokens[0..rows], .parents = parents[0..rows] }};
        try f.round(&part);
        try f.picks(rows, &sampled);
        // the path: from the root, each next token the child whose token the target drew
        var n: usize = 1;
        path[0] = 0;
        var terminal = sampled[0];
        while (out.items.len + n < count and !e.w.isEos(terminal)) {
            const child = for (1..rows) |r| {
                if (parents[r] == @as(i32, @intCast(path[n - 1])) and tokens[r] == terminal) break r;
            } else break;
            path[n] = @intCast(child);
            n += 1;
            terminal = sampled[child];
        }
        try f.commit(&part, &.{path[0..n]});
        // the drafter reads the kept rows' taps, gathered in path order
        var items: [st.max_rows]kern.Copy = undefined;
        const gathered = e.scratch.taps + st.max_rows * taps_width * bf;
        for (path[0..n], 0..) |row, j| items[j] = .{ .dst = gathered + j * taps_width * bf, .src = e.scratch.taps + row * taps_width * bf, .bytes = taps_width * bf };
        try f.ops.upload(e.scratch.copy_items, std.mem.sliceAsBytes(items[0..n]));
        try f.ops.copies(e.scratch.copy_items, n);
        try d.absorb(ctx, gathered, n);
        for (path[1..n]) |row| {
            try out.append(gpa, tokens[row]);
            try context.append(gpa, tokens[row]);
        }
        try out.append(gpa, terminal);
        try context.append(gpa, terminal);
        if (copied.len > 0) copy_rows = nextCopyRows(copy_rows, n == rows, tree_rows, max_rows);
        stats.rounds += 1;
        stats.drafted += rows - 1;
        stats.accepted += n - 1;
    }
}

test "copy windows grow after whole copies and shrink after broken ones" {
    try std.testing.expectEqual(@as(usize, 12), nextCopyRows(12, false, 12, 12));
    try std.testing.expectEqual(@as(usize, 12), nextCopyRows(12, true, 12, 12));
    try std.testing.expectEqual(@as(usize, 16), nextCopyRows(16, false, 12, 64));
    try std.testing.expectEqual(@as(usize, 32), nextCopyRows(16, true, 12, 64));
}

test "copies propose the latest earlier continuation of the last 8 tokens" {
    const gpa = std.testing.allocator;
    var x: CopyIndex = .{ .gpa = gpa };
    defer x.deinit();
    const ctx = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 1, 2, 3, 4, 5, 6, 7, 8 };
    try std.testing.expectEqualSlices(u32, &.{ 9, 10, 11, 12, 13, 14, 15, 16 }, try x.propose(&ctx, 8));
    try std.testing.expectEqual(@as(usize, 0), (try x.propose(ctx[0..15], 8)).len);
}
