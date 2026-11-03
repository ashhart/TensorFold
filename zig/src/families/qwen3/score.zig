//! The reranker's label logits: a prompt's final-normed last row dotted in f64 with head rows read on the host.
const std = @import("std");
const model = @import("model.zig");
const Allocator = std.mem.Allocator;

/// Labels one call may ask for: a reranker asks two, a decision a few options.
pub const labels_most = 16;

pub const Refusal = error{ MassUnsupported, BadPrompt, BadLabels };

/// A refusal's reason as a client reads it.
pub fn problemOf(err: Refusal) []const u8 {
    return switch (err) {
        error.MassUnsupported => "this engine gives no vocabulary mass",
        error.BadPrompt => "a prompt is empty, holds a token outside the vocabulary, or does not fit the window",
        error.BadLabels => "a branch's label count, or a label, does not fit the score head",
    };
}

/// Head rows read so far, by token id, as fp32 host rows: the 1.24 GB head never goes to the device.
pub const Heads = struct {
    rows: std.AutoHashMapUnmanaged(u32, []f32) = .empty,
    io: std.Io,
    /// The checkpoint directory and the head's stored name (`Model.head`).
    dir: []const u8,
    key: []const u8,
    hidden: usize,
    /// Rows the head has: the vocabulary of the checkpoint's config.
    count: usize,

    pub fn of(m: *const model.Model, io: std.Io, dir: []const u8) Heads {
        return .{ .io = io, .dir = dir, .key = m.head, .hidden = m.spec.hidden, .count = m.spec.vocab };
    }

    /// Row `id`, read once from the checkpoint.
    pub fn row(h: *Heads, gpa: Allocator, id: u32) ![]const f32 {
        if (h.rows.get(id)) |r| return r;
        const r = try model.readRows(gpa, h.io, h.dir, h.key, &.{id});
        errdefer gpa.free(r);
        if (r.len != h.hidden) return error.BadHeadRow;
        try h.rows.put(gpa, id, r);
        return r;
    }

    pub fn deinit(h: *Heads, gpa: Allocator) void {
        var it = h.rows.valueIterator();
        while (it.next()) |r| gpa.free(r.*);
        h.rows.deinit(gpa);
    }
};

/// A call's checks before any GPU work: a bad call fails, it never reads out of bounds.
pub fn check(prompt: []const u32, labels: []const u32, logits: []const f64, mass: bool, window: usize, vocab: usize, head_rows: usize) Refusal!void {
    if (mass) return error.MassUnsupported; // only /v1/decisions asks, and a reranker server refuses that route
    if (prompt.len == 0 or prompt.len > window) return error.BadPrompt;
    for (prompt) |id| if (id >= vocab) return error.BadPrompt;
    if (labels.len == 0 or labels.len > labels_most or logits.len != labels.len) return error.BadLabels;
    for (labels) |id| if (id >= head_rows) return error.BadLabels;
}

/// A decision's checks before any GPU work: one count a branch, no empty branch, every prompt and label in range.
pub fn checkDecision(trunk: []const u32, branches: []const []const u32, counts: []const usize, labels: []const u32, logits: []const f64, window: usize, vocab: usize, head_rows: usize) Refusal!void {
    if (branches.len == 0 or counts.len != branches.len) return error.BadLabels;
    for (trunk) |id| if (id >= vocab) return error.BadPrompt;
    var total: usize = 0;
    for (branches, counts) |b, n| {
        if (b.len == 0 or trunk.len + b.len > window) return error.BadPrompt;
        for (b) |id| if (id >= vocab) return error.BadPrompt;
        if (n == 0 or n > labels_most) return error.BadLabels;
        total += n;
    }
    if (total != labels.len or logits.len != labels.len) return error.BadLabels;
    for (labels) |id| if (id >= head_rows) return error.BadLabels;
}

/// Each run's end (past its last prompt), in order: a new run past `window` tokens or `most` prompts; no empty prompt.
pub fn pack(a: Allocator, lengths: []const usize, window: usize, most: usize) (Allocator.Error || Refusal)![]usize {
    var ends: std.ArrayList(usize) = .empty;
    errdefer ends.deinit(a);
    var tokens: usize = 0;
    var count: usize = 0;
    for (lengths, 0..) |n, i| {
        if (n == 0 or n > window) return error.BadPrompt;
        if (count > 0 and (tokens + n > window or count == most)) {
            try ends.append(a, i);
            tokens = 0;
            count = 0;
        }
        tokens += n;
        count += 1;
    }
    if (count > 0) try ends.append(a, lengths.len);
    return ends.toOwnedSlice(a);
}

/// A checked decision through `runner` (`run`, `hidden`): each branch as trunk ++ branch, packed as `pack` cuts them.
pub fn decidePacked(a: Allocator, runner: anytype, heads: anytype, trunk: []const u32, branches: []const []const u32, counts: []const usize, labels: []const u32, logits: []f64, window: usize, most: usize) !void {
    const lengths = try a.alloc(usize, branches.len);
    defer a.free(lengths);
    for (lengths, branches) |*n, b| n.* = trunk.len + b.len;
    const runs = try pack(a, lengths, window, most);
    defer a.free(runs);
    var first: usize = 0;
    var at: usize = 0; // the first branch's labels and logits in this run
    for (runs) |end| {
        var tokens: std.ArrayList(u32) = .empty;
        defer tokens.deinit(a);
        const ends = try a.alloc(u32, end - first);
        defer a.free(ends);
        for (branches[first..end], ends) |b, *e| {
            try tokens.appendSlice(a, trunk);
            try tokens.appendSlice(a, b);
            e.* = @intCast(tokens.items.len);
        }
        const rows = try a.alloc(f32, (end - first) * runner.hidden);
        defer a.free(rows);
        try runner.run(tokens.items, ends, rows);
        for (counts[first..end], 0..) |n, i| {
            try labelLogits(a, heads, rows[i * runner.hidden ..][0..runner.hidden], labels[at..][0..n], logits[at..][0..n]);
            at += n;
        }
        first = end;
    }
}

/// Each label's logit from the last row (its BF16 values widened, as Engine.run gives them): sum in f64.
pub fn labelLogits(gpa: Allocator, heads: anytype, row: []const f32, labels: []const u32, out: []f64) !void {
    if (row.len != heads.hidden) return error.BadRow;
    for (labels, out) |id, *x| {
        const head = try heads.row(gpa, id);
        var s: f64 = 0;
        for (row, head) |v, w| s += @as(f64, v) * @as(f64, w);
        x.* = s;
    }
}

test "head rows are read once each, and a label's logit is the f64 dot of the BF16 row with its head row" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    // lm_head rows: 3 rows of 2 bf16 values, row r = (r + 1, -(r + 1))
    const head = [_]u16{ 0x3f80, 0xbf80, 0x4000, 0xc000, 0x4040, 0xc040 };
    try @import("core").checkpoint_fixture.write(a, io, tmp.dir, "model.safetensors", &.{
        .{ .name = "lm_head.weight", .dtype = "BF16", .shape = &.{ 3, 2 }, .bytes = std.mem.sliceAsBytes(&head) },
    });
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer a.free(dir);
    var heads: Heads = .{ .io = io, .dir = dir, .key = "lm_head.weight", .hidden = 2, .count = 3 };
    defer heads.deinit(a);
    const row = [_]f32{ 1.5, 0.5 };
    var out: [2]f64 = undefined;
    try labelLogits(a, &heads, &row, &.{ 2, 0 }, &out);
    try std.testing.expectEqualSlices(f64, &.{ 3, 1 }, &out); // 1.5 * 3 - 0.5 * 3, 1.5 * 1 - 0.5 * 1
    try labelLogits(a, &heads, &row, &.{2}, out[0..1]);
    try std.testing.expectError(error.BadRow, labelLogits(a, &heads, row[0..1], &.{2}, out[0..1]));
    try std.testing.expectEqual(@as(u32, 2), heads.rows.count());
}

test "a bad call is refused before any work" {
    var out: [2]f64 = undefined;
    try check(&.{ 1, 2 }, &.{ 3, 4 }, &out, false, 8, 10, 10);
    try std.testing.expectError(error.MassUnsupported, check(&.{1}, &.{ 3, 4 }, &out, true, 8, 10, 10));
    try std.testing.expectError(error.BadPrompt, check(&.{}, &.{ 3, 4 }, &out, false, 8, 10, 10));
    try std.testing.expectError(error.BadPrompt, check(&.{ 1, 2, 3 }, &.{ 3, 4 }, &out, false, 2, 10, 10));
    try std.testing.expectError(error.BadPrompt, check(&.{10}, &.{ 3, 4 }, &out, false, 8, 10, 10));
    try std.testing.expectError(error.BadLabels, check(&.{1}, &.{}, out[0..0], false, 8, 10, 10));
    try std.testing.expectError(error.BadLabels, check(&.{1}, &.{ 3, 10 }, &out, false, 8, 10, 10));
    try std.testing.expectError(error.BadLabels, check(&.{1}, &.{3}, &out, false, 8, 10, 10));
}

test "a decision's runs: at most `most` prompts and the window's tokens each, an empty or oversized prompt refused" {
    const a = std.testing.allocator;
    var lengths: [129]usize = undefined;
    @memset(&lengths, 1);
    const one = try pack(a, lengths[0..128], 2048, 128);
    defer a.free(one);
    try std.testing.expectEqualSlices(usize, &.{128}, one);
    const two = try pack(a, &lengths, 2048, 128);
    defer a.free(two);
    try std.testing.expectEqualSlices(usize, &.{ 128, 129 }, two);
    const exact = try pack(a, &.{ 1000, 1048, 1 }, 2048, 128);
    defer a.free(exact);
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, exact);
    const over = try pack(a, &.{ 1000, 1049 }, 2048, 128);
    defer a.free(over);
    try std.testing.expectEqualSlices(usize, &.{ 1, 2 }, over);
    const none = try pack(a, &.{}, 2048, 128);
    defer a.free(none);
    try std.testing.expectEqual(@as(usize, 0), none.len);
    try std.testing.expectError(error.BadPrompt, pack(a, &.{ 5, 2049 }, 2048, 128));
    try std.testing.expectError(error.BadPrompt, pack(a, &.{ 5, 0 }, 2048, 128));
}

test "a bad decision is refused before any work" {
    var out: [3]f64 = undefined;
    const two: []const []const u32 = &.{ &.{2}, &.{ 3, 4 } };
    try checkDecision(&.{1}, two, &.{ 2, 1 }, &.{ 5, 6, 7 }, &out, 8, 10, 10);
    try std.testing.expectError(error.BadLabels, checkDecision(&.{1}, two, &.{3}, &.{ 5, 6, 7 }, &out, 8, 10, 10));
    try std.testing.expectError(error.BadLabels, checkDecision(&.{1}, &.{}, &.{}, &.{}, out[0..0], 8, 10, 10));
    try std.testing.expectError(error.BadPrompt, checkDecision(&.{1}, &.{ &.{2}, &.{} }, &.{ 2, 1 }, &.{ 5, 6, 7 }, &out, 8, 10, 10));
    try std.testing.expectError(error.BadPrompt, checkDecision(&.{ 1, 1, 1 }, two, &.{ 2, 1 }, &.{ 5, 6, 7 }, &out, 4, 10, 10));
    try std.testing.expectError(error.BadPrompt, checkDecision(&.{1}, &.{ &.{2}, &.{10} }, &.{ 2, 1 }, &.{ 5, 6, 7 }, &out, 8, 10, 10));
    try std.testing.expectError(error.BadPrompt, checkDecision(&.{10}, two, &.{ 2, 1 }, &.{ 5, 6, 7 }, &out, 8, 10, 10));
    try std.testing.expectError(error.BadLabels, checkDecision(&.{1}, two, &.{ 2, 0 }, &.{ 5, 6 }, out[0..2], 8, 10, 10));
    try std.testing.expectError(error.BadLabels, checkDecision(&.{1}, two, &.{ 2, 1 }, &.{ 5, 6, 10 }, &out, 8, 10, 10));
    try std.testing.expectError(error.BadLabels, checkDecision(&.{1}, two, &.{ 2, 1 }, &.{ 5, 6, 7 }, out[0..2], 8, 10, 10));
    // each refusal has its own reason for the client
    const all = [_]Refusal{ error.MassUnsupported, error.BadPrompt, error.BadLabels };
    for (all, 0..) |x, i| for (all[i + 1 ..]) |y| try std.testing.expect(!std.mem.eql(u8, problemOf(x), problemOf(y)));
}

/// A fake engine of 2-wide rows: an input's row is (its length, the sum of its tokens); counts its runs.
const FakeRunner = struct {
    hidden: usize = 2,
    runs: usize = 0,

    fn run(r: *FakeRunner, tokens: []const u32, ends: []const u32, out: []f32) !void {
        r.runs += 1;
        var begin: u32 = 0;
        for (ends, 0..) |e, i| {
            var sum: f32 = 0;
            for (tokens[begin..e]) |t| sum += @floatFromInt(t);
            out[2 * i] = @floatFromInt(e - begin);
            out[2 * i + 1] = sum;
            begin = e;
        }
    }
};

/// Head rows of the fake: label l's row is (l, 1), so its logit is length * l + sum.
const FakeHeads = struct {
    hidden: usize = 2,
    store: [16][2]f32 = undefined,

    fn row(h: *FakeHeads, _: Allocator, id: u32) ![]const f32 {
        h.store[id] = .{ @floatFromInt(id), 1 };
        return &h.store[id];
    }
};

test "a packed decision gives each branch's labels the logits of its own whole prompt, across several runs" {
    const a = std.testing.allocator;
    var runner: FakeRunner = .{};
    var heads: FakeHeads = .{};
    const trunk = [_]u32{ 1, 2 };
    const branches: []const []const u32 = &.{ &.{3}, &.{ 4, 5 }, &.{6}, &.{ 7, 8, 9 }, &.{10} };
    const counts = [_]usize{ 2, 1, 2, 1, 2 };
    const labels = [_]u32{ 2, 3, 4, 5, 6, 7, 8, 9 };
    var logits: [8]f64 = undefined;
    // prompts of 3, 4, 3, 5, 3 tokens, window 7, 2 prompts a run: runs [3, 4], [3], [5], [3]
    try decidePacked(a, &runner, &heads, &trunk, branches, &counts, &labels, &logits, 7, 2);
    try std.testing.expectEqual(@as(usize, 4), runner.runs);
    var at: usize = 0;
    for (branches, counts) |b, n| {
        var sum: f64 = 3; // the trunk's 1 + 2
        for (b) |t| sum += @floatFromInt(t);
        const len: f64 = @floatFromInt(trunk.len + b.len);
        for (labels[at..][0..n], logits[at..][0..n]) |l, got| try std.testing.expectEqual(len * @as(f64, @floatFromInt(l)) + sum, got);
        at += n;
    }
}
