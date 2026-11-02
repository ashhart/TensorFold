//! The reranker's label logits: a prompt's final-normed last row dotted in f64 with head rows read on the host.
const std = @import("std");
const model = @import("model.zig");
const Allocator = std.mem.Allocator;

/// Labels one call may ask for: a reranker asks two, a decision a few options.
pub const labels_most = 16;

pub const Refusal = error{ MassUnsupported, BadPrompt, BadLabels };

/// Head rows read so far, by token id, as fp32 host rows: the 1.24 GB head never goes to the device.
pub const Heads = struct {
    rows: std.AutoHashMapUnmanaged(u32, []f32) = .empty,
    /// The checkpoint directory and the head's stored name (`Model.head`).
    dir: []const u8,
    key: []const u8,
    hidden: usize,
    /// Rows the head has: the vocabulary of the checkpoint's config.
    count: usize,

    pub fn of(m: *const model.Model, dir: []const u8) Heads {
        return .{ .dir = dir, .key = m.head, .hidden = m.spec.hidden, .count = m.spec.vocab };
    }

    /// Row `id`, read once from the checkpoint.
    pub fn row(h: *Heads, gpa: Allocator, io: std.Io, id: u32) ![]const f32 {
        if (h.rows.get(id)) |r| return r;
        const r = try model.readRows(gpa, io, h.dir, h.key, &.{id});
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

/// Each label's logit from the last row (its BF16 values widened, as Engine.run gives them): sum in f64.
pub fn labelLogits(gpa: Allocator, io: std.Io, heads: *Heads, row: []const f32, labels: []const u32, out: []f64) !void {
    if (row.len != heads.hidden) return error.BadRow;
    for (labels, out) |id, *x| {
        const head = try heads.row(gpa, io, id);
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
    var heads: Heads = .{ .dir = dir, .key = "lm_head.weight", .hidden = 2, .count = 3 };
    defer heads.deinit(a);
    const row = [_]f32{ 1.5, 0.5 };
    var out: [2]f64 = undefined;
    try labelLogits(a, io, &heads, &row, &.{ 2, 0 }, &out);
    try std.testing.expectEqualSlices(f64, &.{ 3, 1 }, &out); // 1.5 * 3 - 0.5 * 3, 1.5 * 1 - 0.5 * 1
    try labelLogits(a, io, &heads, &row, &.{2}, out[0..1]);
    try std.testing.expectError(error.BadRow, labelLogits(a, io, &heads, row[0..1], &.{2}, out[0..1]));
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
