//! CUDA upload and application of core's shared per-row presence and frequency entries.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const most_path = @import("cuda_round.zig").most_verify;

pub const Rows = struct {
    entries: lanes.penalty_rows.Entries = .{},
    host: ?cuda.HostBuffer = null,
    dev: ?cuda.DeviceBuffer = null,

    pub fn deinit(self: *Rows, gpa: std.mem.Allocator) void {
        self.entries.deinit(gpa);
        if (self.host) |*h| h.free();
        if (self.dev) |*d| d.free();
    }

    pub fn len(self: *const Rows) usize {
        return self.entries.len();
    }

    pub fn window(self: *Rows, gpa: std.mem.Allocator, stream: *const lanes.Stream, cache_len: usize, next: u32) !void {
        self.entries.window(gpa, stream, cache_len, next) catch |err| return switch (err) {
            error.PenaltyPositionOutOfStep => error.InvalidQwenPosition,
            else => err,
        };
    }

    pub fn row(self: *Rows, gpa: std.mem.Allocator, at: u32, p: lanes.Penalties, path: []const u32) !void {
        return self.entries.row(gpa, at, p, path);
    }

    pub fn rowBias(self: *Rows, gpa: std.mem.Allocator, at: u32, p: lanes.Penalties, path: []const u32, bias: []const lanes.logit_bias.Entry) !void {
        return self.entries.rowBias(gpa, at, p, path, bias);
    }

    /// The staged entries taken from `logits` (rows of `vocab` bf16 words) on `stream`; none staged: nothing runs.
    pub fn apply(self: *Rows, d: *const cuda.Driver, draws: *const cuda.sample.Draws, stream: cuda.Stream, logits: u64, vocab: usize) !void {
        const n = self.len();
        if (n == 0) return;
        defer self.clear();
        // the server checks IDs against the tokenizer's vocabulary; the launch writes rows of the head's, unchecked
        if (self.entries.biased) for (self.entries.toks.items) |t| if (t >= vocab) return error.BadLogitBiasToken;
        const bytes = n * 12;
        if (self.host == null or self.host.?.bytes.len < bytes) {
            if (self.host) |*h| h.free();
            self.host = null;
            self.host = try cuda.HostBuffer.alloc(d, 2 * bytes);
        }
        if (self.dev == null or self.dev.?.len < bytes) {
            if (self.dev) |*b| b.free();
            self.dev = null;
            self.dev = try cuda.DeviceBuffer.alloc(d, 2 * bytes);
        }
        const h = self.host.?.bytes;
        @memcpy(h[0 .. n * 4], std.mem.sliceAsBytes(self.entries.rows.items));
        @memcpy(h[n * 4 .. n * 8], std.mem.sliceAsBytes(self.entries.toks.items));
        @memcpy(h[n * 8 .. n * 12], std.mem.sliceAsBytes(self.entries.subs.items));
        // the forward waits on its stream before the next one stages, so the pinned bytes are not reused early
        try self.dev.?.uploadAsync(0, h[0..bytes], stream.handle);
        if (self.entries.biased) try draws.biasRows(stream, logits, vocab, self.dev.?.ptr, n) else try draws.penalize(stream, logits, vocab, self.dev.?.ptr, n);
    }

    pub fn stageWindow(self: *Rows, gpa: std.mem.Allocator, w: anytype, cache_len: usize, first: usize) !void {
        try self.window(gpa, w.stream, cache_len, w.pending);
        var buf: [most_path]u32 = undefined;
        if (w.rows() > most_path + 1) return error.UnsupportedQwenWindow;
        for (0..w.rows()) |r| try self.entries.rowBias(gpa, @intCast(first + r), w.stream.penalties, lanes.penalty_rows.pathOf(&buf, w.tokens, w.parents, r), w.stream.logit_bias);
    }

    pub fn clear(self: *Rows) void {
        self.entries.clear();
    }
};

fn fakeStream(context: []const u32, prompt_len: usize) lanes.Stream {
    var s: lanes.Stream = undefined;
    s.context = .fromOwnedSlice(@constCast(context));
    s.prompt_len = prompt_len;
    return s;
}

test "a row's entries count the reply, the pending token and its own path once each" {
    const gpa = std.testing.allocator;
    var rows: Rows = .{};
    defer rows.deinit(gpa);
    // prompt 9 9 (not counted), reply 5 7 5, pending 7 at cache_len 5
    const s = fakeStream(&.{ 9, 9, 5, 7, 5, 7 }, 2);
    try rows.window(gpa, &s, 5, 7);
    const p: lanes.Penalties = .{ .presence = 1, .frequency = 0.5 };
    try rows.row(gpa, 0, p, &.{});
    try rows.row(gpa, 1, p, &.{ 5, 3 });
    try std.testing.expectEqualSlices(u32, &.{ 0, 0, 1, 1, 1 }, rows.entries.rows.items);
    try std.testing.expectEqualSlices(u32, &.{ 5, 7, 5, 7, 3 }, rows.entries.toks.items);
    try std.testing.expectEqualSlices(f32, &.{ 2, 2, 2.5, 2, 1.5 }, rows.entries.subs.items);
}

test "the window's counts drop a path's tokens after its row" {
    const gpa = std.testing.allocator;
    var rows: Rows = .{};
    defer rows.deinit(gpa);
    const s = fakeStream(&.{ 1, 4 }, 1);
    try rows.window(gpa, &s, 2, 6);
    const p: lanes.Penalties = .{ .frequency = 1 };
    try rows.row(gpa, 0, p, &.{8});
    rows.clear();
    try rows.row(gpa, 1, p, &.{});
    try std.testing.expectEqualSlices(u32, &.{ 4, 6 }, rows.entries.toks.items);
    try std.testing.expectEqualSlices(f32, &.{ 1, 1 }, rows.entries.subs.items);
}

test "a position before the reply or past the committed tokens is refused" {
    const gpa = std.testing.allocator;
    var rows: Rows = .{};
    defer rows.deinit(gpa);
    const s = fakeStream(&.{ 1, 2, 3 }, 2);
    try std.testing.expectError(error.InvalidQwenPosition, rows.window(gpa, &s, 1, 0));
    try std.testing.expectError(error.InvalidQwenPosition, rows.window(gpa, &s, 4, 0));
}

test "a tree row's path follows its parents from the first row" {
    var buf: [8]u32 = undefined;
    const tokens = [_]u32{ 10, 11, 12, 13 };
    const parents = [_]i32{ -1, 0, 0, 1, 3 };
    try std.testing.expectEqualSlices(u32, &.{}, lanes.penalty_rows.pathOf(&buf, &tokens, &parents, 0));
    try std.testing.expectEqualSlices(u32, &.{11}, lanes.penalty_rows.pathOf(&buf, &tokens, &parents, 2));
    try std.testing.expectEqualSlices(u32, &.{ 10, 12, 13 }, lanes.penalty_rows.pathOf(&buf, &tokens, &parents, 4));
    try std.testing.expectEqualSlices(u32, &.{ 10, 11 }, lanes.penalty_rows.pathOf(&buf, &tokens, null, 2));
}

test "bias rows merge each cell once, preserve path penalties and leave another stream unchanged" {
    const a = std.testing.allocator;
    var rows: Rows = .{};
    defer rows.deinit(a);
    const s = fakeStream(&.{ 9, 9, 5, 7, 5, 7 }, 2);
    try rows.window(a, &s, 5, 7);
    const p: lanes.Penalties = .{ .presence = 1, .frequency = 0.5 };
    // sorted, as parsed
    const bias = [_]lanes.logit_bias.Entry{ .{ .token = 3, .value = -100, .ban = true }, .{ .token = 5, .value = 3 } };
    try rows.rowBias(a, 4, p, &.{5}, &bias);
    try rows.rowBias(a, 5, p, &.{3}, &bias);
    try rows.row(a, 6, p, &.{});
    try std.testing.expectEqualSlices(u32, &.{ 4, 4, 4, 5, 5, 5, 6, 6 }, rows.entries.rows.items);
    try std.testing.expectEqualSlices(u32, &.{ 5, 7, 3, 5, 7, 3, 5, 7 }, rows.entries.toks.items);
    try std.testing.expectEqualSlices(f32, &.{ -0.5, 2, std.math.inf(f32), -1, 2, std.math.inf(f32), 2, 2 }, rows.entries.subs.items);
    for (rows.entries.rows.items, rows.entries.toks.items, 0..) |at, token, i| {
        for (rows.entries.rows.items[0..i], rows.entries.toks.items[0..i]) |other_row, other_token|
            try std.testing.expect(at != other_row or token != other_token);
    }
    rows.clear();
    try std.testing.expect(!rows.entries.biased);
    try rows.rowBias(a, 0, .{}, &.{5}, &bias);
    try std.testing.expectEqualSlices(u32, &.{ 3, 5 }, rows.entries.toks.items);
    try std.testing.expectEqualSlices(f32, &.{ std.math.inf(f32), -3 }, rows.entries.subs.items);
}
