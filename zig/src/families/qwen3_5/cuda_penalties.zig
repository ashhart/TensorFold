//! presence and frequency penalties on the CUDA rows: (row, token, amount) entries over a kept row's reply and path.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");

/// The drafts on a row's path at most: a verify forward's rows.
const most_path = @import("cuda_round.zig").most_verify;

/// One forward's entries, staged by the caller and applied by the forward before its draws or its logits copy.
pub const Rows = struct {
    rows: std.ArrayList(u32) = .empty,
    toks: std.ArrayList(u32) = .empty,
    subs: std.ArrayList(f32) = .empty,
    /// The current window's counts: the reply's tokens before it, then its pending token.
    counts: std.AutoArrayHashMapUnmanaged(u32, u32) = .empty,
    host: ?cuda.HostBuffer = null,
    dev: ?cuda.DeviceBuffer = null,

    pub fn deinit(self: *Rows, gpa: std.mem.Allocator) void {
        self.rows.deinit(gpa);
        self.toks.deinit(gpa);
        self.subs.deinit(gpa);
        self.counts.deinit(gpa);
        if (self.host) |*h| h.free();
        if (self.dev) |*d| d.free();
    }

    /// Entries staged for the next forward.
    pub fn len(self: *const Rows) usize {
        return self.rows.items.len;
    }

    /// A window's reply so far: `stream`'s committed tokens before `cache_len`, then `next`, the token at `cache_len`.
    pub fn window(self: *Rows, gpa: std.mem.Allocator, stream: *const lanes.Stream, cache_len: usize, next: u32) !void {
        self.counts.clearRetainingCapacity();
        const context = stream.context.items;
        if (cache_len < stream.prompt_len or cache_len > context.len) return error.InvalidQwenPosition;
        for (context[stream.prompt_len..cache_len]) |t| try bump(gpa, &self.counts, t, 1);
        try bump(gpa, &self.counts, next, 1);
    }

    /// Row `row`'s entries under `p`: the window's counts with `path` (the drafts from the window's first row to it).
    pub fn row(self: *Rows, gpa: std.mem.Allocator, at: u32, p: lanes.Penalties, path: []const u32) !void {
        for (path) |t| try bump(gpa, &self.counts, t, 1);
        defer for (path) |t| bump(gpa, &self.counts, t, -1) catch unreachable;
        const n = self.counts.count();
        try self.rows.ensureUnusedCapacity(gpa, n);
        try self.toks.ensureUnusedCapacity(gpa, n);
        try self.subs.ensureUnusedCapacity(gpa, n);
        for (self.counts.keys(), self.counts.values()) |t, c| if (c > 0) {
            self.rows.appendAssumeCapacity(at);
            self.toks.appendAssumeCapacity(t);
            self.subs.appendAssumeCapacity(p.of(c));
        };
    }

    /// The staged entries taken from `logits` (rows of `vocab` bf16 words) on `stream`; none staged: nothing runs.
    pub fn apply(self: *Rows, d: *const cuda.Driver, draws: *const cuda.sample.Draws, stream: cuda.Stream, logits: u64, vocab: usize) !void {
        const n = self.len();
        if (n == 0) return;
        defer self.clear();
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
        @memcpy(h[0 .. n * 4], std.mem.sliceAsBytes(self.rows.items));
        @memcpy(h[n * 4 .. n * 8], std.mem.sliceAsBytes(self.toks.items));
        @memcpy(h[n * 8 .. n * 12], std.mem.sliceAsBytes(self.subs.items));
        // the forward waits on its stream before the next one stages, so the pinned bytes are not reused early
        try self.dev.?.uploadAsync(0, h[0..bytes], stream.handle);
        try draws.penalize(stream, logits, vocab, self.dev.?.ptr, n);
    }

    /// A window's rows from row `first`: the reply before `cache_len`, the pending token, and each row's own path.
    pub fn stageWindow(self: *Rows, gpa: std.mem.Allocator, w: anytype, cache_len: usize, first: usize) !void {
        try self.window(gpa, w.stream, cache_len, w.pending);
        var buf: [most_path]u32 = undefined;
        if (w.rows() > most_path + 1) return error.UnsupportedQwenWindow;
        for (0..w.rows()) |r| try self.row(gpa, @intCast(first + r), w.stream.penalties, pathOf(&buf, w.tokens, w.parents, r));
    }

    pub fn clear(self: *Rows) void {
        self.rows.clearRetainingCapacity();
        self.toks.clearRetainingCapacity();
        self.subs.clearRetainingCapacity();
    }
};

fn bump(gpa: std.mem.Allocator, counts: *std.AutoArrayHashMapUnmanaged(u32, u32), t: u32, by: i2) !void {
    const gop = try counts.getOrPut(gpa, t);
    if (!gop.found_existing) gop.value_ptr.* = 0;
    gop.value_ptr.* = if (by > 0) gop.value_ptr.* + 1 else gop.value_ptr.* - 1;
}

/// Row `r`'s path in a window of `tokens` drafts (row r > 0 feeds tokens[r - 1]) under `parents` (null: a chain).
pub fn pathOf(out: []u32, tokens: []const u32, parents: ?[]const i32, r: usize) []const u32 {
    var n: usize = 0;
    var x = r;
    while (x > 0) {
        out[n] = tokens[x - 1];
        n += 1;
        x = if (parents) |ps| @intCast(ps[x]) else x - 1;
    }
    std.mem.reverse(u32, out[0..n]);
    return out[0..n];
}

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
    try std.testing.expectEqualSlices(u32, &.{ 0, 0, 1, 1, 1 }, rows.rows.items);
    try std.testing.expectEqualSlices(u32, &.{ 5, 7, 5, 7, 3 }, rows.toks.items);
    try std.testing.expectEqualSlices(f32, &.{ 2, 2, 2.5, 2, 1.5 }, rows.subs.items);
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
    try std.testing.expectEqualSlices(u32, &.{ 4, 6 }, rows.toks.items);
    try std.testing.expectEqualSlices(f32, &.{ 1, 1 }, rows.subs.items);
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
    try std.testing.expectEqualSlices(u32, &.{}, pathOf(&buf, &tokens, &parents, 0));
    try std.testing.expectEqualSlices(u32, &.{11}, pathOf(&buf, &tokens, &parents, 2));
    try std.testing.expectEqualSlices(u32, &.{ 10, 12, 13 }, pathOf(&buf, &tokens, &parents, 4));
    try std.testing.expectEqualSlices(u32, &.{ 10, 11 }, pathOf(&buf, &tokens, null, 2));
}
