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
        @memcpy(h[0 .. n * 4], std.mem.sliceAsBytes(self.entries.rows.items));
        @memcpy(h[n * 4 .. n * 8], std.mem.sliceAsBytes(self.entries.toks.items));
        @memcpy(h[n * 8 .. n * 12], std.mem.sliceAsBytes(self.entries.subs.items));
        // the forward waits on its stream before the next one stages, so the pinned bytes are not reused early
        try self.dev.?.uploadAsync(0, h[0..bytes], stream.handle);
        try draws.penalize(stream, logits, vocab, self.dev.?.ptr, n);
    }

    pub fn stageWindow(self: *Rows, gpa: std.mem.Allocator, w: anytype, cache_len: usize, first: usize) !void {
        try self.window(gpa, w.stream, cache_len, w.pending);
        var buf: [most_path]u32 = undefined;
        if (w.rows() > most_path + 1) return error.UnsupportedQwenWindow;
        for (0..w.rows()) |r| try self.row(gpa, @intCast(first + r), w.stream.penalties, lanes.penalty_rows.pathOf(&buf, w.tokens, w.parents, r));
    }

    pub fn clear(self: *Rows) void {
        self.entries.clear();
    }
};
