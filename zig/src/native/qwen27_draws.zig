//! A sampled window's rows drawn on host threads: each row's own keyed draw, and the walk's first failing row.
const std = @import("std");

/// Threads that draw one window's rows, the caller's included.
pub const threads = 4;
/// Rows one call takes at most: a window's.
pub const max_rows = 64;

/// `job.row(i)` for every row in [0, rows) on up to `threads` threads; the lowest failing row's error, as a walk.
pub fn each(job: anytype, rows: usize) anyerror!void {
    if (rows > max_rows) return error.TooManyDrawRows;
    var errors: [max_rows]?anyerror = @splat(null);
    var next = std.atomic.Value(usize).init(0);
    const W = Worker(@TypeOf(job));
    var helpers: [threads - 1]?std.Thread = @splat(null);
    const wanted = @min(threads, rows) -| 1;
    // a helper that cannot start leaves its rows to the others
    for (helpers[0..wanted]) |*t| t.* = std.Thread.spawn(.{ .stack_size = 1 << 20 }, W.run, .{ job, rows, &next, &errors }) catch null;
    W.run(job, rows, &next, &errors);
    for (helpers[0..wanted]) |t| if (t) |h| h.join();
    for (errors[0..rows]) |e| if (e) |err| return err;
}

/// A window's logits rows, each drawn by `drawFn` at its own position with its own mask, into its own `out` slot.
pub fn Rows(comptime drawFn: anytype, comptime Sampling: type) type {
    return struct {
        gpa: std.mem.Allocator,
        words: [*]const u16,
        vocab: usize,
        width: usize,
        sampling: ?Sampling,
        positions: []const u64,
        out: []u32,
        masks: []const u32,

        pub fn row(r: @This(), i: usize) !void {
            const mask = if (r.masks.len == 0) &.{} else r.masks[i * r.width ..][0..r.width];
            r.out[i] = try @import("qwen27_masks.zig").draw(drawFn, r.gpa, r.words[i * r.vocab ..][0..r.vocab], r.sampling, r.positions[i], mask);
        }
    };
}

fn Worker(comptime Job: type) type {
    return struct {
        fn run(job: Job, rows: usize, next: *std.atomic.Value(usize), errors: *[max_rows]?anyerror) void {
            while (true) {
                const i = next.fetchAdd(1, .monotonic);
                if (i >= rows) return;
                job.row(i) catch |err| {
                    errors[i] = err;
                };
            }
        }
    };
}

const Squares = struct {
    out: []u64,
    fail: []const bool,
    pub fn row(s: Squares, i: usize) !void {
        if (s.fail[i]) return if (i % 2 == 0) error.EvenRow else error.OddRow;
        s.out[i] = @as(u64, i) * i;
    }
};

test "qwen27 draws: every row is drawn once, at any count up to a window" {
    for ([_]usize{ 0, 1, 2, 3, 5, 16, 17, max_rows }) |rows| {
        var out: [max_rows]u64 = @splat(std.math.maxInt(u64));
        const fail: [max_rows]bool = @splat(false);
        try each(Squares{ .out = out[0..rows], .fail = &fail }, rows);
        for (out[0..rows], 0..) |v, i| try std.testing.expectEqual(@as(u64, i) * i, v);
        for (out[rows..]) |v| try std.testing.expectEqual(std.math.maxInt(u64), v);
    }
}

test "qwen27 draws: the lowest failing row's error, as the sequential walk returns" {
    var out: [16]u64 = undefined;
    var fail: [16]bool = @splat(false);
    fail[11] = true;
    fail[6] = true;
    fail[3] = true;
    try std.testing.expectError(error.OddRow, each(Squares{ .out = &out, .fail = &fail }, 16));
    fail[3] = false;
    try std.testing.expectError(error.EvenRow, each(Squares{ .out = &out, .fail = &fail }, 16));
    try std.testing.expectError(error.TooManyDrawRows, each(Squares{ .out = &out, .fail = &fail }, max_rows + 1));
}
