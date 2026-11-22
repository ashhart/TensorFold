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

/// `threads - 1` helpers kept alive between calls: TF_Q27_DRAW_POOL, the engine's own. `each` unchanged beside it.
pub fn Pool(comptime Job: type) type {
    return struct {
        helpers: [threads - 1]?std.Thread = @splat(null),
        mutex: std.Io.Mutex = .init,
        work: std.Io.Condition = .init,
        idle: std.Io.Condition = .init,
        job: ?Job = null,
        rows: usize = 0,
        next: std.atomic.Value(usize) = .init(0),
        errors: [max_rows]?anyerror = @splat(null),
        epoch: u64 = 0,
        pending: usize = 0,
        quit: bool = false,

        /// Started here; a spawn that fails leaves its rows to the others, as `each`'s `catch null` does.
        pub fn start(p: *@This()) void {
            for (&p.helpers) |*t| t.* = std.Thread.spawn(.{ .stack_size = 1 << 20 }, loop, .{p}) catch null;
        }

        /// The engine's teardown: every started helper joined, whichever started.
        pub fn deinit(p: *@This()) void {
            const io = std.Io.Threaded.global_single_threaded.io();
            p.mutex.lockUncancelable(io);
            p.quit = true;
            p.work.broadcast(io);
            p.mutex.unlock(io);
            for (p.helpers) |t| if (t) |h| h.join();
        }

        /// `each`'s contract on the kept helpers: the caller draws beside them, the first failing row's error back.
        pub fn each(p: *@This(), job: Job, rows: usize) anyerror!void {
            if (rows > max_rows) return error.TooManyDrawRows;
            const io = std.Io.Threaded.global_single_threaded.io();
            p.mutex.lockUncancelable(io);
            p.job = job;
            p.rows = rows;
            p.next.store(0, .monotonic);
            p.errors = @splat(null);
            p.pending = 0;
            for (p.helpers) |t| p.pending += @intFromBool(t != null);
            p.epoch += 1;
            p.work.broadcast(io);
            p.mutex.unlock(io);
            Worker(Job).run(job, rows, &p.next, &p.errors);
            p.mutex.lockUncancelable(io);
            while (p.pending != 0) p.idle.waitUncancelable(io, &p.mutex);
            p.mutex.unlock(io);
            for (p.errors[0..rows]) |e| if (e) |err| return err;
        }

        fn loop(p: *@This()) void {
            const io = std.Io.Threaded.global_single_threaded.io();
            p.mutex.lockUncancelable(io);
            var seen: u64 = 0;
            while (true) {
                while (!p.quit and p.epoch == seen) p.work.waitUncancelable(io, &p.mutex);
                if (p.quit) {
                    p.mutex.unlock(io);
                    return;
                }
                seen = p.epoch;
                const job = p.job.?;
                const rows = p.rows;
                p.mutex.unlock(io);
                Worker(Job).run(job, rows, &p.next, &p.errors);
                p.mutex.lockUncancelable(io);
                p.pending -= 1;
                if (p.pending == 0) p.idle.signal(io);
            }
        }
    };
}

/// What a penalized row takes from a copy of its logits first: lanes.penalty_rows.Entries.applyRow by frame row.
pub const Lower = struct { ctx: *const anyopaque, apply: *const fn (ctx: *const anyopaque, row: u32, logits: []u16) void };

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
        pens: ?Lower = null, // presence and frequency penalties by frame row: row i is `row0 + i`
        row0: usize = 0,

        pub fn row(r: @This(), i: usize) !void {
            const mask = if (r.masks.len == 0) &.{} else r.masks[i * r.width ..][0..r.width];
            const words = r.words[i * r.vocab ..][0..r.vocab];
            const draw = @import("qwen27_masks.zig").draw;
            const p = r.pens orelse {
                r.out[i] = try draw(drawFn, r.gpa, words, r.sampling, r.positions[i], mask);
                return;
            };
            const lowered = try r.gpa.dupe(u16, words); // the shared frame's row stays the forward's
            defer r.gpa.free(lowered);
            p.apply(p.ctx, @intCast(r.row0 + i), lowered);
            r.out[i] = try draw(drawFn, r.gpa, lowered, r.sampling, r.positions[i], mask);
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

fn greatest(_: std.mem.Allocator, logits: []const u16, _: ?u8, _: u64, _: []const u32) !u32 {
    var best: usize = 0;
    for (logits, 0..) |w, t| {
        if (@as(f32, @bitCast(@as(u32, w) << 16)) > @as(f32, @bitCast(@as(u32, logits[best]) << 16))) best = t;
    }
    return @intCast(best);
}

/// Frame row 5 loses 1.5 on token 1 (Entries.applyRow's rounding is lanes.penalty_rows' own test).
fn lowerFive(_: *const anyopaque, row: u32, logits: []u16) void {
    if (row == 5) logits[1] = 0x3fc0; // 3 - 1.5
}

test "qwen27 draws: a kept pool draws the same rows as the walk, errors alike, helpers reused" {
    var pool: Pool(Squares) = .{};
    pool.start();
    defer pool.deinit(); // teardown joins the started helpers
    for ([_]usize{ 0, 1, 2, 7, max_rows }) |rows| {
        var out: [max_rows]u64 = @splat(std.math.maxInt(u64));
        const fail: [max_rows]bool = @splat(false);
        try pool.each(Squares{ .out = out[0..rows], .fail = &fail }, rows);
        for (out[0..rows], 0..) |v, i| try std.testing.expectEqual(@as(u64, i) * i, v);
        for (out[rows..]) |v| try std.testing.expectEqual(std.math.maxInt(u64), v);
    }
    var out: [16]u64 = undefined;
    var fail: [16]bool = @splat(false);
    fail[6] = true;
    fail[3] = true;
    try std.testing.expectError(error.OddRow, pool.each(Squares{ .out = &out, .fail = &fail }, 16));
    try std.testing.expectError(error.TooManyDrawRows, pool.each(Squares{ .out = &out, .fail = &fail }, max_rows + 1));
}

test "qwen27 draws: a penalized row draws from its lowered copy, and the frame's rows stay the forward's" {
    const gpa = std.testing.allocator;
    const row = [_]u16{ 0x3f80, 0x4040, 0x4000, 0 }; // 1, 3, 2, 0
    const words = row ++ row;
    var out: [2]u32 = undefined;
    const pens: Lower = .{ .ctx = &words, .apply = lowerFive };
    const job = Rows(greatest, u8){ .gpa = gpa, .words = &words, .vocab = 4, .width = 1, .sampling = null, .positions = &.{ 0, 0 }, .out = &out, .masks = &.{}, .pens = pens, .row0 = 4 };
    try each(job, 2);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2 }, &out); // row 5's 1.5 falls under 2
    try std.testing.expectEqualSlices(u16, &(row ++ row), &words);
}
