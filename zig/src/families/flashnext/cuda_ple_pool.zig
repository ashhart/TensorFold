//! The n-gram gather's parked host workers; parts claimed from one counter, each claim waking one more worker.
const std = @import("std");

/// The most parts a call may have (the caller beside up to 127 workers).
pub const max_parts = 128;

/// One part of a call: `run(ctx, part)`.
pub const Task = struct {
    ctx: *const anyopaque,
    run: *const fn (ctx: *const anyopaque, part: usize) anyerror!void,
};

pub const Pool = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    mutex: std.Io.Mutex = .init, // one call at a time
    threads: [max_parts - 1]std.Thread = undefined,
    go: [max_parts - 1]std.atomic.Value(u32) = undefined, // bumped to wake a worker: to claim parts, or to stop
    spawned: usize = 0,
    stopping: std.atomic.Value(bool) = .init(false),
    // the current call: generation << 32 | parts << 16 | parts claimed (a claim is a compare-exchange on all three)
    state: std.atomic.Value(u64) = .init(0),
    done: std.atomic.Value(u32) = .init(0), // parts finished
    waking: std.atomic.Value(u32) = .init(0), // workers woken
    task: Task = undefined,
    failed: [max_parts]?anyerror = undefined,

    pub fn create(gpa: std.mem.Allocator, io: std.Io) !*Pool {
        const p = try gpa.create(Pool);
        p.* = .{ .gpa = gpa, .io = io };
        return p;
    }

    /// Wakes every worker to exit, joins them and frees the pool.
    pub fn destroy(p: *Pool) void {
        p.stopping.store(true, .release);
        for (p.go[0..p.spawned]) |*g| {
            _ = g.fetchAdd(1, .release);
            p.io.futexWake(u32, &g.raw, 1);
        }
        for (p.threads[0..p.spawned]) |th| th.join();
        p.gpa.destroy(p);
    }

    /// Runs parts 0..parts-1 of `task` and returns when all are done: the first failed part's error, in part order.
    pub fn run(p: *Pool, parts: usize, task: Task) !void {
        if (parts == 0) return;
        if (parts > max_parts) return error.TooManyParts;
        p.mutex.lockUncancelable(p.io);
        defer p.mutex.unlock(p.io);
        while (p.spawned < parts - 1) {
            p.go[p.spawned] = .init(0);
            p.threads[p.spawned] = try std.Thread.spawn(.{}, work, .{ p, p.spawned });
            p.spawned += 1;
        }
        p.task = task;
        @memset(p.failed[0..parts], null);
        p.done.store(0, .monotonic);
        p.waking.store(0, .monotonic);
        const gen = (p.state.load(.monotonic) >> 32) +% 1;
        p.state.store(gen << 32 | @as(u64, parts) << 16, .release);
        p.drain();
        while (true) {
            const done = p.done.load(.acquire);
            if (done == parts) break;
            p.io.futexWaitUncancelable(u32, &p.done.raw, done);
        }
        for (p.failed[0..parts]) |f| if (f) |err| return err;
    }

    /// Claims and runs parts of the current call until none is left unclaimed.
    fn drain(p: *Pool) void {
        var s = p.state.load(.acquire);
        while (true) {
            const parts: u32 = @intCast(s >> 16 & 0xffff);
            const part: u32 = @intCast(s & 0xffff);
            if (part >= parts) return;
            if (p.state.cmpxchgWeak(s, s + 1, .acq_rel, .acquire)) |now| {
                s = now;
                continue;
            }
            if (part + 1 < parts) p.wakeOne(parts - 1);
            p.failed[part] = if (p.task.run(p.task.ctx, part)) null else |err| err;
            if (p.done.fetchAdd(1, .acq_rel) + 1 == parts) p.io.futexWake(u32, &p.done.raw, 1);
            s = p.state.load(.acquire);
        }
    }

    /// Wakes the next parked worker: a call of quick parts stays mostly on the caller, one of slow parts fans out.
    fn wakeOne(p: *Pool, workers: u32) void {
        const w = p.waking.fetchAdd(1, .monotonic);
        if (w >= workers) return;
        _ = p.go[w].fetchAdd(1, .release);
        p.io.futexWake(u32, &p.go[w].raw, 1);
    }

    fn work(p: *Pool, w: usize) void {
        var seen: u32 = 0;
        while (true) {
            const now = p.go[w].load(.acquire);
            if (now == seen) {
                p.io.futexWaitUncancelable(u32, &p.go[w].raw, seen);
                continue;
            }
            seen = now;
            if (p.stopping.load(.acquire)) return;
            p.drain();
        }
    }
};

test "parts run once each across calls of changing size, and the first failed part's error wins" {
    const Job = struct {
        hits: [max_parts]std.atomic.Value(u32) = @splat(.init(0)),
        fail_from: usize = max_parts,

        fn run(ctx: *const anyopaque, part: usize) anyerror!void {
            const j: *@This() = @ptrCast(@alignCast(@constCast(ctx)));
            _ = j.hits[part].fetchAdd(1, .monotonic);
            if (part >= j.fail_from) return if (part == j.fail_from) error.First else error.Later;
        }
    };
    const p = try Pool.create(std.testing.allocator, std.testing.io);
    defer p.destroy();
    var job: Job = .{};
    const task: Task = .{ .ctx = &job, .run = Job.run };
    const sizes = [_]usize{ 4, 16, 1, 128, 16, 4 };
    for (sizes) |n| try p.run(n, task);
    try std.testing.expectEqual(@as(usize, 127), p.spawned);
    for (job.hits, 0..) |h, part| {
        var calls: u32 = 0;
        for (sizes) |n| calls += @intFromBool(part < n);
        try std.testing.expectEqual(calls, h.load(.monotonic));
    }
    job.fail_from = 5;
    try std.testing.expectError(error.First, p.run(64, task));
    try p.run(5, task);
    try std.testing.expectError(error.TooManyParts, p.run(max_parts + 1, task));
}
