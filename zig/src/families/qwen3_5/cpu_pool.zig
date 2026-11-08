//! A fixed pool of worker threads for the CPU forward's operations; tasks own disjoint outputs, so any order serves.
const std = @import("std");

pub const Task = *const fn (ctx: *anyopaque, index: usize, worker: usize) void;

pub const Pool = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    threads: []std.Thread,
    generation: std.atomic.Value(u32) = .init(0),
    /// The next index to claim, tagged with its job's generation (stored first): a claim from the last job is void.
    next: std.atomic.Value(u64) = .init(0),
    pending: std.atomic.Value(u32) = .init(0),
    stop: std.atomic.Value(bool) = .init(false),
    task: Task = undefined,
    ctx: *anyopaque = undefined,
    count: usize = 0,

    /// The caller plus `threads` workers (the host's logical CPUs less one when null, at least one worker).
    pub fn init(gpa: std.mem.Allocator, io: std.Io, threads: ?usize) !*Pool {
        const p = try gpa.create(Pool);
        errdefer gpa.destroy(p);
        const cpus = std.Thread.getCpuCount() catch 2;
        const n = threads orelse @max(1, @min(cpus, max_workers) - 1);
        p.* = .{ .gpa = gpa, .io = io, .threads = try gpa.alloc(std.Thread, n) };
        errdefer gpa.free(p.threads);
        var spawned: usize = 0;
        errdefer p.shutdown(spawned);
        for (p.threads, 1..) |*t, worker| {
            t.* = try std.Thread.spawn(.{}, serve, .{ p, worker });
            spawned += 1;
        }
        return p;
    }

    pub fn deinit(p: *Pool) void {
        p.shutdown(p.threads.len);
        p.gpa.free(p.threads);
        p.gpa.destroy(p);
    }

    /// Threads working together on one job, the caller included.
    pub fn workers(p: *const Pool) usize {
        return p.threads.len + 1;
    }

    fn shutdown(p: *Pool, spawned: usize) void {
        p.stop.store(true, .release);
        _ = p.generation.fetchAdd(1, .release);
        p.io.futexWake(u32, &p.generation.raw, std.math.maxInt(u32));
        for (p.threads[0..spawned]) |t| t.join();
    }

    /// Run `task` for every index in [0, count), the caller working beside the threads; returns once all are done.
    pub fn run(p: *Pool, ctx: *anyopaque, task: Task, count: usize) void {
        if (count == 0) return;
        if (count == 1 or p.threads.len == 0) {
            for (0..count) |i| task(ctx, i, 0);
            return;
        }
        const gen = p.generation.load(.acquire) + 1;
        p.next.store(@as(u64, gen) << 32, .release);
        p.task = task;
        p.ctx = ctx;
        p.count = count;
        p.pending.store(@intCast(count), .release);
        p.generation.store(gen, .release);
        p.io.futexWake(u32, &p.generation.raw, std.math.maxInt(u32));
        p.work(0, gen);
        var spins: usize = 0;
        while (true) {
            const left = p.pending.load(.acquire);
            if (left == 0) return;
            if (spins < spin_limit) {
                spins += 1;
                std.atomic.spinLoopHint();
            } else p.io.futexWaitUncancelable(u32, &p.pending.raw, left);
        }
    }

    /// Claims indices of job `gen` until they run out; a claim from another job is never taken.
    fn work(p: *Pool, worker: usize, gen: u32) void {
        while (true) {
            var claim = p.next.load(.acquire);
            while (true) {
                if (@as(u32, @intCast(claim >> 32)) != gen or (claim & 0xffff_ffff) >= p.count) return;
                claim = p.next.cmpxchgWeak(claim, claim + 1, .acq_rel, .acquire) orelse break;
            }
            p.task(p.ctx, @intCast(claim & 0xffff_ffff), worker);
            if (p.pending.fetchSub(1, .acq_rel) == 1) p.io.futexWake(u32, &p.pending.raw, 1);
        }
    }

    fn serve(p: *Pool, worker: usize) void {
        var seen: u32 = 0;
        while (true) {
            var spins: usize = 0;
            while (p.generation.load(.acquire) == seen) {
                if (spins < spin_limit) {
                    spins += 1;
                    std.atomic.spinLoopHint();
                } else p.io.futexWaitUncancelable(u32, &p.generation.raw, seen);
            }
            if (p.stop.load(.acquire)) return;
            seen = p.generation.load(.acquire);
            p.work(worker, seen);
        }
    }
};

const max_workers = 64;
/// Iterations a thread spins on a change before it sleeps on the futex: short enough to idle cheaply between tokens.
const spin_limit = 20_000;

test "every index runs once, on any worker, the caller returns after the last, and back-to-back jobs never mix" {
    const gpa = std.testing.allocator;
    const Counts = struct {
        hits: [1000]std.atomic.Value(u32) = @splat(.init(0)),
        fn task(ctx: *anyopaque, index: usize, worker: usize) void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            _ = worker;
            _ = self.hits[index].fetchAdd(1, .monotonic);
        }
    };
    var counts: Counts = .{};
    const pool = try Pool.init(gpa, std.testing.io, 3);
    defer pool.deinit();
    for (0..5) |_| {
        pool.run(&counts, Counts.task, 1000);
        pool.run(&counts, Counts.task, 1);
    }
    for (counts.hits, 0..) |h, i| try std.testing.expectEqual(@as(u32, if (i == 0) 10 else 5), h.load(.monotonic));
    var small: Counts = .{};
    for (0..4000) |_| pool.run(&small, Counts.task, 5);
    for (small.hits[0..5]) |h| try std.testing.expectEqual(@as(u32, 4000), h.load(.monotonic));
    for (small.hits[5..]) |h| try std.testing.expectEqual(@as(u32, 0), h.load(.monotonic));
}
