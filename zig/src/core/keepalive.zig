//! While no request runs, a thread commits a tiny command buffer every 400 ms so macOS keeps the model wired.
const std = @import("std");

/// One tick: a tiny command buffer on the engine's own queue, touching its residency sets if it holds any.
pub const Target = struct {
    ctx: *anyopaque,
    tick: *const fn (ctx: *anyopaque) void,
};

/// The ticker: commits every 400 ms while no request runs and `--keep-warm` seconds have not passed.
pub const Keepalive = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    target: Target,
    /// One tick per interval; macOS unwires an idle model after 1-2 s, so this stays well inside it.
    interval_ns: i64 = 400 * std.time.ns_per_ms,
    /// Ticks stop this long after the last request ends: --keep-warm seconds, and 0 never starts the thread.
    window_ns: i64,
    /// Requests running now: the ticker holds its commits while this is above zero.
    busy: std.atomic.Value(u32) = .init(0),
    /// The awake-clock nanosecond the window runs from: the start, then each request's end.
    idle_since: std.atomic.Value(i64) = .init(0),
    stopped: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,

    /// Starts the ticker for an engine's queue. The target must outlive `stop`.
    pub fn start(gpa: std.mem.Allocator, io: std.Io, target: Target, window_ns: i64) !*Keepalive {
        const k = try gpa.create(Keepalive);
        k.* = .{ .gpa = gpa, .io = io, .target = target, .window_ns = window_ns, .idle_since = .init(now(io)) };
        if (std.Thread.spawn(.{}, loop, .{k})) |t| {
            k.thread = t;
        } else |e| {
            gpa.destroy(k);
            return e;
        }
        return k;
    }

    /// A request is running: the ticker stops committing until every request ended.
    pub fn begin(k: *Keepalive) void {
        _ = k.busy.fetchAdd(1, .acq_rel);
    }

    /// The request ended, any outcome: the keep-warm window restarts when the last request back calls this.
    pub fn end(k: *Keepalive) void {
        _ = k.busy.fetchSub(1, .acq_rel);
        k.idle_since.store(now(k.io), .release);
    }

    /// Whether a tick is due now: no request runs and the keep-warm window has not passed.
    pub fn due(k: *const Keepalive, at: i64) bool {
        if (k.window_ns <= 0) return false;
        return k.busy.load(.acquire) == 0 and at - k.idle_since.load(.acquire) <= k.window_ns;
    }

    /// Stops the thread and frees the ticker. Call before the engine's queue goes away.
    pub fn stop(k: *Keepalive) void {
        k.stopped.store(true, .release);
        if (k.thread) |t| t.join();
        k.gpa.destroy(k);
    }

    fn now(io: std.Io) i64 {
        return @intCast(std.Io.Clock.awake.now(io).toNanoseconds());
    }

    fn loop(k: *Keepalive) void {
        while (!k.stopped.load(.acquire)) {
            std.Io.sleep(k.io, .fromNanoseconds(k.interval_ns), .awake) catch {};
            if (k.stopped.load(.acquire)) break;
            if (k.due(now(k.io))) k.target.tick(k.target.ctx);
        }
    }
};

const testing = std.testing;

/// A commit the test gates: each tick parks until the test releases it, so the thread's timing is deterministic.
const Gated = struct {
    commits: std.atomic.Value(u32) = .init(0),
    gate: std.atomic.Value(u32) = .init(0),

    fn tick(ctx: *anyopaque) void {
        const g: *Gated = @ptrCast(@alignCast(ctx));
        _ = g.commits.fetchAdd(1, .monotonic);
        while (g.gate.load(.acquire) == 0) std.Thread.yield() catch {};
    }
};

fn until(value: *const std.atomic.Value(u32), want: u32) !void {
    for (0..2000) |_| {
        if (value.load(.acquire) >= want) return;
        std.Thread.yield() catch {};
        std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    }
    return error.Timeout;
}

test "the ticker commits while idle and holds while a request runs" {
    var gated = Gated{};
    const k = try Keepalive.start(testing.allocator, std.testing.io, .{ .ctx = &gated, .tick = Gated.tick }, 900 * std.time.ns_per_s);
    k.interval_ns = 10 * std.time.ns_per_ms;
    // armed and idle: a commit arrives and parks in the gate
    try until(&gated.commits, 1);
    // a request runs: the parked tick is released, then nothing further commits
    k.begin();
    gated.gate.store(1, .release);
    var settled = false;
    for (0..50) |_| {
        std.Io.sleep(std.testing.io, .fromMilliseconds(5), .awake) catch {};
        if (gated.commits.load(.acquire) == 1) {
            settled = true;
            break;
        }
    }
    try testing.expect(settled); // the released tick left, and busy holds the rest
    for (0..20) |_| {
        std.Io.sleep(std.testing.io, .fromMilliseconds(5), .awake) catch {};
        try testing.expectEqual(@as(u32, 1), gated.commits.load(.acquire));
    }
    // the request ended: commits resume
    k.end();
    try until(&gated.commits, 2);
    gated.gate.store(1, .release);
    Keepalive.stop(k);
}

test "ticks stop once the keep-warm window has passed and resume when a request ends" {
    var commits = std.atomic.Value(u32).init(0);
    const Count = struct {
        fn tick(ctx: *anyopaque) void {
            _ = @as(*std.atomic.Value(u32), @ptrCast(@alignCast(ctx))).fetchAdd(1, .monotonic);
        }
    };
    const k = try Keepalive.start(testing.allocator, std.testing.io, .{ .ctx = &commits, .tick = Count.tick }, 60 * std.time.ns_per_ms);
    k.interval_ns = 10 * std.time.ns_per_ms;
    try until(&commits, 1); // the window is open: commits flow
    std.Io.sleep(std.testing.io, .fromMilliseconds(120), .awake) catch {}; // the window closes at 60 ms
    const frozen = commits.load(.acquire);
    std.Io.sleep(std.testing.io, .fromMilliseconds(100), .awake) catch {};
    try testing.expectEqual(frozen, commits.load(.acquire)); // flat once the window has passed
    k.begin(); // a request ran and ended: the window restarts from its end
    k.end();
    try until(&commits, frozen + 1);
    Keepalive.stop(k);
}

test "the window decides due without a thread" {
    var k: Keepalive = .{ .gpa = testing.allocator, .io = std.testing.io, .target = .{ .ctx = undefined, .tick = undefined }, .window_ns = 100, .idle_since = .init(1000) };
    try testing.expect(k.due(1100)); // inside the window
    try testing.expect(!k.due(1101)); // past it
    k.begin();
    try testing.expect(!k.due(1050)); // a request runs: no tick even inside the window
    k.end();
    try testing.expect(k.due(1050));
    k.window_ns = 0;
    try testing.expect(!k.due(1000)); // keep-warm 0: never due
}
