//! While no request runs, a thread commits a tiny command buffer every 400 ms so macOS keeps the model wired.
const std = @import("std");

/// One tick: a tiny command buffer on the engine's own queue, touching its residency sets if it holds any.
pub const Target = struct {
    ctx: *anyopaque,
    tick: *const fn (ctx: *anyopaque) void,
    /// The command buffer alone, each interval a request runs: a pass committed whole submits nothing for its length.
    probe: ?*const fn (ctx: *anyopaque) void = null,
};

/// The served interval: macOS unwires an idle model after 1-2 s, so a tick each 400 ms stays well inside it.
pub const served_interval_ns: i64 = 400 * std.time.ns_per_ms;

/// The ticker: commits every 400 ms while no request runs and `--keep-warm` seconds have not passed.
pub const Keepalive = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    target: Target,
    /// One tick per interval, set before the thread starts.
    interval_ns: i64 = served_interval_ns,
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
        return startEvery(gpa, io, target, window_ns, served_interval_ns);
    }

    /// `start` with its own tick interval, fixed before the thread first reads it.
    pub fn startEvery(gpa: std.mem.Allocator, io: std.Io, target: Target, window_ns: i64, interval_ns: i64) !*Keepalive {
        const k = try gpa.create(Keepalive);
        k.* = .{ .gpa = gpa, .io = io, .target = target, .interval_ns = interval_ns, .window_ns = window_ns, .idle_since = .init(now(io)) };
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

    /// Whether the interval's probe is due: a request runs and the target has one.
    pub fn probing(k: *const Keepalive) bool {
        return k.target.probe != null and k.busy.load(.acquire) > 0;
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
            if (k.due(now(k.io))) k.target.tick(k.target.ctx) else if (k.probing()) k.target.probe.?(k.target.ctx);
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

    /// A ticker on this gate every 10 ms.
    fn start(g: *Gated, window_ns: i64) !*Keepalive {
        return Keepalive.startEvery(testing.allocator, std.testing.io, .{ .ctx = g, .tick = tick }, window_ns, 10 * std.time.ns_per_ms);
    }

    /// Opens the gate and stops `k`; deferred, so a failed test leaves no ticker on memory the test runner frees.
    fn done(g: *Gated, k: *Keepalive) void {
        g.gate.store(1, .release);
        k.stop();
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
    const k = try gated.start(900 * std.time.ns_per_s);
    defer gated.done(k);
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
}

test "a target with a probe has it committed while a request runs, and its tick only while idle" {
    const Probed = struct {
        var probes: std.atomic.Value(u32) = .init(0);
        fn probe(_: *anyopaque) void {
            _ = probes.fetchAdd(1, .monotonic);
        }
    };
    Probed.probes.store(0, .release);
    var gated = Gated{};
    gated.gate.store(1, .release); // ticks pass straight through
    const k = try Keepalive.startEvery(testing.allocator, std.testing.io, .{ .ctx = &gated, .tick = Gated.tick, .probe = Probed.probe }, 900 * std.time.ns_per_s, std.time.ns_per_ms);
    defer k.stop();
    try until(&gated.commits, 1);
    try testing.expectEqual(@as(u32, 0), Probed.probes.load(.acquire)); // idle: the tick, never the probe
    k.begin();
    try until(&Probed.probes, 3); // a request runs: the probe each interval
    k.end();
    const ticked = gated.commits.load(.acquire);
    try until(&gated.commits, ticked + 2); // the request ended: ticks resume
}

test "a probe is due only while a request runs: not idle inside the window, past it, or once the last request ends" {
    const none = struct {
        fn probe(_: *anyopaque) void {}
    }.probe;
    var k: Keepalive = .{ .gpa = testing.allocator, .io = std.testing.io, .target = .{ .ctx = undefined, .tick = undefined, .probe = none }, .window_ns = 100, .idle_since = .init(1000) };
    try testing.expect(k.due(1050) and !k.probing()); // idle inside the window: the tick's interval, never the probe's
    try testing.expect(!k.due(1101) and !k.probing()); // idle past the window: neither
    k.begin();
    k.begin();
    try testing.expect(k.probing());
    k.end();
    try testing.expect(k.probing()); // one of two requests still runs
    k.end();
    try testing.expect(!k.probing()); // the last request ended
    k.window_ns = 0;
    try testing.expect(!k.probing());
}

test "past the keep-warm window an idle ticker commits nothing, and the probe stops with the last request" {
    const Probed = struct {
        var probes: std.atomic.Value(u32) = .init(0);
        var gate: std.atomic.Value(u32) = .init(0);
        fn probe(_: *anyopaque) void {
            _ = probes.fetchAdd(1, .monotonic);
            while (gate.load(.acquire) == 0) std.Thread.yield() catch {};
        }
        fn done(k: *Keepalive) void {
            gate.store(1, .release);
            k.stop();
        }
    };
    Probed.probes.store(0, .release);
    Probed.gate.store(0, .release);
    var gated = Gated{};
    gated.gate.store(1, .release);
    // a 1 ns window has passed before the first interval, and again one interval after each request ends
    const k = try Keepalive.startEvery(testing.allocator, std.testing.io, .{ .ctx = &gated, .tick = Gated.tick, .probe = Probed.probe }, 1, 5 * std.time.ns_per_ms);
    defer Probed.done(k);
    for (0..20) |_| {
        std.Io.sleep(std.testing.io, .fromMilliseconds(5), .awake) catch {};
        try testing.expectEqual(@as(u32, 0), Probed.probes.load(.acquire)); // idle: no probe, though the target has one
    }
    k.begin();
    try until(&Probed.probes, 1); // a request runs: a probe arrives and parks in its gate
    k.end(); // the last request ends while it is parked
    Probed.gate.store(1, .release);
    for (0..20) |_| {
        std.Io.sleep(std.testing.io, .fromMilliseconds(5), .awake) catch {};
        try testing.expectEqual(@as(u32, 1), Probed.probes.load(.acquire)); // flat once no request runs
    }
    try testing.expectEqual(@as(u32, 0), gated.commits.load(.acquire)); // and the window never opened a tick
}

test "ticks stop once the keep-warm window has passed and resume when a request ends" {
    var gated = Gated{};
    const k = try gated.start(900 * std.time.ns_per_s);
    defer gated.done(k);
    try until(&gated.commits, 1); // the window is open: a commit arrives and parks in the gate
    k.idle_since.store(Keepalive.now(std.testing.io) - k.window_ns - 1, .release); // the window passes while it is parked
    gated.gate.store(1, .release);
    for (0..20) |_| {
        std.Io.sleep(std.testing.io, .fromMilliseconds(5), .awake) catch {};
        try testing.expectEqual(@as(u32, 1), gated.commits.load(.acquire)); // flat once the window has passed
    }
    k.begin(); // a request ran and ended: the window restarts from its end
    k.end();
    try until(&gated.commits, 2);
}

test "the window decides due without a thread" {
    var k: Keepalive = .{ .gpa = testing.allocator, .io = std.testing.io, .target = .{ .ctx = undefined, .tick = undefined }, .window_ns = 100, .idle_since = .init(1000) };
    try testing.expect(k.due(1100)); // inside the window
    try testing.expect(!k.due(1101)); // past it
    k.begin();
    try testing.expect(!k.due(1050)); // a request runs: no tick even inside the window
    try testing.expect(!k.probing()); // and no probe for a target that has none
    k.end();
    try testing.expect(k.due(1050));
    k.window_ns = 0;
    try testing.expect(!k.due(1000)); // keep-warm 0: never due
}
