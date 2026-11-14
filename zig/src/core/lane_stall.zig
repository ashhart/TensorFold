//! --stall-seconds: one engine-loop pass (a round scope; idle waits are outside) open past the deadline is a stall.
const std = @import("std");
const Activity = @import("lane_activity.zig").Activity;

pub const Config = struct { timeout_ns: u64, exit: bool = false, family: []const u8 = "lanes" };

/// The status code --stall-exit leaves with, for a service manager's restart rule.
pub const exit_code: u8 = 70;

pub const Watch = struct {
    enabled: std.atomic.Value(bool) = .init(false),
    unhealthy: std.atomic.Value(bool) = .init(false),
    episodes: std.atomic.Value(u64) = .init(0),
    stopping: std.atomic.Value(bool) = .init(false),
    mutex: std.atomic.Mutex = .unlocked,
    active: usize = 0,
    began_ns: u64 = 0,
    limit_ns: u64 = 0,
    exit: bool = false,
    io: std.Io = undefined,
    family: []const u8 = "",
    gpa: ?std.mem.Allocator = null,
    thread: ?std.Thread = null,
    exit_hook: ?*const fn () void = null, // host tests observe the exit decision without ending the test process

    pub fn start(w: *Watch, a: std.mem.Allocator, io: std.Io, config: Config) !void {
        if (config.timeout_ns == 0) return;
        if (w.thread != null or w.enabled.load(.acquire)) return error.StallAlreadyConfigured;
        w.family = try a.dupe(u8, config.family);
        errdefer {
            a.free(w.family);
            w.family = "";
        }
        w.gpa = a;
        w.io = io;
        w.limit_ns = config.timeout_ns;
        w.exit = config.exit;
        w.stopping.store(false, .release);
        w.thread = try std.Thread.spawn(.{}, run, .{w});
        w.enabled.store(true, .release);
    }

    /// After the engine thread has ended: no scope is open, so nothing reads `family` again.
    pub fn stop(w: *Watch) void {
        w.enabled.store(false, .release);
        w.stopping.store(true, .release);
        if (w.thread) |thread| thread.join();
        w.thread = null;
        if (w.gpa) |a| if (w.family.len > 0) a.free(w.family);
        w.gpa = null;
        w.family = "";
    }

    /// Stall episodes since the engine opened; null while the sentinel is off, so /metrics adds nothing.
    pub fn count(w: *const Watch) ?u64 {
        return if (w.enabled.load(.acquire)) w.episodes.load(.acquire) else null;
    }

    /// The scope observer while the sentinel is on; null (one atomic load a pass) while it is off.
    pub fn activity(w: *Watch) ?Activity {
        return if (w.enabled.load(.acquire)) .{ .ctx = w, .begin = begin, .end = end } else null;
    }

    fn begin(ctx: *anyopaque) void {
        const w: *Watch = @ptrCast(@alignCast(ctx));
        w.beginAt(w.now());
    }
    pub fn beginAt(w: *Watch, now_ns: u64) void {
        w.lock();
        defer w.mutex.unlock();
        if (w.active == 0) w.began_ns = now_ns;
        w.active += 1;
    }
    fn end(ctx: *anyopaque) void {
        const w: *Watch = @ptrCast(@alignCast(ctx));
        w.endScope();
    }
    /// The scope that stalled has ended: health and admission return; the episode stays counted.
    pub fn endScope(w: *Watch) void {
        w.lock();
        defer w.mutex.unlock();
        std.debug.assert(w.active > 0);
        w.active -= 1;
        if (w.active == 0) w.unhealthy.store(false, .release);
    }

    /// True when this check finds a new stall: one count and one log line a scope, however often it is polled.
    pub fn checkAt(w: *Watch, now_ns: u64) bool {
        w.lock();
        const expired = w.limit_ns > 0 and w.active > 0 and now_ns -| w.began_ns >= w.limit_ns;
        const newly = expired and !w.unhealthy.swap(true, .acq_rel);
        if (newly) _ = w.episodes.fetchAdd(1, .acq_rel);
        w.mutex.unlock();
        if (!newly) return false;
        if (!@import("builtin").is_test)
            std.log.warn("{s}: engine stall: a round has run for {d} s (--stall-seconds); /health is 503 and new requests are refused until it ends{s}", .{ w.family, w.limit_ns / std.time.ns_per_s, if (w.exit) "; exiting (--stall-exit)" else "" });
        // health is withdrawn first; the exit runs on this thread, holding no engine lock
        if (w.exit) if (w.exit_hook) |hook| hook() else leave();
        return true;
    }

    /// No atexit handlers: a driver's teardown could wait on the stalled device.
    fn leave() noreturn {
        if (@import("builtin").link_libc) std.c._exit(exit_code);
        std.process.exit(exit_code);
    }
    fn run(w: *Watch) void {
        while (!w.stopping.load(.acquire)) {
            if (w.enabled.load(.acquire)) _ = w.checkAt(w.now());
            std.Io.sleep(w.io, .fromMilliseconds(100), .awake) catch {};
        }
    }
    fn now(w: *const Watch) u64 {
        return @intCast(@max(0, std.Io.Clock.awake.now(w.io).toNanoseconds()));
    }
    fn lock(w: *Watch) void {
        while (!w.mutex.tryLock()) std.atomic.spinLoopHint();
    }
};

test "lane sentinel: idle, short prompt scopes, repeated checks, recovery and another episode" {
    var w: Watch = .{ .limit_ns = 100 };
    try std.testing.expect(!w.checkAt(100000));
    w.beginAt(1000);
    try std.testing.expect(!w.checkAt(1099));
    w.endScope();
    for (0..100) |i| {
        w.beginAt(2000 + i * 100);
        try std.testing.expect(!w.checkAt(2099 + i * 100));
        w.endScope();
    }
    w.beginAt(20000);
    try std.testing.expect(w.checkAt(20100));
    try std.testing.expect(!w.checkAt(20900));
    try std.testing.expect(w.unhealthy.load(.acquire));
    try std.testing.expectEqual(@as(u64, 1), w.episodes.load(.acquire));
    w.endScope();
    try std.testing.expect(!w.unhealthy.load(.acquire));
    w.beginAt(30000);
    try std.testing.expect(w.checkAt(30100));
    w.endScope();
    try std.testing.expectEqual(@as(u64, 2), w.episodes.load(.acquire));
}

test "lane sentinel: exit is opt-in and is decided once for an expired scope" {
    const Exit = struct {
        var calls: usize = 0;
        fn call() void {
            calls += 1;
        }
    };
    Exit.calls = 0;
    var w: Watch = .{ .limit_ns = 100, .exit_hook = Exit.call };
    w.beginAt(0);
    try std.testing.expect(w.checkAt(100));
    try std.testing.expectEqual(@as(usize, 0), Exit.calls);
    w.endScope();
    w.exit = true;
    w.beginAt(200);
    try std.testing.expect(w.checkAt(300));
    try std.testing.expect(!w.checkAt(400));
    try std.testing.expectEqual(@as(usize, 1), Exit.calls);
    w.endScope();
}

test "lane sentinel: off adds no scope observer and no metric value" {
    var w: Watch = .{};
    try std.testing.expect(w.activity() == null);
    try std.testing.expectEqual(@as(?u64, null), w.count());
    try w.start(std.testing.allocator, std.testing.io, .{ .timeout_ns = 0 });
    try std.testing.expect(w.thread == null and w.activity() == null);
    w.stop();
}
