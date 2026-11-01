//! Multirank on host threads over the shm wire: a rank that stops posting ends its peers' waits at the timeout.
const std = @import("std");
const roce = @import("roce_multi.zig");
const bootstrap = @import("bootstrap.zig");
const sock = @import("sock.zig");
const Bootstrap = bootstrap.Bootstrap;

const world = 3;
const n = 256;

const Run = struct {
    port: u16,
    lfd: sock.Fd,
    timeout_ns: u64,
    /// The last rank posts this many gathers, then stops: nothing but the timeout ends the others' next wait.
    stop_after: u32,
    errs: [world]?anyerror = @splat(null),
    done: [world]u32 = @splat(0),
    waited_ns: [world]u64 = @splat(0),
    roces: [world]std.atomic.Value(?*roce.Roce) = @splat(.init(null)),
    over: std.atomic.Value(bool) = .init(false),
    fired: bool = false,

    fn rank(run: *Run, me: u32) void {
        run.errs[me] = run.body(me);
    }

    fn body(run: *Run, me: u32) ?anyerror {
        var b = (if (me == 0) Bootstrap.acceptOn(run.lfd, world, 5 * std.time.ns_per_s) else Bootstrap.open(.{ .world = world, .rank = me, .link_port = run.port, .connect_timeout_ns = 5 * std.time.ns_per_s })) catch |e| return e;
        defer b.close();
        const r = roce.Roce.init(null, &b, "", .{ .wire = .shm, .max_bytes = 4096, .timeout_ns = run.timeout_ns }) catch |e| return e;
        run.roces[me].store(r, .release); // the test tears it down after the watchdog, which may abort it first
        var in: [n]u8 = undefined;
        var out: [world * n]u8 = undefined;
        for (0..run.stop_after + 1) |k| {
            if (k == run.stop_after and me == world - 1) return null;
            @memset(&in, @truncate(31 * @as(usize, me) + k));
            const t0 = sock.nowNs();
            r.hostGather(&in, &out) catch |e| {
                run.waited_ns[me] = sock.nowNs() - t0;
                return e;
            };
            for (0..world) |p| for (out[p * n ..][0..n]) |x| if (x != @as(u8, @truncate(31 * p + k))) return error.WrongBytes;
            run.done[me] += 1;
        }
        return null;
    }

    /// Aborts every rank's wire after `limit_ns` unless the test ended: an unbounded wait fails as `error.Aborted`.
    fn watch(run: *Run, limit_ns: u64) void {
        const t0 = sock.nowNs();
        while (sock.nowNs() - t0 < limit_ns) {
            if (run.over.load(.acquire)) return;
            sock.sleepNs(5 * std.time.ns_per_ms);
        }
        run.fired = true;
        for (&run.roces) |*r| if (r.load(.acquire)) |x| x.abort();
    }
};

test "multirank: a rank that stops posting ends its peers' next gathers at the timeout, after every earlier gather matched" {
    if (comptime @import("builtin").os.tag != .linux) return error.SkipZigTest; // the shm wire: /dev/shm, a pinned proxy
    const t = std.testing;
    const l = try bootstrap.testListener();
    defer sock.close(l.fd);
    var run: Run = .{ .port = l.port, .lfd = l.fd, .timeout_ns = 100 * std.time.ns_per_ms, .stop_after = 10 };
    const watchdog = try std.Thread.spawn(.{}, Run.watch, .{ &run, 30 * run.timeout_ns });
    var ranks: [world - 1]std.Thread = undefined;
    for (&ranks, 1..) |*x, me| x.* = try std.Thread.spawn(.{}, Run.rank, .{ &run, @as(u32, @intCast(me)) });
    run.rank(0);
    for (ranks) |x| x.join();
    run.over.store(true, .release);
    watchdog.join();
    for (&run.roces) |*r| if (r.load(.acquire)) |x| x.deinit();
    try t.expect(!run.fired);
    for (0..world - 1) |me| {
        try t.expectEqual(@as(?anyerror, error.Timeout), run.errs[me]);
        try t.expect(run.waited_ns[me] >= run.timeout_ns and run.waited_ns[me] < 10 * run.timeout_ns);
    }
    try t.expectEqual(@as(?anyerror, null), run.errs[world - 1]);
    for (run.done) |d| try t.expectEqual(run.stop_after, d);
}
