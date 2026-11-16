//! A request that ends while its remote fill is under way tells the fill, which stops waiting on the other machine.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const remote = @import("lane_remote.zig");
const rt = @import("lane_remote_test.zig");
const t = std.testing;

/// A source whose fill waits until it is told the request is gone (or 3 s), as a pull stuck on a slow producer does.
const Waiter = struct {
    started: std.atomic.Value(bool) = .init(false),
    aborted: std.atomic.Value(bool) = .init(false),
    running: std.atomic.Value(u32) = .init(0),

    fn wants(_: *anyopaque, _: remote.Ask) remote.Verdict {
        return .remote;
    }

    fn fill(ptr: *anyopaque, _: []const u32, _: u64, gone: *const std.atomic.Value(bool)) anyerror!?remote.Filled {
        const w: *Waiter = @ptrCast(@alignCast(ptr));
        w.started.store(true, .release);
        const until = std.Io.Clock.awake.now(t.io).toNanoseconds() + 3 * std.time.ns_per_s;
        while (std.Io.Clock.awake.now(t.io).toNanoseconds() < until) {
            if (gone.load(.acquire)) {
                w.aborted.store(true, .release);
                return error.Aborted;
            }
            try std.Io.sleep(t.io, .fromMilliseconds(1), .awake);
        }
        return null;
    }

    fn drop(_: *anyopaque, _: pc.Saved) void {}
};

test "remote prefix: cancelling a request whose fill is under way aborts the fill at once" {
    var cfg = try lanes.Config.init(t.allocator, .{}, 1, 0);
    defer cfg.deinit(t.allocator);
    var fake: lanes.fake.Fake = .{ .gpa = t.allocator };
    defer fake.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(t.allocator, &cfg, fake.backend(), clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(t.allocator, t.io, &core, .{ .lanes = 1 });
    var w: Waiter = .{};
    host.remote = .{ .ptr = &w, .wants = Waiter.wants, .fill = Waiter.fill, .drop = Waiter.drop, .running = &w.running };
    try host.start();
    defer host.stop();
    var seen: rt.Seen = .{};
    try host.engine().submit(1, &.{ .prompt = &rt.six, .max_tokens = 4 }, .{ .ctx = &seen, .event = rt.Seen.event });
    while (!w.started.load(.acquire)) try std.Io.sleep(t.io, .fromMilliseconds(1), .awake);
    const began = std.Io.Clock.awake.now(t.io).toNanoseconds();
    host.engine().cancel(1);
    while (w.running.load(.acquire) > 0 or !seen.done.load(.acquire)) {
        if (std.Io.Clock.awake.now(t.io).toNanoseconds() - began > 2 * std.time.ns_per_s) return error.FillNotAborted;
        try std.Io.sleep(t.io, .fromMilliseconds(1), .awake);
    }
    try t.expect(w.aborted.load(.acquire));
}
