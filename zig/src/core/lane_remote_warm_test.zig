//! A request that waits for its remote fill keeps the model warm: the engine ticks the keepalive while it waits.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const remote = @import("lane_remote.zig");
const t = std.testing;

/// A remote source whose fill takes `delay_ms` (0: it wants nothing), and a keepalive target that counts its ticks.
const Slow = struct {
    delay_ms: u32,
    ticks: std.atomic.Value(u32) = .init(0),
    running: std.atomic.Value(u32) = .init(0),
    done: std.atomic.Value(bool) = .init(false),
    kept: std.ArrayList(u32) = .empty,

    fn wants(ptr: *anyopaque, _: remote.Ask) remote.Verdict {
        const s: *Slow = @ptrCast(@alignCast(ptr));
        return if (s.delay_ms > 0) .remote else .small;
    }
    fn fill(ptr: *anyopaque, prompt: []const u32, _: u64, _: *const std.atomic.Value(bool)) anyerror!?remote.Filled {
        const s: *Slow = @ptrCast(@alignCast(ptr));
        try std.Io.sleep(t.io, .fromMilliseconds(s.delay_ms), .awake);
        try s.kept.appendSlice(t.allocator, prompt[0..4]);
        return .{ .saved = &s.kept, .at = 4 };
    }
    fn drop(_: *anyopaque, _: pc.Saved) void {}
    fn tick(ctx: *anyopaque) void {
        const s: *Slow = @ptrCast(@alignCast(ctx));
        _ = s.ticks.fetchAdd(1, .acq_rel);
    }
    fn event(ctx: *anyopaque, _: api.Id, ev: *const api.Event) void {
        const s: *Slow = @ptrCast(@alignCast(ctx));
        if (ev.* == .finished) s.done.store(true, .release);
    }
};

fn ticksOf(delay_ms: u32) !u32 {
    var cfg = try lanes.Config.init(t.allocator, .{}, 1, 0);
    defer cfg.deinit(t.allocator);
    var fake: lanes.fake.Fake = .{ .gpa = t.allocator };
    defer fake.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(t.allocator, &cfg, fake.backend(), clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(t.allocator, t.io, &core, .{ .lanes = 1 });
    var slow: Slow = .{ .delay_ms = delay_ms };
    defer slow.kept.deinit(t.allocator);
    host.remote = .{ .ptr = &slow, .wants = Slow.wants, .fill = Slow.fill, .drop = Slow.drop, .running = &slow.running };
    host.keepalive_target = .{ .ctx = &slow, .tick = Slow.tick };
    try host.start();
    defer host.stop();
    const prompt = [_]u32{ 11, 12, 13, 14, 15, 16 };
    try host.engine().submit(1, &.{ .prompt = &prompt, .max_tokens = 2 }, .{ .ctx = &slow, .event = Slow.event });
    const until = std.Io.Clock.awake.now(t.io).toNanoseconds() + 10 * std.time.ns_per_s;
    while (!slow.done.load(.acquire) or slow.running.load(.acquire) > 0) {
        if (std.Io.Clock.awake.now(t.io).toNanoseconds() > until) return error.RemoteFixtureTimeout;
        try std.Io.sleep(t.io, .fromMilliseconds(1), .awake);
    }
    return slow.ticks.load(.acquire);
}

test "remote prefix: the engine ticks the keepalive while a fill is awaited, and not when none is" {
    try t.expect(try ticksOf(1100) >= 2); // 400 ms apart over a 1.1 s fill
    try t.expectEqual(@as(u32, 0), try ticksOf(0));
}
