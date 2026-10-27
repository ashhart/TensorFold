//! The loop guard's stop policy through LaneHost: the reply ends at the fire, and its lane and caches are released.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const t = std.testing;

const Box = struct {
    mutex: std.Io.Mutex = .init,
    done: ?api.Reason = null,
    period: ?u32 = null,
    tokens: usize = 0,
    fn event(ctx: *anyopaque, _: api.Id, e: *const api.Event) void {
        const b: *@This() = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(t.io);
        defer b.mutex.unlock(t.io);
        switch (e.*) {
            .tokens => |got| b.tokens += got.len,
            .finished => |f| {
                b.done = f.reason;
                b.period = f.stats.loop_period;
            },
            else => {},
        }
    }
    fn wait(b: *@This()) !api.Reason {
        for (0..20_000) |_| {
            b.mutex.lockUncancelable(t.io);
            const d = b.done;
            b.mutex.unlock(t.io);
            if (d) |r| return r;
            std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
        }
        return error.Timeout;
    }
};

test "loop stop: a stopped reply ends at the fire and frees its lane for the request waiting on its memory" {
    const gpa = t.allocator;
    const looping = [_]u32{ 2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5, 9 };
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    // one lane's memory: the second request runs only once the stopped one gives its caches back
    var target: lanes.fake.Fake = .{ .gpa = gpa, .memory_lanes = 1, .cycle_after = 70, .probe_prompt = looping.len };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, t.io, &core, .{ .lanes = 2 });
    try host.start();
    var stopped = false;
    defer if (!stopped) host.stop();
    const guarded: api.Request = .{ .prompt = &looping, .max_tokens = 4000, .think_close = &.{ 90, 91, 92 }, .think_end = 91, .loop_guard = true, .loop_stop = true };
    const short: api.Request = .{ .prompt = &.{ 3, 1, 4, 1 }, .max_tokens = 8 };
    var first: Box = .{};
    var second: Box = .{};
    const e = host.engine();
    try e.submit(1, &guarded, .{ .ctx = &first, .event = Box.event });
    try e.submit(2, &short, .{ .ctx = &second, .event = Box.event });
    try t.expectEqual(api.Reason.stop, try first.wait());
    try t.expectEqual(@as(?u32, 3), first.period);
    try t.expectEqual(@as(usize, 329), first.tokens); // the reasoning up to the fire: no close, no answer
    try t.expectEqual(api.Reason.length, try second.wait());
    try t.expectEqual(@as(usize, 8), second.tokens);
    host.stop();
    stopped = true;
    try t.expectEqual(@as(usize, 0), target.lanes.count());
}
