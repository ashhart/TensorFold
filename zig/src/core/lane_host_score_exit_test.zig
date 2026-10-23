//! A pending decision must be failed when its engine thread cannot initialize.
const std = @import("std");
const lanes = @import("lanes");
const LaneHost = @import("lane_host.zig").LaneHost;
const api = @import("engine_api.zig");
const t = std.testing;

fn failAfterAsk(ptr: *anyopaque) anyerror!void {
    const h: *LaneHost = @ptrCast(@alignCast(ptr));
    h.lock();
    defer h.unlock();
    while (h.scores.asks.items.len == 0 and !h.closing) h.wake.waitUncancelable(h.io, &h.mutex);
    return error.InjectedInitializationFailure;
}
fn neverScore(_: *anyopaque, _: []const u32, _: []const u32, _: []f64, _: bool) anyerror!f64 {
    return error.UnexpectedScore;
}
const Caller = struct {
    engine: api.Engine,
    result: ?anyerror = null,
    fn run(c: *Caller) void {
        var values: [1]f64 = undefined;
        _ = c.engine.score(&.{1}, &.{2}, &values, false) catch |err| {
            c.result = err;
            return;
        };
    }
};

test "score exit: failed engine initialization fails every pending score without stop" {
    var cfg = try lanes.Config.init(t.allocator, .{}, 1, 0);
    defer cfg.deinit(t.allocator);
    var fake = lanes.fake.Fake{ .gpa = t.allocator };
    defer fake.deinit();
    const backend = fake.backend();
    var vt = backend.vtable.*;
    vt.score = neverScore;
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(t.allocator, &cfg, .{ .ptr = backend.ptr, .vtable = &vt }, clock.clock());
    defer core.deinit();
    var host = LaneHost.init(t.allocator, t.io, &core, .{ .lanes = 1 });
    host.on_thread = .{ .ctx = &host, .run = failAfterAsk };
    try host.start();
    var stopped = false;
    defer if (!stopped) host.stop();
    var caller = Caller{ .engine = host.engine() };
    const thread = try std.Thread.spawn(.{}, Caller.run, .{&caller});
    // Joining the failed worker makes this a state check, without a timing threshold or an orphaned thread.
    host.thread.?.join();
    host.thread = null;
    host.lock();
    const pending = host.scores.asks.items.len;
    const unsignalled = pending > 0 and !host.scores.asks.items[0].done;
    host.unlock();
    // Rescue the caller before asserting, so this regression cannot hang the test process.
    host.stop();
    stopped = true;
    thread.join();
    try t.expectEqual(error.Failed, caller.result.?);
    try t.expect(!unsignalled);
    try t.expectEqual(@as(usize, 0), pending);
}
