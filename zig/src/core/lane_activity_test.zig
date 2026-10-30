//! The real lane host retains work activity while a backend call is blocked and releases it before idle shutdown.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const t = std.testing;
const Case = struct {
    active: std.atomic.Value(u32) = .init(0),
    entered: std.atomic.Value(bool) = .init(false),
    release: std.atomic.Value(bool) = .init(false),
    done: std.atomic.Value(bool) = .init(false),
    unowned: std.atomic.Value(bool) = .init(false),
    fn self(ctx: *anyopaque) *Case {
        return @ptrCast(@alignCast(ctx));
    }
    fn begin(ctx: *anyopaque) void {
        _ = self(ctx).active.fetchAdd(1, .acq_rel);
    }
    fn end(ctx: *anyopaque) void {
        std.debug.assert(self(ctx).active.fetchSub(1, .acq_rel) == 1);
    }
    fn block(ctx: *anyopaque, _: *lanes.Stream, _: usize) void {
        const c = self(ctx);
        c.unowned.store(c.active.load(.acquire) == 0, .release);
        c.entered.store(true, .release);
        while (!c.release.load(.acquire)) std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
    }
    fn event(ctx: *anyopaque, _: api.Id, ev: *const api.Event) void {
        if (ev.* == .finished) self(ctx).done.store(true, .release);
    }
};
fn wait(flag: *const std.atomic.Value(bool)) !void {
    const until = std.Io.Clock.awake.now(t.io).toNanoseconds() + 5 * std.time.ns_per_s;
    while (!flag.load(.acquire)) {
        if (std.Io.Clock.awake.now(t.io).toNanoseconds() > until) return error.ActivityFixtureTimeout;
        try std.Io.sleep(t.io, .fromMilliseconds(1), .awake);
    }
}
test "lane activity: a real blocked prefill is owned until completion and idle stop releases the scope" {
    var cfg = try lanes.Config.init(t.allocator, .{}, 1, 0);
    defer cfg.deinit(t.allocator);
    var c: Case = .{};
    var fake: lanes.fake.Fake = .{ .gpa = t.allocator, .prefill_hook = Case.block, .prefill_hook_ctx = &c };
    defer fake.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(t.allocator, &cfg, fake.backend(), clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(t.allocator, t.io, &core, .{ .lanes = 1 });
    host.activity = .{ .ctx = &c, .begin = Case.begin, .end = Case.end };
    try host.start();
    var stopped = false;
    defer if (!stopped) {
        c.release.store(true, .release);
        host.stop();
    };
    const request: api.Request = .{ .prompt = &.{ 1, 2, 3 }, .max_tokens = 1 };
    try host.engine().submit(1, &request, .{ .ctx = &c, .event = Case.event });
    try wait(&c.entered);
    try t.expect(!c.unowned.load(.acquire));
    try t.expectEqual(@as(u32, 1), c.active.load(.acquire));
    c.release.store(true, .release);
    try wait(&c.done);
    host.stop();
    stopped = true;
    try t.expectEqual(@as(u32, 0), c.active.load(.acquire));
}
const Failed = struct {
    calls: u32 = 0,
    err: ?anyerror = null,
    done: std.atomic.Value(bool) = .init(false),
    message: [32]u8 = undefined,
    len: usize = 0,
    fn heard(ctx: *anyopaque, err: anyerror) void {
        const f: *Failed = @ptrCast(@alignCast(ctx));
        f.calls += 1;
        f.err = err;
    }
    fn explain(_: ?*anyopaque, _: anyerror) ?[]const u8 {
        return "a family's words";
    }
    fn event(ctx: *anyopaque, _: api.Id, ev: *const api.Event) void {
        const f: *Failed = @ptrCast(@alignCast(ctx));
        if (ev.* != .finished) return;
        f.len = @min(f.message.len, ev.finished.message.len);
        @memcpy(f.message[0..f.len], ev.finished.message[0..f.len]);
        f.done.store(true, .release);
    }
};
test "lane activity: a failed round reaches the failure hook while its streams keep the error's name" {
    var cfg = try lanes.Config.init(t.allocator, .{ .exact_width = 8, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(t.allocator);
    var fake: lanes.fake.Fake = .{ .gpa = t.allocator, .fail_round = 3 };
    defer fake.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(t.allocator, &cfg, fake.backend(), clock.clock());
    defer core.deinit();
    var f: Failed = .{};
    var host = api.LaneHost.init(t.allocator, t.io, &core, .{ .lanes = 1 });
    host.explain = .{ .text = Failed.explain };
    host.failure = .{ .ctx = &f, .call = Failed.heard };
    try host.start();
    defer host.stop();
    const request: api.Request = .{ .prompt = &.{ 2, 7, 1, 8 }, .max_tokens = 48 };
    try host.engine().submit(1, &request, .{ .ctx = &f, .event = Failed.event });
    try wait(&f.done);
    try t.expectEqualStrings("GpuFailed", f.message[0..f.len]);
    try t.expectEqual(@as(u32, 1), f.calls);
    try t.expectEqual(error.GpuFailed, f.err.?);
}
