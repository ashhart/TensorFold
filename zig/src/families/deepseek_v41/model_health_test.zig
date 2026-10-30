//! Watchdog and fatal-error hooks attach only when they apply; a failed boot joins its supervisor first.
const std = @import("std");
const api = @import("engine_api");
const cuda = @import("cuda");
const lanes = @import("lanes");
const tp = @import("tp");
const health = @import("model_health.zig");
const Model = @import("cuda_model.zig").Model;
const Supervisor = @import("model_supervisor.zig").Supervisor;
const t = std.testing;
const Case = struct {
    calls: std.atomic.Value(u32) = .init(0),
    fn healthy(_: *anyopaque) bool {
        return true;
    }
    fn fail(ctx: *anyopaque, _: anyerror) void {
        const c: *Case = @ptrCast(@alignCast(ctx));
        _ = c.calls.fetchAdd(1, .acq_rel);
    }
    fn supervisor(c: *Case, stall_ns: u64) Supervisor {
        return .{ .io = t.io, .ctx = c, .healthy = healthy, .fail = fail, .stall_ns = stall_ns };
    }
};
fn hosted(m: *Model, core: *lanes.Engine) api.LaneHost {
    var host = api.LaneHost.init(t.allocator, t.io, core, .{ .lanes = 1 });
    health.watch(m);
    health.attach(m, &host);
    return host;
}
test "model health: with the watchdog off and no Fate, no launch, lane round or fatal error reaches the supervisor" {
    var cfg = try lanes.Config.init(t.allocator, .{}, 1, 0);
    defer cfg.deinit(t.allocator);
    var target: lanes.fake.Fake = .{ .gpa = t.allocator };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(t.allocator, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var c: Case = .{};
    var m: Model = undefined;
    m.progress = .{ .io = t.io };
    m.runner.progress = &m.progress;
    m.f.progress = &m.progress;
    m.session.fate = null;
    m.supervisor = c.supervisor(0);
    const host = hosted(&m, &core);
    try t.expect(m.runner.progress == null and m.f.progress == null);
    try t.expect(host.activity == null and host.failure == null);
    if (host.failure) |f| f.call(f.ctx, error.CudaFailed);
    try t.expect(!m.supervisor.?.check());
    try t.expectEqual(@as(u32, 0), c.calls.load(.acquire));
}
test "model health: an enabled watchdog owns launches and lane rounds, and a Fate takes fatal errors only" {
    var cfg = try lanes.Config.init(t.allocator, .{}, 1, 0);
    defer cfg.deinit(t.allocator);
    var target: lanes.fake.Fake = .{ .gpa = t.allocator };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(t.allocator, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var c: Case = .{};
    var m: Model = undefined;
    m.progress = .{ .io = t.io };
    m.session.fate = tp.Fate.init(.{ .rank = 0, .world = 2, .grace_ns = 0 }, @splat(-1));
    defer m.session.fate.?.close();
    m.supervisor = c.supervisor(std.time.ns_per_s);
    const host = hosted(&m, &core);
    try t.expect(m.runner.progress == &m.progress and m.f.progress == &m.progress);
    const activity = host.activity.?;
    activity.begin(activity.ctx);
    try t.expectEqual(@as(u32, 1), m.progress.active);
    activity.end(activity.ctx);
    try t.expectEqual(@as(u32, 0), m.progress.active);
    const failure = host.failure.?;
    failure.call(failure.ctx, error.GpuFailed);
    try t.expect(!m.supervisor.?.failed.load(.acquire));
    failure.call(failure.ctx, error.CudaFailed);
    try t.expect(m.supervisor.?.failed.load(.acquire));
    try t.expect(m.supervisor.?.check());
    try t.expectEqual(@as(u32, 1), c.calls.load(.acquire));
}
test "model boot: a failed boot joins the supervisor before the session can close" {
    const Sync = struct {
        var model: ?*Model = null;
        var joined: ?bool = null;
        fn context() callconv(.c) cuda.abi.Result {
            joined = model.?.supervisor.?.thread == null;
            return cuda.abi.error_not_ready;
        }
    };
    var driver = @import("load_mock.zig").init();
    driver.api.cuCtxSynchronize = Sync.context;
    var c: Case = .{};
    var m: Model = undefined;
    m.ctx = .{ .d = &driver, .device = 0, .handle = @ptrFromInt(0x5401) };
    m.supervisor = c.supervisor(0);
    try m.supervisor.?.start();
    defer m.supervisor.?.stop();
    Sync.model = &m;
    defer Sync.model = null;
    m.bootAborted(error.OutOfMemory);
    try t.expect(Sync.joined.?);
    try t.expectEqual(@as(u32, 0), c.calls.load(.acquire));
}
