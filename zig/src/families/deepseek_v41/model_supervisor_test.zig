//! Fault notification is independent of request/health polls, and its thread belongs to the model.
const std = @import("std");
const Supervisor = @import("model_supervisor.zig").Supervisor;
const tp = @import("tp");
const t = std.testing;
const Case = struct {
    healthy: std.atomic.Value(bool) = .init(true),
    calls: std.atomic.Value(u32) = .init(0),
    before_health: std.atomic.Value(bool) = .init(false),
    fate: ?*tp.Fate = null,
    failure: ?anyerror = null,
    fn read(ctx: *anyopaque) bool {
        return self(ctx).healthy.load(.acquire);
    }
    fn fail(ctx: *anyopaque, err: anyerror) void {
        const c = self(ctx);
        c.before_health.store(c.healthy.load(.acquire), .release);
        c.failure = err;
        if (c.fate) |f| f.fatal("CUDA health", err);
        _ = c.calls.fetchAdd(1, .acq_rel);
    }
    fn self(ctx: *anyopaque) *Case {
        return @ptrCast(@alignCast(ctx));
    }
    fn supervisor(c: *Case) Supervisor {
        return .{ .io = t.io, .ctx = c, .healthy = read, .fail = fail };
    }
};
var exits: std.atomic.Value(u32) = .init(0);
fn exited(code: u8) void {
    std.debug.assert(code == 70);
    _ = exits.fetchAdd(1, .acq_rel);
}
fn wait(c: *Case) !void {
    const until = std.Io.Clock.awake.now(t.io).toNanoseconds() + 5 * std.time.ns_per_s;
    while (c.calls.load(.acquire) == 0) {
        if (std.Io.Clock.awake.now(t.io).toNanoseconds() >= until) return error.MissingFailureNotification;
        try std.Io.sleep(t.io, .fromMilliseconds(1), .awake);
    }
}
test "model supervisor: a sticky context fault reaches each rank's fate after health is withdrawn" {
    exits.store(0, .release);
    for (0..2) |rank| {
        var f = tp.Fate.init(.{ .rank = @intCast(rank), .world = 2, .grace_ns = 0 }, @splat(-1));
        f.exit_fn = exited;
        defer f.close();
        var c: Case = .{ .fate = &f };
        var s = c.supervisor();
        try t.expect(!s.check());
        c.healthy.store(false, .release);
        try t.expect(s.check());
        try t.expect(s.check());
        try t.expectEqual(@as(u32, 1), c.calls.load(.acquire));
        try t.expect(!c.before_health.load(.acquire));
        try t.expect(std.mem.indexOf(u8, f.reason(), "CudaContextLost") != null);
    }
    try t.expectEqual(@as(u32, 2), exits.load(.acquire));
}
test "model supervisor: a live independent thread notices a fault without a health poll" {
    var c: Case = .{};
    var s = c.supervisor();
    try s.start();
    defer s.stop();
    c.healthy.store(false, .release);
    try wait(&c);
    try t.expectEqual(@as(u32, 1), c.calls.load(.acquire));
}
test "model supervisor: a healthy idle owner stops and joins without firing" {
    var c: Case = .{};
    var s = c.supervisor();
    try s.start();
    s.stop();
    try t.expect(s.thread == null);
    c.healthy.store(false, .release);
    try t.expectEqual(@as(u32, 0), c.calls.load(.acquire));
}

test "model supervisor: stalled owned work exits without health polling while healthy idle work stays alive" {
    var c: Case = .{};
    var p: @import("round_progress.zig").Progress = .{ .io = t.io };
    var s = c.supervisor();
    s.progress = &p;
    s.stall_ns = std.time.ns_per_ms;
    try t.expect(!s.check());
    var work = p.beginAt(.round, 0);
    defer work.end();
    try s.start();
    defer s.stop();
    try wait(&c);
    try t.expectEqual(error.StalledRound, c.failure.?);
    try t.expect(s.failed.load(.acquire));
}
test "model supervisor: an expired work scope withdraws native readiness and admission before failure handling" {
    const lanes = @import("lanes");
    const api = @import("engine_api");
    const cuda = @import("cuda");
    var cfg = try lanes.Config.init(t.allocator, .{}, 1, 0);
    defer cfg.deinit(t.allocator);
    var target: lanes.fake.Fake = .{ .gpa = t.allocator };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(t.allocator, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var c: Case = .{};
    var d: cuda.Driver = undefined;
    var m: @import("cuda_model.zig").Model = undefined;
    m.ctx = .{ .d = &d, .device = 0, .handle = @ptrFromInt(0x5401) };
    m.supervisor = c.supervisor();
    var p: @import("round_progress.zig").Progress = .{ .io = t.io };
    m.supervisor.?.progress = &p;
    m.supervisor.?.stall_ns = 1;
    var host = api.LaneHost.init(t.allocator, t.io, &core, .{ .lanes = 1 });
    host.health = .{ .ctx = &m, .read = @import("model_health.zig").serving };
    const eng = host.engine();
    var status: api.Status = .{};
    eng.status(&status, &.{});
    try t.expect(status.healthy);
    var scope = p.beginAt(.round, 0);
    defer scope.end();
    try t.expect(m.supervisor.?.check());
    eng.status(&status, &.{});
    try t.expect(!status.healthy);
    const Sink = struct {
        fn event(_: *anyopaque, _: api.Id, _: *const api.Event) void {
            @panic("refused work reached a sink");
        }
    };
    const request: api.Request = .{ .prompt = &.{1}, .max_tokens = 1 };
    try t.expectError(error.Closed, eng.submit(1, &request, .{ .ctx = &c, .event = Sink.event }));
    try t.expectEqual(@as(usize, 0), host.queued.items.len);
}

test "model supervisor: the owned fatal callback exits a real child with fail-fast status 70" {
    if (comptime @import("builtin").os.tag != .linux) return error.SkipZigTest;
    const platform = @import("sys");
    const pid = platform.c.fork();
    if (pid < 0) return error.ForkFailed;
    if (pid == 0) {
        const Exit = struct {
            fn fail(_: *anyopaque, _: anyerror) void {
                std.c._exit(70);
            }
        };
        var c: Case = .{};
        var p: @import("round_progress.zig").Progress = .{ .io = t.io };
        var work = p.beginAt(.round, 0);
        defer work.end();
        var s = c.supervisor();
        s.progress = &p;
        s.stall_ns = 1;
        s.fail = Exit.fail;
        _ = s.check();
        std.c._exit(71);
    }
    var status: c_int = 0;
    try t.expectEqual(pid, platform.c.waitpid(pid, &status, 0));
    try t.expectEqual(@as(c_int, 0), status & 127);
    try t.expectEqual(@as(c_int, 70), (status >> 8) & 255);
}

test "model supervisor: fatal transport reports preserve the first error and withdraw readiness before notification" {
    var c: Case = .{};
    var s = c.supervisor();
    s.report(error.BackendFailed);
    s.report(error.CudaFailed);
    try t.expect(s.failed.load(.acquire));
    try t.expectEqual(@as(u32, 0), c.calls.load(.acquire));
    try t.expect(s.check());
    try t.expect(s.check());
    try t.expectEqual(error.BackendFailed, c.failure.?);
    try t.expectEqual(@as(u32, 1), c.calls.load(.acquire));
}
