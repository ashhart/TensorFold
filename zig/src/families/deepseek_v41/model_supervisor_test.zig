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
test "model supervisor: a stop signal is reported once as StopRequested, ahead of a context fault" {
    var c: Case = .{};
    var sig = std.atomic.Value(bool).init(false);
    var s = c.supervisor();
    s.signal = &sig;
    try t.expect(!s.check());
    sig.store(true, .release);
    c.healthy.store(false, .release);
    try t.expect(s.check());
    try t.expect(s.check());
    try t.expectEqual(@as(u32, 1), c.calls.load(.acquire));
    try t.expectEqual(error.StopRequested, c.failure.?);
    try t.expect(s.failed.load(.acquire));
}
var aborts: std.atomic.Value(u32) = .init(0);
fn aborted(_: *anyopaque, _: []const u8) void {
    _ = aborts.fetchAdd(1, .acq_rel);
}
test "model supervisor: a stop on rank 1 aborts both ranks' collectives and ends both through fail-fast" {
    exits.store(0, .release);
    aborts.store(0, .release);
    const l = try tp.bootstrap.testListener();
    defer tp.sock.close(l.fd);
    const deadline = tp.sock.nowNs() + 5 * std.time.ns_per_s;
    var fds: [2][tp.fate.max_ranks]tp.sock.Fd = .{ @splat(-1), @splat(-1) };
    fds[1][0] = try tp.sock.connect("localhost", l.port, deadline);
    fds[0][1] = try tp.sock.accept(l.fd, deadline);
    var fates: [2]tp.Fate = undefined;
    for (&fates, fds, 0..) |*f, fd, rank| {
        f.* = tp.Fate.init(.{ .world = 2, .rank = @intCast(rank), .grace_ns = 0 }, fd);
        f.exit_fn = exited;
        f.onFail(.{ .ctx = f, .call = aborted });
    }
    for (&fates) |*f| try f.start();
    defer for (&fates) |*f| f.close();
    var sig = std.atomic.Value(bool).init(false);
    var c: Case = .{ .fate = &fates[1] };
    var s = c.supervisor();
    s.signal = &sig;
    try t.expect(!s.check());
    const t0 = tp.sock.nowNs();
    sig.store(true, .release);
    try t.expect(s.check());
    while (exits.load(.acquire) < 2) {
        if (tp.sock.nowNs() - t0 > 2 * std.time.ns_per_s) return error.PeerNotStopped;
        tp.sock.sleepNs(std.time.ns_per_ms);
    }
    for (&fates) |*f| f.close();
    try t.expectEqual(@as(u32, 2), aborts.load(.acquire));
    try t.expect(std.mem.indexOf(u8, fates[0].reason(), "rank 1 failed") != null);
    try t.expect(std.mem.indexOf(u8, fates[0].reason(), "StopRequested") != null);
}
test "model supervisor: the stop handler raises the flag the supervisor reads, and a second signal kills" {
    if (comptime @import("builtin").os.tag == .windows) return error.SkipZigTest;
    const sup = @import("model_supervisor.zig");
    const platform = @import("sys");
    var fds: [2]c_int = undefined;
    if (platform.c.pipe(&fds) != 0) return error.PipeFailed;
    defer _ = platform.c.close(fds[0]);
    const pid = platform.c.fork();
    if (pid < 0) {
        _ = platform.c.close(fds[1]);
        return error.ForkFailed;
    }
    if (pid == 0) {
        sup.installStopSignals();
        _ = platform.c.raise(.TERM);
        if (!sup.stop_signal.load(.acquire)) std.c._exit(2);
        var c: Case = .{};
        var s = c.supervisor();
        if (!s.check() or c.failure.? != error.StopRequested) std.c._exit(3);
        _ = platform.c.write(fds[1], "S", 1);
        _ = platform.c.raise(.INT);
        std.c._exit(4);
    }
    _ = platform.c.close(fds[1]); // the write end is the child's alone: a child that never wrote reads 0 bytes
    var status: c_int = 0;
    try t.expectEqual(pid, platform.c.waitpid(pid, &status, 0));
    var byte: [1]u8 = .{0};
    try t.expectEqual(@as(isize, 1), platform.c.read(fds[0], &byte, 1));
    try t.expectEqual(@as(u8, 'S'), byte[0]);
    try t.expectEqual(@as(c_int, @backingInt(std.posix.SIG.INT)), status & 127);
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

test "model supervisor: the real model fatal callback exits without synchronous NCCL abort when fate is absent" {
    if (comptime @import("builtin").os.tag != .linux) return error.SkipZigTest;
    const platform = @import("sys");
    const nccl = @import("cuda").nccl;
    const pid = platform.c.fork();
    if (pid < 0) return error.ForkFailed;
    if (pid == 0) {
        const Abort = struct {
            fn entered(_: nccl.Comm) callconv(.c) nccl.Result {
                std.c._exit(71);
            }
        };
        var lib: nccl.Library = undefined;
        lib.api.ncclCommAbort = Abort.entered;
        var m: @import("cuda_model.zig").Model = undefined;
        m.session.fate = null;
        m.session.mail = null;
        m.session.roce = null;
        m.session.roce_multi = null;
        m.session.nccl.lib = &lib;
        m.session.nccl.borrowed = null;
        m.session.nccl.aborted = .init(false);
        m.session.nccl.comm = null;
        @import("model_health.zig").failed(&m, error.CudaContextLost);
        std.c._exit(72);
    }
    var status: c_int = 0;
    try t.expectEqual(pid, platform.c.waitpid(pid, &status, 0));
    try t.expectEqual(@as(c_int, 0), status & 127);
    try t.expectEqual(@as(c_int, 70), (status >> 8) & 255);
}
