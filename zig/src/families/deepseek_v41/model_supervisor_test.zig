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
    fn read(ctx: *anyopaque) bool {
        return self(ctx).healthy.load(.acquire);
    }
    fn fail(ctx: *anyopaque, err: anyerror) void {
        const c = self(ctx);
        c.before_health.store(c.healthy.load(.acquire), .release);
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
