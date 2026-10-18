//! A lane host's ends: a planner whose admitted callback fails before any stream opens, and ends with no memory left.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const ends = @import("lane_host_end.zig");
const Reason = api.Reason;

/// A request's finished events: how many came, and the last one's reason.
const Box = struct {
    mutex: std.Io.Mutex = .init,
    finished: usize = 0,
    reason: ?Reason = null,
    fn event(ctx: *anyopaque, _: api.Id, e: *const api.Event) void {
        const b: *@This() = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(std.testing.io);
        defer b.mutex.unlock(std.testing.io);
        if (e.* == .finished) {
            b.finished += 1;
            b.reason = e.finished.reason;
        }
    }
    fn wait(b: *@This()) !Reason {
        for (0..10_000) |_| { // 10 s
            b.mutex.lockUncancelable(std.testing.io);
            const r = b.reason;
            b.mutex.unlock(std.testing.io);
            if (r) |x| return x;
            std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        }
        return error.Timeout;
    }
};

const prompt = [_]u32{ 2, 7, 1, 8, 2, 8 };
const request: api.Request = .{ .prompt = &prompt, .max_tokens = 8 };

/// The ends below, called on a host with no engine thread whose allocator fails from `arm` on.
const Rig = struct {
    failing: std.testing.FailingAllocator,
    boxes: [3]Box = .{ .{}, .{}, .{} },

    fn arm(r: *Rig) void {
        r.failing.fail_index = r.failing.alloc_index;
    }
    fn disarm(r: *Rig) void {
        r.failing.fail_index = std.math.maxInt(usize);
    }
    fn expectEach(r: *Rig, reason: Reason) !void {
        for (&r.boxes) |*b| {
            try std.testing.expectEqual(@as(usize, 1), b.finished);
            try std.testing.expectEqual(reason, b.reason.?);
        }
    }
};

const End = enum { fail, cancel, close };

/// Three requests submitted, the first admitted (its stream never opened), then `end` with every allocation failing.
fn endWithoutMemory(end: End) !void {
    var rig: Rig = .{ .failing = std.testing.FailingAllocator.init(std.testing.allocator, .{}) };
    const gpa = rig.failing.allocator();
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 4 });
    defer host.stop();
    const e = host.engine();
    for (&rig.boxes, 1..) |*b, id| try e.submit(id, &request, .{ .ctx = b, .event = Box.event });
    try host.admitted.append(gpa, host.queued.orderedRemove(0));
    if (end == .fail) {
        try host.admitted.appendSlice(gpa, host.queued.items);
        host.queued.clearRetainingCapacity();
    }
    if (end == .cancel) for (1..rig.boxes.len + 1) |id| e.cancel(id);
    rig.arm();
    switch (end) {
        .fail => ends.failAll(&host, "the planner refused"),
        .cancel => ends.takeCancels(&host),
        .close => ends.closeAll(&host),
    }
    rig.disarm();
    try rig.expectEach(if (end == .fail) .failed else .cancelled);
    try std.testing.expectEqual(@as(usize, 0), host.queued.items.len + host.admitted.items.len + host.cancels.items.len);
}

test "a failed round ends every admitted job once, failed, when no allocation can succeed" {
    try endWithoutMemory(.fail);
}

test "named cancellations end their queued and admitted jobs once, cancelled, when no allocation can succeed" {
    try endWithoutMemory(.cancel);
}

test "a close ends every queued and admitted job once, cancelled, when no allocation can succeed" {
    try endWithoutMemory(.close);
}

// last: on the old failAll this reads an unopened stream, which can end the test binary before the others report
test "a planner whose admitted callback fails ends the jobs it admitted, failed once each, opening no stream" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.pieceBackend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    const Plan = struct {
        admits: usize = 0,
        lefts: usize = 0,
        fn self(ctx: *anyopaque) *@This() {
            return @ptrCast(@alignCast(ctx));
        }
        fn beginAdmit(_: *anyopaque) void {}
        fn admit(ctx: *anyopaque, _: api.Rounds.Ask, _: *[]const u8) api.Rounds.Verdict {
            self(ctx).admits += 1;
            return .admit;
        }
        fn admitted(_: *anyopaque) anyerror!void {
            return error.PlannerFull;
        }
        fn arm(_: *anyopaque, _: usize) void {}
        fn began(_: *anyopaque, _: usize, _: bool) void {}
        fn plan(_: *anyopaque, pieces: *std.ArrayList(api.Rounds.Piece), finals: *std.ArrayList(usize)) anyerror!void {
            pieces.clearRetainingCapacity();
            finals.clearRetainingCapacity();
        }
        fn left(ctx: *anyopaque, _: usize) void {
            self(ctx).lefts += 1;
        }
        fn after(_: *anyopaque, _: f64, _: f64) void {}
    };
    var pl: Plan = .{};
    host.rounds = .{ .ctx = &pl, .vtable = &.{ .begin_admit = Plan.beginAdmit, .admit = Plan.admit, .admitted = Plan.admitted, .arm = Plan.arm, .began = Plan.began, .plan = Plan.plan, .left = Plan.left, .after = Plan.after } };
    const e = host.engine();
    var boxes: [2]Box = .{ .{}, .{} };
    for (&boxes, 1..) |*b, id| try e.submit(id, &request, .{ .ctx = b, .event = Box.event });
    try host.start();
    {
        errdefer host.stop(); // a timed-out wait stops the host thread before this frame goes
        for (&boxes) |*b| try std.testing.expectEqual(Reason.failed, try b.wait());
    }
    host.stop();
    for (&boxes) |*b| try std.testing.expectEqual(@as(usize, 1), b.finished);
    try std.testing.expectEqual(@as(usize, 2), pl.admits);
    try std.testing.expectEqual(@as(usize, 2), pl.lefts);
    try std.testing.expectEqual(@as(u32, 0), target.lanes.count()); // no stream was opened, none is left behind
}
