//! Errors at admission and cancellation boundaries must keep every job owned.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const ends = @import("lane_host_end.zig");
const a = std.testing.allocator;
const Box = struct {
    done: std.atomic.Value(usize) = .init(0),
    seen: [18]usize = @splat(0),
    failed: usize = 0,
    cancelled: usize = 0,
    fn event(ctx: *anyopaque, id: api.Id, e: *const api.Event) void {
        const b: *@This() = @ptrCast(@alignCast(ctx));
        if (e.* == .finished) {
            b.seen[id] += 1;
            if (e.finished.reason == .failed) b.failed += 1;
            if (e.finished.reason == .cancelled) b.cancelled += 1;
            _ = b.done.fetchAdd(1, .release);
        }
    }
    fn wait(b: *@This(), n: usize) !void {
        for (0..3000) |_| {
            if (b.done.load(.acquire) == n) return;
            std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        }
        return error.Timeout;
    }
};
const Plan = struct {
    keys: [17]usize = undefined,
    n: usize = 0,
    planned: bool = false,
    fail_admitted: bool = false,
    left_n: usize = 0,
    fn self(ctx: *anyopaque) *@This() {
        return @ptrCast(@alignCast(ctx));
    }
    fn begin(_: *anyopaque) void {}
    fn admit(ctx: *anyopaque, ask: api.Rounds.Ask, _: *[]const u8) api.Rounds.Verdict {
        const p = self(ctx);
        p.keys[p.n] = ask.key;
        p.n += 1;
        return .admit;
    }
    fn admitted(ctx: *anyopaque) anyerror!void {
        if (self(ctx).fail_admitted) return error.AdmittedFailed;
    }
    fn key(_: *anyopaque, _: usize) void {}
    fn began(_: *anyopaque, _: usize, _: bool) void {}
    fn plan(ctx: *anyopaque, pieces: *std.ArrayList(api.Rounds.Piece), finals: *std.ArrayList(usize)) anyerror!void {
        const p = self(ctx);
        pieces.clearRetainingCapacity();
        finals.clearRetainingCapacity();
        if (p.planned) return;
        for (p.keys[0..p.n]) |k| {
            try pieces.append(a, .{ .key = k, .start = 0, .end = 2, .save = false });
            try finals.append(a, k);
        }
        p.planned = true;
    }
    fn left(ctx: *anyopaque, _: usize) void {
        self(ctx).left_n += 1;
    }
    fn after(_: *anyopaque, _: f64, _: f64) void {}
    fn rounds(p: *@This()) api.Rounds {
        return .{ .ctx = p, .vtable = &.{ .begin_admit = begin, .admit = admit, .admitted = admitted, .arm = key, .began = began, .plan = plan, .left = left, .after = after } };
    }
};
const Hooks = struct {
    var releases: usize = 0;
    var prepared: [17]*lanes.Stream = undefined;
    var n: usize = 0;
    var calls: usize = 0;
    var fail_call: usize = 0;
    var finished: usize = 0;
    fn release(ctx: *anyopaque, s: *lanes.Stream) void {
        releases += 1;
        const f: *lanes.fake.Fake = @ptrCast(@alignCast(ctx));
        f.pieceBackend().vtable.release(ctx, s);
    }
    fn finals(_: *anyopaque, ss: []const *lanes.Stream) anyerror!void {
        calls += 1;
        if (calls == fail_call) return error.FinalsFailed;
        for (ss) |s| {
            prepared[n] = s;
            n += 1;
        }
    }
    fn finish(ctx: *anyopaque, s: *lanes.Stream) anyerror!void {
        var found = false;
        for (prepared[0..n]) |p| found = found or p == s;
        if (!found) return error.FinalNotPrepared;
        finished += 1;
        const f: *lanes.fake.Fake = @ptrCast(@alignCast(ctx));
        return f.pieceBackend().vtable.finish.?(ctx, s);
    }
    fn reset() void {
        releases = 0;
        n = 0;
        calls = 0;
        fail_call = 0;
        finished = 0;
    }
};

test "Ownership rollback: admitted callback error ends an ordinary unopened job without backend release" {
    var cfg = try lanes.Config.init(a, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(a);
    var fake: lanes.fake.Fake = .{ .gpa = a };
    defer fake.deinit();
    Hooks.reset();
    var be = fake.pieceBackend();
    var vt = be.vtable.*;
    vt.release = Hooks.release;
    be.vtable = &vt;
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, be, clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(a, std.testing.io, &core, .{ .lanes = 1 });
    var p: Plan = .{ .fail_admitted = true };
    host.rounds = p.rounds();
    var b: Box = .{};
    const req: api.Request = .{ .prompt = &.{ 1, 2, 3 }, .max_tokens = 1 };
    try host.engine().submit(1, &req, .{ .ctx = &b, .event = Box.event });
    try host.start();
    {
        errdefer host.stop();
        try b.wait(1);
    }
    host.stop();
    try std.testing.expectEqual(@as(usize, 1), b.failed);
    try std.testing.expectEqual(@as(usize, 1), p.left_n);
    try std.testing.expectEqual(@as(usize, 0), Hooks.releases);
    try std.testing.expectEqual(@as(usize, 0), fake.lanes.count());
}

fn cancellation(fail_offset: usize, close: bool) !void {
    var cfg = try lanes.Config.init(a, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(a);
    var fake: lanes.fake.Fake = .{ .gpa = a };
    defer fake.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, fake.pieceBackend(), clock.clock());
    defer core.deinit();
    var f = std.testing.FailingAllocator.init(a, .{});
    var host = api.LaneHost.init(f.allocator(), std.testing.io, &core, .{ .lanes = 2 });
    var b: Box = .{};
    defer {
        f.fail_index = std.math.maxInt(usize);
        ends.closeAll(&host);
        host.stop();
    }
    const req: api.Request = .{ .prompt = &.{ 1, 2, 3 }, .max_tokens = 1 };
    for (1..4) |id| try host.engine().submit(@intCast(id), &req, .{ .ctx = &b, .event = Box.event });
    // Retain pointers solely for cleaning up the old remove-before-append regression after its assertion fails.
    const jobs = try a.dupe(*api.LaneHost.Job, host.queued.items);
    defer a.free(jobs);
    defer if (b.done.load(.acquire) != 3) {
        for (jobs, 1..) |job, id| {
            var owned = false;
            for (host.queued.items) |q| owned = owned or q == job;
            if (!owned and b.seen[id] == 0) host.finish(job, .cancelled, "red cleanup");
        }
    };
    if (!close) for (1..4) |id| host.engine().cancel(@intCast(id));
    f.fail_index = f.alloc_index + fail_offset;
    if (close) ends.closeAll(&host) else ends.takeCancels(&host);
    try std.testing.expectEqual(@as(usize, 3), b.cancelled);
    for (b.seen[1..4]) |n| try std.testing.expectEqual(@as(usize, 1), n);
    try std.testing.expectEqual(@as(usize, 0), host.queued.items.len);
    try std.testing.expectEqual(@as(usize, 0), host.admitted.items.len);
}
test "Ownership rollback: cancellation survives ID copy allocation failure" {
    try cancellation(0, false);
}
test "Ownership rollback: cancellation survives dropped list growth failure" {
    try cancellation(1, false);
}
test "Ownership rollback: close cancels queued jobs without any allocation" {
    try cancellation(0, true);
}

fn seventeenFinals(fail: bool) !void {
    var cfg = try lanes.Config.init(a, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true, .replay_runs = true, .max_streams = 17 }, 8, 7);
    defer cfg.deinit(a);
    var fake: lanes.fake.Fake = .{ .gpa = a };
    defer fake.deinit();
    Hooks.reset();
    if (fail) Hooks.fail_call = 2;
    var be = fake.pieceBackend();
    var vt = be.vtable.*;
    vt.finals = Hooks.finals;
    vt.finish = Hooks.finish;
    vt.release = Hooks.release;
    be.vtable = &vt;
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, be, clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(a, std.testing.io, &core, .{ .lanes = 17 });
    var p: Plan = .{};
    host.rounds = p.rounds();
    var b: Box = .{};
    const req: api.Request = .{ .prompt = &.{ 1, 2, 3 }, .max_tokens = 1 };
    for (1..18) |id| try host.engine().submit(@intCast(id), &req, .{ .ctx = &b, .event = Box.event });
    try host.start();
    {
        errdefer host.stop();
        try b.wait(17);
    }
    host.stop();
    try std.testing.expectEqual(@as(usize, 2), Hooks.calls);
    try std.testing.expectEqual(@as(usize, 17), p.left_n);
    for (b.seen[1..18]) |n| try std.testing.expectEqual(@as(usize, 1), n);
    try std.testing.expectEqual(@as(usize, if (fail) 17 else 0), b.failed);
    try std.testing.expectEqual(@as(usize, if (fail) 0 else 17), Hooks.finished);
    try std.testing.expectEqual(@as(usize, 0), fake.lanes.count());
}
test "Ownership rollback: seventeen finals are all prepared before finish" {
    try seventeenFinals(false);
}
test "Ownership rollback: second finals batch error ends every admitted owner" {
    try seventeenFinals(true);
}
