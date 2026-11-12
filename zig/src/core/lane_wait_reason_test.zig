//! Wait reasons in the load ledger: stored on the admission path, read by the publish, one for each queued request.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const admit = @import("lane_host_admit.zig");
const ends = @import("lane_host_end.zig");
const Ledger = api.load_ledger.Ledger;
const Requests = api.load_ledger.Requests;
const Wait = api.Rounds.Wait;
const Request = api.Request;

const Rig = struct {
    cfg: lanes.Config,
    target: lanes.fake.Fake,
    clock: lanes.fake.FixedClock = .{},
    core: lanes.Engine = undefined,
    host: LaneHost = undefined,

    fn init(r: *Rig, info: api.Info, pieces: bool) !void {
        const gpa = std.testing.allocator;
        r.cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
        r.target = .{ .gpa = gpa };
        r.clock = .{};
        r.core = lanes.Engine.init(gpa, &r.cfg, if (pieces) r.target.pieceBackend() else r.target.backend(), r.clock.clock());
        r.host = LaneHost.init(gpa, std.testing.io, &r.core, info);
    }

    fn deinit(r: *Rig) void {
        r.host.stop();
        r.core.deinit();
        r.target.deinit();
        r.cfg.deinit(std.testing.allocator);
    }

    fn submit(r: *Rig, id: api.Id, request: *const Request) !void {
        try r.host.engine().submit(id, request, .{ .ctx = r, .event = ignore });
    }

    fn ignore(_: *anyopaque, _: api.Id, _: *const api.Event) void {}
};

/// A DeepSeek-V4.1-like planner: a long prompt (first token 1) is passed over, any other waits for `second`.
const Plan = struct {
    second: Wait = .kv_pool,
    admits: u32 = 0,

    fn self(ctx: *anyopaque) *Plan {
        return @ptrCast(@alignCast(ctx));
    }
    fn none(_: *anyopaque) void {}
    fn admitFn(ctx: *anyopaque, ask: api.Rounds.Ask, _: *[]const u8) api.Rounds.Verdict {
        const p = self(ctx);
        p.admits += 1;
        return switch (ask.prompt[0]) {
            1 => .{ .skip = .prefill_concurrent },
            else => .{ .wait = p.second },
        };
    }
    fn admitted(_: *anyopaque) anyerror!void {}
    fn key(_: *anyopaque, _: usize) void {}
    fn began(_: *anyopaque, _: usize, _: bool) void {}
    fn plan(_: *anyopaque, pieces: *std.ArrayList(api.Rounds.Piece), finals: *std.ArrayList(usize)) anyerror!void {
        pieces.clearRetainingCapacity();
        finals.clearRetainingCapacity();
    }
    fn after(_: *anyopaque, _: f64, _: f64) void {}
    fn rounds(p: *Plan) api.Rounds {
        return .{ .ctx = p, .vtable = &.{ .begin_admit = none, .admit = admitFn, .admitted = admitted, .arm = key, .began = began, .plan = plan, .left = key, .after = after } };
    }
};

fn at(r: Requests, w: Wait) u32 {
    return r.waiting_by_reason[@backingInt(w)];
}

const long = [_]u32{ 1, 5, 9, 2 };
const second = [_]u32{ 2, 6, 5, 3 };
const third = [_]u32{ 3, 5, 8, 9 };
const fourth = [_]u32{ 4, 7, 1, 8 };

test "wait reason ledger: a planner's kv_pool, memory_floor and prefill_concurrent reach the ledger, not engine_busy" {
    var rig: Rig = undefined;
    try rig.init(.{ .lanes = 2 }, true);
    defer rig.deinit();
    var pl: Plan = .{};
    rig.host.rounds = pl.rounds();
    var ledger: Ledger = .init(std.testing.io);
    ledger.attach(rig.host.engine());
    defer ledger.detach(false);
    const ra: Request = .{ .prompt = &long, .max_tokens = 4 };
    const rb: Request = .{ .prompt = &second, .max_tokens = 4 };
    const rc: Request = .{ .prompt = &third, .max_tokens = 4 };
    try rig.submit(1, &ra);
    try rig.submit(2, &rb);
    try rig.submit(3, &rc);
    errdefer ends.closeAll(&rig.host);
    try std.testing.expectEqual(@as(u32, 3), at(ledger.read().requests.?, .engine_busy)); // not examined yet
    _ = api.Rounds.Test.step(&rig.host);
    var r = ledger.read().requests.?;
    try std.testing.expectEqual(@as(u32, 3), r.waiting);
    try std.testing.expectEqual(@as(u32, 1), at(r, .prefill_concurrent));
    try std.testing.expectEqual(@as(u32, 2), at(r, .kv_pool)); // the wait and the request behind it (FIFO)
    try std.testing.expectEqual(@as(u32, 0), at(r, .engine_busy));
    pl.second = .memory_floor;
    _ = api.Rounds.Test.step(&rig.host);
    r = ledger.read().requests.?;
    try std.testing.expectEqual(@as(u32, 2), at(r, .memory_floor));
    try std.testing.expectEqual(@as(u32, 0), at(r, .memory_pressure)); // the floor is not pressure
    try std.testing.expectEqual(@as(u32, 0), at(r, .kv_pool));
    try std.testing.expectEqual(@as(u32, 1), at(r, .prefill_concurrent));
    try std.testing.expectEqual(@as(u32, 0), at(r, .engine_busy));
    ends.closeAll(&rig.host);
    try std.testing.expectEqual(@as(u64, 0), ledger.read().requests.?.waitingTotal());
}

test "wait reason ledger: a fresh host publishes every reason, each at zero" {
    var rig: Rig = undefined;
    try rig.init(.{ .lanes = 2 }, false);
    defer rig.deinit();
    var ledger: Ledger = .init(std.testing.io);
    ledger.attach(rig.host.engine());
    defer ledger.detach(false);
    const r = ledger.read().requests.?;
    try std.testing.expectEqual(std.enums.values(Wait).len, r.waiting_by_reason.len);
    for (r.waiting_by_reason) |n| try std.testing.expectEqual(@as(u32, 0), n);
    try std.testing.expectEqual(@as(u32, 0), r.waiting);
}

test "wait reason ledger: the reasons sum to waiting through an admission and a cancel while queued" {
    var rig: Rig = undefined;
    try rig.init(.{ .lanes = 4 }, true);
    defer rig.deinit();
    var pl: Plan = .{};
    rig.host.rounds = pl.rounds();
    var ledger: Ledger = .init(std.testing.io);
    ledger.attach(rig.host.engine());
    defer ledger.detach(false);
    const ra: Request = .{ .prompt = &fourth, .max_tokens = 4 };
    const rb: Request = .{ .prompt = &long, .max_tokens = 4 };
    const rc: Request = .{ .prompt = &second, .max_tokens = 4 };
    const rd: Request = .{ .prompt = &third, .max_tokens = 4 };
    try rig.submit(1, &ra);
    try rig.submit(2, &rb);
    try rig.submit(3, &rc);
    try rig.submit(4, &rd);
    errdefer ends.closeAll(&rig.host);
    // the head leaves the queue for a lane: its reason goes with it
    const taken = rig.host.takeJob();
    try std.testing.expect(taken == .job);
    _ = api.Rounds.Test.step(&rig.host);
    var r = ledger.read().requests.?;
    try std.testing.expectEqual(@as(u32, 1), r.running);
    try std.testing.expectEqual(@as(u32, 3), r.waiting);
    try std.testing.expectEqual(@as(u32, 1), at(r, .prefill_concurrent));
    try std.testing.expectEqual(@as(u32, 2), at(r, .kv_pool));
    try std.testing.expectEqual(@as(u64, r.waiting), r.waitingTotal());
    // the request passed over is cancelled while queued: its reason leaves with it
    rig.host.engine().cancel(2);
    ends.takeCancels(&rig.host);
    r = ledger.read().requests.?;
    try std.testing.expectEqual(@as(u32, 2), r.waiting);
    try std.testing.expectEqual(@as(u32, 0), at(r, .prefill_concurrent));
    try std.testing.expectEqual(@as(u32, 2), at(r, .kv_pool));
    try std.testing.expectEqual(@as(u64, r.waiting), r.waitingTotal());
    ends.closeAll(&rig.host);
    r = ledger.read().requests.?;
    try std.testing.expectEqual(@as(u32, 0), r.running);
    try std.testing.expectEqual(@as(u64, 0), r.waitingTotal());
}

/// A family pool that counts how often the host asks it.
const Counted = struct {
    fn fits(ctx: *anyopaque, _: u32, _: u32) bool {
        const calls: *u32 = @ptrCast(@alignCast(ctx));
        calls.* += 1;
        return true;
    }
};

test "wait reason ledger: publishing the ledger reads stored reasons and never asks the family's pool" {
    var calls: u32 = 0;
    var rig: Rig = undefined;
    try rig.init(.{ .lanes = 1, .admit = .{ .ctx = &calls, .fits = Counted.fits } }, false);
    defer rig.deinit();
    var ledger: Ledger = .init(std.testing.io);
    ledger.attach(rig.host.engine());
    defer ledger.detach(false);
    const ra: Request = .{ .prompt = &second, .max_tokens = 4 };
    const rb: Request = .{ .prompt = &third, .max_tokens = 4 };
    const rc: Request = .{ .prompt = &fourth, .max_tokens = 4 };
    try rig.submit(1, &ra);
    try rig.submit(2, &rb);
    try rig.submit(3, &rc);
    errdefer ends.closeAll(&rig.host);
    for (0..8) |_| { // each unlock publishes: the requests group is built again every time
        rig.host.lock();
        rig.host.unlock();
    }
    try std.testing.expectEqual(@as(u32, 3), at(ledger.read().requests.?, .engine_busy));
    // an admission attempt decides the reason; the publishes after it only read it
    rig.host.lock();
    rig.host.stale = true;
    try std.testing.expect(admit.pick(&rig.host) == null);
    rig.host.stale = false;
    rig.host.unlock();
    for (0..8) |_| {
        rig.host.lock();
        rig.host.unlock();
    }
    const r = ledger.read().requests.?;
    try std.testing.expectEqual(@as(u32, 3), at(r, .engine_unavailable));
    try std.testing.expectEqual(@as(u64, 3), r.waitingTotal());
    try std.testing.expectEqual(@as(u32, 0), calls);
    ends.closeAll(&rig.host);
}
