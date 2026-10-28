//! The work group's accounting step by step, on a hand-moved clock: what each call, report and wait adds.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Job = LaneHost.Job;
const load = @import("lane_host_load.zig");

/// An awake clock the test moves by hand.
const Hand = struct {
    var ns: std.atomic.Value(i64) = .init(1);
    var table: std.Io.VTable = undefined;
    fn now(ctx: ?*anyopaque, kind: std.Io.Clock) std.Io.Timestamp {
        if (kind == .awake) return .{ .nanoseconds = ns.load(.monotonic) };
        return std.testing.io.vtable.now(ctx, kind);
    }
    fn io() std.Io {
        table = std.testing.io.vtable.*;
        table.now = now;
        return .{ .userdata = std.testing.io.userdata, .vtable = &table };
    }
    fn wait(d: i64) void {
        _ = ns.fetchAdd(d, .monotonic);
    }
};

/// A host that never starts: the test plays its engine thread.
const Host = struct {
    cfg: lanes.Config,
    target: lanes.fake.Fake,
    clock: lanes.fake.FixedClock = .{},
    core: lanes.Engine = undefined,
    host: LaneHost = undefined,

    fn init(x: *Host) !void {
        const gpa = std.testing.allocator;
        x.cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
        x.target = .{ .gpa = gpa };
        x.clock = .{};
        x.core = lanes.Engine.init(gpa, &x.cfg, x.target.backend(), x.clock.clock()); // reports fill progress
        x.host = LaneHost.init(gpa, Hand.io(), &x.core, .{ .lanes = 2 });
    }
    fn deinit(x: *Host) void {
        x.host.stop();
        x.core.deinit();
        x.target.deinit();
        x.cfg.deinit(std.testing.allocator);
    }
    fn job(x: *Host, r: *const api.Request, hint: u32) Job {
        var j: Job = .{ .host = &x.host, .id = 1, .request = r, .sink = undefined };
        j.stream.fill_at = .init(hint);
        j.stream.cached = 0;
        j.stream.reuse_failed = false;
        load.began(&x.host, &j);
        return j;
    }
    fn work(x: *Host) [2]u64 {
        return .{ x.host.load.prefill_compute, x.host.load.busy_ns };
    }
    fn report(x: *Host, j: *Job) void {
        x.host.lock();
        load.credit(&x.host, j);
        x.host.unlock();
    }
};

const prompt: [1000]u32 = @splat(5);
const request: api.Request = .{ .prompt = &prompt, .max_tokens = 4 };

test "waiting between a pass's calls is not busy time; each chunk's work moves with its own time" {
    var x: Host = undefined;
    try x.init();
    defer x.deinit();
    var j = x.job(&request, 0);
    load.calling(&x.host, &j);
    Hand.wait(800);
    j.stream.fill_at.store(800, .monotonic);
    load.progress(&x.host, &j);
    try std.testing.expectEqual([2]u64{ 800, 800 }, x.work());
    Hand.wait(500); // another stream's work: not this pass's
    load.calling(&x.host, &j);
    Hand.wait(200);
    j.stream.fill_at.store(1000, .monotonic);
    x.report(&j);
    try std.testing.expectEqual([2]u64{ 1000, 1000 }, x.work());
}

test "a start in doubt counts nothing until the report, which counts the whole prompt with the whole time" {
    var x: Host = undefined;
    try x.init();
    defer x.deinit();
    var j = x.job(&request, 400); // a reuse hint the backend rejected without saying so
    load.calling(&x.host, &j);
    Hand.wait(800);
    j.stream.fill_at.store(800, .monotonic);
    load.progress(&x.host, &j);
    try std.testing.expectEqual([2]u64{ 0, 0 }, x.work());
    load.calling(&x.host, &j);
    Hand.wait(200);
    x.report(&j);
    try std.testing.expectEqual([2]u64{ 1000, 1000 }, x.work());
}

test "a confirmed restore counts from its end; a rejected one counts from zero" {
    var x: Host = undefined;
    try x.init();
    defer x.deinit();
    var kept = x.job(&request, 400);
    kept.stream.cached = 400;
    load.calling(&x.host, &kept);
    Hand.wait(400);
    kept.stream.fill_at.store(800, .monotonic);
    load.progress(&x.host, &kept);
    try std.testing.expectEqual([2]u64{ 400, 400 }, x.work());
    x.report(&kept); // 200 more computed after 0 ns: tokens move, and only the time since the last take
    try std.testing.expectEqual(@as(u64, 600), x.host.load.prefill_compute);
    try std.testing.expectEqual(@as(u64, 400), x.host.load.prefill_cache);
    var rejected = x.job(&request, 400);
    rejected.stream.reuse_failed = true;
    rejected.stream.fill_at.store(0, .monotonic);
    load.calling(&x.host, &rejected);
    Hand.wait(256);
    rejected.stream.fill_at.store(256, .monotonic);
    load.progress(&x.host, &rejected);
    try std.testing.expectEqual([2]u64{ 856, 656 }, x.work());
}

test "an error return counts nothing; an early report inside a call takes the time up to it" {
    var x: Host = undefined;
    try x.init();
    defer x.deinit();
    var failed = x.job(&request, 0);
    load.calling(&x.host, &failed);
    Hand.wait(300);
    failed.stream.fill_at.store(300, .monotonic); // stored, then the call returned an error
    x.host.lock();
    load.ended(&failed);
    x.host.unlock();
    try std.testing.expectEqual([2]u64{ 0, 0 }, x.work());
    var early = x.job(&request, 0);
    load.calling(&x.host, &early);
    Hand.wait(300);
    x.report(&early); // the first token inside the call
    Hand.wait(100);
    load.progress(&x.host, &early); // after the call returns: nothing more
    try std.testing.expectEqual([2]u64{ 1000, 300 }, x.work());
}

test "a lone driver's unfinished prompt keeps its call's time across other work until the report" {
    var x: Host = undefined;
    try x.init();
    defer x.deinit();
    var j = x.job(&request, 0);
    load.calling(&x.host, &j);
    Hand.wait(800);
    load.returned(&x.host, &j); // handed to the fill queue
    Hand.wait(500); // another job runs
    load.calling(&x.host, &j);
    Hand.wait(200);
    x.report(&j);
    try std.testing.expectEqual([2]u64{ 1000, 1000 }, x.work());
}

test "a batch counts every completed prompt, a later first-token failure included, and its call once" {
    var x: Host = undefined;
    try x.init();
    defer x.deinit();
    var a = x.job(&request, 0);
    var b = x.job(&request, 0);
    var c = x.job(&request, 0);
    const t0 = load.clock(&x.host);
    Hand.wait(900);
    const jobs = [_]*Job{ &a, &b, &c };
    load.creditBatch(&x.host, &jobs, &.{ true, true, false }, t0); // b's first token failed after its prompt ran
    try std.testing.expectEqual([2]u64{ 2000, 900 }, x.work());
    x.report(&a); // its own report finds it counted
    try std.testing.expectEqual([2]u64{ 2000, 900 }, x.work());
}

test "an unconfirmed reuse hint leaves the whole prompt to fill; a confirmed restore leaves the rest" {
    var x: Host = undefined;
    try x.init();
    defer x.deinit();
    var ledger: api.load_ledger.Ledger = .init(std.testing.io);
    defer ledger.detach(false);
    var j: Job = .{ .host = &x.host, .id = 1, .request = &request, .sink = undefined };
    j.stream.fill_at = .init(900); // the backend may yet refuse the 900-token restore
    j.stream.cached = 0;
    j.stream.reuse_failed = false;
    try x.host.admitted.append(std.testing.allocator, &j);
    defer {
        x.host.lock();
        _ = x.host.admitted.pop();
        x.host.unlock();
    }
    ledger.attach(x.host.engine());
    try std.testing.expectEqual(@as(?u64, 0), ledger.read().requests.?.filling_left);
    load.began(&x.host, &j); // a blocking prefill unlocks no more: began's own publish must show the fill
    try std.testing.expectEqual(@as(?u64, 1000), ledger.read().requests.?.filling_left); // an upper bound, not 100
    j.stream.cached = 900; // the first chunk returned: the restore held
    load.calling(&x.host, &j);
    load.progress(&x.host, &j);
    x.host.lock();
    x.host.unlock();
    try std.testing.expectEqual(@as(?u64, 100), ledger.read().requests.?.filling_left);
}
