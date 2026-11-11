//! The remote gate: a prompt that finds the pair busy is prefilled here at once, not queued behind the pull under way.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const remote = @import("lane_remote.zig");
const route = @import("lane_remote_route.zig");
const Route = remote.Route;
const t = std.testing;

const s: u64 = std.time.ns_per_s;

test "route: a prompt that finds the pair busy goes elsewhere unless waiting for it is faster, and a wait is bounded" {
    var r: Route = .{};
    const e = Route.estimate(8191, 8192); // 9.2 s pulled, 8.6 s prefilled here
    try t.expectEqual(.entered, r.enter(0, 0, e));
    try t.expect(r.busy(1 * s, e));
    try t.expectEqual(.local, r.enter(1 * s, 1 * s, e)); // 8.2 s of the pull left, then 9.2 s of its own: the Mac is faster
    const long = Route.estimate(8191, 40000); // 42 s here
    try t.expect(!r.busy(8 * s, long));
    try t.expectEqual(.wait, r.enter(8 * s, 8 * s, long)); // 1.2 s of the pull left and 9.2 s of its own beat 42 s
    try t.expectEqual(.wait, r.enter(40 * s, 8 * s, long)); // the pull overran; 32 s waited and 9.2 s to go still fit
    try t.expectEqual(.local, r.enter(41 * s, 8 * s, long)); // 33 s and 9.2 s do not: the wait ends
    r.leave();
    try t.expect(!r.busy(41 * s, e));
    try t.expectEqual(.entered, r.enter(41 * s, 8 * s, long));
}

test "route: always queues every prompt behind the pull under way, as one pull at a time did" {
    var r: Route = .{ .mode = .always };
    const e = Route.estimate(8191, 8192);
    try t.expectEqual(.entered, r.enter(0, 0, e));
    try t.expect(!r.busy(1 * s, e));
    try t.expectEqual(.wait, r.enter(1 * s, 1 * s, e));
    try t.expectEqual(.wait, r.enter(1000 * s, 1 * s, e));
    r.leave();
    try t.expectEqual(.entered, r.enter(1001 * s, 1 * s, e));
}

/// A pull that holds the pair for `hold_ms`.
const Holder = struct {
    route: *Route,
    hold_ms: i64,
    entered: std.atomic.Value(bool) = .init(false),
    closing: std.atomic.Value(bool) = .init(false),

    fn run(h: *Holder) void {
        _ = h.route.admit(t.io, Route.estimate(8191, 8192), &h.closing) orelse return;
        h.entered.store(true, .release);
        std.Io.sleep(t.io, .fromMilliseconds(h.hold_ms), .awake) catch {};
        h.route.leave();
    }
};

/// How long a second pull took to get the pair (null: it was sent elsewhere) while another held it for 300 ms.
fn second(mode: route.Mode) !struct { waited_ms: i64, entered: bool } {
    var r: Route = .{ .mode = mode };
    var h: Holder = .{ .route = &r, .hold_ms = 300 };
    const thread = try std.Thread.spawn(.{}, Holder.run, .{&h});
    defer thread.join();
    while (!h.entered.load(.acquire)) try std.Io.sleep(t.io, .fromMilliseconds(1), .awake);
    var closing: std.atomic.Value(bool) = .init(false);
    const began = Route.nowNs(t.io);
    const got = r.admit(t.io, Route.estimate(8191, 8192), &closing);
    const took: i64 = @intCast((Route.nowNs(t.io) - began) / std.time.ns_per_ms);
    if (got != null) r.leave();
    return .{ .waited_ms = took, .entered = got != null };
}

test "route: the second of two pulls is sent elsewhere at once, and under always waits out the first" {
    const auto = try second(.auto);
    try t.expect(!auto.entered);
    try t.expect(auto.waited_ms < 100);
    const always = try second(.always);
    try t.expect(always.entered);
    try t.expect(always.waited_ms >= 200);
}

/// The pair as a source: `wants` and `fill` over the gate a family's pull keeps, `delay_ms` per pull.
const Pair = struct {
    route: Route,
    at: u32 = 4,
    delay_ms: i64 = 600,
    fills: std.atomic.Value(u32) = .init(0),
    drops: std.atomic.Value(u32) = .init(0),
    running: std.atomic.Value(u32) = .init(0),
    closing: std.atomic.Value(bool) = .init(false),

    fn wants(ptr: *anyopaque, len: u64, cached: u64) bool {
        const p: *Pair = @ptrCast(@alignCast(ptr));
        return p.at > cached and !p.route.busy(Route.nowNs(t.io), Route.estimate(p.at, len - cached));
    }

    fn fill(ptr: *anyopaque, prompt: []const u32, cached: u64) anyerror!?remote.Filled {
        const p: *Pair = @ptrCast(@alignCast(ptr));
        _ = p.route.admit(t.io, Route.estimate(p.at, prompt.len - cached), &p.closing) orelse return null;
        defer p.route.leave();
        _ = p.fills.fetchAdd(1, .acq_rel);
        try std.Io.sleep(t.io, .fromMilliseconds(p.delay_ms), .awake);
        const kept = try t.allocator.create(std.ArrayList(u32));
        kept.* = .empty;
        errdefer t.allocator.destroy(kept);
        try kept.appendSlice(t.allocator, prompt[0..p.at]);
        return .{ .saved = kept, .at = p.at };
    }

    fn drop(ptr: *anyopaque, saved: pc.Saved) void {
        const p: *Pair = @ptrCast(@alignCast(ptr));
        const kept: *std.ArrayList(u32) = @ptrCast(@alignCast(saved));
        kept.deinit(t.allocator);
        t.allocator.destroy(kept);
        _ = p.drops.fetchAdd(1, .acq_rel);
    }
};

/// A request's own clock: when its prompt was in, and whether it ended.
const Timed = struct {
    began: i96 = 0,
    prefilled_ms: std.atomic.Value(i64) = .init(-1),
    done: std.atomic.Value(bool) = .init(false),

    fn event(ctx: *anyopaque, _: api.Id, ev: *const api.Event) void {
        const x: *Timed = @ptrCast(@alignCast(ctx));
        switch (ev.*) {
            .prefilled => x.prefilled_ms.store(@intCast(@divTrunc(std.Io.Clock.awake.now(t.io).toNanoseconds() - x.began, std.time.ns_per_ms)), .release),
            .finished => x.done.store(true, .release),
            else => {},
        }
    }
};

fn wait(flag: *const std.atomic.Value(bool)) !void {
    const until = std.Io.Clock.awake.now(t.io).toNanoseconds() + 10 * std.time.ns_per_s;
    while (!flag.load(.acquire)) {
        if (std.Io.Clock.awake.now(t.io).toNanoseconds() > until) return error.RemoteFixtureTimeout;
        try std.Io.sleep(t.io, .fromMilliseconds(1), .awake);
    }
}

/// Two prompts, the second sent 50 ms into the first one's pull: its `prefilled` time (ms) and the pulls made.
fn twoPrompts(mode: route.Mode) !struct { prefilled_ms: i64, fills: u32 } {
    var cfg = try lanes.Config.init(t.allocator, .{}, 1, 0);
    defer cfg.deinit(t.allocator);
    var fake: lanes.fake.Fake = .{ .gpa = t.allocator };
    defer fake.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(t.allocator, &cfg, fake.backend(), clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(t.allocator, t.io, &core, .{ .lanes = 2 });
    var pair: Pair = .{ .route = .{ .mode = mode } };
    host.remote = .{ .ptr = &pair, .wants = Pair.wants, .fill = Pair.fill, .drop = Pair.drop, .running = &pair.running };
    try host.start();
    defer host.stop();
    const one = [_]u32{ 11, 12, 13, 14, 15, 16 };
    const two = [_]u32{ 21, 22, 23, 24, 25, 26 };
    var a: Timed = .{ .began = std.Io.Clock.awake.now(t.io).toNanoseconds() };
    try host.engine().submit(1, &.{ .prompt = &one, .max_tokens = 2 }, .{ .ctx = &a, .event = Timed.event });
    while (pair.fills.load(.acquire) == 0) try std.Io.sleep(t.io, .fromMilliseconds(1), .awake);
    try std.Io.sleep(t.io, .fromMilliseconds(50), .awake);
    var b: Timed = .{ .began = std.Io.Clock.awake.now(t.io).toNanoseconds() };
    try host.engine().submit(2, &.{ .prompt = &two, .max_tokens = 2 }, .{ .ctx = &b, .event = Timed.event });
    try wait(&a.done);
    try wait(&b.done);
    while (pair.running.load(.acquire) > 0) try std.Io.sleep(t.io, .fromMilliseconds(1), .awake);
    return .{ .prefilled_ms = b.prefilled_ms.load(.acquire), .fills = pair.fills.load(.acquire) };
}

test "route: on the lane host a second prompt is prefilled here while the pair holds the first, and under always waits for it" {
    const auto = try twoPrompts(.auto);
    try t.expect(auto.prefilled_ms >= 0 and auto.prefilled_ms < 300); // the pull takes 600 ms; the Mac was not idle for it
    try t.expectEqual(@as(u32, 1), auto.fills);
    const always = try twoPrompts(.always);
    try t.expect(always.prefilled_ms >= 500); // behind the first pull, then its own
    try t.expectEqual(@as(u32, 2), always.fills);
}
