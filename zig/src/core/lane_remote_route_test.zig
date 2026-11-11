//! The remote gate and decider: a prompt goes to the pair only when it pays, never behind a pull costing more.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const remote = @import("lane_remote.zig");
const route = @import("lane_remote_route.zig");
const Route = remote.Route;
const Verdict = remote.Verdict;
const t = std.testing;

const s: u64 = std.time.ns_per_s;

/// A prompt of `len` tokens, none cached, with `decoding` other streams, weighed with all but its last token pullable.
fn weigh(r: *Route, now: u64, len: u64, decoding: u32) Verdict {
    return r.weigh(now, .{ .len = len, .cached = 0, .decoding = decoding }, len - 1);
}

test "route: a prompt that finds the pair busy goes elsewhere unless waiting for it is faster, and a wait is bounded" {
    var r: Route = .{};
    const e = r.estimate(8191, 8192); // 9.2 s pulled, 8.6 s prefilled here
    try t.expectEqual(.entered, r.enter(0, 0, e));
    try t.expectEqual(Verdict.busy, weigh(&r, 1 * s, 8192, 0));
    try t.expectEqual(.local, r.enter(1 * s, 1 * s, e)); // 8.2 s of the pull left, then 9.2 s of its own: the Mac is faster
    const long = r.estimate(39999, 40000); // 41 s pulled, 42.1 s here
    try t.expectEqual(Verdict.busy, weigh(&r, 8 * s, 40000, 0)); // 1.2 s of the pull left and 41 s of its own: 42.2 s
    try t.expectEqual(Verdict.remote, weigh(&r, 9 * s, 40000, 0)); // 0.2 s left: 41.2 s beats 42.1 s
    try t.expectEqual(.wait, r.enter(9 * s, 9 * s, long));
    try t.expectEqual(.wait, r.enter(10 * s, 9 * s, long)); // the pull overran its estimate: 1 s waited and 41 s to go still fit
    try t.expectEqual(.local, r.enter(11 * s, 9 * s, long)); // 2 s and 41 s do not: the wait ends
    r.leave();
    try t.expectEqual(Verdict.remote, weigh(&r, 12 * s, 40000, 0));
    try t.expectEqual(.entered, r.enter(12 * s, 9 * s, long));
}

test "route: always queues every prompt behind the pull under way, as one pull at a time did" {
    var r: Route = .{ .mode = .always };
    const e = r.estimate(8191, 8192);
    try t.expectEqual(.entered, r.enter(0, 0, e));
    try t.expectEqual(Verdict.remote, weigh(&r, 1 * s, 8192, 0));
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
        _ = h.route.admit(t.io, h.route.estimate(8191, 8192), &h.closing) orelse return;
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
    const got = r.admit(t.io, r.estimate(8191, 8192), &closing);
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

test "weigh: always sends every prompt, and auto only one the pair is predicted to serve no slower, or while a stream decodes" {
    var always: Route = .{ .mode = .always };
    try t.expectEqual(Verdict.remote, weigh(&always, 0, 8192, 0));
    var r: Route = .{};
    try t.expectEqual(Verdict.slower, weigh(&r, 0, 8192, 0)); // 9.2 s pulled against 8.6 s here
    try t.expectEqual(Verdict.remote, weigh(&r, 0, 8192, 1)); // a prompt pass here would stall the stream
    try t.expectEqual(Verdict.remote, weigh(&r, 0, 40000, 0)); // 41 s against 42 s
    try t.expectEqual(Verdict.slower, r.weigh(0, .{ .len = 8192, .cached = 4000, .decoding = 0 }, 8191)); // 4.2 s here
}

test "weigh: a busy pair is not waited for beyond what prefilling here costs, even with a stream decoding" {
    var r: Route = .{};
    try t.expectEqual(.entered, r.enter(0, 0, r.estimate(8191, 8192))); // under way, 9.2 s to go
    try t.expectEqual(Verdict.busy, weigh(&r, 1 * s, 8192, 1));
    try t.expectEqual(Verdict.busy, weigh(&r, 1 * s, 60000, 0)); // 8.2 s of the pull and 61 s of its own, against 63 s
    try t.expectEqual(Verdict.remote, weigh(&r, 9 * s, 60000, 0)); // 0.2 s of the pull left: worth waiting for
    r.leave();
    try t.expectEqual(Verdict.slower, weigh(&r, 10 * s, 8192, 0));
}

test "observe: measured rates replace the seeds a third to three times at a time, and thin or empty passes say nothing" {
    var r: Route = .{};
    try t.expectEqual(Verdict.slower, weigh(&r, 0, 8192, 0));
    r.observe(.{ .pass = .remote, .rows = 8191, .ns = 4100 * std.time.ns_per_ms }); // 2000 tokens a second end to end
    r.observe(.{ .pass = .tail, .rows = 1, .ns = 500 * std.time.ns_per_ms });
    try t.expectEqual(Verdict.remote, weigh(&r, 0, 8192, 0)); // 4.1 s and 0.5 s against 8.6 s
    r.observe(.{ .pass = .local, .rows = 8192, .ns = 4000 * std.time.ns_per_ms }); // this Mac does 2048 a second
    try t.expectEqual(Verdict.slower, weigh(&r, 0, 8192, 0)); // 4.6 s pulled against 4.0 s here
    const before = r.local_tps.v;
    r.observe(.{ .pass = .local, .rows = 512, .ns = 1 }); // too few rows
    r.observe(.{ .pass = .local, .rows = 8192, .ns = 0 }); // no time
    try t.expectEqual(before, r.local_tps.v);
    r.observe(.{ .pass = .local, .rows = 8192, .ns = 1 }); // absurdly fast: three times the standing rate, blended
    try t.expectApproxEqAbs(0.7 * before + 0.3 * 3 * before, r.local_tps.v, 1e-6);
}

test "knobs: TF_PREFILL_ROUTE, TF_PREFILL_LOCAL_TPS and TF_PREFILL_REMOTE_TPS default to auto and the notes' rates, and refuse bad text" {
    const d = try Route.fromText(null, null, null);
    try t.expectEqual(route.Mode.auto, d.mode);
    try t.expectEqual(@as(f64, 950), d.local_tps.v);
    try t.expectEqual(@as(f64, 1000), d.remote_tps.v);
    const set = try Route.fromText("always", "2000", " 3000.5");
    try t.expectEqual(route.Mode.always, set.mode);
    try t.expectEqual(@as(f64, 2000), set.local_tps.v);
    try t.expectEqual(@as(f64, 3000.5), set.remote_tps.v);
    try t.expectEqual(route.Mode.auto, (try Route.fromText("", "", "")).mode); // blank is unset
    try t.expectError(error.InvalidEnvironment, Route.fromText("sometimes", null, null));
    try t.expectError(error.InvalidEnvironment, Route.fromText(null, "0", null));
    try t.expectError(error.InvalidEnvironment, Route.fromText(null, null, "fast"));
}

/// What a recording source saw, and the verdict it answers.
const Rec = struct {
    verdict: Verdict,
    looks: u32 = 0,
    decoding: u32 = 0,
    running: std.atomic.Value(u32) = .init(0),

    fn wants(ptr: *anyopaque, ask: remote.Ask) Verdict {
        const x: *Rec = @ptrCast(@alignCast(ptr));
        x.looks += 1;
        x.decoding = ask.decoding;
        return x.verdict;
    }

    fn fill(_: *anyopaque, _: []const u32, _: u64) anyerror!?remote.Filled {
        return error.NotUsed;
    }

    fn drop(_: *anyopaque, _: pc.Saved) void {}
};

test "queue: a transient verdict is weighed again and said once, a standing one is weighed once, and decoding counts the others" {
    var cfg = try lanes.Config.init(t.allocator, .{}, 1, 0);
    defer cfg.deinit(t.allocator);
    var fake: lanes.fake.Fake = .{ .gpa = t.allocator };
    defer fake.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(t.allocator, &cfg, fake.backend(), clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(t.allocator, t.io, &core, .{ .lanes = 4 });
    defer host.queued.deinit(t.allocator);
    defer host.admitted.deinit(t.allocator);
    defer host.filling.deinit(t.allocator);
    var rec: Rec = .{ .verdict = .busy };
    host.remote = .{ .ptr = &rec, .wants = Rec.wants, .fill = Rec.fill, .drop = Rec.drop, .running = &rec.running };
    const tokens: [8]u32 = @splat(3);
    const request: api.Request = .{ .prompt = &tokens, .max_tokens = 1 };
    var queued: api.LaneHost.Job = .{ .host = &host, .id = 1, .request = &request, .sink = .{ .ctx = &host, .event = noop } };
    var others: [3]api.LaneHost.Job = undefined; // only their count is read: two decoding, one still filling
    try host.queued.append(t.allocator, &queued);
    for (&others) |*o| try host.admitted.append(t.allocator, o);
    try host.filling.append(t.allocator, &others[2]);
    remote.queue(&host);
    remote.queue(&host);
    try t.expectEqual(@as(u32, 2), rec.looks); // busy: looked at again
    try t.expectEqual(@as(u32, 2), rec.decoding);
    try t.expect(!queued.remote_looked and queued.ticket == null);
    try t.expectEqual(Verdict.busy, queued.remote_said);
    rec.verdict = .small;
    remote.queue(&host);
    remote.queue(&host);
    try t.expectEqual(@as(u32, 3), rec.looks); // too small: once, for good
    try t.expect(queued.remote_looked and queued.ticket == null);
}

fn noop(_: *anyopaque, _: api.Id, _: *const api.Event) void {}

/// The pair as a source: `wants` and `fill` over the gate a family's pull keeps, `delay_ms` per pull.
const Pair = struct {
    route: Route,
    at: u32 = 4,
    delay_ms: i64 = 600,
    fills: std.atomic.Value(u32) = .init(0),
    drops: std.atomic.Value(u32) = .init(0),
    running: std.atomic.Value(u32) = .init(0),
    closing: std.atomic.Value(bool) = .init(false),
    seen: [4]route.Measured = undefined,
    seen_n: std.atomic.Value(u32) = .init(0),

    fn wants(ptr: *anyopaque, ask: remote.Ask) Verdict {
        const p: *Pair = @ptrCast(@alignCast(ptr));
        if (p.at <= ask.cached) return .small;
        return p.route.weigh(Route.nowNs(t.io), ask, p.at);
    }

    fn fill(ptr: *anyopaque, prompt: []const u32, cached: u64) anyerror!?remote.Filled {
        const p: *Pair = @ptrCast(@alignCast(ptr));
        _ = p.route.admit(t.io, p.route.estimate(p.at, prompt.len - cached), &p.closing) orelse return null;
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

    fn measured(ptr: *anyopaque, m: route.Measured) void {
        const p: *Pair = @ptrCast(@alignCast(ptr));
        const n = p.seen_n.fetchAdd(1, .acq_rel);
        if (n < p.seen.len) p.seen[n] = m;
    }

    fn source(p: *Pair) remote.Source {
        return .{ .ptr = p, .wants = wants, .fill = fill, .drop = drop, .running = &p.running, .measured = measured };
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
    // 6 token prompts, a pull of 4: at 5 tokens a second an idle pair is faster, a busy one not worth waiting for
    pair.route.local_tps = .{ .v = 5 };
    host.remote = pair.source();
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

/// One request through a host whose pair pulls `at` tokens of a prompt of `len`; the passes the host reported.
fn learned(len: u32, at: u32) !struct { n: u32, first: route.Measured } {
    var cfg = try lanes.Config.init(t.allocator, .{}, 1, 0);
    defer cfg.deinit(t.allocator);
    var fake: lanes.fake.Fake = .{ .gpa = t.allocator };
    defer fake.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(t.allocator, &cfg, fake.backend(), clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(t.allocator, t.io, &core, .{ .lanes = 1 });
    var pair: Pair = .{ .route = .{ .mode = .always }, .at = at, .delay_ms = 1 };
    host.remote = pair.source();
    try host.start();
    defer host.stop();
    const prompt = try t.allocator.alloc(u32, len);
    defer t.allocator.free(prompt);
    for (prompt, 0..) |*x, i| x.* = @intCast(i % 50 + 1);
    var a: Timed = .{ .began = std.Io.Clock.awake.now(t.io).toNanoseconds() };
    try host.engine().submit(1, &.{ .prompt = prompt, .max_tokens = 2 }, .{ .ctx = &a, .event = Timed.event });
    try wait(&a.done);
    while (pair.running.load(.acquire) > 0) try std.Io.sleep(t.io, .fromMilliseconds(1), .awake);
    return .{ .n = pair.seen_n.load(.acquire), .first = pair.seen[0] };
}

test "measured: the host reports its own prompt pass, and the last-token pass after a pull, to the source" {
    const own = try learned(1100, 0);
    try t.expectEqual(@as(u32, 1), own.n);
    try t.expectEqual(route.Pass.local, own.first.pass);
    try t.expectEqual(@as(u64, 1100), own.first.rows);
    const pulled = try learned(1100, 1000);
    try t.expectEqual(@as(u32, 1), pulled.n);
    try t.expectEqual(route.Pass.tail, pulled.first.pass);
    try t.expectEqual(@as(u64, 100), pulled.first.rows);
}
