//! Slow readers on the lane host: a stream sits out while its client is behind, others keep their pace, stalls end.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const backlog = @import("lane_host_backlog.zig");
const LaneHost = api.LaneHost;
const Request = api.Request;
const Reason = api.Reason;
const gpa = std.testing.allocator;
const io = std.testing.io;

/// A client: the tokens it was sent, how many it has not read (Request.unread), and the most it ever left unread.
const Reader = struct {
    mutex: std.Io.Mutex = .init,
    tokens: std.ArrayList(u32) = .empty,
    unread: std.atomic.Value(u32) = .init(0),
    most: std.atomic.Value(u32) = .init(0),
    done: ?Reason = null,
    eager: bool = true, // reads each event as it lands (a normal client)

    fn event(ctx: *anyopaque, _: api.Id, e: *const api.Event) void {
        const r: *@This() = @ptrCast(@alignCast(ctx));
        r.mutex.lockUncancelable(io);
        defer r.mutex.unlock(io);
        switch (e.*) {
            .tokens => |t| {
                r.tokens.appendSlice(gpa, t) catch {};
                const now = r.unread.fetchAdd(@intCast(t.len), .acq_rel) + @as(u32, @intCast(t.len));
                _ = r.most.fetchMax(now, .acq_rel);
                if (r.eager) _ = r.unread.fetchSub(@intCast(t.len), .acq_rel);
            },
            .finished => |f| r.done = f.reason,
            else => {},
        }
    }
    fn count(r: *@This()) usize {
        r.mutex.lockUncancelable(io);
        defer r.mutex.unlock(io);
        return r.tokens.items.len;
    }
    fn finished(r: *@This()) ?Reason {
        r.mutex.lockUncancelable(io);
        defer r.mutex.unlock(io);
        return r.done;
    }
    fn wait(r: *@This()) Reason {
        while (true) {
            if (r.finished()) |d| return d;
            std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
        }
    }
    /// A slow client: two tokens every `ms` milliseconds until its reply ends.
    fn trickle(r: *@This(), ms: u32) void {
        while (r.finished() == null) {
            std.Io.sleep(io, .fromMilliseconds(ms), .awake) catch {};
            const n = @min(r.unread.load(.acquire), 2);
            _ = r.unread.fetchSub(n, .acq_rel);
        }
    }
    fn request(r: *@This(), prompt: []const u32, max: u32, drafts: bool) Request {
        return .{ .prompt = prompt, .max_tokens = max, .drafts = drafts, .unread = &r.unread };
    }
};

/// A lone driver that decodes by the fake's rule and hands its stream back whenever the host's yield asks.
const Lone = struct {
    target: *lanes.fake.Fake,
    handed: u32 = 0, // hand-backs in the middle of a reply

    fn run(ctx: *anyopaque, s: *lanes.Stream, hooks: api.LoneHooks) anyerror!bool {
        const l: *Lone = @ptrCast(@alignCast(ctx));
        if (!hooks.resumed) {
            _ = try l.target.backend().opening(gpa, s);
            hooks.committed(hooks.ctx);
            s.pending = s.context.items[s.context.items.len - 1];
            s.cache_len = s.prompt_len; // as a backend's hand-over leaves it: the first token not yet fed
            return true;
        }
        const history = &l.target.lanes.getPtr(s).?.history;
        while (!s.finished) {
            if (hooks.yield(hooks.ctx)) {
                l.handed += 1;
                return true; // the fake's cache holds every token but the pending one, as the lane core takes it
            }
            try history.append(gpa, s.pending.?);
            s.cache_len += 1;
            const t = lanes.fake.next(history.items, null, history.items.len);
            _ = try s.commit(gpa, &.{t}, &.{});
            s.pending = t;
            hooks.committed(hooks.ctx);
        }
        l.target.backend().release(s);
        return false;
    }
};

/// `drafted`: a drafted family's windowed rounds (head drafts, 16-row windows); else one-token steps queued ahead.
const Options = struct { lanes: u32 = 1, limits: backlog.Limits = .{}, ring_len: ?usize = null, lone: bool = false, drafted: bool = false };

const Rig = struct {
    cfg: lanes.Config,
    target: lanes.fake.Fake,
    clock: lanes.fake.FixedClock = .{},
    core: lanes.Engine,
    host: LaneHost,
    lone: Lone = undefined,
    stopped: bool = false,

    fn init(r: *Rig, o: Options) !void {
        var costs: [16]lanes.config.Cost = undefined;
        for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 5.0 + 0.8 * @as(f64, @floatFromInt(w)) };
        r.cfg = if (o.drafted)
            try lanes.Config.init(gpa, .{ .exact_width = 16, .gpu_tokens = true, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 4, .window_costs = &costs, .mtp_step_ms = 0.5, .hidden_rows = true, .batch_rows = 32, .max_streams = 8, .draft_streams = true }, 16, 15)
        else
            try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
        r.target = .{ .gpa = gpa, .ring_len = o.ring_len };
        r.clock = .{};
        r.stopped = false;
        r.core = lanes.Engine.init(gpa, &r.cfg, r.target.backend(), r.clock.clock());
        r.host = LaneHost.init(gpa, io, &r.core, .{ .lanes = o.lanes });
        r.host.unread_limits = o.limits;
        r.lone = .{ .target = &r.target };
        if (o.lone) r.host.lone = .{ .ctx = &r.lone, .run = Lone.run, .resumes = true };
        try r.host.start();
    }
    fn stop(r: *Rig) void {
        if (!r.stopped) r.host.stop();
        r.stopped = true;
    }
    fn deinit(r: *Rig) void {
        r.stop();
        r.core.deinit();
        r.target.deinit();
        r.cfg.deinit(gpa);
    }
    fn pauses(r: *Rig) u64 {
        return r.host.unread_limits.pauses.load(.monotonic);
    }
};

const p1 = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6 };
const p2 = [_]u32{ 2, 7, 1, 8, 2, 8, 1, 8 };

/// One request with an eager reader on its own host: the reference tokens (the caller frees them).
fn reference(prompt: []const u32, max: u32) !std.ArrayList(u32) {
    var rig: Rig = undefined;
    try rig.init(.{});
    defer rig.deinit();
    var r: Reader = .{};
    const req = r.request(prompt, max, true);
    try rig.host.engine().submit(1, &req, .{ .ctx = &r, .event = Reader.event });
    try std.testing.expectEqual(Reason.length, r.wait());
    return r.tokens;
}

test "slow readers: a normal reader never pauses, even with a small limit" {
    var r: Reader = .{};
    defer r.tokens.deinit(gpa);
    var rig: Rig = undefined;
    try rig.init(.{ .limits = .{ .limit = 8 } });
    defer rig.deinit();
    const req = r.request(&p1, 200, true);
    try rig.host.engine().submit(1, &req, .{ .ctx = &r, .event = Reader.event });
    try std.testing.expectEqual(Reason.length, r.wait());
    try std.testing.expectEqual(@as(u64, 0), rig.pauses());
    try std.testing.expectEqual(@as(usize, 200), r.tokens.items.len);
}

test "slow readers: the backlog stays within a round of the limit and the text is a fast reader's, byte for byte" {
    var fast = try reference(&p1, 200);
    defer fast.deinit(gpa);
    const limit = 8;
    for ([_]bool{ false, true }) |drafted| {
        var r: Reader = .{ .eager = false };
        defer r.tokens.deinit(gpa);
        var rig: Rig = undefined;
        try rig.init(.{ .limits = .{ .limit = limit }, .drafted = drafted });
        defer rig.deinit();
        const req = r.request(&p1, 200, true);
        try rig.host.engine().submit(1, &req, .{ .ctx = &r, .event = Reader.event });
        const t = try std.Thread.spawn(.{}, Reader.trickle, .{ &r, 1 });
        const reason = r.wait();
        t.join();
        try std.testing.expectEqual(Reason.length, reason);
        try std.testing.expect(rig.pauses() > 0);
        // the limit, plus at most the one round that crossed it (a window of the config's width) and one queued step
        try std.testing.expect(r.most.load(.acquire) <= limit + rig.cfg.family_width + 1);
        try std.testing.expectEqualSlices(u32, fast.items, r.tokens.items);
    }
}

test "slow readers: a stalled reader holds only its own stream, the other finishes, then the slow one ends unchanged" {
    var solo = try reference(&p2, 120);
    defer solo.deinit(gpa);
    var slow_ref = try reference(&p1, 120);
    defer slow_ref.deinit(gpa);
    var slow: Reader = .{ .eager = false };
    defer slow.tokens.deinit(gpa);
    var quick: Reader = .{};
    defer quick.tokens.deinit(gpa);
    var rig: Rig = undefined;
    try rig.init(.{ .lanes = 2, .limits = .{ .limit = 8 } });
    defer rig.deinit();
    const e = rig.host.engine();
    const a = slow.request(&p1, 120, true);
    const b = quick.request(&p2, 120, true);
    try e.submit(1, &a, .{ .ctx = &slow, .event = Reader.event });
    try e.submit(2, &b, .{ .ctx = &quick, .event = Reader.event });
    // the slow client reads nothing: the quick one still finishes, all its tokens, while the slow one waits
    try std.testing.expectEqual(Reason.length, quick.wait());
    try std.testing.expect(slow.finished() == null);
    try std.testing.expect(slow.count() < 120);
    try std.testing.expectEqualSlices(u32, solo.items, quick.tokens.items);
    // the client catches up: its stream resumes and ends as it would have
    const t = try std.Thread.spawn(.{}, Reader.trickle, .{ &slow, 1 });
    const reason = slow.wait();
    t.join();
    try std.testing.expectEqual(Reason.length, reason);
    try std.testing.expectEqualSlices(u32, slow_ref.items, slow.tokens.items);
}

test "slow readers: a held stream's step queued ahead lands first, so a peer's draws never meet it in the ring" {
    var slow_ref = try reference(&p1, 60);
    defer slow_ref.deinit(gpa);
    var quick_ref = try reference(&p2, 60);
    defer quick_ref.deinit(gpa);
    var slow: Reader = .{ .eager = false };
    defer slow.tokens.deinit(gpa);
    var quick: Reader = .{};
    defer quick.tokens.deinit(gpa);
    var rig: Rig = undefined;
    // GLM's ring of four draws: an unread draw holds its slot, and a draw meeting it fails the round
    try rig.init(.{ .lanes = 2, .limits = .{ .limit = 8 }, .ring_len = 4 });
    defer rig.deinit();
    const e = rig.host.engine();
    // drafts off: every round is a step queued ahead, so the slow stream holds a draw whenever it stops
    const a = slow.request(&p1, 60, false);
    try e.submit(1, &a, .{ .ctx = &slow, .event = Reader.event });
    var waited: u32 = 0;
    while (rig.pauses() == 0) : (waited += 1) {
        if (waited == 5000) return error.NeverHeld;
        std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    }
    const b = quick.request(&p2, 60, false);
    try e.submit(2, &b, .{ .ctx = &quick, .event = Reader.event });
    try std.testing.expectEqual(Reason.length, quick.wait());
    try std.testing.expectEqualSlices(u32, quick_ref.items, quick.tokens.items);
    const t = try std.Thread.spawn(.{}, Reader.trickle, .{ &slow, 1 });
    const reason = slow.wait();
    t.join();
    try std.testing.expectEqual(Reason.length, reason);
    try std.testing.expectEqualSlices(u32, slow_ref.items, slow.tokens.items);
}

test "slow readers: a reader stalled past the timeout is cancelled as a disconnect is, its lane to the next request" {
    var next_ref = try reference(&p2, 40);
    defer next_ref.deinit(gpa);
    var stuck: Reader = .{ .eager = false };
    defer stuck.tokens.deinit(gpa);
    var next: Reader = .{};
    defer next.tokens.deinit(gpa);
    var rig: Rig = undefined;
    try rig.init(.{ .limits = .{ .limit = 8, .stall_ns = 50 * std.time.ns_per_ms } });
    defer rig.deinit();
    const e = rig.host.engine();
    const a = stuck.request(&p1, 10_000, true);
    const b = next.request(&p2, 40, true);
    try e.submit(1, &a, .{ .ctx = &stuck, .event = Reader.event });
    try e.submit(2, &b, .{ .ctx = &next, .event = Reader.event }); // one lane: it waits behind the stuck stream
    try std.testing.expectEqual(Reason.cancelled, stuck.wait());
    try std.testing.expect(stuck.count() < 10_000);
    try std.testing.expectEqual(@as(u64, 1), rig.host.unread_limits.stalls.load(.monotonic));
    try std.testing.expectEqual(Reason.length, next.wait());
    try std.testing.expectEqualSlices(u32, next_ref.items, next.tokens.items);
    rig.stop();
    // the held stream's 50 ms passed in 2 ms waits, not in empty rounds spun while the queued request waited
    try std.testing.expect(rig.core.steps < 400);
}

test "lone driver: a reader behind hands its stream back, the host holds it, and its reply is a fast reader's" {
    var fast = try reference(&p1, 200);
    defer fast.deinit(gpa);
    const limit = 8;
    var r: Reader = .{ .eager = false };
    defer r.tokens.deinit(gpa);
    var rig: Rig = undefined;
    try rig.init(.{ .limits = .{ .limit = limit }, .lone = true, .drafted = true });
    defer rig.deinit();
    const req = r.request(&p1, 200, true);
    try rig.host.engine().submit(1, &req, .{ .ctx = &r, .event = Reader.event });
    const t = try std.Thread.spawn(.{}, Reader.trickle, .{ &r, 1 });
    const reason = r.wait();
    t.join();
    try std.testing.expectEqual(Reason.length, reason);
    rig.stop();
    try std.testing.expect(rig.lone.handed > 0);
    try std.testing.expect(rig.pauses() > 0);
    try std.testing.expect(r.most.load(.acquire) <= limit + rig.cfg.family_width + 1);
    try std.testing.expectEqualSlices(u32, fast.items, r.tokens.items);
}

test "slow readers: the host lifts only its own holds, lands a queued step first and releases at half the limit" {
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, io, &core, .{ .lanes = 1 }); // never started: watch runs on this thread
    defer host.stop();
    host.unread_limits = .{ .limit = 8, .stall_ns = 100 };
    var unread: std.atomic.Value(u32) = .init(0);
    const req: Request = .{ .prompt = &p1, .max_tokens = 40, .unread = &unread };
    var job: LaneHost.Job = .{ .host = &host, .id = 7, .request = &req, .sink = undefined, .started = true };
    job.stream = try lanes.Stream.init(gpa, .{ .id = "a", .prompt = &p1, .max_new = 40 });
    defer job.stream.deinit(gpa);
    try host.admitted.append(gpa, &job);
    defer host.admitted.clearRetainingCapacity();
    const s = &job.stream;
    // its backend's pause (a larger session to free) stays, with the reader caught up or behind
    s.paused = true;
    _ = backlog.watch(&host, 0);
    try std.testing.expect(s.paused and !job.backlog.held);
    unread.store(8, .monotonic);
    try std.testing.expectEqual(@as(usize, 0), backlog.watch(&host, 0).held);
    try std.testing.expect(s.paused and !job.backlog.held);
    // the backend lifts it with a step queued ahead: that step lands in the stream's next round, before any hold
    s.paused = false;
    s.inflight = 3;
    _ = backlog.watch(&host, 0);
    try std.testing.expect(!s.paused and s.mode == .drain);
    s.inflight = null;
    try std.testing.expectEqual(@as(usize, 1), backlog.watch(&host, 10).held);
    try std.testing.expect(s.paused and job.backlog.held);
    try std.testing.expectEqual(@as(u64, 1), host.unread_limits.pauses.load(.monotonic));
    // a read restarts the stall's clock; the hold lasts until half the limit is left
    unread.store(5, .monotonic);
    _ = backlog.watch(&host, 90);
    _ = backlog.watch(&host, 150);
    try std.testing.expect(s.paused and host.cancels.items.len == 0);
    unread.store(4, .monotonic);
    try std.testing.expectEqual(@as(usize, 1), backlog.watch(&host, 150).running);
    try std.testing.expect(!s.paused and !job.backlog.held);
    // held again and read no more: one cancel, named once
    unread.store(9, .monotonic);
    _ = backlog.watch(&host, 1000);
    _ = backlog.watch(&host, 1100);
    _ = backlog.watch(&host, 1200);
    try std.testing.expectEqualSlices(api.Id, &.{7}, host.cancels.items);
    try std.testing.expectEqual(@as(u64, 1), host.unread_limits.stalls.load(.monotonic));
}
