//! The lone driver: reclaiming the lane core's last stream between rounds, and a halt during its prompt pass.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Reason = api.Reason;
const Id = api.Id;
const Event = api.Event;
const Request = api.Request;

const gpa = std.testing.allocator;

const Box = struct {
    mutex: std.Io.Mutex = .init,
    tokens: std.ArrayList(u32) = .empty,
    done: ?Reason = null,
    message: []const u8 = "",

    fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
        const b: *Box = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(std.testing.io);
        defer b.mutex.unlock(std.testing.io);
        switch (e.*) {
            .tokens => |t| b.tokens.appendSlice(gpa, t) catch {},
            .finished => |f| {
                b.done = f.reason;
                b.message = f.message;
            },
            else => {},
        }
    }

    fn wait(b: *Box) Reason {
        while (true) {
            b.mutex.lockUncancelable(std.testing.io);
            const d = b.done;
            b.mutex.unlock(std.testing.io);
            if (d) |r| return r;
            std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        }
    }
};

/// Hands a new stream over after its first token; a resumed one decodes by the fake's one-token path (handover.zig).
const Lone = struct {
    be: lanes.backend.Backend,
    opened: u32 = 0,
    resumed: u32 = 0,
    /// Resumed tokens after which the driver signals `paused` and waits for `go` (0: never), for hand-back tests.
    pause_at: u32 = 0,
    commits: u32 = 0,
    paused: std.Io.Event = .unset,
    go: std.Io.Event = .unset,
    /// When set, an opening waits for `open` first: requests submitted meanwhile wait in the queue, not behind a reply.
    hold_open: bool = false,
    open: std.Io.Event = .unset,

    fn run(ctx: *anyopaque, s: *lanes.Stream, hooks: api.LoneHooks) anyerror!bool {
        const l: *Lone = @ptrCast(@alignCast(ctx));
        if (!hooks.resumed) {
            if (l.hold_open) l.open.waitUncancelable(std.testing.io);
            l.opened += 1;
            _ = try l.be.opening(gpa, s);
            hooks.committed(hooks.ctx);
            s.pending = s.context.items[s.context.items.len - 1];
            s.cache_len = s.prompt_len; // as a backend's hand-over leaves it: the first token not yet fed
            return true;
        }
        l.resumed += 1;
        while (!s.finished) {
            if (hooks.yield(hooks.ctx)) return true;
            // feed the pending token and draw the next at the new length, as the lane core's queueNext does
            s.cache_len += 1;
            const t = try l.be.read(try l.be.queue(s, .{ .value = s.pending.? }, s.cache_len));
            _ = try s.commit(gpa, &.{t}, &.{});
            s.pending = t;
            hooks.committed(hooks.ctx);
            l.commits += 1;
            if (l.commits == l.pause_at) {
                l.paused.set(std.testing.io);
                l.go.waitUncancelable(std.testing.io);
            }
        }
        l.be.release(s);
        return false;
    }
};

const Runs = struct { opened: u32, resumed: u32 };

fn serve(request: *const Request, resumes: bool) !Runs {
    return serveImages(request, resumes, true);
}

fn serveImages(request: *const Request, resumes: bool, images: bool) !Runs {
    return serveDriver(request, resumes, images, false);
}

fn draftedConfig(costs: *[16]lanes.config.Cost) !lanes.Config {
    return lanes.Config.init(gpa, .{ .exact_width = 16, .gpu_tokens = true, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 4, .window_costs = costs, .mtp_step_ms = 0.5, .hidden_rows = true, .batch_rows = 32, .max_streams = 8, .draft_streams = true }, 16, 15);
}

fn expectSolo(request: *const Request, concurrent: []const u32) !void {
    var costs: [16]lanes.config.Cost = undefined;
    for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 5.0 + 0.8 * @as(f64, @floatFromInt(w)) };
    var cfg = try draftedConfig(&costs);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    var driver: Lone = .{ .be = target.backend() };
    host.lone = .{ .ctx = &driver, .run = Lone.run, .resumes = true };
    try host.start();
    defer host.stop();
    var box: Box = .{};
    defer box.tokens.deinit(gpa);
    try host.engine().submit(1, request, .{ .ctx = &box, .event = Box.event });
    try std.testing.expectEqual(Reason.length, box.wait());
    try std.testing.expectEqualSlices(u32, box.tokens.items, concurrent);
}

/// Every token the fake's own rule after the prompt and the tokens before it, and the whole budget.
fn expectRule(request: *const Request, got: []const u32) !void {
    var history: std.ArrayList(u32) = .empty;
    defer history.deinit(gpa);
    try history.appendSlice(gpa, request.prompt);
    for (got) |t| {
        try std.testing.expectEqual(lanes.fake.next(history.items, null, history.items.len), t);
        try history.append(gpa, t);
    }
    try std.testing.expectEqual(@as(usize, request.max_tokens), got.len);
}

fn serveDriver(request: *const Request, resumes: bool, images: bool, plain: bool) !Runs {
    // a drafted family's windowed rounds, as the lone driver's backends run them (lanes/engine_test.zig's model)
    var costs: [16]lanes.config.Cost = undefined;
    for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 5.0 + 0.8 * @as(f64, @floatFromInt(w)) };
    var cfg = try draftedConfig(&costs);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    var lone: Lone = .{ .be = target.backend() };
    host.lone = .{ .ctx = &lone, .run = Lone.run, .resumes = resumes, .images = images, .plain = plain };
    try host.start();
    defer host.stop();
    var box: Box = .{};
    defer box.tokens.deinit(gpa);
    try host.engine().submit(1, request, .{ .ctx = &box, .event = Box.event });
    try std.testing.expectEqual(Reason.length, box.wait());
    var history: std.ArrayList(u32) = .empty;
    defer history.deinit(gpa);
    try history.appendSlice(gpa, request.prompt);
    for (box.tokens.items) |t| {
        try std.testing.expectEqual(lanes.fake.next(history.items, null, history.items.len), t);
        try history.append(gpa, t);
    }
    try std.testing.expectEqual(@as(usize, request.max_tokens), box.tokens.items.len);
    try std.testing.expectEqual(@as(usize, 0), target.lanes.count());
    return .{ .opened = lone.opened, .resumed = lone.resumed };
}

const prompt = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6 };

test "the lane core's last stream goes back to the lone driver, its tokens served once each, in order" {
    const runs = try serve(&.{ .prompt = &prompt, .max_tokens = 24 }, true);
    try std.testing.expectEqual(@as(u32, 1), runs.opened);
    try std.testing.expectEqual(@as(u32, 1), runs.resumed);
}

test "a stream the lone driver does not take stays in the lane core to its end" {
    const runs = try serve(&.{ .prompt = &prompt, .max_tokens = 24, .drafts = false }, true);
    try std.testing.expectEqual(@as(u32, 0), runs.opened);
    try std.testing.expectEqual(@as(u32, 0), runs.resumed);
}

test "a penalized stream stays in the lane core: no lone driver takes presence or frequency penalties" {
    const runs = try serve(&.{ .prompt = &prompt, .max_tokens = 24, .penalties = .{ .presence = 1.5, .frequency = 0.5 } }, true);
    try std.testing.expectEqual(@as(u32, 0), runs.opened);
    try std.testing.expectEqual(@as(u32, 0), runs.resumed);
}

test "a lone driver that takes plain streams opens one with drafts off" {
    const runs = try serveDriver(&.{ .prompt = &prompt, .max_tokens = 24, .drafts = false }, false, true, true);
    try std.testing.expectEqual(@as(u32, 1), runs.opened);
    try std.testing.expectEqual(@as(u32, 0), runs.resumed);
}

test "a lone driver that only opens streams never gets the lane core's last stream back" {
    const runs = try serve(&.{ .prompt = &prompt, .max_tokens = 24 }, false);
    try std.testing.expectEqual(@as(u32, 1), runs.opened);
    try std.testing.expectEqual(@as(u32, 0), runs.resumed);
}

test "halt fails a request in the lone driver's prompt pass with the reason" {
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    // the stream's lane is released once, by the driver's failed opening, not again when the halt ends the request
    const Releases = struct {
        n: u32 = 0,
        fn count(ctx: *anyopaque, _: *lanes.Stream) void {
            const r: *@This() = @ptrCast(@alignCast(ctx));
            r.n += 1;
        }
    };
    var releases: Releases = .{};
    target.release_hook = Releases.count;
    target.release_hook_ctx = &releases;
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    // the driver holds its prompt pass until the halt is requested; the pass then finds the host closing
    const Held = struct {
        be: lanes.backend.Backend,
        entered: std.Io.Event = .unset,
        go: std.Io.Event = .unset,
        fn run(ctx: *anyopaque, s: *lanes.Stream, _: api.LoneHooks) anyerror!bool {
            const l: *@This() = @ptrCast(@alignCast(ctx));
            l.entered.set(std.testing.io);
            l.go.waitUncancelable(std.testing.io);
            _ = try l.be.opening(gpa, s); // error.Cancelled, the stream released
            return error.NotCancelled;
        }
    };
    var held: Held = .{ .be = target.backend() };
    host.lone = .{ .ctx = &held, .run = Held.run };
    try host.start();
    defer host.stop();
    var box: Box = .{};
    defer box.tokens.deinit(gpa);
    const request: Request = .{ .prompt = &prompt, .max_tokens = 24 };
    try host.engine().submit(1, &request, .{ .ctx = &box, .event = Box.event });
    held.entered.waitUncancelable(std.testing.io);
    const halter = try std.Thread.spawn(.{}, LaneHost.halt, .{ &host, "server restarting: retry shortly" });
    host.mutex.lockUncancelable(std.testing.io);
    while (host.halt_reason == null) host.wake.waitTimeout(std.testing.io, &host.mutex, .{ .duration = .{ .raw = .fromMilliseconds(1), .clock = .awake } }) catch {}; // timed: the lane thread also waits on wake
    host.mutex.unlock(std.testing.io);
    held.go.set(std.testing.io);
    halter.join();
    try std.testing.expectEqual(@as(?Reason, .failed), box.done);
    try std.testing.expectEqualStrings("server restarting: retry shortly", box.message);
    try std.testing.expectEqual(@as(usize, 0), box.tokens.items.len);
    try std.testing.expectEqual(@as(usize, 0), target.lanes.count());
    try std.testing.expectEqual(@as(u32, 1), releases.n);
}

test "the lone driver's image opt-out keeps both opening and reclaim on the lane core" {
    const images = [_]api.Image{.{ .pixels = &.{}, .gh = 1, .gw = 1, .at = 1, .tokens = 1 }};
    const request: Request = .{ .prompt = &prompt, .max_tokens = 24, .images = &images };
    const denied = try serveImages(&request, true, false);
    try std.testing.expectEqual(@as(u32, 0), denied.opened);
    try std.testing.expectEqual(@as(u32, 0), denied.resumed);
    const allowed = try serve(&request, true);
    try std.testing.expectEqual(@as(u32, 1), allowed.opened);
    try std.testing.expectEqual(@as(u32, 1), allowed.resumed);
    const text = try serveImages(&.{ .prompt = &prompt, .max_tokens = 24 }, true, false);
    try std.testing.expectEqual(@as(u32, 1), text.opened);
    try std.testing.expectEqual(@as(u32, 1), text.resumed);
}

/// A request's tokens and stats, with the order in which replies finished, for queue tests.
const LogBox = struct {
    mutex: std.Io.Mutex = .init,
    seq: *std.atomic.Value(usize),
    host: ?*LaneHost = null,
    peer_id: ?Id = null,
    peer_at_first: ?u64 = null,
    peer_at_finish: ?u64 = null,
    order: usize = 0,
    tokens: std.ArrayList(u32) = .empty,
    events: std.ArrayList(usize) = .empty,
    stats: api.Stats = .{},
    done: ?Reason = null,

    fn peerRounds(b: *@This()) ?u64 {
        const h = b.host orelse return null;
        const id = b.peer_id orelse return null;
        h.mutex.lockUncancelable(std.testing.io);
        defer h.mutex.unlock(std.testing.io);
        for (h.admitted.items) |j| if (j.id == id) return j.stream.rounds;
        return null;
    }

    fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
        const b: *@This() = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(std.testing.io);
        defer b.mutex.unlock(std.testing.io);
        switch (e.*) {
            .tokens => |t| {
                if (b.peer_at_first == null) b.peer_at_first = b.peerRounds();
                b.tokens.appendSlice(gpa, t) catch {};
                b.events.append(gpa, t.len) catch {};
            },
            .finished => |f| {
                b.peer_at_finish = b.peerRounds();
                b.done = f.reason;
                b.stats = f.stats;
                b.order = b.seq.fetchAdd(1, .monotonic);
            },
            else => {},
        }
    }

    fn wait(b: *@This()) Reason {
        while (true) {
            b.mutex.lockUncancelable(std.testing.io);
            const d = b.done;
            b.mutex.unlock(std.testing.io);
            if (d) |r| return r;
            std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        }
    }

    fn deinit(b: *@This()) void {
        b.tokens.deinit(gpa);
        b.events.deinit(gpa);
    }
};

test "lone driver: a third tiny prompt at two lanes starts within two rounds and shares later rounds" {
    const Wait = @import("wait_reason.zig").Wait;
    var costs: [16]lanes.config.Cost = undefined;
    for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 5.0 + 0.8 * @as(f64, @floatFromInt(w)) };
    var cfg = try draftedConfig(&costs);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    // its opening waits for A, B and C: a fake reply takes microseconds and could otherwise end before B arrives
    var driver: Lone = .{ .be = target.backend(), .hold_open = true };
    host.lone = .{ .ctx = &driver, .run = Lone.run, .resumes = true };
    try host.start();
    defer host.stop();

    var seq = std.atomic.Value(usize).init(0);
    var a: LogBox = .{ .seq = &seq, .host = &host, .peer_id = 2 };
    var b: LogBox = .{ .seq = &seq };
    var c: LogBox = .{ .seq = &seq, .host = &host, .peer_id = 2 };
    var d: LogBox = .{ .seq = &seq };
    defer {
        a.deinit();
        b.deinit();
        c.deinit();
        d.deinit();
    }
    const ra: Request = .{ .prompt = &prompt, .max_tokens = 200 };
    const rb: Request = .{ .prompt = &prompt, .max_tokens = 400 };
    const rc: Request = .{ .prompt = &prompt, .max_tokens = 24 };
    const rd: Request = .{ .prompt = &prompt, .max_tokens = 8 };
    try host.engine().submit(1, &ra, .{ .ctx = &a, .event = LogBox.event });
    try host.engine().submit(2, &rb, .{ .ctx = &b, .event = LogBox.event });
    try host.engine().submit(3, &rc, .{ .ctx = &c, .event = LogBox.event });
    driver.open.set(std.testing.io);

    // A and B hold both lanes; C waits with a named reason; the loop waits on the state alone, not a time bound.
    while (true) {
        host.mutex.lockUncancelable(std.testing.io);
        const reason: ?Wait = if (host.queued.items.len > 0) host.queued.items[0].wait_reason else null;
        host.mutex.unlock(std.testing.io);
        if (reason == .engine_busy) break;
        std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    }
    host.mutex.lockUncancelable(std.testing.io);
    try std.testing.expect(host.queued.items.len > 0);
    try std.testing.expectEqual(Wait.engine_busy, host.queued.items[0].wait_reason);
    host.mutex.unlock(std.testing.io);

    // A fourth request joins behind C and must not pass it.
    try host.engine().submit(4, &rd, .{ .ctx = &d, .event = LogBox.event });
    try std.testing.expectEqual(Reason.length, a.wait());
    try std.testing.expectEqual(Reason.length, c.wait());
    try std.testing.expectEqual(Reason.length, d.wait());
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expect(c.stats.queued_seconds > 0);
    try std.testing.expect(c.stats.rounds > 0);
    try std.testing.expect(c.order < d.order);

    // Survivor rounds are global scheduling evidence: C must neither wait past two rounds nor be skipped thereafter.
    try std.testing.expect(a.peer_at_finish != null and c.peer_at_first != null and c.peer_at_finish != null);
    const rounds_to_first = c.peer_at_first.? - a.peer_at_finish.?;
    try std.testing.expect(rounds_to_first <= 2);
    const survivor_rounds = c.peer_at_finish.? - c.peer_at_first.?;
    try std.testing.expect(survivor_rounds <= c.stats.rounds + 1);
    try std.testing.expect(c.stats.rounds <= 2 * @as(u64, @intCast(c.tokens.items.len)) + 2);

    try expectRule(&rc, c.tokens.items);
    try std.testing.expect(c.events.items.len >= 2);
    for ([_]*const Request{ &ra, &rb, &rd }, [_][]const u32{ a.tokens.items, b.tokens.items, d.tokens.items }) |r, got| try expectRule(r, got);
    try expectSolo(&ra, a.tokens.items);
    try expectSolo(&rb, b.tokens.items);
    try expectSolo(&rc, c.tokens.items);
    try expectSolo(&rd, d.tokens.items);
}

test "lone driver: a stream it hands back after tokens of its own goes on from them, as its solo run" {
    var costs: [16]lanes.config.Cost = undefined;
    for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 5.0 + 0.8 * @as(f64, @floatFromInt(w)) };
    var cfg = try draftedConfig(&costs);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    // A alone: opened, handed over, reclaimed after a round; its driver lands 5 tokens, then B arrives
    var driver: Lone = .{ .be = target.backend(), .pause_at = 5 };
    host.lone = .{ .ctx = &driver, .run = Lone.run, .resumes = true };
    try host.start();
    defer host.stop();
    var seq = std.atomic.Value(usize).init(0);
    var a: LogBox = .{ .seq = &seq };
    var b: LogBox = .{ .seq = &seq };
    defer {
        a.deinit();
        b.deinit();
    }
    const ra: Request = .{ .prompt = &prompt, .max_tokens = 60 };
    const rb: Request = .{ .prompt = &prompt, .max_tokens = 24 };
    try host.engine().submit(1, &ra, .{ .ctx = &a, .event = LogBox.event });
    driver.paused.waitUncancelable(std.testing.io);
    try host.engine().submit(2, &rb, .{ .ctx = &b, .event = LogBox.event });
    driver.go.set(std.testing.io); // its next check sees B waiting: A goes back to the lane core mid-reply
    try std.testing.expectEqual(Reason.length, a.wait());
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expect(driver.resumed >= 1 and driver.commits >= 5);
    try expectRule(&ra, a.tokens.items);
    try expectRule(&rb, b.tokens.items);
    try expectSolo(&ra, a.tokens.items);
    try expectSolo(&rb, b.tokens.items);
}
