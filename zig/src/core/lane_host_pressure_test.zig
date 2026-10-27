//! The lane host under a fake pressure source: admission held and resumed, 503s on critical, the cache shed, logs once.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const mp = @import("memory_pressure.zig");
const LaneHost = api.LaneHost;
const Reason = api.Reason;
const Request = api.Request;
const io = std.testing.io;

/// A pressure source the test thread sets while the engine thread reads it.
const Dial = struct {
    level: std.atomic.Value(u8) = .init(@backingInt(mp.Level.normal)),
    fn set(d: *Dial, l: mp.Level) void {
        d.level.store(@backingInt(l), .release);
    }
    fn read(ctx: ?*anyopaque) ?mp.Level {
        const d: *Dial = @ptrCast(@alignCast(ctx.?));
        return @fromBackingInt(@intCast(d.level.load(.acquire)));
    }
};

/// The governor's log lines, as changes.
const Lines = struct {
    mutex: std.Io.Mutex = .init,
    seen: [16]mp.Change = undefined,
    n: usize = 0,
    fn line(ctx: ?*anyopaque, c: mp.Change, _: ?u64) void {
        const l: *Lines = @ptrCast(@alignCast(ctx.?));
        l.mutex.lockUncancelable(io);
        defer l.mutex.unlock(io);
        l.seen[l.n] = c;
        l.n += 1;
    }
};

/// A request's events: its tokens, cached prompt tokens and end, and the order its first token and end came in.
const Box = struct {
    mutex: std.Io.Mutex = .init,
    tokens: [64]u32 = undefined,
    n: usize = 0,
    cached: ?u32 = null,
    done: ?Reason = null,
    order: ?*std.atomic.Value(u32) = null,
    first_at: u32 = 0,
    done_at: u32 = 0,
    fn event(ctx: *anyopaque, _: api.Id, e: *const api.Event) void {
        const b: *Box = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(io);
        defer b.mutex.unlock(io);
        const tick = if (b.order) |o| o.fetchAdd(1, .acq_rel) + 1 else 0;
        switch (e.*) {
            .prefilled => |c| b.cached = c,
            .tokens => |t| {
                if (b.n == 0) b.first_at = tick;
                for (t) |x| if (b.n < b.tokens.len) {
                    b.tokens[b.n] = x;
                    b.n += 1;
                };
            },
            .finished => |f| {
                b.done = f.reason;
                b.done_at = tick;
            },
            else => {},
        }
    }
    fn count(b: *Box) usize {
        b.mutex.lockUncancelable(io);
        defer b.mutex.unlock(io);
        return b.n;
    }
    fn wait(b: *Box) !Reason {
        for (0..4000) |_| {
            b.mutex.lockUncancelable(io);
            const d = b.done;
            b.mutex.unlock(io);
            if (d) |r| return r;
            std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
        }
        return error.RequestTimedOut;
    }
    fn got(b: *Box) []const u32 {
        return b.tokens[0..b.n];
    }
};

/// The fake target's kept states, counted where the test thread reads them.
const Snaps = struct {
    f: *lanes.fake.Fake,
    live: std.atomic.Value(usize) = .init(0),
    fn of(ctx: *anyopaque) *Snaps {
        return @ptrCast(@alignCast(ctx));
    }
    fn bytes(_: *anyopaque, at: u32) u64 {
        return 64 + 4 * @as(u64, at);
    }
    fn save(ctx: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!pc.Saved {
        const s = of(ctx);
        const saved = try s.f.save(@ptrCast(@alignCast(owner.?)), at);
        _ = s.live.fetchAdd(1, .acq_rel);
        return saved;
    }
    fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {
        return error.BackendRestores; // lane-core backends restore inside their own prompt pass
    }
    fn drop(ctx: *anyopaque, saved: pc.Saved) void {
        const s = of(ctx);
        _ = s.live.fetchSub(1, .acq_rel);
        s.f.drop(saved);
    }
    fn liveWithin(s: *Snaps, want: usize, ms: usize) bool {
        for (0..ms) |_| {
            if (s.live.load(.acquire) == want) return true;
            std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
        }
        return s.live.load(.acquire) == want;
    }
};

/// A lane host on the fake target with a prompt cache, governed by `dial` when given.
const Rig = struct {
    cfg: lanes.Config,
    target: lanes.fake.Fake,
    clock: lanes.fake.FixedClock = .{},
    core: lanes.Engine,
    snaps: Snaps,
    store: pc.Store,
    host: LaneHost,
    stopped: bool = false,

    fn init(r: *Rig, lanes_n: u32, dial: ?*Dial, lines: ?*Lines, percent: u8) !void {
        const gpa = std.testing.allocator;
        r.cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
        r.target = .{ .gpa = gpa };
        r.clock = .{};
        r.core = lanes.Engine.init(gpa, &r.cfg, r.target.backend(), r.clock.clock());
        r.snaps = .{ .f = &r.target };
        r.stopped = false;
        r.store = pc.Store.init(gpa, .{ .ptr = &r.snaps, .vtable = &.{ .bytes = Snaps.bytes, .save = Snaps.save, .restore = Snaps.restore, .drop = Snaps.drop } }, .{ .min_gap = 4, .min_prompt = 0 }, 1 << 20);
        r.host = LaneHost.init(gpa, io, &r.core, .{ .lanes = lanes_n });
        r.host.cache = &r.store;
        if (dial) |d| r.host.pressure = .{ .source = .{ .ctx = d, .read = Dial.read }, .percent = percent, .calm_ns = 0, .every_ns = 0, .log = if (lines) |l| .{ .ctx = l, .line = Lines.line } else .{} };
        try r.host.start();
    }

    /// The engine thread joined: the store is the test's to read.
    fn halt(r: *Rig) void {
        if (!r.stopped) r.host.stop();
        r.stopped = true;
    }

    fn deinit(r: *Rig) void {
        r.halt();
        r.store.deinit();
        r.core.deinit();
        r.target.deinit();
        r.cfg.deinit(std.testing.allocator);
    }

    fn submit(r: *Rig, id: api.Id, req: *const Request, box: *Box) !void {
        try r.host.engine().submit(id, req, .{ .ctx = box, .event = Box.event });
    }

    /// Waits for the engine thread to govern at `want`.
    fn reach(r: *Rig, want: mp.Level) !void {
        for (0..4000) |_| {
            r.host.lock();
            const now = r.host.pressure.?.level;
            r.host.unlock();
            if (now == want) return;
            std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
        }
        return error.LevelNotReached;
    }
};

/// The fake target's greedy reply to `prompt`, `n` tokens: what a request's solo run gives.
fn solo(prompt: []const u32, n: usize, out: []u32) []const u32 {
    var history: [128]u32 = undefined;
    @memcpy(history[0..prompt.len], prompt);
    for (0..n) |i| {
        const len = prompt.len + i;
        out[i] = lanes.fake.next(history[0..len], null, len);
        history[len] = out[i];
    }
    return out[0..n];
}

const t1 = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3 };
const t2 = [_]u32{ 2, 7, 1, 8, 2, 8, 1, 8, 2, 8 };
const t3 = [_]u32{ 1, 4, 1, 4, 2, 1, 3, 5, 6, 2 };
const short = [_]u32{ 8, 6, 7, 5, 3, 0, 9 };

/// What a fixed workload leaves: each reply, its cached tokens, and the store's counts and states.
const Outcome = struct {
    tokens: [4][16]u32 = @splat(@splat(0)),
    cached: [4]?u32 = @splat(null),
    counts: pc.Counts = .{},
    ats: [8]u32 = @splat(0),
    held: u64 = 0,
};

fn workload(governed: bool) !Outcome {
    var dial: Dial = .{};
    var r: Rig = undefined;
    try r.init(3, if (governed) &dial else null, null, 0);
    defer r.deinit();
    var out: Outcome = .{};
    var boxes: [4]Box = @splat(.{});
    const first: Request = .{ .prompt = &t1, .max_tokens = 6, .history_len = 8 };
    try r.submit(1, &first, &boxes[0]);
    try std.testing.expectEqual(Reason.length, try boxes[0].wait());
    var next: [16]u32 = undefined;
    @memcpy(next[0..10], &t1);
    @memcpy(next[10..14], boxes[0].got()[0..4]);
    next[14], next[15] = .{ 7, 3 };
    const turn: Request = .{ .prompt = &next, .max_tokens = 8, .history_len = 14 };
    try r.submit(2, &turn, &boxes[1]);
    try std.testing.expectEqual(Reason.length, try boxes[1].wait());
    const pair = [_]Request{ .{ .prompt = &t2, .max_tokens = 6, .history_len = 8 }, .{ .prompt = &t3, .max_tokens = 6, .history_len = 8 } };
    try r.submit(3, &pair[0], &boxes[2]);
    try r.submit(4, &pair[1], &boxes[3]);
    for (boxes[2..]) |*b| try std.testing.expectEqual(Reason.length, try b.wait());
    for (&boxes, 0..) |*b, i| {
        @memcpy(out.tokens[i][0..b.n], b.got());
        out.cached[i] = b.cached;
    }
    r.halt();
    out.counts = r.store.counts;
    out.held = r.store.held;
    for (r.store.entries.items, 0..) |e, i| out.ats[i] = e.at;
    std.mem.sort(u32, out.ats[0..r.store.entries.items.len], {}, std.sort.asc(u32));
    return out;
}

test "memory pressure: with pressure normal, admission and the prompt cache are exactly as without a governor" {
    const plain = try workload(false);
    const governed = try workload(true);
    try std.testing.expectEqualDeep(plain, governed);
    try std.testing.expectEqual(@as(?u32, 8), governed.cached[1]); // the workload did resume and keep
    try std.testing.expect(governed.counts.kept >= 4);
}

test "memory pressure: normal, warning, critical and normal again hold, refuse and resume admission, logged once" {
    var dial: Dial = .{};
    var lines: Lines = .{};
    var r: Rig = undefined;
    try r.init(3, &dial, &lines, 0);
    defer r.deinit();
    var first: Box = .{};
    const keeps: Request = .{ .prompt = &t1, .max_tokens = 6, .history_len = 8 };
    try r.submit(1, &keeps, &first);
    try std.testing.expectEqual(Reason.length, try first.wait());
    try std.testing.expectEqual(@as(usize, 1), r.snaps.live.load(.acquire));
    var running: Box = .{};
    const long: Request = .{ .prompt = &short, .max_tokens = 100000 };
    try r.submit(2, &long, &running);
    for (0..2000) |_| if (running.count() == 0) std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    try std.testing.expect(running.count() > 0);
    dial.set(.warning);
    try r.reach(.warning);
    try std.testing.expect(r.snaps.liveWithin(0, 2000)); // shed to the watermark (0% here) while the stream runs
    var held: Box = .{};
    const later: Request = .{ .prompt = &t2, .max_tokens = 6 };
    try r.submit(3, &later, &held); // queued, not refused
    std.Io.sleep(io, .fromMilliseconds(100), .awake) catch {};
    try std.testing.expectEqual(@as(usize, 0), held.count()); // no second stream beside the running one
    dial.set(.critical);
    try r.reach(.critical);
    var refused: Box = .{};
    try std.testing.expectError(error.MemoryPressure, r.submit(4, &later, &refused));
    try std.testing.expect(refused.done == null);
    dial.set(.normal);
    try r.reach(.normal);
    try std.testing.expectEqual(Reason.length, try held.wait()); // admitted again beside the running stream
    var want: [6]u32 = undefined;
    try std.testing.expectEqualSlices(u32, solo(&t2, 6, &want), held.got()); // equal to its solo run
    var again: Box = .{};
    try r.submit(5, &keeps, &again);
    try std.testing.expectEqual(Reason.length, try again.wait());
    try std.testing.expectEqual(@as(?u32, 0), again.cached); // its state was shed: a fresh pass, the same reply
    try std.testing.expectEqualSlices(u32, first.got(), again.got());
    try std.testing.expect(r.snaps.liveWithin(1, 2000)); // kept again
    r.host.engine().cancel(2);
    try std.testing.expectEqual(Reason.cancelled, try running.wait());
    lines.mutex.lockUncancelable(io);
    defer lines.mutex.unlock(io);
    const expected = [_]mp.Change{ .{ .from = .normal, .to = .warning }, .{ .from = .warning, .to = .critical }, .{ .from = .critical, .to = .normal } };
    try std.testing.expectEqualSlices(mp.Change, &expected, lines.seen[0..lines.n]);
}

test "memory pressure: under warning a request still runs when none does, one at a time, keeping no state" {
    var dial: Dial = .{};
    dial.set(.warning);
    var r: Rig = undefined;
    try r.init(3, &dial, null, 50);
    defer r.deinit();
    try r.reach(.warning);
    var order: std.atomic.Value(u32) = .init(0);
    var boxes: [2]Box = .{ .{ .order = &order }, .{ .order = &order } };
    const asks = [_]Request{ .{ .prompt = &t1, .max_tokens = 6, .history_len = 8 }, .{ .prompt = &t3, .max_tokens = 6, .history_len = 8 } };
    try r.submit(1, &asks[0], &boxes[0]);
    try r.submit(2, &asks[1], &boxes[1]);
    for (&boxes) |*b| try std.testing.expectEqual(Reason.length, try b.wait());
    const a, const b = .{ &boxes[0], &boxes[1] };
    try std.testing.expect(a.done_at < b.first_at or b.done_at < a.first_at); // the second began after the first ended
    var want: [6]u32 = undefined;
    try std.testing.expectEqualSlices(u32, solo(&t1, 6, &want), a.got());
    try std.testing.expectEqualSlices(u32, solo(&t3, 6, &want), b.got());
    try std.testing.expectEqual(@as(usize, 0), r.snaps.live.load(.acquire));
    try std.testing.expectEqual(@as(u64, 0), r.store.counts.kept);
}
