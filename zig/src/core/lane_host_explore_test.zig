//! Schedule exploration for the lane host: a seed picks arrivals, cancels, room and pressure at each backend call.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const fill = @import("lane_host_fill.zig");
const pressure = @import("memory_pressure.zig");
const options = @import("explore_options");
const be = lanes.backend;
const Reason = api.Reason;
const gpa = std.testing.allocator;
const io = std.testing.io;

/// Where the fake target yields: a prompt call, a prompt step, a batched pass, a first draw, a one-token step, a round.
const Point = enum { prefill, step, many, first, queue, verify };

/// What the seed may do at a yield point.
const Act = enum { none, arrive, cancel, room, pressure };

/// An invariant a yield point saw broken, reported once the schedule ends.
const Broken = error{ ChunkOutOfOrder, LanesPastHost, RowsPastBudget };

/// One request as its client saw it: tokens, finished events, and whether the explorer cancelled it.
const Box = struct {
    mutex: std.Io.Mutex = .init,
    tokens: std.ArrayList(u32) = .empty,
    done: ?Reason = null,
    ends: u32 = 0, // finished events: exactly one
    late: bool = false, // an event came after the finish
    cancel_sent: bool = false,

    fn event(ctx: *anyopaque, _: api.Id, e: *const api.Event) void {
        const b: *Box = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(io);
        defer b.mutex.unlock(io);
        if (b.ends > 0) b.late = true;
        switch (e.*) {
            .tokens => |t| b.tokens.appendSlice(gpa, t) catch {},
            .finished => |f| {
                b.done = f.reason;
                b.ends += 1;
            },
            else => {},
        }
    }

    fn finished(b: *Box) bool {
        b.mutex.lockUncancelable(io);
        defer b.mutex.unlock(io);
        return b.ends > 0;
    }
};

/// The seeded scheduler: requests not yet sent, and the choices it makes on the engine thread at each yield point.
const Explorer = struct {
    rng: std.Random.DefaultPrng,
    mutex: std.Io.Mutex = .init, // `next`: the engine thread's arrivals at yield points, the test thread's when idle
    engine: api.Engine,
    core: *lanes.Engine,
    target: *lanes.fake.Fake,
    requests: []const api.Request,
    boxes: []Box,
    calls: []usize, // each request's prompt calls so far (its chunk index next)
    next: usize = 0, // the next request to arrive
    lanes: usize, // the host's lanes: the backend never holds more
    budget: u64, // prompt rows a visit beside decoders (0: none)
    rows: u64 = 0, // prompt rows stepped beside a decoder since the core's last round
    steps: u64 = 0, // the core's rounds when `rows` last counted
    dial: pressure.Level = .normal, // the memory pressure the host reads
    broken: ?Broken = null, // the first invariant a yield point saw broken
    digest: std.hash.Wyhash, // every yield point and act, then every reply: a replay must match it
    tally: *Tally,

    fn below(x: *Explorer, n: usize) usize {
        return x.rng.random().uintLessThan(usize, n);
    }

    fn fail(x: *Explorer, e: Broken) void {
        if (x.broken == null) x.broken = e;
    }

    fn mix(x: *Explorer, values: []const u64) void {
        x.digest.update(std.mem.sliceAsBytes(values));
    }

    fn sent(x: *Explorer) usize {
        x.mutex.lockUncancelable(io);
        defer x.mutex.unlock(io);
        return x.next;
    }

    /// The next request, from the engine thread at a yield point or the test thread while the engine idles.
    fn send(x: *Explorer) void {
        x.mutex.lockUncancelable(io);
        defer x.mutex.unlock(io);
        if (x.next >= x.requests.len) return;
        const i = x.next;
        x.next += 1;
        const b = &x.boxes[i];
        x.engine.submit(@intCast(i + 1), &x.requests[i], .{ .ctx = b, .event = Box.event }) catch {
            b.mutex.lockUncancelable(io);
            defer b.mutex.unlock(io);
            b.done = .failed; // refused: the check after the schedule fails it
            b.ends += 1;
        };
    }

    /// One yield point: the backend's lanes checked against the host's, then the act the seed picks there.
    fn point(x: *Explorer, at: Point) void {
        if (x.target.lanes.count() > x.lanes) x.fail(error.LanesPastHost);
        var act: Act = switch (x.below(20)) {
            0...5 => .arrive,
            6, 7 => .cancel,
            8 => .room,
            9 => .pressure,
            else => .none,
        };
        var arg: u64 = 0;
        switch (act) {
            .none => {},
            .arrive => if (x.sent() < x.requests.len) x.send() else {
                act = .none;
            },
            .cancel => if (x.sent() > 0) {
                const i = x.below(x.sent()); // a request still queued, filling, decoding or already ended
                x.boxes[i].cancel_sent = true;
                x.engine.cancel(@intCast(i + 1));
                arg = i;
            } else {
                act = .none;
            },
            // the backend's room for live lanes moves, as when other memory comes and goes: a pass past it is refused
            .room => {
                arg = x.below(4);
                x.target.memory_lanes = if (arg < 3) arg + 1 else null;
            },
            .pressure => {
                arg = x.below(2);
                x.dial = if (arg == 0) .warning else .normal; // warning: one stream admitted at a time
            },
        }
        x.tally.acts[@backingInt(at)][@backingInt(act)] += 1;
        x.mix(&.{ @backingInt(at), @backingInt(act), arg });
    }

    /// Before a prompt step: rows stepped beside a decoder since the core's last round stay under the budget.
    fn stepBegins(x: *Explorer) bool {
        var decoding = false;
        for (x.core.live.items) |s| decoding = decoding or (!s.finished and !s.paused);
        if (!decoding or x.core.steps != x.steps) x.rows = 0; // a visit with no decoder counts none, as the host's
        x.steps = x.core.steps;
        if (x.budget > 0 and x.rows >= x.budget) x.fail(error.RowsPastBudget);
        return decoding;
    }

    /// The host's memory pressure reading: the dial the seed turns.
    fn reading(ctx: ?*anyopaque) ?pressure.Level {
        const x: *Explorer = @ptrCast(@alignCast(ctx.?));
        return x.dial;
    }

    /// Whether every request sent has ended; if so, and some are left, the test thread sends the next (true: all done).
    fn idle(x: *Explorer) bool {
        const n = x.sent();
        for (x.boxes[0..n]) |*b| if (!b.finished()) return false;
        if (n == x.requests.len) return true;
        x.send(); // no yield point comes while nothing is in flight: this send races none of the engine thread's
        return false;
    }

    /// A prompt call for `s`: its chunk index must be its request's next (named by its prompt's last token).
    fn chunk(ctx: *anyopaque, s: *lanes.Stream, at: usize) void {
        const x: *Explorer = @ptrCast(@alignCast(ctx));
        const p = s.prompt();
        const i: usize = p[p.len - 1] - first_token;
        if (i < x.calls.len) {
            if (x.calls[i] != at and at != 0) x.fail(error.ChunkOutOfOrder); // a retry starts over at 0
            x.calls[i] = at + 1;
        }
    }
};

/// What a sweep's schedules did, so a run shows what it explored.
const Tally = struct {
    acts: [6][5]usize = @splat(@splat(0)), // by yield point, then act
    shared: usize = 0, // rounds verified for several streams together
    drafted: usize = 0, // verified windows with draft rows
    refused: usize = 0, // prompt passes refused for room
    cancelled: usize = 0, // requests that ended cancelled

    fn print(t: *const Tally, kind: Kind, seeds: usize) void {
        std.debug.print("lane host explore ({s}): {d} seeds, {d} shared rounds, {d} drafted windows, {d} room refusals, {d} ended cancelled\n", .{ @tagName(kind), seeds, t.shared, t.drafted, t.refused, t.cancelled });
        for (t.acts, 0..) |row, p| {
            var all: usize = 0;
            for (row) |n| all += n;
            std.debug.print("  {s}: {d} yields, {d} arrivals, {d} cancels, {d} room, {d} pressure\n", .{ @tagName(@as(Point, @fromBackingInt(@intCast(p)))), all, row[1], row[2], row[3], row[4] });
        }
    }
};

/// The explorer the wrapped target calls (one schedule runs at a time).
var current: ?*Explorer = null;

/// The target's own vtable, which the wrapped one calls through.
var inner_vt: *const be.Backend.VTable = undefined;

/// The target's own vtable with a yield point before each call a schedule can interleave with.
const Wrapped = struct {
    fn yield(at: Point) void {
        if (current) |x| x.point(at);
    }
    fn refused(e: anyerror) anyerror {
        if (e == error.LaneMemory) if (current) |x| {
            x.tally.refused += 1;
        };
        return e;
    }
    fn prefill(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        yield(.prefill);
        return inner_vt.prefill(ptr, s) catch |e| refused(e);
    }
    fn prefillStep(ptr: *anyopaque, s: *lanes.Stream, max_rows: u32) anyerror!bool {
        yield(.step);
        const x = current orelse return inner_vt.prefill_step.?(ptr, s, max_rows);
        const decoding = x.stepBegins();
        const from = s.fill_at.load(.monotonic);
        const done = inner_vt.prefill_step.?(ptr, s, max_rows) catch |e| return refused(e);
        if (decoding) x.rows += s.fill_at.load(.monotonic) -| from;
        return done;
    }
    fn prefillMany(ptr: *anyopaque, ss: []const *lanes.Stream) anyerror!bool {
        yield(.many);
        const took = try inner_vt.prefill_many.?(ptr, ss);
        if (!took) if (current) |x| {
            x.tally.refused += 1;
        };
        return took;
    }
    fn first(ptr: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
        yield(.first);
        return inner_vt.first(ptr, s, position);
    }
    fn queue(ptr: *anyopaque, s: *lanes.Stream, feed: be.Feed, position: u64) anyerror!u64 {
        yield(.queue);
        return inner_vt.queue(ptr, s, feed, position);
    }
    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        yield(.verify);
        if (current) |x| {
            x.tally.shared += @intFromBool(windows.len > 1);
            for (windows) |w| x.tally.drafted += @intFromBool(w.held > 0 or w.tokens.len > 0);
        }
        return inner_vt.verify(ptr, windows, out);
    }
    /// `inner` with its calls wrapped (`inner_vt` set to its own table).
    fn of(inner: be.Backend, table: *be.Backend.VTable) be.Backend {
        inner_vt = inner.vtable;
        table.* = inner.vtable.*;
        table.prefill = prefill;
        if (inner.vtable.prefill_step != null) table.prefill_step = prefillStep;
        if (inner.vtable.prefill_many != null) table.prefill_many = prefillMany;
        table.first = first;
        table.queue = queue;
        table.verify = verify;
        var out = inner;
        out.vtable = table;
        return out;
    }
};

/// Prompts end in `first_token + i` for request i: the hook names a stream's request by it.
const first_token = 40;

/// A reply as a solo run of its request gives it: the fake target's own continuation (a cancelled one's prefix).
fn expectOwn(q: api.Request, tokens: []const u32) !void {
    var history: std.ArrayList(u32) = .empty;
    defer history.deinit(gpa);
    try history.appendSlice(gpa, q.prompt);
    for (tokens) |t| {
        try std.testing.expectEqual(lanes.fake.next(history.items, q.sampling, history.items.len), t);
        try history.append(gpa, t);
    }
}

const Kind = enum { many, service };

/// One schedule: the scenario's requests sent as the seed says, then every invariant; the schedule's digest.
fn explore(kind: Kind, seed: u64, tally: *Tally) !u64 {
    var setup = std.Random.DefaultPrng.init(seed ^ 0x9e3779b97f4a7c15);
    const r = setup.random();
    const many = kind == .many;
    const head = r.boolean(); // a draft head: drafted rounds whose wrong drafts roll back
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = r.boolean(), .hidden_rows = true, .mtp = head, .speculate = head, .speculate_early = r.boolean(), .drafts = 3, .draft_streams = r.boolean() }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = if (many) 0 else 2 + r.uintLessThan(usize, 3) };
    defer target.deinit();
    const inner = switch (kind) {
        .many => if (r.boolean()) target.batched() else target.batchedSteps(),
        .service => target.stepped(),
    };
    var table: be.Backend.VTable = undefined;
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, Wrapped.of(inner, &table), clock.clock());
    defer core.deinit();
    const slots = 1 + r.uintLessThan(u32, 4);
    var host = api.LaneHost.init(gpa, io, &core, .{ .lanes = slots });
    // the environment's prompt order, share and waits pinned: a seed is one schedule on any machine
    const budgets = [_]u64{ 0, 24, fill.Share.short_rows };
    host.share = .{ .rows = if (many) 0 else budgets[r.uintLessThan(usize, budgets.len)] };
    host.srpt = .{};
    host.prefix = .{ .ns = 0 };
    const n = 2 + r.uintLessThan(usize, 5);
    var prompts: [6][48]u32 = undefined;
    var requests: [6]api.Request = undefined;
    var boxes: [6]Box = @splat(.{});
    defer for (&boxes) |*b| b.tokens.deinit(gpa);
    var calls: [6]usize = @splat(0);
    const drafts = r.boolean();
    for (0..n) |i| {
        const len: usize = if (many) 5 else if (i == 0) 4 else 48;
        @memset(prompts[i][0..len], @intCast(2 + i % 5));
        prompts[i][len - 1] = @intCast(first_token + i);
        const sampling: ?api.Sampling = if (r.boolean()) .{ .seed = seed +% i, .temperature = 0.8, .top_k = 8 } else null;
        requests[i] = .{ .prompt = prompts[i][0..len], .max_tokens = if (!many and i == 0) 24 else 8, .drafts = drafts, .sampling = sampling };
    }
    var x: Explorer = .{ .rng = .init(seed), .engine = host.engine(), .core = &core, .target = &target, .requests = requests[0..n], .boxes = boxes[0..n], .calls = calls[0..n], .lanes = slots, .budget = host.share.rows, .digest = .init(seed), .tally = tally };
    host.pressure = .{ .source = .{ .ctx = &x, .read = Explorer.reading }, .calm_ns = 0, .every_ns = 0 };
    target.prefill_hook = Explorer.chunk;
    target.prefill_hook_ctx = &x;
    current = &x;
    defer current = null;
    for (0..1 + r.uintLessThan(usize, n)) |_| x.send();
    try host.start();
    var running = true;
    defer if (running) host.stop();
    // the test thread sends the rest only when nothing is in flight: the engine thread's own sends stay seeded
    var waited: usize = 0;
    while (!x.idle()) : (waited += 1) {
        if (waited > 20_000) return error.ScheduleStalled;
        std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    }
    host.stop();
    running = false;
    if (x.broken) |e| return e;
    for (boxes[0..n], requests[0..n]) |*b, q| {
        if (b.ends != 1 or b.late) return error.NotFinishedOnce;
        const reason = b.done.?;
        switch (reason) {
            .length => try std.testing.expectEqual(@as(usize, q.max_tokens), b.tokens.items.len),
            .cancelled => if (!b.cancel_sent) return error.CancelledUnasked,
            else => return error.UnexpectedReason,
        }
        if (b.tokens.items.len > q.max_tokens) return error.TokensPastMax;
        tally.cancelled += @intFromBool(reason == .cancelled);
        try expectOwn(q, b.tokens.items);
        x.mix(&.{ @backingInt(reason), b.tokens.items.len });
        x.digest.update(std.mem.sliceAsBytes(b.tokens.items));
    }
    try std.testing.expectEqual(@as(usize, 0), target.lanes.count()); // every lane released
    try std.testing.expectEqual(@as(usize, 0), core.activeCount());
    return x.digest.final();
}

/// The sweep's seeds: `options.seeds` of them from here.
const first_seed: u64 = 0x5eed_0000;

/// Every seed in the sweep (or the one -Dexplore-seed names); a failure names its seed so it replays.
fn sweep(kind: Kind) !void {
    const level = std.testing.log_level;
    std.testing.log_level = .err; // a refused batched pass logs a warning
    defer std.testing.log_level = level;
    var tally: Tally = .{};
    if (options.seed) |s| return failing(kind, s, explore(kind, s, &tally));
    for (0..options.seeds) |i| try failing(kind, first_seed + i, explore(kind, first_seed + i, &tally));
    if (options.tally) tally.print(kind, options.seeds);
}

/// A schedule's result, its seed and replay line printed when it failed.
fn failing(kind: Kind, seed: u64, result: anyerror!u64) !void {
    _ = result catch |err| {
        std.debug.print("lane host explore ({s}): seed {d} failed ({s}); replay: zig build test-lane-explore-host -Dexplore-seed={d}\n", .{ @tagName(kind), seed, @errorName(err), seed });
        return err;
    };
}

test "explore: queued requests through batched prompt passes, under seeded arrivals, cancels, room and pressure" {
    try sweep(.many);
}

test "explore: a decoder beside chunked prompts and the prompt-row budget, under seeded arrivals, cancels, room and pressure" {
    try sweep(.service);
}

test "explore: a seed replays its schedule, every yield point's act and every reply the same" {
    const level = std.testing.log_level;
    std.testing.log_level = .err;
    defer std.testing.log_level = level;
    var tally: Tally = .{};
    for ([_]Kind{ .many, .service }) |kind| for (0..if (options.seed == null) @min(options.seeds, 16) else 1) |i| {
        const seed = options.seed orelse first_seed + i;
        const a = explore(kind, seed, &tally) catch |err| return failing(kind, seed, err);
        if (a != try explore(kind, seed, &tally)) return failing(kind, seed, error.ScheduleNotReplayed);
    };
}
