//! A lone stream's copy windows priced from its own timed rounds (#3390): off where they lose, back once they land.
const std = @import("std");
const cfgs = @import("config.zig");
const Engine = @import("engine.zig").Engine;
const sm = @import("stream.zig");
const fake = @import("fake.zig");
const be = @import("backend.zig");
const Proposer = @import("proposer.zig").Proposer;
const ev = @import("events.zig");
const win = @import("windows.zig");
const lone_copy = @import("lone_copy.zig");
const copy_price = @import("copy_price.zig");
const lone_costs = @import("lone_costs.zig");

const gpa = std.testing.allocator;

/// The forwards as timed at load, on a short scratch context: 5 ms plus 0.2 ms a row.
const startup = blk: {
    var c: [16]cfgs.Cost = undefined;
    for (&c, 1..) |*x, w| x.* = .{ .width = w, .ms = 5 + 0.2 * @as(f64, @floatFromInt(w)) };
    break :blk c;
};

/// A round's ms: a fixed part plus each verified row's (at a long context each row walks the whole cache).
const long_context = [2]f64{ 50, 40 };
const short_context = [2]f64{ 10, 0.1 };

const Clock = struct {
    target: *const fake.Fake,
    times: [2]f64,
    draft_ms: f64 = 0, // each head drafting since the round started
    drafts_at: usize = 0,

    fn clock(c: *Clock) be.Clock {
        return .{ .ptr = c, .vtable = &.{ .start = start, .elapsed_ms = elapsed } };
    }

    fn start(ptr: *anyopaque) void {
        const c: *Clock = @ptrCast(@alignCast(ptr));
        c.drafts_at = c.target.drafts_calls;
    }

    fn elapsed(ptr: *anyopaque, _: be.Mark) f64 {
        const c: *Clock = @ptrCast(@alignCast(ptr));
        const drafts: f64 = @floatFromInt(c.target.drafts_calls - c.drafts_at);
        return c.times[0] + c.times[1] * @as(f64, @floatFromInt(c.target.last_rows)) + c.draft_ms * drafts;
    }
};

/// The target's next 15 tokens at every ask: from context length `land_from` on they land, before it the first misses.
const Copier = struct {
    out: [15]u32 = undefined,
    land_from: usize = std.math.maxInt(usize),
    quiet_until: usize = 0, // no proposal before this context length

    fn propose(ptr: *anyopaque, context: []const u32, most: i64) anyerror![]const u32 {
        const c: *Copier = @ptrCast(@alignCast(ptr));
        if (most < 2 or context.len < c.quiet_until) return &.{};
        const history = try gpa.alloc(u32, context.len + c.out.len);
        defer gpa.free(history);
        @memcpy(history[0..context.len], context);
        const n: usize = @intCast(@min(c.out.len, most));
        for (c.out[0..n], 0..) |*t, k| {
            t.* = fake.next(history[0 .. context.len + k], null, 0);
            history[context.len + k] = t.*;
        }
        if (context.len < c.land_from) c.out[0] = (c.out[0] + 1) % fake.vocab;
        return c.out[0..n];
    }

    fn lastMatch(_: *anyopaque) i64 {
        return 100;
    }

    const vtable: Proposer.VTable = .{ .propose = propose, .last_match = lastMatch };
};

/// Copy windows the target verified (no held drafts, host tokens), before and from the window position `split`.
const Watch = struct {
    var inner: *const fn (*anyopaque, []const be.Window, []be.Verified) anyerror!void = undefined;
    var split: u64 = std.math.maxInt(u64);
    var windows: [2]u64 = @splat(0);
    var copies: [2]u64 = @splat(0);
    var copy_rows: u64 = 0;
    var shared: [3]u64 = @splat(0); // shared rounds' windows, copy windows and copy rows
    var first: ?usize = null; // the first copy window's rows

    fn verify(ptr: *anyopaque, ws: []const be.Window, out: []be.Verified) anyerror!void {
        for (ws) |w| {
            const late = @intFromBool(w.positions[0] >= split);
            windows[late] += 1;
            if (ws.len > 1) shared[0] += 1;
            if (w.held > 0 or w.tokens.len == 0) continue;
            if (first == null) first = w.tokens.len;
            copies[late] += 1;
            copy_rows += w.tokens.len;
            if (ws.len > 1) {
                shared[1] += 1;
                shared[2] += w.tokens.len;
            }
        }
        return inner(ptr, ws, out);
    }
};

const Run = struct {
    tokens: [4][160]u32 = undefined,
    lens: [4]usize = @splat(0),
    rounds: u64 = 0,
    drafted: u64 = 0,
    accepted: u64 = 0,
    windows: [2]u64 = @splat(0), // Watch's counters for this run
    copies: [2]u64 = @splat(0),
    copy_rows: u64 = 0,
    shared: [3]u64 = @splat(0),
};

const prompts = [_][]const u32{ &.{ 2, 7, 1, 8, 2, 8, 1, 8 }, &.{ 3, 1, 4, 1, 5, 9, 2, 6, 5 }, &.{ 1, 4, 1, 4, 2, 1, 3, 5 }, &.{ 1, 7, 3, 2, 0, 5, 0, 8, 1 } };
const ids = [_][]const u8{ "a", "b", "c", "d" };

/// The engine's decision trail of the next run, when set.
var trail: ?*ev.Log = null;

/// `streams` streams on one engine, each with its own copier when `copier` is set (drafts off without one).
fn run(m: cfgs.Model, times: [2]f64, copier: ?Copier, streams: usize, max_new: u32) !Run {
    var cfg = try cfgs.Config.init(gpa, m, 16, 15);
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var backend = target.backend();
    var table = backend.vtable.*;
    Watch.inner = table.verify;
    table.verify = Watch.verify;
    backend.vtable = &table;
    Watch.windows = @splat(0);
    Watch.copies = @splat(0);
    Watch.copy_rows = 0;
    Watch.shared = @splat(0);
    var clock: Clock = .{ .target = &target, .times = times };
    var e = Engine.init(gpa, &cfg, backend, clock.clock());
    defer e.deinit();
    e.log = trail;
    var copiers: [4]Copier = @splat(copier orelse .{});
    var ss: [4]sm.Stream = undefined;
    for (ss[0..streams], copiers[0..streams], prompts[0..streams], ids[0..streams]) |*s, *c, p, id| s.* = try sm.Stream.init(gpa, .{ .id = id, .prompt = p, .max_new = max_new, .drafts = copier != null, .proposer = .{ .ptr = c, .vtable = &Copier.vtable } });
    defer for (ss[0..streams]) |*s| s.deinit(gpa);
    for (ss[0..streams]) |*s| try e.addStream(s);
    while (e.activeCount() > 0) try e.step();
    var out: Run = .{ .windows = Watch.windows, .copies = Watch.copies, .copy_rows = Watch.copy_rows, .shared = Watch.shared };
    for (ss[0..streams], 0..) |*s, i| {
        out.lens[i] = s.emitted().len;
        @memcpy(out.tokens[i][0..out.lens[i]], s.emitted());
        out.rounds += s.rounds;
        out.drafted += s.drafted;
        out.accepted += s.accepted;
    }
    return out;
}

/// hip_lanes.zig facts in small: 16-row windows, timed forwards; plain rounds compete with a head, lone copies always.
fn hip(head: bool) cfgs.Model {
    return .{ .exact_width = 16, .mtp = head, .speculate = head, .speculate_early = false, .drafts = 4, .plain_guard = head, .copy_guard = true, .window_costs = &startup, .mtp_step_ms = if (head) 0.5 else 0, .hidden_rows = true, .batch_rows = 32, .max_streams = 4 };
}

fn expectPlain(m: cfgs.Model, times: [2]f64, got: Run, streams: usize, max_new: u32) !void {
    const plain = try run(m, times, null, streams, max_new);
    for (0..streams) |i| try std.testing.expectEqualSlices(u32, plain.tokens[i][0..plain.lens[i]], got.tokens[i][0..got.lens[i]]);
}

test "lone copy price: copies that lose at a long context are priced off, the head drafts, the reply as plain decode" {
    const got = try run(hip(true), long_context, .{}, 1, 96);
    try expectPlain(hip(true), long_context, got, 1, 96);
    try std.testing.expect(got.copy_rows < got.lens[0]); // a few copies measure them, then a probe now and then
    try std.testing.expect(got.drafted - got.copy_rows > got.lens[0] / 2); // the head's drafts take their rounds
}

test "lone copy price: a copies-only stream times its rounds and prices its copies off the same way" {
    const got = try run(hip(false), long_context, .{}, 1, 96);
    try expectPlain(hip(false), long_context, got, 1, 96);
    try std.testing.expect(got.copy_rows < got.lens[0]);
    try std.testing.expect(got.copies[0] * 4 < got.windows[0]);
}

test "lone copy price: copies priced off come back after a backoff once they land" {
    for ([_]bool{ true, false }) |head| {
        const land_from = prompts[0].len + 40;
        Watch.split = land_from;
        defer Watch.split = std.math.maxInt(u64);
        const got = try run(hip(head), short_context, .{ .land_from = land_from }, 1, 144);
        try expectPlain(hip(head), short_context, got, 1, 144);
        try std.testing.expect(got.windows[0] >= got.copies[0] + 8); // missing copies left off for a backoff's wait at least
        try std.testing.expect(got.windows[1] * 6 < got.lens[0] - 40); // landing ones carry the rest, many rows a round
    }
}

test "lone copy price: at a short context copies that land stay whole" {
    for ([_]bool{ true, false }) |head| {
        const got = try run(hip(head), short_context, .{ .land_from = 0 }, 1, 96);
        try expectPlain(hip(head), short_context, got, 1, 96);
        try std.testing.expect(got.rounds * 8 < got.lens[0]);
        try std.testing.expect(got.copy_rows + 15 >= 15 * (got.copies[0] + got.copies[1]));
    }
}

/// What a run leaves that pricing could move: rounds, drafted and kept rows, copy windows and their rows.
fn digest(got: Run) [5]u64 {
    return .{ got.rounds, got.drafted, got.accepted, got.copies[0] + got.copies[1], got.copy_rows };
}

/// Values from main e7e24243d, before lone copies were priced (the same configurations, clocks and copier).
fn pinned(expected: [5]u64, got: Run) !void {
    try std.testing.expectEqual(expected, digest(got));
}

test "lone copy price: head windows without a copy proposed keep main's depths" {
    var silent = hip(true);
    silent.copy_drafts = false; // no copy is ever proposed
    try pinned(.{ 40, 82, 55, 0, 0 }, try run(silent, long_context, .{}, 1, 96));
    try pinned(.{ 39, 138, 56, 0, 0 }, try run(silent, short_context, .{}, 1, 96));
}

test "lone copy price: root copies and GLM's own copy rates stay as on main" {
    var root = hip(true);
    root.root_copies = true;
    try pinned(.{ 52, 424, 43, 26, 365 }, try run(root, long_context, .{}, 1, 96));
    var glm = root; // lanes/../glm/cuda_copy_policy.zig: root and context copies priced by each stream's own rate
    glm.context_copies = true;
    glm.own_copy_rates = true;
    glm.copy_match = 4;
    try pinned(.{ 28, 255, 67, 16, 229 }, try run(glm, long_context, .{ .land_from = prompts[0].len + 40 }, 1, 96));
}

test "lone copy price: context copies on a lone head window stay as on main" {
    var context = hip(true);
    context.context_copies = true;
    try pinned(.{ 38, 562, 57, 38, 562 }, try run(context, long_context, .{ .land_from = prompts[0].len + 40 }, 1, 96));
}

test "lone copy price: shared rounds' copies stay as on main (the streams' lone rounds after one ends excluded)" {
    // main e7e24243d: shared rounds' windows, copy windows and copy rows
    for ([_]bool{ true, false }, [_][3]u64{ .{ 124, 122, 1644 }, .{ 126, 122, 1644 } }) |head, expected| {
        const got = try run(hip(head), long_context, .{}, 2, 64);
        try std.testing.expectEqual(expected, got.shared);
    }
}

/// A hash of the trail's lines up to the first step without a shared round, and how many lines that is.
fn sharedPhase(log: *ev.Log) ![2]u64 {
    var end: usize = log.events.items.len;
    var step: ?usize = null;
    var turns = false;
    for (log.events.items, 0..) |fields, i| {
        const kind = for (fields) |x| {
            if (std.mem.eql(u8, x.k, "ev")) break x.v.str;
        } else "";
        if (std.mem.eql(u8, kind, "turns")) turns = true;
        if (!std.mem.eql(u8, kind, "step")) continue;
        if (step) |at| if (!turns) {
            end = at;
            break;
        };
        step = i;
        turns = false;
    } else if (step) |at| if (!turns) {
        end = at; // the run's last step was a lone one
    };
    return hashed(log, end);
}

/// A hash of the trail's first `end` lines, and `end`.
fn hashed(log: *ev.Log, end: usize) ![2]u64 {
    var h = std.hash.Wyhash.init(0);
    for (0..end) |i| {
        const text = try log.line(gpa, i);
        defer gpa.free(text);
        h.update(text);
        h.update("\n");
    }
    return .{ h.final(), end };
}

test "lone copy price: HIP's facts without a head keep main's shared rounds, two and four streams" {
    // main e7e24243d, its hip_lanes.zig facts without a head (no plain_guard): the shared phase's trail, hashed
    const cases = [_]struct { streams: usize, copier: Copier, expected: [2]u64 }{
        .{ .streams = 2, .copier = .{}, .expected = .{ 12160614949009965908, 1276 } },
        .{ .streams = 2, .copier = .{ .land_from = 48 }, .expected = .{ 612887639395430184, 445 } },
        .{ .streams = 4, .copier = .{}, .expected = .{ 5045447432178392522, 1790 } },
        .{ .streams = 4, .copier = .{ .land_from = 80 }, .expected = .{ 5351245168654378449, 1104 } },
    };
    for (cases) |c| {
        var log = ev.Log.init(gpa);
        defer log.deinit();
        trail = &log;
        defer trail = null;
        _ = try run(hip(false), long_context, c.copier, c.streams, 128);
        const got = try sharedPhase(&log);
        try std.testing.expectEqual(c.expected, got);
    }
}

test "lone copy price: copy_guard follows plain_guard unless a family sets it" {
    for ([_]bool{ false, true }) |plain| for ([_]?bool{ null, false, true }) |copy| {
        var cfg = try cfgs.Config.init(gpa, .{ .exact_width = 16, .plain_guard = plain, .copy_guard = copy }, 16, 15);
        defer cfg.deinit(gpa);
        try std.testing.expectEqual(plain, cfg.plain_guard);
        try std.testing.expectEqual(copy orelse plain, cfg.copy_guard);
    };
}

test "lone copy price: a head without plain_guard prices copies against its own rounds, not the load-time forward" {
    var m = hip(true);
    m.plain_guard = false; // copy_guard alone: the head's depths keep the rule's greedy rounds, plain is never timed
    const got = try run(m, long_context, .{ .land_from = 0 }, 1, 96);
    try expectPlain(m, long_context, got, 1, 96);
    try std.testing.expect(got.rounds * 8 < got.lens[0]); // landing copies stay on, many rows a round
}

test "lone copy price: copyAhead reads the price the plan takes, backoff probes included, and counts nothing" {
    var cfg = try cfgs.Config.init(gpa, hip(true), 16, 15);
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: Clock = .{ .target = &target, .times = long_context };
    var e = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer e.deinit();
    var c: Copier = .{};
    var s = try sm.Stream.init(gpa, .{ .id = "a", .prompt = prompts[0], .max_new = 96, .proposer = .{ .ptr = &c, .vtable = &Copier.vtable } });
    defer s.deinit(gpa);
    // a 90 ms plain round and a 15-row copy of 690 ms at a long context, after eight copies that missed
    try e.rule.observeCost(0, 90, false, win.who(&s));
    try copy_price.observeCopy(&e.rule, 15, 690, false, win.who(&s));
    for (0..8) |_| win.observeCopied(&e, &s, false, 0, 15);
    var probes: [2]usize = .{ 0, 0 };
    var found: usize = 0;
    for (1..41) |take| {
        _ = e.arena.reset(.retain_capacity);
        const w = win.who(&s);
        const ahead = win.copyAhead(&e, &w);
        try std.testing.expectEqual(ahead, win.copyAhead(&e, &w)); // a second read moves nothing
        const kept = try lone_copy.rows(&e, &s, try win.copyProposal(&e, &s), true);
        try std.testing.expectEqual(ahead, kept > 0);
        if (kept == 0) continue;
        try std.testing.expect(found < probes.len);
        probes[found] = take;
        found += 1;
    }
    try std.testing.expectEqualSlices(usize, &.{ 8, 24 }, &probes); // after 8, then 16 more, priced-off copies
}

/// A request's reply, its first copy window's rows, and its copy windows and rows.
const Request = struct { tokens: [160]u32 = undefined, len: usize = 0, first: ?usize = null, copies: u64 = 0, copy_rows: u64 = 0 };

/// A request `serve` sends: its prompt, the tokens its copier stays quiet for, its reply's length and its round times.
const Ask = struct { prompt: []const u32, quiet: usize, max_new: u32, times: [2]f64, drafts: bool = true };

/// One engine serving `asks` as lone requests one after another, each with its own copier and clock.
fn serve(m: cfgs.Model, asks: []const Ask, out: []Request) !void {
    var cfg = try cfgs.Config.init(gpa, m, 16, 15);
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var backend = target.backend();
    var table = backend.vtable.*;
    Watch.inner = table.verify;
    table.verify = Watch.verify;
    backend.vtable = &table;
    var clock: Clock = .{ .target = &target, .times = asks[0].times };
    var e = Engine.init(gpa, &cfg, backend, clock.clock());
    defer e.deinit();
    e.log = trail;
    for (asks, out, ids[0..asks.len]) |ask, *r, id| {
        clock.times = ask.times;
        Watch.copies = @splat(0);
        Watch.copy_rows = 0;
        Watch.first = null;
        var c: Copier = .{ .land_from = 0, .quiet_until = ask.prompt.len + ask.quiet };
        var s = try sm.Stream.init(gpa, .{ .id = id, .prompt = ask.prompt, .max_new = ask.max_new, .drafts = ask.drafts, .proposer = .{ .ptr = &c, .vtable = &Copier.vtable } });
        defer s.deinit(gpa);
        try e.addStream(&s);
        while (e.activeCount() > 0) try e.step();
        r.len = s.emitted().len;
        @memcpy(r.tokens[0..r.len], s.emitted());
        r.first = Watch.first;
        r.copies = Watch.copies[0] + Watch.copies[1];
        r.copy_rows = Watch.copy_rows;
    }
}

/// `out.len` requests of `prompts` in order on one engine, each copier quiet for its first `quiet` tokens.
fn sequential(m: cfgs.Model, times: [2]f64, quiet: usize, out: []Request, max_new: u32) !void {
    var asks: [prompts.len]Ask = undefined;
    for (asks[0..out.len], prompts[0..out.len]) |*ask, p| ask.* = .{ .prompt = p, .quiet = quiet, .max_new = max_new, .times = times };
    return serve(m, asks[0..out.len], out);
}

test "lone copy price: a request's first copies are priced from the copy rounds the request before it timed" {
    for ([_]bool{ false, true }) |head| {
        var got: [2]Request = @splat(.{});
        try sequential(hip(head), short_context, 4, &got, 64);
        const plain = try run(hip(head), short_context, null, 2, 64);
        for (got, 0..) |r, i| try std.testing.expectEqualSlices(u32, plain.tokens[i][0..plain.lens[i]], r.tokens[0..r.len]);
        // the second request starts from the first's timed copies: its first copy is whole, as the first's later ones
        try std.testing.expectEqual(@as(?usize, 15), got[1].first);
        // without a head the first request timed its plain rounds before any copy: the untimed price cut its first one
        if (!head) try std.testing.expect(got[0].first.? < 15);
    }
}

test "lone copy price: kept copy prices need two timed copies on a lone stream, leave a head's times, keep a stream's own" {
    for ([_]bool{ false, true }) |head| {
        var cfg = try cfgs.Config.init(gpa, hip(head), 16, 15);
        defer cfg.deinit(gpa);
        var target: fake.Fake = .{ .gpa = gpa };
        defer target.deinit();
        var clock: Clock = .{ .target = &target, .times = short_context };
        var e = Engine.init(gpa, &cfg, target.backend(), clock.clock());
        defer e.deinit();
        var done = try sm.Stream.init(gpa, .{ .id = "a", .prompt = prompts[0], .max_new = 8 });
        defer done.deinit(gpa);
        try copy_price.observeCopy(&e.rule, 15, 11.6, false, win.who(&done));
        try copy_price.observeOwn(&e.rule, 0, 10.1, false, win.who(&done));
        done.depth.?.copy_band = 0; // as its first priced round left it (lone_copy.zig's seed)
        lone_copy.keep(&e, &done);
        try std.testing.expect(e.lone_kept[0][0] == null); // one timed copy round, as a warm-up's few tokens: nothing kept
        try copy_price.observeCopy(&e.rule, 15, 11.6, false, win.who(&done));
        done.shared_round = true;
        lone_copy.keep(&e, &done);
        try std.testing.expect(e.lone_kept[0][0] == null); // a round beside another stream: its times were not its own
        done.shared_round = false;
        lone_copy.keep(&e, &done);
        var next = try sm.Stream.init(gpa, .{ .id = "b", .prompt = prompts[1], .max_new = 8 });
        defer next.deinit(gpa);
        _ = try e.rule.rates(win.who(&next));
        try lone_costs.seedCopies(gpa, lone_costs.slot(&e.lone_kept, &next).*, &next);
        try std.testing.expectEqual(@as(?f64, 11.6), next.depth.?.copy_ms.?.get(15));
        // a head's own round times are its depth rule's, never carried by copy pricing
        try std.testing.expectEqual(@as(?f64, if (head) null else 10.1), if (next.depth.?.ms) |t| t.get(0) else null);
        const long = try gpa.alloc(u32, 8192);
        defer gpa.free(long);
        @memset(long, 1);
        var far = try sm.Stream.init(gpa, .{ .id = "c", .prompt = long, .max_new = 8 });
        defer far.deinit(gpa);
        _ = try e.rule.rates(win.who(&far));
        try lone_costs.seedCopies(gpa, lone_costs.slot(&e.lone_kept, &far).*, &far);
        try std.testing.expect(far.depth.?.copy_ms == null); // another context band starts unpriced
        var timed = try sm.Stream.init(gpa, .{ .id = "d", .prompt = prompts[2], .max_new = 8 });
        defer timed.deinit(gpa);
        try copy_price.observeCopy(&e.rule, 15, 30, false, win.who(&timed));
        try lone_costs.seedCopies(gpa, lone_costs.slot(&e.lone_kept, &timed).*, &timed);
        try std.testing.expectEqual(@as(?f64, 30), timed.depth.?.copy_ms.?.get(15)); // its own copy times stay
        try std.testing.expectEqual(@as(?f64, if (head) null else 10.1), if (timed.depth.?.ms) |t| t.get(0) else null);
    }
}

test "lone copy price: engines that leave copies unpriced decide two requests in a row as before" {
    // e5478c4ab: the trail of two requests in a row on one engine, hashed, where the copy gate is off
    var tile = hip(true);
    tile.lone_cost_tile = 2; // CUDA's lone window prices, which keep their own calibration (lanes/lone_costs.zig)
    var head = hip(true);
    head.copy_guard = false;
    var copies = hip(false);
    copies.copy_guard = false;
    const cases = [_]struct { m: cfgs.Model, expected: [2]u64 }{
        .{ .m = tile, .expected = .{ 14840352178686162370, 75 } },
        .{ .m = head, .expected = .{ 12942564931587194512, 69 } },
        .{ .m = copies, .expected = .{ 14652143946053147409, 62 } },
    };
    for (cases) |c| {
        var log = ev.Log.init(gpa);
        defer log.deinit();
        trail = &log;
        defer trail = null;
        var got: [2]Request = @splat(.{});
        try sequential(c.m, short_context, 4, &got, 64);
        try std.testing.expectEqual(c.expected, try hashed(&log, log.events.items.len));
    }
}

test "lone copy price: a request reads the copy prices of the one before that started in its band, not a short warm-up's" {
    // a warm-up timing one copy at a long context's round times, then two from 8,150 tokens past 8,192 at a short one's
    const long = try gpa.alloc(u32, 8150);
    defer gpa.free(long);
    for (long, 0..) |*t, i| t.* = @intCast(i % fake.vocab);
    for ([_]bool{ false, true }) |head| {
        var asks = [_]Ask{
            .{ .prompt = prompts[0], .quiet = 2, .max_new = 8, .times = long_context },
            .{ .prompt = long, .quiet = 4, .max_new = 64, .times = short_context },
            .{ .prompt = long, .quiet = 4, .max_new = 64, .times = short_context },
        };
        var got: [3]Request = @splat(.{});
        try serve(hip(head), &asks, &got);
        try std.testing.expectEqual(@as(u64, 1), got[0].copies);
        var alone: [2]Request = @splat(.{});
        try serve(hip(head), asks[1..], &alone);
        for (&asks) |*ask| ask.drafts = false;
        var plain: [3]Request = @splat(.{});
        try serve(hip(head), &asks, &plain);
        for (got, plain) |r, p| try std.testing.expectEqualSlices(u32, p.tokens[0..p.len], r.tokens[0..r.len]);
        // the warm-up's one timed copy replaces nothing: the long replies decide as on an engine that never served it
        for (got[1..], alone) |r, x| try std.testing.expectEqual([3]u64{ x.first orelse 0, x.copies, x.copy_rows }, [3]u64{ r.first orelse 0, r.copies, r.copy_rows });
        // the second long reply starts from the first's timed copies: its first copy whole, as the first's later ones
        try std.testing.expectEqual(@as(?usize, 15), got[2].first);
    }
}

test "lone copy price: a head stream's copy rounds are timed by their verify, without the head's next drafting" {
    var cfg = try cfgs.Config.init(gpa, hip(true), 16, 15);
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: Clock = .{ .target = &target, .times = short_context, .draft_ms = 30 }; // a slow head: 30 ms a drafting
    var e = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer e.deinit();
    var c: Copier = .{ .land_from = 0, .quiet_until = prompts[0].len + 4 };
    var s = try sm.Stream.init(gpa, .{ .id = "a", .prompt = prompts[0], .max_new = 64, .proposer = .{ .ptr = &c, .vtable = &Copier.vtable } });
    defer s.deinit(gpa);
    try e.addStream(&s);
    while (e.activeCount() > 0) try e.step();
    try std.testing.expect(target.drafts_calls > 0);
    // the kept copy prices hold each whole copy's verify time, 10 ms plus 0.1 ms a row, and none of the drafting
    const kept = e.lone_kept[0][lone_costs.band(prompts[0].len)] orelse return error.NothingKept;
    try std.testing.expectEqual(@as(?f64, short_context[0] + short_context[1] * 16), kept.copy_ms.get(15));
}

/// A dense model's forwards: a row costs 11.6 ms past a 65 ms round, so a 15-row copy costs about three plain rounds.
const dense = blk: {
    var c: [16]cfgs.Cost = undefined;
    for (&c, 1..) |*x, w| x.* = .{ .width = w, .ms = 65 + 11.6 * @as(f64, @floatFromInt(w)) };
    break :blk c;
};

test "lone copy price: a request's first copy is priced at the landing rate the requests before it saw" {
    var m = hip(false);
    m.window_costs = &dense;
    var asks = [_]Ask{
        .{ .prompt = prompts[0], .quiet = 0, .max_new = 64, .times = .{ 65, 11.6 } },
        .{ .prompt = prompts[1], .quiet = 0, .max_new = 64, .times = .{ 65, 11.6 } },
    };
    var got: [2]Request = @splat(.{});
    try serve(m, &asks, &got);
    for (&asks) |*ask| ask.drafts = false;
    var plain: [2]Request = @splat(.{});
    try serve(m, &asks, &plain);
    for (got, plain) |r, p| try std.testing.expectEqualSlices(u32, p.tokens[0..p.len], r.tokens[0..r.len]);
    // the first request's first copy meets the prior's landing rate: its expected rows stop paying for a full copy
    try std.testing.expect(got[0].first.? < 15);
    // the second starts at the rate the first's landed copies taught: its first copy is whole
    try std.testing.expectEqual(@as(?usize, 15), got[1].first);
}
