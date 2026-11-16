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

    fn clock(c: *Clock) be.Clock {
        return .{ .ptr = c, .vtable = &.{ .start = start, .elapsed_ms = elapsed } };
    }

    fn start(_: *anyopaque) void {}

    fn elapsed(ptr: *anyopaque, _: be.Mark) f64 {
        const c: *Clock = @ptrCast(@alignCast(ptr));
        return c.times[0] + c.times[1] * @as(f64, @floatFromInt(c.target.last_rows));
    }
};

/// The target's next 15 tokens at every ask: from context length `land_from` on they land, before it the first misses.
const Copier = struct {
    out: [15]u32 = undefined,
    land_from: usize = std.math.maxInt(usize),

    fn propose(ptr: *anyopaque, context: []const u32, most: i64) anyerror![]const u32 {
        const c: *Copier = @ptrCast(@alignCast(ptr));
        if (most < 2) return &.{};
        var history: [256]u32 = undefined;
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

    fn verify(ptr: *anyopaque, ws: []const be.Window, out: []be.Verified) anyerror!void {
        for (ws) |w| {
            const late = @intFromBool(w.positions[0] >= split);
            windows[late] += 1;
            if (ws.len > 1) shared[0] += 1;
            if (w.held > 0 or w.tokens.len == 0) continue;
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
