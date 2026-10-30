//! Live single-stream prices must charge an extra target tile only when its accepted tokens repay it.
const std = @import("std");
const Config = @import("config.zig").Config;
const Engine = @import("engine.zig").Engine;
const Stream = @import("stream.zig").Stream;
const fake = @import("fake.zig");
const win = @import("windows.zig");
const a = std.testing.allocator;

fn setup() !Config {
    return Config.init(a, .{ .exact_width = 64, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 63, .plain_guard = true, .draft_probabilities = true, .hidden_rows = true, .batch_rows = 128, .window_costs = &.{ .{ .width = 1, .ms = 80 }, .{ .width = 16, .ms = 80 }, .{ .width = 17, .ms = 81 }, .{ .width = 64, .ms = 90 } }, .lone_cost_tile = 16 }, 64, 63);
}

fn selected(wide_ms: f64, alone: bool, timed: bool, neighbor_ms: ?f64) !usize {
    var cfg = try setup();
    defer cfg.deinit(a);
    cfg.timed = timed;
    var target: fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var e = Engine.init(a, &cfg, target.backend(), clock.clock());
    defer e.deinit();
    e.alone = alone;
    var s = try Stream.init(a, .{ .id = "own-width", .prompt = &.{ 3, 1, 4 }, .max_new = 128 });
    defer s.deinit(a);
    // Near-certain prefixes are synthetic controls; the two round prices are supplied timing witnesses.
    _ = try e.rule.rates(win.who(&s));
    @memset(s.depth.?.p, 0.99);
    try e.rule.observeCost(15, 94.5, false, win.who(&s));
    try e.rule.observeCost(16, wide_ms, false, win.who(&s));
    if (neighbor_ms) |ms| {
        var other = try Stream.init(a, .{ .id = "neighbor", .prompt = &.{7}, .max_new = 128 });
        defer other.deinit(a);
        try e.rule.observeCost(16, ms, false, win.who(&other));
    }
    const tokens: [16]u32 = @splat(7);
    var plans = [_]win.Plan{.{ .stream = &s, .position = 3, .kind = .head, .tokens = &tokens }};
    try win.allocate(&e, &plans);
    return plans[0].count();
}

test "lone price: a marginal second tile loses on actual round time, but wins when it pays" {
    try std.testing.expectEqual(@as(usize, 15), try selected(107.5, true, true, null));
    try std.testing.expectEqual(@as(usize, 16), try selected(95, true, true, null));
}

test "lone price: another stream's measurements cannot change this stream's width" {
    try std.testing.expectEqual(@as(usize, 15), try selected(107.5, true, true, 1));
    try std.testing.expectEqual(@as(usize, 15), try selected(107.5, true, true, 1000));
}

test "lone price: untimed and shared allocation retain the startup cost policy" {
    try std.testing.expectEqual(@as(usize, 16), try selected(107.5, false, true, null));
    try std.testing.expectEqual(@as(usize, 16), try selected(107.5, true, false, null));
}

const Table = @import("table.zig").Table;
const own = @import("lone_costs.zig");

test "lone price: unseen widths retain measured tile jumps and full rounds are not charged twice" {
    var startup: Table = .{};
    defer startup.deinit(a);
    var samples: Table = .{};
    defer samples.deinit(a);
    try startup.put(a, 15, 80);
    try startup.put(a, 16, 80);
    try startup.put(a, 17, 100);
    try samples.put(a, 15, 94.5);
    try std.testing.expectEqual(@as(?f64, 94.5), own.estimate(startup, samples, 16, 200));
    try std.testing.expectEqual(@as(?f64, 94.5), own.estimate(startup, samples, 15, 200));
    try std.testing.expectEqual(@as(?f64, 114.5), own.estimate(startup, samples, 17, 200));
    for ([_]f64{ 0, -1, std.math.nan(f64), std.math.inf(f64) }) |bad| {
        try samples.put(a, 16, bad);
        try std.testing.expectEqual(@as(?f64, 114.5), own.estimate(startup, samples, 17, 200));
    }
}

test "lone price: tile probes are bounded, respect available rows, and stop after calibration" {
    var cfg = try setup();
    defer cfg.deinit(a);
    var target: fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var e = Engine.init(a, &cfg, target.backend(), clock.clock());
    defer e.deinit();
    var s = try Stream.init(a, .{ .id = "probe-widths", .prompt = &.{3}, .max_new = 128 });
    defer s.deinit(a);
    const tokens: [63]u32 = @splat(7);
    var p: win.Plan = .{ .stream = &s, .position = 1, .kind = .head, .tokens = &tokens };
    for ([_]u32{ 0, 15, 31, 47, 63 }) |count| {
        s.rounds += 8;
        try std.testing.expectEqual(@as(?u32, count), own.probe(&e, p));
        try e.rule.observeCost(count, 100, false, win.who(&s));
        own.observe(&s, count + 1, count + 1, 100);
        s.rounds += 1;
        try std.testing.expectEqual(@as(?u32, null), own.probe(&e, p));
        s.rounds -= 1;
    }
    s.rounds += 8;
    try std.testing.expectEqual(@as(?u32, null), own.probe(&e, p));
    p.tokens = tokens[0..16];
    try std.testing.expectEqual(@as(?u32, 16), own.probe(&e, p));
    try e.rule.observeCost(16, 100, false, win.who(&s));
    own.observe(&s, 17, 17, 100);
    try std.testing.expectEqual(@as(?u32, null), own.probe(&e, p));
}

const be = @import("backend.zig");
const Clock = struct {
    target: *const fake.Fake,
    fn start(_: *anyopaque) void {}
    fn elapsed(ptr: *anyopaque, _: be.Mark) f64 {
        const c: *@This() = @ptrCast(@alignCast(ptr));
        return if (c.target.last_rows <= 16) 94.5 else 107.5;
    }
    fn clock(c: *@This()) be.Clock {
        return .{ .ptr = c, .vtable = &.{ .start = start, .elapsed_ms = elapsed } };
    }
};

fn emitted(drafted: bool, live: bool, cohort: usize, seed: ?u64, restored: usize) ![][]u32 {
    var cfg = try setup();
    defer cfg.deinit(a);
    cfg.lone_cost_tile = if (live) 16 else 0;
    var target: fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var clock: Clock = .{ .target = &target };
    var e = Engine.init(a, &cfg, target.backend(), clock.clock());
    defer e.deinit();
    const prompt = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5 };
    var saved: std.ArrayList(u32) = .empty;
    defer saved.deinit(a);
    try saved.appendSlice(a, prompt[0..restored]);
    const streams = try a.alloc(Stream, cohort);
    defer a.free(streams);
    var made: usize = 0;
    defer for (streams[0..made]) |*s| s.deinit(a);
    for (streams, 0..) |*s, i| {
        s.* = try Stream.init(a, .{ .id = "equal-widths", .prompt = &prompt, .max_new = 96, .drafts = drafted, .sampling = if (seed) |k| .{ .seed = k + i, .temperature = 0.8 } else null, .reuse = .{ .saved = if (restored > 0) &saved else null, .at = @intCast(restored) } });
        made += 1;
        try e.addStream(s);
    }
    var steps: usize = 0;
    var learned = false;
    var scheduled = false;
    while (e.activeCount() > 0 and steps < 4096) : (steps += 1) {
        try e.step();
        for (streams) |s| scheduled = scheduled or s.cost_probe_after > 8;
        for (streams) |s| if (s.depth) |state| {
            if (state.ms) |ms| {
                learned = learned or !ms.empty();
            }
        };
    }
    if (drafted and live and cohort == 1) {
        try std.testing.expect(learned);
        try std.testing.expect(scheduled);
    }
    try std.testing.expectEqual(@as(usize, 0), e.activeCount());
    const out = try a.alloc([]u32, cohort);
    for (out, streams) |*o, *s| {
        o.* = try a.dupe(u32, s.emitted());
    }
    return out;
}

fn free(out: [][]u32) void {
    for (out) |x| a.free(x);
    a.free(out);
}

test "lone price: changing widths keeps drafted, plain, resumed, fresh and concurrent outputs exact" {
    for ([_]?u64{ null, 41 }) |seed| for ([_]bool{ false, true }) |live| {
        for ([_]usize{ 0, 5 }) |restored| for ([_]usize{ 1, 2, 4 }) |cohort| {
            const actual = try emitted(true, live, cohort, seed, restored);
            defer free(actual);
            for (actual, 0..) |tokens, i| {
                const plain = try emitted(false, false, 1, if (seed) |k| k + i else null, 0);
                defer free(plain);
                try std.testing.expectEqualSlices(u32, plain[0], tokens);
            }
        };
    };
}

test "lone price: an explicit head-lane shape and confidence override keep their own policy" {
    var cfg = try setup();
    defer cfg.deinit(a);
    var target: fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var e = Engine.init(a, &cfg, target.backend(), clock.clock());
    defer e.deinit();
    e.alone = true;
    var s = try Stream.init(a, .{ .id = "owned-policy", .prompt = &.{3}, .max_new = 128 });
    defer s.deinit(a);
    var p: win.Plan = .{ .stream = &s, .position = 1, .kind = .head, .tokens = &.{7} };
    try std.testing.expect(own.enabled(&e, p));
    const shape = @import("shape.zig");
    const l = try shape.best(a, shape.Odds.init(shape.start_odds, shape.start_weight), 4, 4);
    defer l.deinit(a);
    p.lanes = &l;
    try std.testing.expect(!own.enabled(&e, p));
    p.lanes = null;
    cfg.draft_confidence = 0.7;
    try std.testing.expect(!own.enabled(&e, p));
}

test "lone price: varying offers cannot phase-lock a profitable unseen width out of calibration" {
    var cfg = try setup();
    defer cfg.deinit(a);
    for (1..65) |rows| try cfg.shared_costs.put(a, @intCast(rows), if (rows <= 16) 10 else 32);
    var target: fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var e = Engine.init(a, &cfg, target.backend(), clock.clock());
    defer e.deinit();
    e.alone = true;
    var s = try Stream.init(a, .{ .id = "variable-offers", .prompt = &.{3}, .max_new = 8192 });
    defer s.deinit(a);
    _ = try e.rule.rates(win.who(&s));
    @memset(s.depth.?.p, 1);
    try e.rule.observeCost(0, 20, false, win.who(&s));
    try e.rule.observeCost(15, 20, false, win.who(&s));
    const tokens: [31]u32 = @splat(7);
    var wide_offers: usize = 0;
    var wide_trials: usize = 0;
    for (1..129) |completed| {
        s.rounds = completed;
        const rows: usize = if (completed % 8 == 0) 16 else 32;
        wide_offers += @intFromBool(rows == 32);
        var plans = [_]win.Plan{.{ .stream = &s, .position = 1, .kind = .head, .tokens = tokens[0 .. rows - 1] }};
        try win.allocate(&e, &plans);
        const actual = plans[0].count() + 1;
        wide_trials += @intFromBool(actual == 32);
        try e.rule.observeCost(@intCast(actual - 1), if (actual == 32) 35 else 20, false, win.who(&s));
        s.rounds += 1;
        own.observe(&s, plans[0].cost_probe_rows, actual, if (actual == 32) 35 else 20);
    }
    try std.testing.expectEqual(@as(usize, 112), wide_offers);
    try std.testing.expectEqual(@as(usize, 105), wide_trials);
    try std.testing.expect(s.depth.?.ms.?.get(31) != null);
}

test "lone price: probe debt survives short offers and unexecuted selections, then waits eight completed rounds" {
    var cfg = try setup();
    defer cfg.deinit(a);
    for (1..65) |rows| try cfg.shared_costs.put(a, @intCast(rows), if (rows <= 16) 10 else 32);
    var target: fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var e = Engine.init(a, &cfg, target.backend(), clock.clock());
    defer e.deinit();
    e.alone = true;
    var s = try Stream.init(a, .{ .id = "probe-debt", .prompt = &.{3}, .max_new = 8192 });
    defer s.deinit(a);
    _ = try e.rule.rates(win.who(&s));
    @memset(s.depth.?.p, 1);
    try e.rule.observeCost(0, 20, false, win.who(&s));
    try e.rule.observeCost(15, 20, false, win.who(&s));
    const tokens: [47]u32 = @splat(7);
    s.rounds = 8;
    var short = [_]win.Plan{.{ .stream = &s, .position = 1, .kind = .head, .tokens = tokens[0..15] }};
    try win.allocate(&e, &short);
    try std.testing.expectEqual(@as(?u32, null), short[0].cost_probe_rows);
    try std.testing.expectEqual(@as(u64, 8), s.cost_probe_after);
    s.rounds = 9;
    var wide = [_]win.Plan{.{ .stream = &s, .position = 1, .kind = .head, .tokens = tokens[0..31] }};
    try win.allocate(&e, &wide);
    try std.testing.expectEqual(@as(?u32, 32), wide[0].cost_probe_rows);
    try std.testing.expectEqual(@as(u64, 8), s.cost_probe_after);
    own.observe(&s, wide[0].cost_probe_rows, 16, 20);
    try std.testing.expectEqual(@as(u64, 8), s.cost_probe_after);
    try e.rule.observeCost(31, 35, false, win.who(&s));
    s.rounds = 10;
    own.observe(&s, wide[0].cost_probe_rows, 32, 35);
    try std.testing.expectEqual(@as(u64, 18), s.cost_probe_after);
    const next: win.Plan = .{ .stream = &s, .position = 1, .kind = .head, .tokens = &tokens };
    s.rounds = 17;
    try std.testing.expectEqual(@as(?u32, null), own.probe(&e, next));
    s.rounds = 18;
    try std.testing.expectEqual(@as(?u32, 47), own.probe(&e, next));
    own.observe(&s, null, 48, 100);
    try std.testing.expectEqual(@as(u64, 18), s.cost_probe_after);
}

fn measured(e: *Engine, s: *Stream, rounds: u64) !void {
    _ = try e.rule.rates(win.who(s));
    try e.rule.observeCost(15, 94.5, false, win.who(s));
    try e.rule.observeCost(16, 107.5, false, win.who(s));
    s.rounds = rounds;
    s.cost_probe_after = rounds + 40;
    s.cost_probe_spacing = 64;
    s.cost_probes[16] = .{ .after = rounds + 100, .spacing = 128, .retry = true };
}

test "lone price: a finished lone stream's prices and backed-off deadlines start the next stream" {
    var cfg = try setup();
    defer cfg.deinit(a);
    var target: fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var e = Engine.init(a, &cfg, target.backend(), clock.clock());
    defer e.deinit();
    var done = try Stream.init(a, .{ .id = "done", .prompt = &.{ 3, 1, 4 }, .max_new = 128 });
    defer done.deinit(a);
    try measured(&e, &done, 60);
    try own.keep(a, &e.lone_kept[0][0], &done);
    var next = try Stream.init(a, .{ .id = "next", .prompt = &.{ 2, 7 }, .max_new = 128 });
    defer next.deinit(a);
    next.rounds = 2;
    _ = try e.rule.rates(win.who(&next));
    try own.seed(a, e.lone_kept[0][0], &next);
    try std.testing.expectEqual(@as(?f64, 94.5), next.depth.?.ms.?.get(15));
    try std.testing.expectEqual(@as(?f64, 107.5), next.depth.?.ms.?.get(16));
    try std.testing.expectEqual(@as(u64, 42), next.cost_probe_after);
    try std.testing.expectEqual(@as(u64, 64), next.cost_probe_spacing);
    try std.testing.expectEqual(@as(u64, 102), next.cost_probes[16].after);
    try std.testing.expectEqual(@as(u64, 128), next.cost_probes[16].spacing);
    try std.testing.expect(next.cost_probes[16].retry);
}

test "lone price: a stream that shared a round leaves nothing, and a priced stream keeps its own" {
    var cfg = try setup();
    defer cfg.deinit(a);
    var target: fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var e = Engine.init(a, &cfg, target.backend(), clock.clock());
    defer e.deinit();
    var shared = try Stream.init(a, .{ .id = "shared", .prompt = &.{1}, .max_new = 128 });
    defer shared.deinit(a);
    try measured(&e, &shared, 30);
    shared.shared_round = true;
    try own.keep(a, &e.lone_kept[0][0], &shared);
    try std.testing.expect(e.lone_kept[0][0] == null);
    shared.shared_round = false;
    try own.keep(a, &e.lone_kept[0][0], &shared);
    var priced = try Stream.init(a, .{ .id = "priced", .prompt = &.{2}, .max_new = 128 });
    defer priced.deinit(a);
    _ = try e.rule.rates(win.who(&priced));
    try e.rule.observeCost(15, 70, false, win.who(&priced));
    try own.seed(a, e.lone_kept[0][0], &priced);
    try std.testing.expectEqual(@as(?f64, 70), priced.depth.?.ms.?.get(15));
    try std.testing.expectEqual(@as(?f64, null), priced.depth.?.ms.?.get(16));
    try std.testing.expectEqual(@as(u64, 8), priced.cost_probe_after);
}

test "lone price: a long prompt's calibration starts only long streams, and leaves a short one's in place" {
    var cfg = try setup();
    defer cfg.deinit(a);
    var target: fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var e = Engine.init(a, &cfg, target.backend(), clock.clock());
    defer e.deinit();
    var short = try Stream.init(a, .{ .id = "short", .prompt = &.{ 3, 1, 4 }, .max_new = 128 });
    defer short.deinit(a);
    try measured(&e, &short, 60);
    try own.keep(a, own.slot(&e.lone_kept, &short), &short);
    // a 200,000-token prompt: its wide rounds timed slow, and its losing tile probe backed off far past a short reply
    const prompt = try a.alloc(u32, 200_000);
    defer a.free(prompt);
    @memset(prompt, 5);
    var long = try Stream.init(a, .{ .id = "long", .prompt = prompt, .max_new = 128 });
    defer long.deinit(a);
    _ = try e.rule.rates(win.who(&long));
    try e.rule.observeCost(15, 94.5, false, win.who(&long));
    try e.rule.observeCost(16, 400, false, win.who(&long));
    long.rounds = 60;
    long.cost_probes[16] = .{ .after = 5000, .spacing = 4096, .retry = true };
    try own.keep(a, own.slot(&e.lone_kept, &long), &long);
    try std.testing.expectEqual(@as(usize, 0), own.band(3));
    try std.testing.expectEqual(@as(usize, 5), own.band(200_000));
    var next = try Stream.init(a, .{ .id = "next", .prompt = &.{ 2, 7 }, .max_new = 128 });
    defer next.deinit(a);
    _ = try e.rule.rates(win.who(&next));
    try own.seed(a, own.slot(&e.lone_kept, &next).*, &next);
    try std.testing.expectEqual(@as(?f64, 107.5), next.depth.?.ms.?.get(16));
    try std.testing.expectEqual(@as(u64, 100), next.cost_probes[16].after);
    var later = try Stream.init(a, .{ .id = "later", .prompt = prompt, .max_new = 128 });
    defer later.deinit(a);
    _ = try e.rule.rates(win.who(&later));
    try own.seed(a, own.slot(&e.lone_kept, &later).*, &later);
    try std.testing.expectEqual(@as(?f64, 400), later.depth.?.ms.?.get(16));
}

test "lone price: deadlines a run of streams backed off start the next stream at most carry_most rounds out" {
    var cfg = try setup();
    defer cfg.deinit(a);
    var target: fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var e = Engine.init(a, &cfg, target.backend(), clock.clock());
    defer e.deinit();
    var done = try Stream.init(a, .{ .id = "done", .prompt = &.{ 3, 1, 4 }, .max_new = 128 });
    defer done.deinit(a);
    try measured(&e, &done, 60);
    // a long run's losing one-row trials: each doubled the spacing, and the deadline sits far past any reply
    done.cost_probe_after = 60 + (1 << 20);
    done.cost_probe_spacing = 1 << 20;
    done.cost_probes[16] = .{ .after = 60 + (1 << 21), .spacing = 1 << 21, .retry = true };
    try own.keep(a, own.slot(&e.lone_kept, &done), &done);
    var next = try Stream.init(a, .{ .id = "next", .prompt = &.{ 2, 7 }, .max_new = 128 });
    defer next.deinit(a);
    next.rounds = 2;
    _ = try e.rule.rates(win.who(&next));
    try own.seed(a, own.slot(&e.lone_kept, &next).*, &next);
    try std.testing.expectEqual(@as(?f64, 107.5), next.depth.?.ms.?.get(16));
    try std.testing.expectEqual(2 + own.carry_most, next.cost_probe_after);
    try std.testing.expectEqual(own.carry_most, next.cost_probe_spacing);
    try std.testing.expectEqual(2 + own.carry_most, next.cost_probes[16].after);
    try std.testing.expectEqual(own.carry_most, next.cost_probes[16].spacing);
}
