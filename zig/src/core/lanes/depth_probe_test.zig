//! Shared depth exploration through the real grant, window trim and depth observation, with synthetic prices.
const std = @import("std");
const depth = @import("depth.zig");
const Config = @import("config.zig").Config;
const Cost = @import("config.zig").Cost;
const Engine = @import("engine.zig").Engine;
const Stream = @import("stream.zig").Stream;
const win = @import("windows.zig");
const fake = @import("fake.zig");
const t = std.testing;
const gpa = t.allocator;
const prior = [_]f64{ 0.8, 0.72, 0.68, 0.62, 0.58, 0.55, 0.5, 0.5 };

fn noCopy(_: *anyopaque, _: *const depth.Who) bool {
    return false;
}

const Replay = struct {
    deepest: u32 = 0,
    crossed: [8]u32 = @splat(0),
    last: [8]u32 = @splat(0),
    trimmed: u32 = 0,
    untried_stays_prior: bool = true,
};

// These complete linear fixtures reproduce a prior-limited grant; they are not measured device tables.
fn replay(n: usize, intercept: f64, slope: f64, head_ms: f64, enabled: bool) !Replay {
    var costs: [32]Cost = undefined;
    for (&costs, 1..) |*c, row| c.* = .{ .width = @intCast(row), .ms = intercept + slope * @as(f64, @floatFromInt(row)) };
    var cfg = try Config.init(gpa, .{ .exact_width = 32, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 8, .draft_prior = &prior, .window_costs = &costs, .mtp_step_ms = head_ms, .hidden_rows = true }, 32, 31);
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var e = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer e.deinit();
    if (!enabled) e.rule.k.depth_probe_every = 0;
    try e.rule.observeOverhead(@intCast(n), n, cfg.shared_costs.get(@intCast(n)).?, false, 0);
    var streams: [8]Stream = undefined;
    var made: usize = 0;
    defer for (streams[0..made]) |*s| s.deinit(gpa);
    for (streams[0..n]) |*s| {
        s.* = try Stream.init(gpa, .{ .id = "replay", .prompt = &.{1}, .max_new = 100000, .eos = &.{} });
        made += 1;
    }
    var got: Replay = .{};
    var produced: [8]u32 = @splat(0);
    for (1..129) |round| {
        _ = e.arena.reset(.retain_capacity);
        var whos: [8]depth.Who = undefined;
        for (streams[0..n], whos[0..n]) |*s, *w| w.* = win.who(s, e.cfg.k.hold_costs.len > 0);
        var offered: [8]i64 = undefined;
        var unused: u8 = 0;
        try e.rule.budgets(whos[0..n], .{ .ptr = &unused, .copy = noCopy }, false, offered[0..n]);
        var plans: [8]win.Plan = undefined;
        for (plans[0..n], streams[0..n], offered[0..n]) |*p, *s, count| p.* = .{ .stream = s, .position = 0, .kind = .head, .held = @intCast(count) };
        try win.allocate(&e, plans[0..n]);
        var rows: usize = n;
        for (plans[0..n], whos[0..n], offered[0..n], 0..) |p, w, count, i| {
            const kept: u32 = @intCast(p.count());
            try t.expect(kept <= count);
            got.trimmed += @intFromBool(kept < count);
            got.deepest = @max(got.deepest, kept);
            got.last[i] = kept;
            produced[i] += 1 + kept;
            if (got.crossed[i] == 0 and produced[i] >= 400) got.crossed[i] = @intCast(round);
            try e.rule.observeDepth(w, kept, kept);
            rows += kept;
        }
        try t.expect(rows <= 32);
    }
    if (got.deepest < prior.len) {
        for (streams[0..n]) |s| got.untried_stays_prior = got.untried_stays_prior and s.depth.?.p[got.deepest] == prior[got.deepest];
    }
    return got;
}

const Seen = struct { deepest: u32, crossed: []const u32, last: []const u32, trimmed: u32 };

fn expectSeen(want: Seen, got: Replay) !void {
    try t.expectEqual(want.deepest, got.deepest);
    try t.expectEqualSlices(u32, want.crossed, got.crossed[0..want.crossed.len]);
    try t.expectEqualSlices(u32, want.last, got.last[0..want.last.len]);
    try t.expectEqual(want.trimmed, got.trimmed);
}

const replays = [2][2][2]Seen{
    .{
        .{
            .{ .deepest = 6, .crossed = &.{ 62, 62 }, .last = &.{ 6, 6 }, .trimmed = 0 },
            .{ .deepest = 8, .crossed = &.{ 59, 57 }, .last = &.{ 7, 8 }, .trimmed = 11 },
        },
        .{
            .{ .deepest = 4, .crossed = &.{ 83, 83, 83, 83 }, .last = &.{ 4, 4, 4, 4 }, .trimmed = 0 },
            .{ .deepest = 5, .crossed = &.{ 75, 83, 83, 76 }, .last = &.{ 5, 5, 5, 5 }, .trimmed = 9 },
        },
    },
    .{
        .{
            .{ .deepest = 6, .crossed = &.{ 61, 61 }, .last = &.{ 6, 6 }, .trimmed = 0 },
            .{ .deepest = 8, .crossed = &.{ 54, 56 }, .last = &.{ 8, 8 }, .trimmed = 2 },
        },
        .{
            .{ .deepest = 4, .crossed = &.{ 82, 82, 82, 82 }, .last = &.{ 4, 4, 4, 4 }, .trimmed = 0 },
            .{ .deepest = 6, .crossed = &.{ 75, 71, 72, 74 }, .last = &.{ 5, 6, 5, 5 }, .trimmed = 9 },
        },
    },
};

test "shared depth probe: grants learn past a prior cap through ordinary window pricing" {
    for ([_][2]f64{ .{ 50, 3 }, .{ 55, 2.8 } }, replays) |price, by_n| {
        for ([_]usize{ 2, 4 }, by_n) |n, want| {
            const old = try replay(n, price[0], price[1], 1, false);
            const next = try replay(n, price[0], price[1], 1, true);
            try expectSeen(want[0], old);
            try expectSeen(want[1], next);
            try t.expectEqual(@as(u32, if (n == 2) 6 else 4), old.deepest);
            try t.expect(old.untried_stays_prior);
            try t.expect(next.deepest > old.deepest);
            try t.expect(next.trimmed > 0);
            for (next.crossed[0..n]) |round| try t.expect(round > 0 and round <= 128);
        }
    }
}

test "shared depth probe: eight streams already filling 32 rows stay unchanged" {
    const old = try replay(8, 100, 0, 0, false);
    const next = try replay(8, 100, 0, 0, true);
    try t.expectEqualDeep(old, next);
    try t.expectEqual(@as(u32, 3), next.deepest);
    const full: [8]u32 = @splat(100);
    const three: [8]u32 = @splat(3);
    try expectSeen(.{ .deepest = 3, .crossed = &full, .last = &three, .trimmed = 0 }, next);
}

test "shared depth probe: every eighth grant adds at most one row and rotates streams" {
    const table: @import("table.zig").Table = .{};
    var r: depth.Rule = .{ .gpa = gpa, .prior = &prior, .most_drafts = 8, .family_costs = &table, .shared_costs = &table, .mtp_step_ms = 0, .plain_guard = false, .node_probabilities = false, .batch_rows = 32 };
    defer r.deinit();
    var states: [4]?depth.State = @splat(null);
    defer for (&states) |*s| if (s.*) |*v| v.deinit(gpa);
    var whos: [4]depth.Who = undefined;
    for (&whos, &states) |*w, *s| w.* = .{ .state = s, .draft_room = 100, .finished = false, .forced = false };
    // Flat untimed pricing fills the cap, so use complete costs which choose a shallow initial grant.
    var costs: @import("table.zig").Table = .{};
    defer costs.deinit(gpa);
    for (1..33) |row| try costs.put(gpa, @intCast(row), 50 + 3 * @as(f64, @floatFromInt(row)));
    r.shared_costs = &costs;
    var before: [4]i64 = undefined;
    var unused: u8 = 0;
    const probe: depth.Probe = .{ .ptr = &unused, .copy = noCopy };
    r.k.depth_probe_every = 0;
    try r.budgets(&whos, probe, false, &before);
    r.k.depth_probe_every = 8;
    r.shared_rounds = 0;
    var seen: [4]bool = @splat(false);
    for (1..33) |round| {
        var after: [4]i64 = undefined;
        try r.budgets(&whos, probe, false, &after);
        var extras: i64 = 0;
        for (before, after, 0..) |b, a, i| {
            try t.expect(a == b or a == b + 1);
            extras += a - b;
            if (a > b) seen[i] = true;
        }
        try t.expectEqual(@as(i64, @intFromBool(round % 8 == 0)), extras);
    }
    for (seen) |s| try t.expect(s);
}

test "a depth's acceptance estimate moves by the configured weight of the newest round" {
    const empty: @import("table.zig").Table = .{};
    var rule: depth.Rule = .{ .gpa = gpa, .k = .{ .depth_rate = 0.08 }, .prior = &.{ 0.5, 0.5 }, .most_drafts = 2, .family_costs = &empty, .shared_costs = &empty, .mtp_step_ms = 0.5, .plain_guard = true, .node_probabilities = false, .batch_rows = 8 };
    defer rule.deinit();
    var st: ?depth.State = null;
    defer if (st) |*s| s.deinit(gpa);
    const who: depth.Who = .{ .state = &st, .draft_room = 100, .finished = false, .forced = false };
    try rule.observeDepth(who, 2, 1); // the first draft landed, the second did not
    try t.expectApproxEqAbs(@as(f64, 0.54), st.?.p[0], 1e-12);
    try t.expectApproxEqAbs(@as(f64, 0.46), st.?.p[1], 1e-12);
}

test "depth picks the most tokens a ms and probes deeper" {
    var fc: @import("table.zig").Table = .{};
    defer fc.deinit(gpa);
    for (1..5) |w| try fc.put(gpa, @intCast(w), 5.0 + @as(f64, @floatFromInt(w)));
    const shared: @import("table.zig").Table = .{};
    var rule: depth.Rule = .{ .gpa = gpa, .prior = &.{ 0.9, 0.9, 0.9 }, .most_drafts = 3, .family_costs = &fc, .shared_costs = &shared, .mtp_step_ms = 0.5, .plain_guard = false, .node_probabilities = false, .batch_rows = 8 };
    defer rule.deinit();
    var st: ?depth.State = null;
    defer if (st) |*s| s.deinit(gpa);
    const who: depth.Who = .{ .state = &st, .draft_room = 100, .finished = false, .forced = false };
    try t.expectEqual(@as(i64, 3), try rule.depth(who));
}

test "costed forced rounds retain a due plain-recovery trial until a draft can run" {
    var slots = [_]?f64{ null, 10, 10 };
    const costs: @import("table.zig").Table = .{ .slots = &slots };
    var rule: depth.Rule = .{ .gpa = gpa, .k = .{ .hold_costs = &.{.{ .streams = 1, .depth = 1, .ms = 100 }} }, .prior = &.{0.5}, .most_drafts = 1, .family_costs = &costs, .shared_costs = &costs, .mtp_step_ms = 100, .plain_guard = true, .node_probabilities = false, .batch_rows = 2 };
    defer rule.deinit();
    var state: ?depth.State = .{ .p = try gpa.dupe(f64, &.{0.5}), .rounds = 19, .plain = 7, .wait = 8 };
    defer state.?.deinit(gpa);
    var whos = [_]depth.Who{.{ .state = &state, .draft_room = 64, .finished = false, .forced = true }};
    var unused: u8 = 0;
    const probe: depth.Probe = .{ .ptr = &unused, .copy = noCopy };
    var out: [1]i64 = undefined;
    for (0..3) |_| {
        try rule.budgets(&whos, probe, false, &out);
        try t.expectEqual(@as(i64, 0), out[0]);
        try t.expectEqual(@as(?u64, 7), state.?.plain);
        try t.expectEqual(@as(?u64, 8), state.?.wait);
        try t.expectEqual(@as(u64, 19), state.?.rounds);
    }
    try t.expectEqual(@as(i64, 0), try rule.headDepth(&whos[0], null, probe));
    try t.expectEqual(@as(?u64, 7), state.?.plain);
    try t.expectEqual(@as(?u64, 8), state.?.wait);
    try t.expectEqual(@as(u64, 19), state.?.rounds);
    whos[0].forced = false;
    try rule.budgets(&whos, probe, false, &out);
    try t.expectEqual(@as(i64, 1), out[0]);
    try t.expectEqual(@as(?u64, 0), state.?.plain);
    try t.expectEqual(@as(?u64, 16), state.?.wait);
    try t.expectEqual(@as(u64, 20), state.?.rounds);
}

test "drafts-off shared rounds advance unpriced depth probes but not costed probes" {
    var costs: [8]Cost = undefined;
    for (&costs, 1..) |*c, rows| c.* = .{ .width = @intCast(rows), .ms = 6 + 2 * @as(f64, @floatFromInt(rows)) };
    for ([_]bool{ false, true }) |priced| {
        var cfg = try Config.init(gpa, .{
            .exact_width = 8,
            .mtp = true,
            .speculate = true,
            .speculate_early = false,
            .drafts = 3,
            .draft_prior = &.{0.5},
            .window_costs = &costs,
            .mtp_step_ms = 0,
            .hidden_rows = true,
            .batch_rows = 8,
            .max_streams = 2,
            .draft_streams = true,
            .hold_costs = if (priced) &.{.{ .streams = 1, .depth = 1, .ms = 0.01 }} else &.{},
        }, 8, 7);
        defer cfg.deinit(gpa);
        cfg.timed = false;
        var target: fake.Fake = .{ .gpa = gpa };
        defer target.deinit();
        var clock: fake.FixedClock = .{};
        var e = Engine.init(gpa, &cfg, target.backend(), clock.clock());
        defer e.deinit();
        e.rule.k.depth_probe_every = 8;
        try e.rule.observeOverhead(2, 2, cfg.shared_costs.get(2).?, false, 0);
        var streams: [2]Stream = undefined;
        var made: usize = 0;
        defer for (streams[0..made]) |*s| s.deinit(gpa);
        for (&streams) |*s| {
            s.* = try Stream.init(gpa, .{ .id = "cadence", .prompt = &.{1}, .max_new = 100, .eos = &.{}, .drafts = false });
            made += 1;
            try e.addStream(s);
        }
        for (0..7) |_| try e.step();
        try t.expectEqual(@as(u64, 7), e.shared_rounds);
        try t.expectEqual(@as(u64, if (priced) 0 else 7), e.rule.shared_rounds);
        try t.expectEqual(@as(usize, 0), target.drafts_calls);
        for (&streams) |*s| s.drafts = true;
        try e.step();
        // The first offered shared head is a probe only if the seven target-only rounds counted.
        try t.expectEqual(@as(u32, if (priced) 1 else 2), streams[0].next.?.count);
        try t.expectEqual(@as(u32, 1), streams[1].next.?.count);
        try t.expectEqual(@as(usize, if (priced) 1 else 2), target.lanes.get(&streams[0]).?.held.items.len);
        try t.expectEqual(@as(usize, 1), target.lanes.get(&streams[1]).?.held.items.len);
        try t.expectEqual(@as(usize, 1), target.drafts_calls);
        try t.expectEqual(@as(usize, 2), target.drafts_most);
        try t.expectEqual(@as(u64, if (priced) 1 else 8), e.rule.shared_rounds);
    }
}

test "target-only shared reservations preserve unpriced depth probe cadence" {
    var costs: [8]Cost = undefined;
    for (&costs, 1..) |*c, rows| c.* = .{ .width = @intCast(rows), .ms = 6 + 2 * @as(f64, @floatFromInt(rows)) };
    for ([_]bool{ false, true }) |priced| for ([_]bool{ false, true }) |limited| for ([_]bool{ false, true }) |granted| {
        var cfg = try Config.init(gpa, .{
            .exact_width = 8,
            .mtp = true,
            .speculate = true,
            .speculate_early = false,
            .drafts = 3,
            .draft_prior = &.{0.5},
            .window_costs = &costs,
            .mtp_step_ms = 0,
            .hidden_rows = true,
            .batch_rows = 8,
            .max_streams = 2,
            .draft_streams = true,
            .allocate_shared_drafts = true,
            .limit_shared_drafts = limited,
            .draft_probabilities = granted,
            .granted_depth = granted,
            .hold_costs = if (priced) &.{.{ .streams = 1, .depth = 1, .ms = 0.01 }} else &.{},
        }, 8, 7);
        defer cfg.deinit(gpa);
        cfg.timed = false;
        var target: fake.Fake = .{ .gpa = gpa };
        defer target.deinit();
        var clock: fake.FixedClock = .{};
        var e = Engine.init(gpa, &cfg, target.backend(), clock.clock());
        defer e.deinit();
        e.rule.k.depth_probe_every = 8;
        var streams: [2]Stream = undefined;
        var made: usize = 0;
        defer for (streams[0..made]) |*s| s.deinit(gpa);
        for (&streams) |*s| {
            s.* = try Stream.init(gpa, .{ .id = "shared-cadence", .prompt = &.{1}, .max_new = 100, .eos = &.{}, .drafts = false });
            made += 1;
            try e.addStream(s);
        }
        for (0..7) |_| try e.step();
        try t.expectEqual(@as(u64, 7), e.shared_rounds);
        // Main reaches rule.budgets through shared reservations only for a legacy depth cap.
        try t.expectEqual(@as(u64, if (!priced and limited and !granted) 7 else 0), e.rule.shared_rounds);
        try t.expectEqual(@as(usize, 0), target.drafts_calls);
    };
}
