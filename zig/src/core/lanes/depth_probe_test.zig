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
        for (streams[0..n], whos[0..n]) |*s, *w| w.* = win.who(s);
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
    std.debug.print("DEPTH_REPLAY n={d} intercept={d} slope={d} head={d} probe={} deepest={d} crossed={any} last={any} trimmed={d}\n", .{ n, intercept, slope, head_ms, enabled, got.deepest, got.crossed[0..n], got.last[0..n], got.trimmed });
    return got;
}

test "shared depth probe: grants learn past a prior cap through ordinary window pricing" {
    for ([_][2]f64{ .{ 50, 3 }, .{ 55, 2.8 } }) |price| {
        for ([_]usize{ 2, 4 }) |n| {
            const old = try replay(n, price[0], price[1], 1, false);
            const next = try replay(n, price[0], price[1], 1, true);
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
