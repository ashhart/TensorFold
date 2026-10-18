//! Losing calibration uses an expected-output repayment bound and retains overdue widths.
const std = @import("std");
const lanes = @import("lanes.zig");
const win = @import("windows.zig");
const own = @import("lone_costs.zig");
const a = std.testing.allocator;

fn trials(costs: []const f64, waits: []const u64) !void {
    var cfg = try lanes.Config.init(a, .{ .exact_width = 64, .mtp = true, .speculate = true, .drafts = 63, .plain_guard = true, .draft_probabilities = true, .hidden_rows = true, .batch_rows = 128, .lone_cost_tile = 16 }, 64, 63);
    defer cfg.deinit(a);
    for (1..65) |rows| try cfg.shared_costs.put(a, @intCast(rows), if (rows <= 16) 10 else 80);
    var target: lanes.fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var e = lanes.Engine.init(a, &cfg, target.backend(), clock.clock());
    defer e.deinit();
    e.alone = true;
    var s = try lanes.Stream.init(a, .{ .id = "backoff", .prompt = &.{3}, .max_new = 8192 });
    defer s.deinit(a);
    @memset(try e.rule.rates(win.who(&s)), 1);
    try e.rule.observeCost(0, 20, false, win.who(&s));
    try e.rule.observeCost(15, 20, false, win.who(&s));
    const tokens: [63]u32 = @splat(7);
    for (costs, waits, 0..) |ms, wait, i| {
        s.rounds = s.cost_probe_after;
        const rows = 32 + 16 * i;
        var p = [_]win.Plan{.{ .stream = &s, .position = 1, .kind = .head, .tokens = tokens[0 .. rows - 1] }};
        try win.allocate(&e, &p);
        try std.testing.expectEqual(@as(?u32, @intCast(rows)), p[0].cost_probe_rows);
        try e.rule.observeCost(@intCast(rows - 1), ms, false, win.who(&s));
        s.rounds += 1;
        own.observe(&s, p[0].cost_probe_rows, rows, ms);
        try std.testing.expectEqual(s.rounds + wait, s.cost_probe_after);
        s.rounds = s.cost_probe_after - 1;
        try std.testing.expectEqual(@as(?u32, null), own.probe(&e, p[0]));
    }
}

test "lone backoff: different widths have independent loss spacing and a winning trial resets its own" {
    try trials(&.{ 80, 120, 60 }, &.{ 16, 16, 8 });
}

test "lone backoff: a narrow expected advantage must repay the entire trial before another probe" {
    // Expected 16/20 beats 32/40.5: a 0.25 ms saving needs 162 rounds to cover 40.5 ms.
    try trials(&.{40.5}, &.{162});
}

fn recovered(losses: usize, ordinary_first: bool) !void {
    var cfg = try lanes.Config.init(a, .{ .exact_width = 64, .mtp = true, .speculate = true, .drafts = 63, .plain_guard = true, .draft_probabilities = true, .hidden_rows = true, .batch_rows = 128, .lone_cost_tile = 16 }, 64, 63);
    defer cfg.deinit(a);
    for (1..65) |rows| try cfg.shared_costs.put(a, @intCast(rows), if (rows <= 16) 10 else if (ordinary_first) 20 else 80);
    var target: lanes.fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var e = lanes.Engine.init(a, &cfg, target.backend(), clock.clock());
    defer e.deinit();
    e.alone = true;
    var s = try lanes.Stream.init(a, .{ .id = "same-width-recovery", .prompt = &.{3}, .max_new = 8192 });
    defer s.deinit(a);
    @memset(try e.rule.rates(win.who(&s)), 1);
    try e.rule.observeCost(0, 20, false, win.who(&s));
    try e.rule.observeCost(15, 20, false, win.who(&s));
    const tokens: [31]u32 = @splat(7);
    var attempts: usize = 0;
    var retried = false;
    for (1..129) |completed| {
        s.rounds = completed;
        var p = [_]win.Plan{.{ .stream = &s, .position = 1, .kind = .head, .tokens = &tokens }};
        try win.allocate(&e, &p);
        const rows = p[0].count() + 1;
        const ms: f64 = if (rows < 32) 20 else if (attempts < losses) 80 else 35;
        attempts += @intFromBool(rows == 32);
        try e.rule.observeCost(@intCast(rows - 1), ms, false, win.who(&s));
        s.rounds += 1;
        const recovery = p[0].cost_probe_rows == 32 and attempts == losses + 1;
        if (recovery) try std.testing.expect(s.depth.?.ms.?.get(31).? > ms);
        own.observe(&s, p[0].cost_probe_rows, rows, ms);
        if (p[0].cost_probe_rows == 32) {
            if (attempts <= losses) {
                try std.testing.expect(s.cost_probes[32].retry);
                try std.testing.expectEqual(@as(u64, 8) << @intCast(attempts), s.cost_probes[32].spacing);
            } else {
                try std.testing.expectEqual(@as(u64, 8), s.cost_probes[32].spacing);
                try std.testing.expect(!s.cost_probes[32].retry);
                try std.testing.expectEqual(@as(?f64, ms), s.depth.?.ms.?.get(31));
                retried = true;
            }
        }
    }
    try std.testing.expect(attempts > losses and retried);
    try std.testing.expectEqual(@as(?f64, 35), s.depth.?.ms.?.get(31));
}

test "lone backoff: the same width recovers after one bad trial instead of starving forever" {
    try recovered(1, false);
}
test "lone backoff: repeated losses at one width double only its cooldown before fresh-time recovery" {
    try recovered(2, false);
}

test "lone backoff: an ordinary bad first sample ages into a same-width recovery trial" {
    try recovered(1, true);
}
