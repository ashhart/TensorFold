//! Request budgets and measured-only table consistency over recording warm-load clocks.
const std = @import("std");
const rule = @import("cost_rule.zig");
const warm = rule.warm_prices;
const t = std.testing;
const Timer = struct {
    setups: usize = 0,
    setup_ms: f64 = 0,
    wall: f64 = 0,
    value: f64 = 10,
    seen: [8]usize = undefined,
    n: usize = 0,
    pub fn prepare(s: *@This(), _: anytype, _: []const u32) !void {
        s.setups += 1;
        s.wall += s.setup_ms;
    }
    pub fn elapsedMs(s: *@This()) f64 {
        return s.wall;
    }
    pub fn time(s: *@This(), i: usize) !f64 {
        s.seen[s.n] = i;
        s.n += 1;
        s.wall += s.value;
        return s.value;
    }
};
fn prepared(timer: *Timer, ms: []f64, widths: []const u32, cursor: *u64) !warm.Stats {
    if (comptime @hasDecl(warm, "prepared")) return warm.prepared(timer, ms, widths, cursor, 0, warm.Local{}, true);
    // The incoming consumers set up before invoking the budgeted policy.
    try timer.prepare({}, widths);
    return rule.healWindows(timer, ms, widths, cursor);
}

test "Metal review: no-fit warm load does no setup or reads and leaves its cursor" {
    var timer: Timer = .{};
    var ms = [_]f64{ 10, 210 };
    var cursor: u64 = 17;
    _ = try prepared(&timer, &ms, &.{ 1, 64 }, &cursor);
    try t.expectEqual(@as(usize, 0), timer.setups);
    try t.expectEqual(@as(usize, 0), timer.n);
    try t.expectEqual(@as(u64, 17), cursor);
}

test "Metal review: a useful 150ms entry is not starved by a discarded widest warm-up" {
    var timer: Timer = .{ .value = 140 };
    var ms = [_]f64{150};
    var cursor: u64 = 0;
    const stats = try prepared(&timer, &ms, &.{64}, &cursor);
    try t.expectEqual(@as(f64, 140), ms[0]);
    try t.expectEqual(@as(usize, 1), timer.n);
    try t.expectEqual(@as(u32, 1), stats.lowered);
}

test "Metal review: setup wall time can stop all reads without advancing the cursor" {
    var timer: Timer = .{ .setup_ms = 195 };
    var ms = [_]f64{ 10, 20 };
    var cursor: u64 = 0;
    _ = try prepared(&timer, &ms, &.{ 1, 2 }, &cursor);
    try t.expectEqual(@as(usize, 1), timer.setups);
    try t.expectEqual(@as(usize, 0), timer.n);
    try t.expectEqual(@as(u64, 0), cursor);
}

test "Metal review: real 16/17 32/33 48/49 pairs run narrower first" {
    var widths: [64]u32 = undefined;
    for (&widths, 0..) |*w, i| w.* = @intCast(i + 1);
    for ([_]u64{ 0, 1, 2 }) |phase| {
        var timer: Timer = .{};
        var ms: [64]f64 = @splat(11);
        var cursor = phase;
        _ = try rule.healWindows(&timer, &ms, &widths, &cursor);
        const edge: usize = @intCast(16 * (1 + phase));
        const calls = timer.seen[0..timer.n];
        const narrow = std.mem.indexOfScalar(usize, calls, edge - 1);
        const wider = std.mem.indexOfScalar(usize, calls, edge);
        try t.expect(narrow != null and wider != null);
        try t.expect(narrow.? < wider.?);
    }
}

test "Metal review: partial healing cannot undercut an unhealed predecessor" {
    var timer: Timer = .{ .value = 10 };
    var ms = [_]f64{ 10.8, 11.8, 12.8, 13.8 };
    var cursor: u64 = 1;
    _ = try rule.healWindows(&timer, &ms, &.{ 1, 2, 3, 4 }, &cursor);
    for (1..ms.len) |i| try t.expect(ms[i] >= ms[i - 1]);
}

test "Metal review: zero and nonfinite samples cannot erase seven good readings" {
    const Bad = struct {
        calls: usize = 0,
        invalid: f64,
        pub fn time(s: *@This(), _: usize) !f64 {
            s.calls += 1;
            return if (s.calls == 4) s.invalid else 10;
        }
    };
    for ([_]f64{ 0, -1, std.math.nan(f64), std.math.inf(f64) }) |bad| {
        var timer: Bad = .{ .invalid = bad };
        var ms: [1]f64 = undefined;
        try rule.measureWindows(&timer, &ms, false);
        try t.expectEqual(@as(f64, 10), ms[0]);
    }
}

test "Metal review: both entries in a two-wide high plateau are retimed" {
    var timer: Timer = .{};
    var ms = [_]f64{ 10, 10, 10.5, 10.5, 10, 10 };
    try rule.smooth(&timer, &ms);
    try t.expectEqualSlices(f64, &.{ 10, 10, 10, 10, 10, 10 }, &ms);
}
