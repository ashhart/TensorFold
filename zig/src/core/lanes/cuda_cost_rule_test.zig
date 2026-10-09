//! Scripted equivalence with the CUDA loop, without importing a device module.
const std = @import("std");
const rule = @import("cost_rule.zig");

const rows = 3;
const Raw = struct { verify: [rows + 1]f64 = @splat(0), level: f64 = 0 };

const Script = struct {
    values: [rows][64]f64 = undefined,
    calls: [rows]usize = @splat(0),
    chains: [rows][2]f64 = .{ .{ 2, 16 }, .{ 4, 11 }, .{ 3, 13 } },
    chain_at: usize = 0,
    events: [512]u8 = undefined,
    count: usize = 0,
    reject_level: bool = false,

    fn init(base: [rows]f64, after: [rows]f64) Script {
        var s: Script = .{};
        for (&s.values, base, after) |*values, first, next| {
            for (values, 0..) |*value, i| {
                value.* = if (i % 8 == 0) 0.001 else (if (i < 8) first else next) + @as(f64, @floatFromInt((i - 1) % 7));
            }
        }
        return s;
    }

    fn record(s: *Script, event: u8) void {
        s.events[s.count] = event;
        s.count += 1;
    }

    const Windows = struct {
        s: *Script,
        pub fn time(w: Windows, i: usize) !f64 {
            const at = w.s.calls[i];
            w.s.calls[i] += 1;
            w.s.record(@intCast(i + 1));
            return w.s.values[i][at];
        }
    };

    const Chains = struct {
        s: *Script,
        pub fn time(c: Chains, levels: usize) !f64 {
            c.s.record(@intCast(100 + levels));
            if (c.s.reject_level) return error.LevelRejected;
            const value = c.s.chains[c.s.chain_at][if (levels == 1) 0 else 1];
            if (levels == 8) c.s.chain_at += 1;
            return value;
        }
    };
};

fn legacyFastest(w: Script.Windows, i: usize) !f64 {
    var best = std.math.inf(f64);
    for (0..8) |j| {
        const value = try w.time(i);
        if (j > 0) best = @min(best, value);
    }
    return best;
}

fn legacy(s: *Script, raw: *Raw, keep: bool) !void {
    const w: Script.Windows = .{ .s = s };
    const h: Script.Chains = .{ .s = s };
    var chains: [2]f64 = @splat(std.math.inf(f64));
    for (1..rows + 1) |r| {
        const value = try legacyFastest(w, r - 1);
        raw.verify[r] = if (keep) @min(raw.verify[r], value) else value;
        for ([_]usize{ 1, 8 }, &chains) |levels, *best| best.* = @min(best.*, try h.time(levels));
    }
    if (!keep) raw.verify[0] = 0;
    for (0..2) |_| {
        var steady = true;
        for (2..rows + 1) |r| if (raw.verify[r] < raw.verify[r - 1] * (1 - 0.03)) {
            steady = false;
            raw.verify[r - 1] = @min(raw.verify[r - 1], try legacyFastest(w, r - 2));
            raw.verify[r] = @min(raw.verify[r], try legacyFastest(w, r - 1));
        };
        if (steady) break;
    }
    const level = (chains[1] - chains[0]) / 7;
    raw.level = if (keep) @min(raw.level, level) else level;
}

fn shared(s: *Script, raw: *Raw, keep: bool) !void {
    const windows: Script.Windows = .{ .s = s };
    var chains: rule.ChainLevel(Script.Chains) = .{ .timer = .{ .s = s } };
    try rule.measureWidths(windows, &chains, raw.verify[1..], keep);
    if (!keep) raw.verify[0] = 0;
    try rule.smooth(windows, raw.verify[1..]);
    const level = chains.level();
    raw.level = if (keep) @min(raw.level, level) else level;
}

fn equivalent(fixture: Script, initial: Raw, keep: bool) !struct { script: Script, raw: Raw } {
    var old = fixture;
    var new = fixture;
    var old_raw = initial;
    var new_raw = initial;
    try legacy(&old, &old_raw, keep);
    try shared(&new, &new_raw, keep);
    try std.testing.expectEqualSlices(f64, &old_raw.verify, &new_raw.verify);
    try std.testing.expectEqual(old_raw.level, new_raw.level);
    try std.testing.expectEqualSlices(usize, &old.calls, &new.calls);
    try std.testing.expectEqualSlices(u8, old.events[0..old.count], new.events[0..new.count]);
    return .{ .script = new, .raw = new_raw };
}

test "CUDA initial pass keeps seven timed minima and interleaves one then eight levels" {
    const result = try equivalent(Script.init(.{ 10, 20, 30 }, .{ 10, 20, 30 }), .{ .verify = @splat(123) }, false);
    try std.testing.expectEqualSlices(f64, &.{ 0, 10, 20, 30 }, &result.raw.verify);
    try std.testing.expectEqual(@as(f64, 9.0 / 7.0), result.raw.level);
    try std.testing.expect(result.raw.level != 1);
    for (0..rows) |i| {
        const start = i * 10;
        for (result.script.events[start .. start + 8]) |event| try std.testing.expectEqual(@as(u8, @intCast(i + 1)), event);
        try std.testing.expectEqualSlices(u8, &.{ 101, 108 }, result.script.events[start + 8 .. start + 10]);
    }
}

test "CUDA retry retains per-width and level minima, including verify index zero" {
    var fixture = Script.init(.{ 12, 18, 40 }, .{ 12, 18, 40 });
    fixture.chains = .{ .{ 2, 19 }, .{ 4, 18 }, .{ 3, 20 } };
    const result = try equivalent(fixture, .{ .verify = .{ 123, 10, 20, 30 }, .level = 1 }, true);
    try std.testing.expectEqualSlices(f64, &.{ 123, 10, 18, 30 }, &result.raw.verify);
    try std.testing.expectEqual(@as(f64, 1), result.raw.level);
    var faster_fixture = Script.init(.{ 9, 18, 30 }, .{ 9, 18, 30 });
    faster_fixture.chains = .{ .{ 1, 4 }, .{ 2, 5 }, .{ 3, 6 } };
    const faster = try equivalent(faster_fixture, result.raw, true);
    try std.testing.expectEqual(@as(f64, 3.0 / 7.0), faster.raw.level);
}

test "CUDA dip retiming matches the legacy loop without resampling levels" {
    const corrected = try equivalent(Script.init(.{ 10, 8, 12 }, .{ 7, 8, 12 }), .{}, false);
    try std.testing.expectEqualSlices(f64, &.{ 0, 7, 8, 12 }, &corrected.raw.verify);
    try std.testing.expectEqualSlices(usize, &.{ 16, 16, 8 }, &corrected.script.calls);
    try std.testing.expectEqual(@as(usize, rows), corrected.script.chain_at);
    const persistent = try equivalent(Script.init(.{ 10, 8, 6 }, .{ 10, 8, 6 }), .{}, false);
    try std.testing.expectEqualSlices(usize, &.{ 24, 40, 24 }, &persistent.script.calls);
    try std.testing.expectEqual(@as(usize, rows), persistent.script.chain_at);
}

test "CUDA independent chain minima include a slow end stretch and propagate timer errors" {
    var fixture = Script.init(.{ 10, 20, 30 }, .{ 10, 20, 30 });
    fixture.chains = .{ .{ 1, 3.38 }, .{ 1.2, 3.5 }, .{ 2, 6.27 } };
    const result = try equivalent(fixture, .{}, false);
    try std.testing.expectApproxEqAbs(@as(f64, 0.34), result.raw.level, 0.0000001);
    fixture.reject_level = true;
    var raw: Raw = .{};
    try std.testing.expectError(error.LevelRejected, shared(&fixture, &raw, false));
}

test "CUDA drift comparison preserves its zero-reference policy" {
    for ([_][2]f64{ .{ 0, 0 }, .{ 1, 0 }, .{ 10, 10 }, .{ 11.5, 10 }, .{ 11.6, 10 }, .{ -1, 0 }, .{ 0, -1 } }) |pair| {
        const old = @abs(pair[0] - pair[1]) > 0.15 * pair[1];
        try std.testing.expectEqual(old, rule.changed(pair[0], pair[1]));
    }
    try std.testing.expect(!rule.changed(0, 0));
    try std.testing.expect(rule.drifted(&.{0}, &.{0}));
}
