//! Timings steer windows, never bits; noise only adds time, so an entry keeps its fastest run and outliers are timed again.
const std = @import("std");

/// Timed runs an entry keeps the fastest of, after one untimed.
pub const reps = 7;
/// An entry timed more than this fraction under the one before it is noise.
pub const dip = 0.03;
/// A table this far from the reference at any entry is timed a second time.
pub const drift = 0.15;

/// The fastest of `reps` runs after one untimed; `ctx.time(i)` times entry `i` once, in ms.
pub fn fastest(ctx: anytype, i: usize) !f64 {
    _ = try ctx.time(i);
    var best = std.math.inf(f64);
    for (0..reps) |_| best = @min(best, try ctx.time(i));
    return best;
}

/// An entry timed more than `dip` under the one before is timed again with it, two sweeps at most, each keeping its faster value.
pub fn smooth(ctx: anytype, ms: []f64) !void {
    for (0..2) |_| {
        var dipped = false;
        for (1..ms.len) |i| {
            if (ms[i] >= ms[i - 1] * (1 - dip)) continue;
            ms[i - 1] = @min(ms[i - 1], try fastest(ctx, i - 1));
            ms[i] = @min(ms[i], try fastest(ctx, i));
            dipped = true;
        }
        if (!dipped) return;
    }
}

/// True when any entry is more than `drift` from the reference's, or the reference has another shape.
pub fn drifted(ms: []const f64, reference: []const f64) bool {
    if (ms.len != reference.len) return true;
    for (ms, reference) |m, r| if (!(r > 0) or @abs(m - r) > drift * r) return true;
    return false;
}

/// A second full pass, each entry keeping its faster value.
pub fn again(ctx: anytype, ms: []f64) !void {
    for (ms, 0..) |*m, i| m.* = @min(m.*, try fastest(ctx, i));
}

/// Scripted runs per entry, consumed in order, for the tests.
const Script = struct {
    runs: []const []const f64,
    next: []usize,
    pub fn time(s: Script, i: usize) !f64 {
        const at = s.next[i];
        s.next[i] += 1;
        return s.runs[i][at];
    }
};

test "an entry keeps its fastest timed run, never the untimed one" {
    var next = [_]usize{0};
    const s: Script = .{ .runs = &.{&.{ 1.0, 9, 5, 6, 4.5, 7, 8, 5 }}, .next = &next };
    try std.testing.expectEqual(@as(f64, 4.5), try fastest(s, 0));
    try std.testing.expectEqual(@as(usize, 8), next[0]);
}

test "a dip times both entries again and keeps the faster values; a real dip stays" {
    const first = [_]f64{ 4, 4, 4, 4, 4, 4, 4, 4 };
    const second = [_]f64{ 6, 6, 6, 6, 6, 6, 6, 6 };
    var next = [_]usize{ 0, 0 };
    const s: Script = .{ .runs = &.{ &first, &second }, .next = &next };
    var ms = [_]f64{ 6.5, 6 };
    try smooth(s, &ms);
    try std.testing.expectEqualSlices(f64, &.{ 4, 6 }, &ms);
    const slow: [16]f64 = @splat(6.5);
    const fast: [16]f64 = @splat(6);
    var real_next = [_]usize{ 0, 0 };
    const real: Script = .{ .runs = &.{ &slow, &fast }, .next = &real_next };
    var kept = [_]f64{ 6.5, 6 };
    try smooth(real, &kept);
    try std.testing.expectEqualSlices(f64, &.{ 6.5, 6 }, &kept);
    try std.testing.expectEqual(@as(usize, 16), real_next[0]);
}

test "drift is any entry more than 15 percent from the reference, or another shape" {
    try std.testing.expect(!drifted(&.{ 10, 20 }, &.{ 11, 18 }));
    try std.testing.expect(drifted(&.{ 10, 20 }, &.{ 10, 24 }));
    try std.testing.expect(drifted(&.{10}, &.{ 10, 20 }));
    try std.testing.expect(drifted(&.{10}, &.{0}));
}
