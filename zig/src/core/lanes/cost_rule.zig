//! Timings steer windows, never bits; retain each entry's fastest run and retime outliers.
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

/// Retime both entries around a dip up to twice, retaining their faster values.
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
    for (ms, reference) |m, r| if (!(r > 0) or changed(m, r)) return true;
    return false;
}

/// Compare a measured cost with its reference without imposing a positive-reference policy.
pub fn changed(ms: f64, reference: f64) bool {
    return @abs(ms - reference) > drift * reference;
}

/// Whether entries `at` and the head (sampled as `measure` does) time within drift, a drifted reading timed twice.
pub fn probeHolds(windows: anytype, heads: anytype, at: []const usize, reference: []const f64, head: ?f64) !bool {
    var best = std.math.inf(f64);
    for (at) |i| {
        if (i >= reference.len or !(reference[i] > 0)) return false;
        if (!try within(windows, i, try fastest(windows, i), reference[i])) return false;
        if (head != null) best = @min(best, try heads.time(0));
    }
    const h = head orelse return true;
    if (!(h > 0)) return false;
    return within(heads, 0, @min(best, try fastest(heads, 0)), h);
}

fn within(ctx: anytype, i: usize, ms: f64, reference: f64) !bool {
    return !changed(ms, reference) or !changed(@min(ms, try fastest(ctx, i)), reference);
}

/// A second full pass, each entry keeping its faster value.
pub fn again(ctx: anytype, ms: []f64) !void {
    for (ms, 0..) |*m, i| m.* = @min(m.*, try fastest(ctx, i));
}

/// Sample each width, then the observer; retries retain each width's faster value.
pub fn measureWidths(windows: anytype, observer: anytype, ms: []f64, keep: bool) !void {
    for (ms, 0..) |*m, i| {
        const value = try fastest(windows, i);
        m.* = if (keep) @min(m.*, value) else value;
        try observer.afterWidth(i);
    }
}

/// Sample the head between widths, keeping its whole-pass minimum; retries retain each entry's faster value.
pub fn measure(windows: anytype, heads: anytype, ms: []f64, head_ms: ?*f64, keep: bool) !void {
    const Observer = struct {
        heads: @TypeOf(heads),
        enabled: bool,
        best: f64 = std.math.inf(f64),

        pub fn afterWidth(o: *@This(), _: usize) !void {
            if (o.enabled) o.best = @min(o.best, try o.heads.time(0));
        }
    };
    var observer: Observer = .{ .heads = heads, .enabled = head_ms != null };
    try measureWidths(windows, &observer, ms, keep);
    if (head_ms) |h| h.* = if (keep) @min(h.*, observer.best) else observer.best;
}

/// Keep independent minima for one- and eight-level chains sampled between widths.
pub fn ChainLevel(comptime Timer: type) type {
    return struct {
        timer: Timer,
        best: [2]f64 = @splat(std.math.inf(f64)),

        pub fn afterWidth(c: *@This(), _: usize) !void {
            for ([_]usize{ 1, 8 }, &c.best) |levels, *best| best.* = @min(best.*, try c.timer.time(levels));
        }

        pub fn level(c: @This()) f64 {
            return (c.best[1] - c.best[0]) / 7;
        }
    };
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

test "a probe holds while every probed entry stays within drift, and stops at the first that does not" {
    const near = [_]f64{ 0, 10.5, 10.4, 10.6, 10.5, 10.5, 10.5, 10.5 };
    const slow: [16]f64 = @splat(30);
    const close: [8]f64 = @splat(21);
    const runs = [_][]const f64{ &near, &slow, &close };
    var next = [_]usize{ 0, 0, 0 };
    const s: Script = .{ .runs = &runs, .next = &next };
    try std.testing.expect(try probeHolds(s, s, &.{ 0, 2 }, &.{ 10, 20, 20 }, null));
    try std.testing.expect(!try probeHolds(s, s, &.{ 1, 2 }, &.{ 10, 20, 20 }, null));
    try std.testing.expectEqual(@as(usize, 16), next[1]); // a reading past drift is timed a second time
    try std.testing.expectEqual(@as(usize, 8), next[2]); // entry 2 is not timed again once entry 1 drifts
    try std.testing.expect(!try probeHolds(s, s, &.{3}, &.{ 10, 20, 20 }, null));
    try std.testing.expect(!try probeHolds(s, s, &.{0}, &.{ 0, 20, 20 }, null));
}

test "one slow reading of a kept entry or head is timed again; a slow build stays refused" {
    const once = [_]f64{ 0, 30, 30, 30, 30, 30, 30, 30, 0, 10.2, 10.2, 10.2, 10.2, 10.2, 10.2, 10.2 };
    const runs = [_][]const f64{&once};
    var next = [_]usize{0};
    const s: Script = .{ .runs = &runs, .next = &next };
    try std.testing.expect(try probeHolds(s, s, &.{0}, &.{10}, null));
    // the head: one sample after the window, then a fastest-of that reads 24% slow, then a second fastest-of
    const window = [_]f64{ 0, 10, 10, 10, 10, 10, 10, 10 };
    const head = [_]f64{ 1.8, 0, 1.84, 1.84, 1.84, 1.84, 1.84, 1.84, 1.84, 0, 1.49, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5 };
    var at = [_]usize{ 0, 0 };
    const w: Script = .{ .runs = &.{&window}, .next = at[0..1] };
    const h: Script = .{ .runs = &.{&head}, .next = at[1..2] };
    try std.testing.expect(try probeHolds(w, h, &.{0}, &.{10}, 1.484));
    try std.testing.expectEqual(@as(usize, 17), at[1]);
    const slow_head: [24]f64 = @splat(1.8);
    var slow_at = [_]usize{ 0, 0 };
    const w2: Script = .{ .runs = &.{&window}, .next = slow_at[0..1] };
    const h2: Script = .{ .runs = &.{&slow_head}, .next = slow_at[1..2] };
    try std.testing.expect(!try probeHolds(w2, h2, &.{0}, &.{10}, 1.484));
    var none_at = [_]usize{ 0, 0 };
    const w3: Script = .{ .runs = &.{&window}, .next = none_at[0..1] };
    try std.testing.expect(!try probeHolds(w3, h2, &.{0}, &.{10}, 0));
}

test "the head reads as the fastest of its samples after each probed window and its own runs" {
    const window: [16]f64 = @splat(10);
    const head = [_]f64{ 1.49, 1.5, 0, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8 };
    var at = [_]usize{ 0, 0 };
    const w: Script = .{ .runs = &.{&window}, .next = at[0..1] };
    const h: Script = .{ .runs = &.{&head}, .next = at[1..2] };
    const entries = [_]usize{ 0, 0 };
    try std.testing.expect(try probeHolds(w, h, &entries, &.{10}, 1.484));
    try std.testing.expectEqual(@as(usize, 10), at[1]); // two samples and one fastest-of, no second timing
}

test {
    _ = @import("cost_rule_test.zig");
    _ = @import("cuda_cost_rule_test.zig");
}
