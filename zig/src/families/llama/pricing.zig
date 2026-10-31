//! What a drafted round is worth: rounds' times by their rows, and the depth landing the most tokens a millisecond.
const std = @import("std");

/// The newest sample's weight in the running times.
const rate = 0.1;

pub fn ema(kept: ?f64, sample: f64) f64 {
    return if (kept) |k| k + rate * (sample - k) else sample;
}

/// A round's milliseconds by its window's rows, a line fitted to the recent rounds (a window costs more with rows).
pub const Fit = struct {
    samples: u32 = 0,
    x: f64 = 0, // rows
    y: f64 = 0, // ms
    xx: f64 = 0,
    xy: f64 = 0,

    pub fn add(f: *Fit, rows: f64, ms: f64) void {
        if (f.samples == 0) {
            f.* = .{ .samples = 1, .x = rows, .y = ms, .xx = rows * rows, .xy = rows * ms };
            return;
        }
        f.samples += 1;
        f.x = ema(f.x, rows);
        f.y = ema(f.y, ms);
        f.xx = ema(f.xx, rows * rows);
        f.xy = ema(f.xy, rows * ms);
    }

    /// Milliseconds a window of `rows` takes: the fitted line, or the mean while the rows seen hardly vary.
    pub fn at(f: Fit, rows: f64) f64 {
        const spread = f.xx - f.x * f.x;
        if (spread < 0.25) return f.y;
        const slope = @max(0, (f.xy - f.x * f.y) / spread);
        return @max(f.y + slope * (rows - f.x), 0.5 * f.y);
    }
};

/// The depth with the most tokens a millisecond: a round lands 1 + p + ... + p^k tokens for k + 1 rows and k steps.
pub fn depth(p: f64, rounds: Fit, step_ms: f64, most: usize) usize {
    var best: usize = 1;
    var best_rate: f64 = 0;
    var expect: f64 = 1;
    var pk: f64 = 1;
    for (1..most + 1) |k| {
        pk *= p;
        expect += pk;
        const kf: f64 = @floatFromInt(k);
        const r = expect / (rounds.at(kf + 1) + kf * step_ms);
        if (r > best_rate) {
            best_rate = r;
            best = k;
        }
    }
    return best;
}

/// A shared round's depth for each of `streams` streams: most tokens a millisecond for the round, 0 if drafting costs.
pub fn depthShared(p: f64, rounds: Fit, step_ms: f64, most: usize, streams: usize) usize {
    const s: f64 = @floatFromInt(streams);
    var best: usize = 0;
    var best_rate = s / rounds.at(s);
    var expect: f64 = 1;
    var pk: f64 = 1;
    for (1..most + 1) |k| {
        pk *= p;
        expect += pk;
        const kf: f64 = @floatFromInt(k);
        const r = s * expect / (rounds.at(s * (kf + 1)) + s * kf * step_ms);
        if (r > best_rate) {
            best_rate = r;
            best = k;
        }
    }
    return best;
}

test "llama pricing: the depth pays for its steps and rows, more at a higher acceptance, none past the ask" {
    // flat 60 ms rounds and 3 ms steps: at p 0.7 drafting about 5-6 pays best; at 0.95 the window's 15; at 0.1 one
    var flat: Fit = .{};
    flat.add(4, 60);
    try std.testing.expect(depth(0.7, flat, 3, 15) >= 4 and depth(0.7, flat, 3, 15) <= 7);
    try std.testing.expectEqual(@as(usize, 15), depth(0.95, flat, 3, 15));
    try std.testing.expectEqual(@as(usize, 1), depth(0.1, flat, 3, 15));
    try std.testing.expectEqual(@as(usize, 3), depth(0.95, flat, 3, 3));
    // dear steps pull the depth down
    try std.testing.expect(depth(0.7, flat, 30, 15) <= 2);
    // rounds that grow 5 ms a row (as the M5's windows do) pull it down too
    var sloped: Fit = .{};
    for (0..40) |i| sloped.add(@floatFromInt(2 + i % 8), 50 + 5 * @as(f64, @floatFromInt(2 + i % 8)));
    try std.testing.expectApproxEqAbs(@as(f64, 100), sloped.at(10), 3);
    try std.testing.expect(depth(0.7, sloped, 3, 15) < depth(0.7, flat, 3, 15));
    try std.testing.expect(depth(0.95, sloped, 3, 15) >= 8);
    // one row count seen: the mean stands in for the line
    var one: Fit = .{};
    one.add(4, 60);
    one.add(4, 70);
    try std.testing.expectApproxEqAbs(one.y, one.at(12), 1e-9);
}

test "llama pricing: a shared round drafts where it pays for every stream, and not where it costs more" {
    // 60 ms rounds growing 5 ms a row, 3 ms steps: two streams at a high acceptance still draft
    var sloped: Fit = .{};
    for (0..40) |i| sloped.add(@floatFromInt(2 + i % 8), 50 + 5 * @as(f64, @floatFromInt(2 + i % 8)));
    try std.testing.expect(depthShared(0.9, sloped, 3, 15, 2) >= 2);
    // a low acceptance: the round runs undrafted
    try std.testing.expectEqual(@as(usize, 0), depthShared(0.2, sloped, 3, 15, 2));
    // more streams make a row dearer for the round: never deeper than for fewer
    try std.testing.expect(depthShared(0.7, sloped, 3, 15, 4) <= depthShared(0.7, sloped, 3, 15, 2));
    try std.testing.expect(depthShared(0.99, sloped, 3, 3, 2) <= 3);
}
