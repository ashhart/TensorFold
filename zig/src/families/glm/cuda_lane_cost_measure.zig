//! The load-time expert race is shared; only MTP consumes measured window and step prices.
const std = @import("std");
const rule = @import("lanes").cost_rule;
pub const widths = [_]u32{ 1, 2, 3, 4, 6, 8, 12, 16, 24, 32 };
const margin = 0.01;
/// Untimed sweeps of every width before one is timed (both forms when racing): a TP=4 load's first 4 s ran 60% slow.
pub const sweeps = 5;
const levels = [2]usize{ 1, 8 };
/// `fact` holds a bit a width, set where that width's windows run on the factored form.
pub const Kept = extern struct { ms: [widths.len]f64, step_ms: f64, fact: u64 };
/// Every width on the factored form.
pub const all: u64 = (1 << widths.len) - 1;

/// Every width on one form.
pub fn every(factored: bool) u64 {
    return if (factored) all else 0;
}

/// Warm the target, race its expert forms by width unless `pin` names one, then time the prices this drafter consumes.
pub fn measure(timer: anytype, block: bool, pin: ?bool) !Kept {
    const Window = struct {
        timer: @TypeOf(timer),
        pub fn time(w: @This(), i: usize) !f64 {
            return w.timer.window(i);
        }
    };
    const Draft = struct {
        timer: @TypeOf(timer),
        pub fn time(d: @This(), depth: usize) !f64 {
            return d.timer.draft(depth);
        }
    };
    const w: Window = .{ .timer = timer };
    const d: Draft = .{ .timer = timer };
    var k: Kept = .{ .ms = @splat(0), .step_ms = 0, .fact = every(pin orelse false) };
    if (pin) |f| {
        try timer.form(every(f));
        if (block) return k;
        for (0..2 * sweeps) |_| for (0..widths.len) |i| {
            _ = try w.time(i);
        };
        for (&k.ms, 0..) |*m, i| m.* = try rule.fastest(w, i);
    } else {
        // a width runs factored only where its windows beat classic's by the margin; its time is its price
        const forms = try race(timer, w);
        for (forms[0], forms[1], 0..) |classic, factored, i| {
            if (factored < classic * (1 - margin)) k.fact |= @as(u64, 1) << @intCast(i);
        }
        try timer.form(k.fact);
        if (block) return k;
        for (&k.ms, forms[0], forms[1], 0..) |*m, classic, factored, i| m.* = if (on(k.fact, i)) factored else classic;
    }
    try rule.smooth(w, &k.ms);
    const chain = [2]f64{ try rule.fastest(d, levels[0]), try rule.fastest(d, levels[1]) };
    k.step_ms = @max(0, (chain[1] - chain[0]) / @as(f64, @floatFromInt(levels[1] - levels[0])));
    return k;
}

/// Each form's fastest window by width: sweeps, then turns both ways, so no load's start, drift or switch picks one.
fn race(timer: anytype, w: anytype) ![2][widths.len]f64 {
    const both = [2]u64{ 0, all };
    for (0..sweeps) |_| for (0..widths.len) |i| for (both) |f| {
        try timer.form(f);
        _ = try w.time(i);
    };
    var ms: [2][widths.len]f64 = @splat(@splat(std.math.inf(f64)));
    for (0..widths.len) |i| for (0..rule.reps) |r| for (0..2) |k| {
        const p = k ^ (r & 1); // classic first on even rounds, factored on odd ones: a switch weighs on both alike
        try timer.form(both[p]);
        ms[p][i] = @min(ms[p][i], try w.time(i));
    };
    return ms;
}

fn on(fact: u64, i: usize) bool {
    return (fact >> @intCast(i)) & 1 != 0;
}

/// A bit a row count from 1: a window takes the form of the narrowest timed width that holds it, past 32 the widest's.
pub fn rows(fact: u64) u64 {
    var out: u64 = 0;
    var i: usize = 0;
    for (1..65) |r| {
        while (i + 1 < widths.len and widths[i] < r) i += 1;
        if (on(fact, i)) out |= @as(u64, 1) << @intCast(r - 1);
    }
    return out;
}

/// A form-only entry cannot masquerade as a window table, or the other way around.
pub fn reusable(k: Kept, block: bool) bool {
    if (k.fact & ~all != 0) return false;
    if (block) {
        for (k.ms) |m| if (m != 0) return false;
        return k.step_ms == 0;
    }
    for (k.ms) |m| if (!(m > 0) or !std.math.isFinite(m)) return false;
    return k.step_ms >= 0 and std.math.isFinite(k.step_ms);
}
