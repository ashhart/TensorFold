//! The load-time expert race is shared; only MTP consumes measured window and step prices.
const std = @import("std");
const rule = @import("lanes").cost_rule;
pub const widths = [_]u32{ 1, 2, 3, 4, 6, 8, 12, 16, 24, 32 };
const race = [_]usize{ 1, 3, 5 };
const margin = 0.01;
const warm = 8;
const levels = [2]usize{ 1, 8 };
pub const Kept = extern struct { ms: [widths.len]f64, step_ms: f64, fact: u64 };

/// Warm the target, race its expert forms unless `pin` names one, then measure only the prices this drafter consumes.
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
    if (pin) |f| try timer.form(f);
    for (0..warm) |_| _ = try w.time(widths.len - 1);
    const fact = pin orelse race: {
        var forms: [2]f64 = .{ 0, 0 };
        for (&forms, [_]bool{ false, true }) |*sum, f| {
            try timer.form(f);
            for (race) |i| sum.* += try rule.fastest(w, i);
            if (!block) sum.* += try rule.fastest(d, levels[0]);
        }
        const won = forms[1] < forms[0] * (1 - margin);
        try timer.form(won);
        break :race won;
    };
    var k: Kept = .{ .ms = @splat(0), .step_ms = 0, .fact = @intFromBool(fact) };
    if (block) return k;
    for (&k.ms, 0..) |*m, i| m.* = try rule.fastest(w, i);
    try rule.smooth(w, &k.ms);
    const chain = [2]f64{ try rule.fastest(d, levels[0]), try rule.fastest(d, levels[1]) };
    k.step_ms = @max(0, (chain[1] - chain[0]) / @as(f64, @floatFromInt(levels[1] - levels[0])));
    return k;
}

/// A form-only entry cannot masquerade as a window table, or the other way around.
pub fn reusable(k: Kept, block: bool) bool {
    if (k.fact > 1) return false;
    if (block) {
        for (k.ms) |m| if (m != 0) return false;
        return k.step_ms == 0;
    }
    for (k.ms) |m| if (!(m > 0) or !std.math.isFinite(m)) return false;
    return k.step_ms >= 0 and std.math.isFinite(k.step_ms);
}
