//! Opt-in redraws of whole speech replies that the silence or register guard tripped, or that stopped short.
const std = @import("std");
const api = @import("engine_api");
const clock = @import("clock.zig");
const drama_allow = @import("drama_allow.zig");

pub const Policy = struct {
    /// --silence-redraws as given; null when omitted, so an explicit 0 still names a speech-only flag.
    max: ?u8 = null,
    /// Seconds from receipt that every redraw must fit; null leaves only the client's X-Deadline-S.
    budget_s: ?f64 = null,
    /// Null keeps the input; an empty list makes it plain; nonempty lists use the delivery allowlist.
    controls: ?drama_allow.Policy = null,
    /// A stop with less audio than this share of the input's expected length is redrawn; needs `short_chars_per_s`.
    short_ratio: ?f64 = null,
    /// Input characters per second of audio in complete replies of the served voice; the expected length's rate.
    short_chars_per_s: ?f64 = null,
    /// The register guard's thresholds: off unless all are set.
    register: @import("speech_register.zig").Rule = .{},

    pub fn count(p: Policy) u8 {
        return p.max orelse 0;
    }

    pub fn validate(p: Policy, silence: api.speech_guard.Config) !void {
        if (p.budget_s) |s| if (!std.math.isFinite(s) or s <= 0) return error.BadRedrawBudget;
        if (p.short_ratio) |r| if (!(r > 0 and r < 1)) return error.BadShortReplyRatio;
        if (p.short_chars_per_s) |c| if (!std.math.isFinite(c) or c <= 0) return error.BadShortReplyRate;
        if (p.short_ratio != null and p.short_chars_per_s == null) return error.MissingShortReplyRate;
        if (p.short_chars_per_s != null and p.short_ratio == null) return error.MissingShortReplyRatio;
        try p.register.validate();
        if (p.count() == 0 and (p.controls != null or p.short_ratio != null or p.budget_s != null and !p.register.on())) return error.MissingRedrawCount;
        if (p.count() > 0 and !silence.enabled()) return error.MissingSilenceGuard;
    }

    pub fn refusal(p: Policy, model_type: []const u8) ?[]const u8 {
        if ((p.max != null or p.register.on()) and !std.mem.eql(u8, model_type, "fish_omni"))
            return "--silence-redraws, the register flags and their options are for Drama speech: serve without them";
        return null;
    }

    /// Whether a take of `input` that ended `reason` with `samples` at `rate` is redrawn: silenced, or a short stop.
    pub fn trips(p: Policy, reason: api.Reason, samples: usize, rate: u32, input: []const u8) bool {
        if (reason == .silence) return true;
        const ratio = p.short_ratio orelse return false;
        // bytes bound code points from above, so a short input is never counted
        if (reason != .stop or input.len < short_min_chars) return false;
        const chars = std.unicode.utf8CountCodepoints(input) catch return false;
        if (chars < short_min_chars) return false;
        const seconds = @as(f64, @floatFromInt(samples)) / @as(f64, @floatFromInt(rate));
        return seconds < ratio * @as(f64, @floatFromInt(chars)) / p.short_chars_per_s.?;
    }
};

/// Inputs of fewer characters are never judged short: a fixed lead-in and tail dominate their duration.
pub const short_min_chars = 200;

/// The lower of the server budget and a request's X-Deadline-S, a positive finite number of seconds.
pub fn budget(server_s: ?f64, header: ?[]const u8) error{BadDeadline}!?f64 {
    const text = std.mem.trim(u8, header orelse return server_s, " \t");
    const s = std.fmt.parseFloat(f64, text) catch return error.BadDeadline;
    if (!std.math.isFinite(s) or s <= 0) return error.BadDeadline;
    return @min(s, server_s orelse s);
}

/// Whether t + d fits `budget_s`: t runs from `received` to `now`, d is the last take's run from `started` to `now`.
pub fn fits(budget_s: ?f64, received: i96, started: i96, now: i96) bool {
    const limit = budget_s orelse return true;
    return clock.seconds(now - received) + clock.seconds(now - started) <= limit;
}

test "redraw policy is off by default; redraws need the guard, a budget redraws or the register guard" {
    const guard: api.speech_guard.Config = .{ .floor_dbfs = -60, .seconds = 1 };
    const register: @import("speech_register.zig").Rule = .{ .st = 6, .high = 50, .high_hz = 300 };
    try (Policy{}).validate(.{});
    try (Policy{ .max = 2, .budget_s = 30, .controls = .{}, .short_ratio = 0.25, .short_chars_per_s = 10 }).validate(guard);
    try (Policy{ .budget_s = 30, .register = register }).validate(.{});
    try std.testing.expectError(error.MissingSilenceGuard, (Policy{ .max = 1 }).validate(.{}));
    try std.testing.expectError(error.MissingRedrawCount, (Policy{ .budget_s = 30 }).validate(guard));
    try std.testing.expectError(error.MissingRedrawCount, (Policy{ .controls = .{} }).validate(guard));
    try std.testing.expectError(error.MissingRedrawCount, (Policy{ .controls = .{}, .register = register }).validate(guard));
    try std.testing.expectError(error.MissingRedrawCount, (Policy{ .short_ratio = 0.25, .short_chars_per_s = 10 }).validate(guard));
    try std.testing.expectError(error.MissingShortReplyRate, (Policy{ .max = 1, .short_ratio = 0.25 }).validate(guard));
    try std.testing.expectError(error.MissingShortReplyRatio, (Policy{ .max = 1, .short_chars_per_s = 10 }).validate(guard));
    for ([_]f64{ 0, -1, std.math.nan(f64), std.math.inf(f64) }) |bad| {
        try std.testing.expectError(error.BadRedrawBudget, (Policy{ .max = 1, .budget_s = bad }).validate(guard));
        try std.testing.expectError(error.BadShortReplyRatio, (Policy{ .max = 1, .short_ratio = bad, .short_chars_per_s = 10 }).validate(guard));
        try std.testing.expectError(error.BadShortReplyRate, (Policy{ .max = 1, .short_ratio = 0.25, .short_chars_per_s = bad }).validate(guard));
    }
    try std.testing.expectError(error.BadShortReplyRatio, (Policy{ .max = 1, .short_ratio = 1, .short_chars_per_s = 10 }).validate(guard));
}

test "a stop of enough characters trips only under the ratio of its expected length; silence always trips" {
    const p: Policy = .{ .max = 1, .short_ratio = 0.25, .short_chars_per_s = 10 };
    // 200 characters at 10 a second are 20 s expected: under a ratio of 0.25, a stop under 5 s of audio is short
    const letters: [200]u8 = @splat('a');
    const input: []const u8 = &letters;
    try std.testing.expect(p.trips(.stop, 4_999, 1000, input));
    try std.testing.expect(!p.trips(.stop, 5_000, 1000, input));
    try std.testing.expect(!p.trips(.length, 1, 1000, input));
    try std.testing.expect(p.trips(.silence, 60_000, 1000, input));
    try std.testing.expect(!p.trips(.stop, 1, 1000, input[1..]));
    // characters are code points: 199 of two bytes each are too few, 200 are enough
    const accents: [200][2]u8 = @splat(.{ 0xc3, 0xa9 });
    try std.testing.expect(!p.trips(.stop, 1, 1000, std.mem.sliceAsBytes(accents[1..])));
    try std.testing.expect(p.trips(.stop, 1, 1000, std.mem.sliceAsBytes(&accents)));
    const off: Policy = .{ .max = 1 };
    try std.testing.expect(!off.trips(.stop, 1, 1000, input));
    try std.testing.expect(off.trips(.silence, 1, 1000, input));
}

test "an explicit redraw count, even 0, or the register guard is refused on text and accepted on Drama speech" {
    for ([_]Policy{ .{ .max = 0 }, .{ .max = 2 }, .{ .register = .{ .st = 6, .high = 50, .high_hz = 300 } } }) |p| {
        try std.testing.expect(p.refusal("nemotron_h") != null);
        try std.testing.expect(p.refusal("fish_omni") == null);
    }
    try std.testing.expect((Policy{}).refusal("nemotron_h") == null);
}

test "the budget is the lower of the server's and a well-formed X-Deadline-S" {
    try std.testing.expectEqual(@as(?f64, null), try budget(null, null));
    try std.testing.expectEqual(@as(?f64, 120), try budget(120, null));
    try std.testing.expectEqual(@as(?f64, 2.5), try budget(null, " 2.5 "));
    try std.testing.expectEqual(@as(?f64, 30), try budget(120, "30"));
    try std.testing.expectEqual(@as(?f64, 120), try budget(120, "600"));
    for ([_][]const u8{ "", "0", "-3", "nan", "inf", "soon" }) |bad| try std.testing.expectError(error.BadDeadline, budget(120, bad));
}

test "a redraw fits when elapsed time plus the last take's duration is within the budget" {
    const s = std.time.ns_per_s;
    try std.testing.expect(fits(null, 0, 0, 1000 * s));
    try std.testing.expect(fits(34, 0, 16 * s, 25 * s));
    try std.testing.expect(!fits(34, 0, 9 * s, 25 * s));
    try std.testing.expect(fits(34, 0, 0, 17 * s));
    try std.testing.expect(!fits(34, 0, 0, 17 * s + 1));
}
