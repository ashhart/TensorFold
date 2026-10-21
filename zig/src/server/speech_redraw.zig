//! Opt-in redraws of whole speech replies that the silence guard ended.
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

    pub fn count(p: Policy) u8 {
        return p.max orelse 0;
    }

    pub fn validate(p: Policy, silence: api.speech_guard.Config) !void {
        if (p.budget_s) |s| if (!std.math.isFinite(s) or s <= 0) return error.BadRedrawBudget;
        if (p.count() == 0 and (p.budget_s != null or p.controls != null)) return error.MissingRedrawCount;
        if (p.count() > 0 and !silence.enabled()) return error.MissingSilenceGuard;
    }

    pub fn refusal(p: Policy, model_type: []const u8) ?[]const u8 {
        if (p.max != null and !std.mem.eql(u8, model_type, "fish_omni"))
            return "--silence-redraws and its options are for Drama speech: serve without them";
        return null;
    }
};

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

test "redraw policy is off by default and needs a count and an enabled guard" {
    const guard: api.speech_guard.Config = .{ .floor_dbfs = -60, .seconds = 1 };
    try (Policy{}).validate(.{});
    try (Policy{ .max = 2, .budget_s = 30, .controls = .{} }).validate(guard);
    try std.testing.expectError(error.MissingSilenceGuard, (Policy{ .max = 1 }).validate(.{}));
    try std.testing.expectError(error.MissingRedrawCount, (Policy{ .budget_s = 30 }).validate(guard));
    try std.testing.expectError(error.MissingRedrawCount, (Policy{ .controls = .{} }).validate(guard));
    for ([_]f64{ 0, -1, std.math.nan(f64), std.math.inf(f64) }) |bad|
        try std.testing.expectError(error.BadRedrawBudget, (Policy{ .max = 1, .budget_s = bad }).validate(guard));
}

test "an explicit redraw count, even 0, is refused on text and accepted on Drama speech" {
    for ([_]Policy{ .{ .max = 0 }, .{ .max = 2 } }) |p| {
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
