//! TENSORFOLD_PREFILL_ORDER=srpt: the side with less estimated work left takes the next round.
const std = @import("std");
const lanes = @import("lanes");

pub const Srpt = struct {
    on: bool = false, // TENSORFOLD_PREFILL_ORDER=srpt; off keeps the fixed share and main's order
    floor: f64 = 0.1, // the long prompt's least share of wall time beside replies: it always makes progress
    fill_ns_row: f64 = 0, // EMA of a prompt row's fill time; 0 until a step is timed
    round_ns: f64 = 0, // EMA of a decode round's time; 0 until a round is timed
    round_tokens: f64 = 0, // EMA of the tokens a round commits, all streams together

    pub const alpha = 0.25; // the weight a new sample carries in each EMA

    pub fn fromEnv() Srpt {
        return if (std.c.getenv("TENSORFOLD_PREFILL_ORDER")) |text| fromText(std.mem.span(text)) else .{};
    }

    /// `srpt`; anything else is off.
    pub fn fromText(text: []const u8) Srpt {
        return .{ .on = std.mem.eql(u8, std.mem.trim(u8, text, " \t"), "srpt") };
    }

    /// A prompt step that filled `rows` in `ns`: its rows' fill time. No rows shown is no sample.
    pub fn filled(s: *Srpt, rows: u64, ns: i64) void {
        if (rows == 0 or ns <= 0) return;
        s.fill_ns_row = ema(s.fill_ns_row, @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(rows)));
    }

    /// A decode round of `ns` that committed `tokens` over every live stream.
    pub fn rounded(s: *Srpt, ns: i64, tokens: u64) void {
        if (ns <= 0 or tokens == 0) return;
        s.round_ns = ema(s.round_ns, @as(f64, @floatFromInt(ns)));
        s.round_tokens = ema(s.round_tokens, @as(f64, @floatFromInt(tokens)));
    }

    fn ema(old: f64, sample: f64) f64 {
        return if (old > 0) old + alpha * (sample - old) else sample;
    }

    /// Whether some reply's estimated time left is under the long prompt's; false until both are sampled.
    pub fn decodeFirst(s: *const Srpt, fill_rows_left: u64, reply_tokens_left: []const i64) bool {
        if (!s.on or s.fill_ns_row <= 0 or s.round_ns <= 0 or s.round_tokens <= 0) return false;
        var runnable: usize = 0;
        for (reply_tokens_left) |t| runnable += @intFromBool(t > 0);
        if (runnable == 0) return false;
        const per_round = @max(s.round_tokens / @as(f64, @floatFromInt(runnable)), 1e-9);
        const fill_ns = @as(f64, @floatFromInt(fill_rows_left)) * s.fill_ns_row;
        for (reply_tokens_left) |t| {
            if (t <= 0) continue;
            if (@as(f64, @floatFromInt(t)) * s.round_ns / per_round < fill_ns) return true;
        }
        return false;
    }

    /// The prompt's share of this step: the floor when a reply goes first, else the whole step.
    pub fn share(s: *const Srpt, decode_first: bool) f64 {
        return if (decode_first) s.floor else 1;
    }
};

/// The runnable replies' tokens left, into `buf`; replies past its end are left out.
pub fn replyTokens(live: []const *lanes.Stream, buf: []i64) []const i64 {
    var n: usize = 0;
    for (live) |s| {
        if (n == buf.len) break;
        if (s.finished or s.paused) continue;
        buf[n] = s.budgetLeft();
        n += 1;
    }
    return buf[0..n];
}

/// The tokens every live stream has emitted: a round's committed tokens are the difference across it.
pub fn emittedTotal(live: []const *lanes.Stream) u64 {
    var n: u64 = 0;
    for (live) |s| n +|= s.emitted().len;
    return n;
}

/// A decode round of `ns` that ran over `live`, `before` being emittedTotal taken before it.
pub fn round(s: *Srpt, ns: i64, live: []const *lanes.Stream, before: u64) void {
    s.rounded(ns, emittedTotal(live) -| before);
}

test "TENSORFOLD_PREFILL_ORDER: srpt is on, anything else or unset is off" {
    const env = @import("sys").env;
    const name = "TENSORFOLD_PREFILL_ORDER";
    _ = env.unsetenv(name);
    try std.testing.expect(!Srpt.fromEnv().on);
    _ = env.setenv(name, "srpt", 1);
    defer _ = env.unsetenv(name);
    try std.testing.expect(Srpt.fromEnv().on);
    for ([_][]const u8{ "", "SRPT", "fcfs", "1", "srpt2" }) |text| try std.testing.expect(!Srpt.fromText(text).on);
    try std.testing.expect(Srpt.fromText(" srpt\t").on);
}

test "srpt: no sample yet is no order; a short reply goes first, a long one does not" {
    var s: Srpt = .{ .on = true };
    try std.testing.expect(!s.decodeFirst(900_000, &.{200}));
    s.filled(1000, 500_000_000); // 0.5 ms a row
    try std.testing.expect(!s.decodeFirst(900_000, &.{200})); // the round is still unsampled
    s.rounded(60_000_000, 2); // 60 ms a round, 2.5 tokens a round after the EMA
    s.rounded(60_000_000, 4);
    try std.testing.expectEqual(@as(f64, 2.5), s.round_tokens);
    try std.testing.expect(s.decodeFirst(900_000, &.{200})); // 200 tokens: 4.8 s against 450 s of fill
    try std.testing.expect(!s.decodeFirst(1_000, &.{1_000_000})); // 24,000 s against 0.5 s of fill
    try std.testing.expect(!s.decodeFirst(900_000, &.{ 0, -1 })); // nothing runnable
    try std.testing.expectEqual(@as(f64, 0.1), s.share(true));
    try std.testing.expectEqual(@as(f64, 1), s.share(false));
}

test "srpt: a step at the floor owes nine times its time; a whole step owes nothing" {
    var s: Srpt = .{ .on = true };
    var share: @import("lane_host_fill.zig").Share = .{};
    share.steppedAt(1000, true, s.share(true));
    try std.testing.expectEqual(@as(i64, 9000), share.debt_ns);
    share.steppedAt(1000, true, s.share(false));
    try std.testing.expectEqual(@as(i64, 9000), share.debt_ns);
    var whole: @import("lane_host_fill.zig").Share = .{};
    whole.steppedAt(1000, true, 1);
    try std.testing.expectEqual(@as(i64, 0), whole.debt_ns);
    whole.steppedAt(1000, false, 0.1);
    try std.testing.expectEqual(@as(i64, 0), whole.debt_ns);
}

test "srpt off: the host's share and debt are main's" {
    const env = @import("sys").env;
    _ = env.unsetenv("TENSORFOLD_PREFILL_ORDER");
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    const host = @import("lane_host.zig").LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    try std.testing.expect(!host.srpt.on);
    try std.testing.expectEqual(@as(f64, 1), host.share.p);
}
