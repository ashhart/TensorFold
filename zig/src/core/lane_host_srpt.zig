//! TENSORFOLD_PREFILL_ORDER=srpt: the side with less estimated work left takes the next round.
const std = @import("std");
const lanes = @import("lanes");

pub const Srpt = struct {
    on: bool = false, // TENSORFOLD_PREFILL_ORDER=srpt; off keeps the fixed share and main's order
    floor: f64 = 0.1, // the long prompt's least share of wall time beside replies: it always makes progress
    fill_ns_row: f64 = 0, // EMA of a prompt row's fill time; 0 until a step is timed
    round_ns: f64 = 0, // a decode round's time, averaged over `round_alpha`; 0 until a round is timed
    round_tokens: f64 = 0, // the tokens a round commits, all streams together, averaged with it
    rounds: u32 = 0, // rounds sampled: their plain mean until `round_alpha` takes over
    think_ema: f64 = 0, // EMA of the think blocks this server has seen close, in tokens; 0 until one has

    pub const alpha = 0.25; // the weight a new fill sample carries
    pub const round_alpha = 1.0 / 32.0; // drafts land in bursts: a round's tokens swing threefold, a stretch of them doesn't
    pub const margin = 0.25; // a reply goes first when it would end within this part of the fill left; near a tie the prompt keeps its lead

    pub fn fromEnv() Srpt {
        return if (std.c.getenv("TENSORFOLD_PREFILL_ORDER")) |text| fromText(std.mem.span(text)) else .{};
    }

    /// A family's default order: `on` unless the env names one.
    pub fn withDefault(on: bool) Srpt {
        return if (std.c.getenv("TENSORFOLD_PREFILL_ORDER")) |text| fromText(std.mem.span(text)) else .{ .on = on };
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
        s.rounds +|= 1;
        const w = @max(round_alpha, 1.0 / @as(f64, @floatFromInt(s.rounds)));
        s.round_ns += w * (@as(f64, @floatFromInt(ns)) - s.round_ns);
        s.round_tokens += w * (@as(f64, @floatFromInt(tokens)) - s.round_tokens);
    }

    /// A think block of `len` tokens has closed: reasoning replies are estimated by it from here on.
    pub fn thought(s: *Srpt, len: usize) void {
        s.think_ema = ema(s.think_ema, @floatFromInt(len));
    }

    /// A stream whose think block has closed joins the average once, also when its reply ended in that round.
    pub fn seen(s: *Srpt, t: *lanes.Stream) void {
        const at = t.answer_from orelse return;
        if (t.think_seen) return;
        t.think_seen = true;
        s.thought(at);
    }

    /// A reasoning reply's work left: its answer quota plus the reasoning left by the think blocks seen, else `left`.
    pub fn estimate(s: *const Srpt, left: i64, answer_max: u32, answer_from: ?usize, emitted_len: usize) i64 {
        if (answer_max == 0 or answer_from != null or s.think_ema <= 0) return left;
        const done: f64 = @floatFromInt(emitted_len);
        const reasoning_left = if (done < s.think_ema) s.think_ema - done else done;
        const work = @as(f64, @floatFromInt(answer_max)) + reasoning_left;
        return @min(left, @as(i64, @intFromFloat(@min(work, 9.0e15))));
    }

    fn ema(old: f64, sample: f64) f64 {
        return if (old > 0) old + alpha * (sample - old) else sample;
    }

    /// Whether a reply would end within `margin` of the long prompt's estimated time left; false before both samples.
    pub fn decodeFirst(s: *const Srpt, fill_rows_left: u64, reply_tokens_left: []const i64) bool {
        if (!s.on or s.fill_ns_row <= 0 or s.round_ns <= 0 or s.round_tokens <= 0) return false;
        var runnable: usize = 0;
        for (reply_tokens_left) |t| runnable += @intFromBool(t > 0);
        if (runnable == 0) return false;
        const per_round = @max(s.round_tokens / @as(f64, @floatFromInt(runnable)), 1e-9);
        const fill_ns = @as(f64, @floatFromInt(fill_rows_left)) * s.fill_ns_row;
        for (reply_tokens_left) |t| {
            if (t <= 0) continue;
            if (@as(f64, @floatFromInt(t)) * s.round_ns / per_round < fill_ns * margin) return true;
        }
        return false;
    }

    /// Whether a prompt of `rows` rows to fill can lend replies more: `lent_ns` so far, `margin` of its time at most.
    pub fn lends(s: *const Srpt, rows: u64, lent_ns: f64) bool {
        return lent_ns < @as(f64, @floatFromInt(rows)) * s.fill_ns_row * margin;
    }

    /// The prompt's share of a step: `p`, at most the floor when a reply goes first; a whole prompt stays whole.
    pub fn share(s: *const Srpt, decode_first: bool, p: f64, whole: bool) f64 {
        return if (decode_first and !whole) @min(p, s.floor) else p;
    }
};

/// The runnable replies' tokens left (a reasoning one's by `Srpt.estimate`) into `buf`; later replies are left out.
pub fn replyTokens(srpt: *const Srpt, live: []const *lanes.Stream, buf: []i64) []const i64 {
    var n: usize = 0;
    for (live) |s| {
        if (n == buf.len) break;
        if (s.finished or s.paused) continue;
        buf[n] = srpt.estimate(s.budgetLeft(), s.answer_max, s.answer_from, s.emitted().len);
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
    for (live) |t| s.seen(t);
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

test "TENSORFOLD_PREFILL_ORDER: a family's default stands until the env names an order" {
    const env = @import("sys").env;
    const name = "TENSORFOLD_PREFILL_ORDER";
    _ = env.unsetenv(name);
    try std.testing.expect(Srpt.withDefault(true).on);
    try std.testing.expect(!Srpt.withDefault(false).on);
    _ = env.setenv(name, "fifo", 1);
    defer _ = env.unsetenv(name);
    try std.testing.expect(!Srpt.withDefault(true).on);
    _ = env.setenv(name, "srpt", 1);
    try std.testing.expect(Srpt.withDefault(false).on);
}

test "srpt: no sample yet is no order; a short reply goes first, a long one does not" {
    var s: Srpt = .{ .on = true };
    try std.testing.expect(!s.decodeFirst(900_000, &.{200}));
    s.filled(1000, 500_000_000); // 0.5 ms a row
    try std.testing.expect(!s.decodeFirst(900_000, &.{200})); // the round is still unsampled
    s.rounded(60_000_000, 2); // 60 ms a round, 3 tokens a round: the first rounds' plain mean
    s.rounded(60_000_000, 4);
    try std.testing.expectEqual(@as(f64, 3), s.round_tokens);
    try std.testing.expect(s.decodeFirst(900_000, &.{200})); // 200 tokens: 4 s against 450 s of fill
    try std.testing.expect(!s.decodeFirst(1_000, &.{1_000_000})); // 20,000 s against 0.5 s of fill
    try std.testing.expect(!s.decodeFirst(900_000, &.{6_000})); // 120 s of reply against 450 s: past the margin, the prompt leads
    try std.testing.expect(s.decodeFirst(900_000, &.{5_000})); // 100 s: inside it
    try std.testing.expect(!s.decodeFirst(900_000, &.{ 0, -1 })); // nothing runnable
    try std.testing.expectEqual(@as(f64, 0.1), s.share(true, 1, false));
    try std.testing.expectEqual(@as(f64, 1), s.share(false, 1, false));
}

test "srpt: a burst of landed drafts doesn't swing the estimate" {
    var s: Srpt = .{ .on = true };
    s.filled(1000, 3_000_000_000); // 3 ms a row: 32,000 rows left is 96 s
    for (0..64) |i| s.rounded(40_000_000, if (i % 2 == 0) 1 else 5); // 3 tokens a round on average
    const before = s.round_tokens;
    s.rounded(40_000_000, 5);
    s.rounded(40_000_000, 5);
    try std.testing.expect(@abs(s.round_tokens - before) < 0.2);
    for (0..8) |_| {
        s.rounded(40_000_000, 1);
        s.rounded(40_000_000, 5);
        try std.testing.expect(!s.decodeFirst(32_000, &.{8_000})); // 107 s of reply against 96 s of fill: the prompt, every time
        try std.testing.expect(s.decodeFirst(32_000, &.{1_000})); // 13 s: the reply, every time
    }
}

test "srpt: a prompt lends at most a quarter of its own time, however many replies keep coming" {
    var s: Srpt = .{ .on = true };
    s.filled(1000, 3_000_000_000); // 3 ms a row: 8,000 rows is 24 s, so 6 s to lend
    try std.testing.expect(s.lends(8_000, 0));
    try std.testing.expect(s.lends(8_000, 5_999_000_000));
    try std.testing.expect(!s.lends(8_000, 6_000_000_000));
    try std.testing.expect(!s.lends(0, 0));
}

test "srpt: a step at the floor owes nine times its time; a whole step owes nothing" {
    var s: Srpt = .{ .on = true };
    var share: @import("lane_host_fill.zig").Share = .{};
    share.steppedAt(1000, true, s.share(true, 1, false));
    try std.testing.expectEqual(@as(i64, 9000), share.debt_ns);
    share.steppedAt(1000, true, s.share(false, 1, false));
    try std.testing.expectEqual(@as(i64, 9000), share.debt_ns);
    var whole: @import("lane_host_fill.zig").Share = .{};
    whole.steppedAt(1000, true, 1);
    try std.testing.expectEqual(@as(i64, 0), whole.debt_ns);
    whole.steppedAt(1000, false, 0.1);
    try std.testing.expectEqual(@as(i64, 0), whole.debt_ns);
}

test "srpt honours --decode-share: its share when the prompt goes first, the floor at most otherwise, 0 whole" {
    const s: Srpt = .{ .on = true };
    try std.testing.expectEqual(@as(f64, 0.5), s.share(false, 0.5, false));
    try std.testing.expectEqual(@as(f64, 0.1), s.share(true, 0.5, false));
    try std.testing.expectEqual(@as(f64, 0.05), s.share(true, 0.05, false));
    try std.testing.expectEqual(@as(f64, 1), s.share(true, 1, true));
    try std.testing.expectEqual(@as(f64, 1), s.share(false, 1, true));
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

test "srpt: a reply that reasons is its answer quota plus the reasoning left, once a think block has been seen" {
    var s: Srpt = .{ .on = true };
    try std.testing.expectEqual(@as(i64, 262_000), s.estimate(262_000, 2_000, null, 100)); // no think block seen: budget
    s.thought(1_000);
    try std.testing.expectEqual(@as(i64, 2_900), s.estimate(262_000, 2_000, null, 100)); // 900 of the reasoning left
    try std.testing.expectEqual(@as(i64, 5_000), s.estimate(262_000, 2_000, null, 3_000)); // past the average: again as long
    try std.testing.expectEqual(@as(i64, 1_500), s.estimate(1_500, 2_000, null, 100)); // never above the budget
    try std.testing.expectEqual(@as(i64, 262_000), s.estimate(262_000, 0, null, 100)); // no quota: the budget stands
    try std.testing.expectEqual(@as(i64, 1_200), s.estimate(1_200, 2_000, 700, 900)); // answer begun: budget exact
}

test "srpt: a reasoning reply with a short think block goes first beside a long prompt, a long one does not" {
    var s: Srpt = .{ .on = true };
    s.filled(1000, 1_000_000_000); // 1 ms a row: 110,000 rows is 110 s, so a reply of 27 s or less goes first
    s.rounded(30_000_000, 3); // 30 ms a round, 3 tokens a round: 100 tokens a second
    try std.testing.expect(!s.decodeFirst(110_000, &.{262_000})); // today: the rest of the window
    s.thought(500);
    try std.testing.expect(s.decodeFirst(110_000, &.{s.estimate(262_000, 1_000, null, 100)})); // 1,400 tokens: 14 s
    s.thought(20_000);
    s.thought(20_000);
    s.thought(20_000);
    try std.testing.expect(!s.decodeFirst(110_000, &.{s.estimate(262_000, 1_000, null, 100)})); // reasons as long as before
}

test "srpt: a reasoning stream under a reserve counts as main before the first think block, then goes first" {
    const gpa = std.testing.allocator;
    var s: Srpt = .{ .on = true };
    s.filled(1000, 1_000_000_000); // 1 ms a row: 110,000 rows is 110 s, so a reply of 27 s or less goes first
    s.rounded(30_000_000, 3); // 100 tokens a second
    var reply = try lanes.Stream.init(gpa, .{ .id = "r", .prompt = &.{1}, .max_new = 262_444, .think_end = 91, .answer_max = 300 });
    defer reply.deinit(gpa);
    _ = try reply.commit(gpa, &.{ 5, 6, 7 }, &.{});
    const live = [_]*lanes.Stream{&reply};
    var buf: [4]i64 = undefined;
    try std.testing.expectEqual(@as(i64, 262_441), replyTokens(&s, &live, &buf)[0]); // no sample: budgetLeft, as on main
    try std.testing.expect(!s.decodeFirst(110_000, replyTokens(&s, &live, &buf)));
    s.thought(500);
    try std.testing.expectEqual(@as(i64, 797), replyTokens(&s, &live, &buf)[0]); // 300 and 497 reasoning left
    try std.testing.expect(s.decodeFirst(110_000, replyTokens(&s, &live, &buf)));
    _ = try reply.commit(gpa, &.{ 8, 91 }, &.{}); // the think block has closed: the answer quota is exact
    try std.testing.expectEqual(@as(i64, 300), replyTokens(&s, &live, &buf)[0]);
}

test "srpt: a think block's length joins the average once, when its reply has closed it" {
    const gpa = std.testing.allocator;
    var s: Srpt = .{ .on = true };
    var a = try lanes.Stream.init(gpa, .{ .id = "a", .prompt = &.{1}, .max_new = 100, .think_end = 91, .answer_max = 10 });
    defer a.deinit(gpa);
    var b = try lanes.Stream.init(gpa, .{ .id = "b", .prompt = &.{1}, .max_new = 100, .think_end = 91, .answer_max = 10 });
    defer b.deinit(gpa);
    _ = try a.commit(gpa, &.{ 5, 6, 7, 91 }, &.{});
    const live = [_]*lanes.Stream{ &a, &b };
    round(&s, 1_000_000, &live, 0);
    try std.testing.expectEqual(@as(f64, 4), s.think_ema);
    round(&s, 1_000_000, &live, 0);
    try std.testing.expectEqual(@as(f64, 4), s.think_ema); // not counted twice, and b is still reasoning
    _ = try b.commit(gpa, &.{ 5, 6, 7, 8, 9, 10, 11, 91 }, &.{});
    round(&s, 1_000_000, &live, 0);
    try std.testing.expectEqual(@as(f64, 5), s.think_ema); // 4 + (8 - 4) / 4
}

test "srpt: a think block that closes in the round that ends its reply joins the average" {
    const gpa = std.testing.allocator;
    var s: Srpt = .{ .on = true };
    var r = try lanes.Stream.init(gpa, .{ .id = "r", .prompt = &.{1}, .max_new = 5, .think_end = 91, .answer_max = 2 });
    defer r.deinit(gpa);
    _ = try r.commit(gpa, &.{ 5, 6, 7, 91, 8 }, &.{});
    try std.testing.expect(r.finished);
    s.seen(&r);
    try std.testing.expectEqual(@as(f64, 4), s.think_ema);
    s.seen(&r);
    try std.testing.expectEqual(@as(f64, 4), s.think_ema);
}
