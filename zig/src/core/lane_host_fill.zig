//! Prefer foreground and least remaining prefill work, with bounded waiting for every admitted fill.
const std = @import("std");
const LaneHost = @import("lane_host.zig").LaneHost;

pub const preference_rounds = 8;

/// After eight skipped choices, oldest wait wins; each fill runs within eight plus the admitted lane count.
pub fn next(h: *const LaneHost) usize {
    const at = pick(h);
    for (h.filling.items, 0..) |j, i| j.fill_wait = if (i == at) 0 else j.fill_wait +| 1;
    return at;
}

/// The fill `next` takes, its waits untouched.
fn pick(h: *const LaneHost) usize {
    var at: usize = 0;
    for (h.filling.items, 0..) |j, i| {
        const best = h.filling.items[at];
        const aged = j.fill_wait >= preference_rounds;
        const best_aged = best.fill_wait >= preference_rounds;
        if (aged != best_aged) {
            if (aged) at = i;
        } else if (aged) {
            if (j.fill_wait > best.fill_wait) at = i;
        } else if (j.request.background != best.request.background) {
            if (!j.request.background) at = i;
        } else if (remaining(h, j) < remaining(h, best)) at = i;
    }
    return at;
}

fn remaining(h: *const LaneHost, j: *const LaneHost.Job) u64 {
    const completed = if (j.fill_began and h.core.backend.vtable.reports_fill)
        @max(j.stream.cached, j.stream.fill_at.load(.monotonic))
    else
        j.stream.cached;
    return j.stream.prompt_len -| completed;
}

/// Takes `job` off the filling list, wherever it is.
pub fn unfill(h: *LaneHost, job: anytype) void {
    for (h.filling.items, 0..) |j, i| if (j == job) {
        _ = h.filling.orderedRemove(i);
        return;
    };
}

/// Prompt steps' share p of wall time beside decoders (TENSORFOLD_PREFILL_SHARE): a step of t owes them t (1 - p) / p.
pub const Share = struct {
    p: f64 = 1,
    debt_ns: i64 = 0, // decode time owed before the next step of a long prompt

    pub const short_rows = 512;
    pub const min_share = 0.01; // below it a step's debt is minutes of decoding, and is no share

    pub fn fromEnv() Share {
        return fromText(std.mem.span(std.c.getenv("TENSORFOLD_PREFILL_SHARE") orelse return .{}));
    }

    /// A share from its text: anything but a number in [min_share, 1] is the default, 1.
    pub fn fromText(text: []const u8) Share {
        const p = std.fmt.parseFloat(f64, text) catch return .{};
        return .{ .p = if (p >= min_share and p <= 1) p else 1 };
    }

    /// A step that took `ns` while `decoding` streams were live: the decode time it owes them.
    pub fn stepped(s: *Share, ns: i64, decoding: bool) void {
        if (s.p >= 1 or !decoding) return;
        const owed = @as(f64, @floatFromInt(ns)) * (1 - s.p) / s.p;
        s.debt_ns +|= @intFromFloat(@min(owed, @as(f64, @floatFromInt(std.math.maxInt(i64) / 2))));
    }

    /// A decode round of `ns` pays the debt down.
    pub fn rounded(s: *Share, ns: i64) void {
        s.debt_ns = @max(0, s.debt_ns - ns);
    }
};

/// Prompts before the next round: none waiting is main's round; short ones go whole; then a long one's step when due.
pub fn run(h: *LaneHost) void {
    if (h.filling.items.len == 0) {
        h.share.debt_ns = 0;
        return;
    }
    // short prompts back to back, as a whole pass at admission went: a burst of them waits on no decode round
    var burst: usize = 0; // a lane count of short prompts begun, then a round: replies ending with their prompt free lanes without end
    while (h.filling.items.len > 0 and short(h.filling.items[pick(h)])) {
        if (!h.filling.items[pick(h)].fill_began) {
            if (burst == h.info_.lanes and runnable(h) > 0) return;
            burst += 1;
        }
        h.fillAt(next(h));
        @import("lane_host_end.zig").takeCancels(h);
        while (h.admitOne()) {}
    }
    if (h.filling.items.len == 0 or !due(h)) return;
    if (h.share.p >= 1) return h.fillAt(next(h));
    const decoding = runnable(h) > 0;
    const t0 = std.Io.Clock.awake.now(h.io);
    h.fillAt(next(h));
    h.share.stepped(@intCast(t0.durationTo(std.Io.Clock.awake.now(h.io)).toNanoseconds()), decoding);
}

/// A prompt whose whole pass is at most `short_rows` rows (a long prompt's last rows are not).
fn short(j: *const LaneHost.Job) bool {
    return j.stream.prompt_len -| j.stream.cached <= Share.short_rows;
}

/// A long prompt's step runs now: none owed, or nothing decoding.
pub fn due(h: *LaneHost) bool {
    return h.share.debt_ns <= 0 or runnable(h) == 0;
}

/// Streams the next decode round runs: unfinished and not paused.
fn runnable(h: *const LaneHost) usize {
    var n: usize = 0;
    for (h.core.live.items) |s| n += @intFromBool(!s.finished and !s.paused);
    return n;
}

test "a share of 0.5 owes the decoders a step's time; 1 owes nothing; rounds pay it down" {
    var s: Share = .{ .p = 0.5 };
    s.stepped(1000, true);
    try std.testing.expectEqual(@as(i64, 1000), s.debt_ns);
    s.stepped(1000, false);
    try std.testing.expectEqual(@as(i64, 1000), s.debt_ns);
    s.rounded(400);
    s.rounded(800);
    try std.testing.expectEqual(@as(i64, 0), s.debt_ns);
    var one: Share = .{};
    one.stepped(1000, true);
    try std.testing.expectEqual(@as(i64, 0), one.debt_ns);
    var q: Share = .{ .p = 0.25 };
    q.stepped(1000, true);
    try std.testing.expectEqual(@as(i64, 3000), q.debt_ns);
}

test "a share below the floor, NaN and inf fall back to 1; the floor and 1 stand" {
    for ([_][]const u8{ "1e-12", "0.0099", "0", "-0.5", "nan", "inf", "-inf", "2", "x", "" }) |text| {
        try std.testing.expectEqual(@as(f64, 1), Share.fromText(text).p);
    }
    try std.testing.expectEqual(@as(f64, 0.01), Share.fromText("0.01").p);
    try std.testing.expectEqual(@as(f64, 1), Share.fromText("1").p);
}

test "a step's debt saturates instead of overflowing" {
    var s: Share = .{ .p = Share.min_share };
    s.stepped(std.math.maxInt(i64), true);
    try std.testing.expect(s.debt_ns > 0);
    s.stepped(std.math.maxInt(i64), true);
    try std.testing.expectEqual(@as(i64, std.math.maxInt(i64)), s.debt_ns);
    s.rounded(1000);
    try std.testing.expect(s.debt_ns < std.math.maxInt(i64));
}
