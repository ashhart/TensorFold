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

    pub fn fromEnv() Share {
        const v = std.c.getenv("TENSORFOLD_PREFILL_SHARE") orelse return .{};
        const p = std.fmt.parseFloat(f64, std.mem.span(v)) catch return .{};
        return .{ .p = if (p > 0 and p <= 1) p else 1 };
    }

    /// A step that took `ns` while `decoding` streams were live: the decode time it owes them.
    pub fn stepped(s: *Share, ns: i64, decoding: bool) void {
        if (s.p >= 1 or !decoding) return;
        s.debt_ns += @intFromFloat(@as(f64, @floatFromInt(ns)) * (1 - s.p) / s.p);
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
    while (h.filling.items.len > 0 and short(h.filling.items[pick(h)])) {
        h.fillAt(next(h));
        while (h.admitOne()) {}
    }
    if (h.filling.items.len == 0 or !due(h)) return;
    if (h.share.p >= 1) return h.fillAt(next(h));
    const decoding = h.core.activeCount() > 0;
    const t0 = std.Io.Clock.awake.now(h.io);
    h.fillAt(next(h));
    h.share.stepped(@intCast(t0.durationTo(std.Io.Clock.awake.now(h.io)).toNanoseconds()), decoding);
}

/// A prompt whose whole pass is at most `short_rows` rows (a long prompt's last rows are not).
fn short(j: *const LaneHost.Job) bool {
    return j.stream.prompt_len -| j.stream.cached <= Share.short_rows;
}

/// A long prompt's step runs now: none owed, or nothing decoding.
fn due(h: *LaneHost) bool {
    return h.share.debt_ns <= 0 or h.core.activeCount() == 0;
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
