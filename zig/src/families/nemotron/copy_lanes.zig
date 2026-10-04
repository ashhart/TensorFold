//! Copy lanes for a lone stream's GPU round: a chain copied after the context's matched suffix, priced by its rows' hits.
const std = @import("std");

pub const max_rows = 30;

/// A round's cost in ms by window rows (the load-time window table, read between its timed widths).
pub const Costs = struct {
    rows: []const f64, // ms at each timed width
    widths: []const usize,

    pub fn at(c: Costs, rows: usize) f64 {
        var i: usize = 0;
        while (i + 1 < c.widths.len and c.widths[i + 1] <= rows) i += 1;
        if (i + 1 == c.widths.len or c.widths[i] >= rows) return c.rows[i] * @as(f64, @floatFromInt(rows)) / @as(f64, @floatFromInt(@max(c.widths[i], 1)));
        const span: f64 = @floatFromInt(c.widths[i + 1] - c.widths[i]);
        const t = @as(f64, @floatFromInt(rows - c.widths[i])) / span;
        return c.rows[i] + t * (c.rows[i + 1] - c.rows[i]);
    }
};

/// How copied rows landed, by their place in the copy (decayed counts; a row with few offers leans on them all).
pub const Hits = struct {
    offered: [max_rows]f64 = @splat(0),
    landed: [max_rows]f64 = @splat(0),

    const prior = 2.0; // pseudo-offers at the pooled share
    const prior_share = 0.95; // the pooled share before any copy
    const keep = 0.97;

    /// A copy of `rows` rows after `before` rows that had to land first, in a window that kept `kept` rows.
    pub fn add(h: *Hits, before: u32, rows: u32, kept: u32) void {
        for (1..@min(rows, max_rows) + 1) |i| {
            if (before + i > kept) break;
            h.offered[i - 1] = keep * h.offered[i - 1] + 1;
            h.landed[i - 1] = keep * h.landed[i - 1] + @as(f64, @floatFromInt(@intFromBool(before + i + 1 <= kept)));
        }
    }

    pub fn pooled(h: *const Hits) f64 {
        var l: f64 = 0;
        var o: f64 = 0;
        for (h.landed, h.offered) |x, y| {
            l += x;
            o += y;
        }
        return if (o > 0) l / o else prior_share;
    }

    /// Row i's chance to land once the rows before it did.
    pub fn share(h: *const Hits, i: usize, pool: f64) f64 {
        return (h.landed[i] + prior * pool) / (h.offered[i] + prior);
    }
};

pub const Copy = struct {
    most: usize, // copy rows a window takes at most
    window: Hits = .{}, // copies that start the window
    tail: Hits = .{}, // copies after the head's drafts
    head_rate: f64 = 0, // tokens a ms of the stream's head windows (recent)
    since_match: usize = 1 << 20, // rounds since the GPU last found a long match
    rounds: usize = 0,
    head_tokens: f64 = 0, // tokens and modelled ms of the head's windows, and of every window
    head_ms: f64 = 0,
    all_ms: f64 = 0,

    /// A round's copy rows (0: a head window), tail rows, rows, kept tokens, long match, modelled ms, its head part's ms.
    pub fn observe(c: *Copy, copy_rows: u32, tail_rows: u32, window_rows: u32, kept: u32, matched: bool, ms: f64, head_ms: f64) void {
        c.rounds += 1;
        c.since_match = if (matched) 0 else c.since_match + 1;
        c.all_ms += ms;
        if (copy_rows > 0) return c.window.add(0, copy_rows, kept);
        if (tail_rows > 0) c.tail.add(window_rows - 1 - tail_rows, tail_rows, kept);
        const head_kept = @min(kept, window_rows - tail_rows);
        c.head_tokens += @floatFromInt(head_kept);
        c.head_ms += head_ms;
        if (head_ms > 0) {
            const r = @as(f64, @floatFromInt(head_kept)) / head_ms;
            c.head_rate = if (c.head_rate == 0) r else c.head_rate + 0.1 * (r - c.head_rate);
        }
    }

    /// The share of the stream's time its head windows took (1 before any copy), by the window table.
    pub fn headShare(c: *const Copy) f64 {
        return if (c.all_ms > 0 and c.head_ms > 0) c.head_ms / c.all_ms else 1;
    }

    /// Copy rows for the next window (0: the head's) with the most tokens a ms; none without a long match lately.
    pub fn rows(c: *const Copy, costs: Costs, over_ms: f64) u32 {
        if (c.since_match > 16) return 0;
        const pool = c.window.pooled();
        const most = c.most;
        var best: usize = 0;
        var best_rate = c.head_rate;
        var expected: f64 = 1;
        var reach: f64 = 1;
        for (1..most + 1) |m| {
            reach *= c.window.share(m - 1, pool);
            expected += reach;
            const rate = expected / (costs.at(1 + m) + over_ms);
            if (rate > best_rate) {
                best = m;
                best_rate = rate;
            }
        }
        // every sixteenth round a short copy, so a stream's copy rows stay measured
        if (best < 2 and c.rounds % 16 == 0) best = @min(4, most);
        return if (best >= 2) @intCast(best) else 0;
    }

    /// Whether the head's drafts may be continued by a copy: while those copies' first rows land often enough.
    pub fn tails(c: *const Copy) bool {
        return c.tail.share(0, c.tail.pooled()) >= 0.5 or c.rounds % 16 == 0;
    }
};

test "copy rows that keep landing price a long window; ones that miss price none" {
    const widths = [_]usize{ 1, 2, 4, 8, 16, 24, 32 };
    const ms = [_]f64{ 5, 6.3, 8.3, 11.6, 17, 24, 30 };
    const costs = Costs{ .rows = &ms, .widths = &widths };
    var good = Copy{ .most = 24, .head_rate = 0.25, .since_match = 0 };
    for (0..40) |_| good.observe(20, 0, 21, 21, true, 0, 0);
    try std.testing.expect(good.rows(costs, 0.5) >= 12);
    var bad = Copy{ .most = 24, .head_rate = 0.25, .since_match = 0 };
    for (0..40) |_| bad.observe(8, 0, 9, 1, true, 0, 0);
    try std.testing.expectEqual(@as(u32, 0), bad.rows(costs, 0.5) * @intFromBool(bad.rounds % 16 != 0));
    try std.testing.expectApproxEqAbs(@as(f64, 12.275), costs.at(9), 1e-9);
}
