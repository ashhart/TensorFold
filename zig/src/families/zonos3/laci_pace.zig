//! LACI's duration targets (zonos3.tts.laci: DurationTarget, RateBuckets, PaceMonitor): a take held to a length.
const std = @import("std");
const detect = @import("laci_detect.zig");
const Allocator = std.mem.Allocator;

pub const tolerance = 0.10;
pub const slack_frames = 3;
pub const smooth_steps = 5;
pub const max_retries = 2;
/// The clamp of the duration-ratio estimate a retry bucket is computed from.
pub const ratio_range = [2]f64{ 0.5, 2.0 };
pub const min_stall_frames = 15;

/// The take's target length in generated rows; `adjust_rate`: a miss regenerates in the bucket the ratio asks for.
pub const Target = struct { frames: f64, adjust_rate: bool = true };

/// A speaking-rate scale: the condition row's index and buckets; bucket i is [edges[i-first-1], edges[i-first]).
pub const Rates = struct {
    index: usize,
    edges: []const f64,
    n_buckets: i64,
    first: i64 = 0,

    pub fn last(r: Rates) i64 {
        return r.first + @as(i64, @intCast(r.edges.len));
    }

    /// The condition row of a bucket: itself, or "unknown" (n_buckets) for a bucket off the row.
    pub fn row(r: Rates, bucket: i64) u32 {
        return @intCast(if (bucket >= 0 and bucket < r.n_buckets) bucket else r.n_buckets);
    }

    /// A bucket's representative rate: its midpoint; the open ends extend by half a neighbour.
    pub fn rate(r: Rates, bucket: i64) f64 {
        const e = r.edges;
        const i = bucket - r.first;
        if (i <= 0) return e[0] - (e[1] - e[0]) / 2;
        if (i >= e.len) return e[e.len - 1] + (e[e.len - 1] - e[e.len - 2]) / 2;
        const k: usize = @intCast(i);
        return (e[k - 1] + e[k]) / 2;
    }

    /// A pace retry's bucket: the rate that makes the ratio 1, at least a bucket away; null off the scale.
    pub fn retryBucket(r: Rates, bucket: i64, ratio: f64) ?i64 {
        const x = r.rate(bucket) * ratio;
        // bisect.bisect_right
        var n: usize = 0;
        while (n < r.edges.len and r.edges[n] <= x) n += 1;
        var want = r.first + @as(i64, @intCast(n));
        want = if (ratio > 1) @max(want, bucket + 1) else @min(want, bucket - 1);
        return if (want >= r.first and want <= r.last()) want else null;
    }
};

/// A duration miss, as the session records it; `ratio`: the take's estimated duration over the target.
pub const Miss = struct { failure: detect.Failure, ratio: f64 };

pub const Seen = union(enum) { miss: Miss, stall: detect.Failure };

/// The duration check over the reading position (PaceMonitor): read too early is pace_fast, too late pace_slow.
pub const Monitor = struct {
    a: Allocator,
    target: f64,
    origin: usize,
    min_rollback_row: usize,
    margin: f64,
    c: detect.Config,
    /// Kinds already recorded as ignored.
    ignored_fast: bool = false,
    ignored_slow: bool = false,
    read_row: ?usize = null,
    rows: std.ArrayList(usize) = .empty,
    pos: std.ArrayList(usize) = .empty,
    best: ?f64 = null,
    best_row: ?usize = null,

    pub fn init(a: Allocator, target_frames: f64, origin_row: usize, min_rollback_row: usize, c: detect.Config) !Monitor {
        if (!(target_frames > 0)) return error.BadDurationTarget;
        return .{ .a = a, .target = target_frames, .origin = origin_row, .min_rollback_row = min_rollback_row, .margin = tolerance * target_frames + slack_frames, .c = c };
    }

    pub fn deinit(m: *Monitor) void {
        m.rows.deinit(m.a);
        m.pos.deinit(m.a);
    }

    /// Rows without an advance that make a stall: stall_factor bins at the target pace, at least min_stall_frames.
    fn stallFrames(m: *const Monitor, text_len: usize) f64 {
        const per_token = m.target / @as(f64, @floatFromInt(@max(1, text_len)));
        return @max(m.c.stall_factor * @as(f64, @floatFromInt(m.c.bin_tokens)) * per_token, min_stall_frames);
    }

    /// statistics.median of the positions from `lo` to `hi`, exclusive.
    fn median(m: *const Monitor, lo: usize, hi: usize) f64 {
        var buf: [smooth_steps]usize = undefined;
        const w = buf[0 .. hi - lo];
        @memcpy(w, m.pos.items[lo..hi]);
        std.mem.sort(usize, w, {}, std.sort.asc(usize));
        const mid = w.len / 2;
        if (w.len % 2 == 1) return @floatFromInt(w[mid]);
        return (@as(f64, @floatFromInt(w[mid - 1])) + @as(f64, @floatFromInt(w[mid]))) / 2.0;
    }

    pub fn observe(m: *Monitor, row: usize, pos: usize, text_len: usize, text_done: bool, frontier: usize) !?Seen {
        if (m.rows.items.len != 0 and row <= m.rows.items[m.rows.items.len - 1]) return error.PaceRowsMustIncrease;
        try m.rows.append(m.a, row);
        try m.pos.append(m.a, pos);
        if (m.read_row != null) return null;
        const n = m.pos.items.len;
        const smooth = m.median(n -| smooth_steps, n);
        if (m.best == null or smooth > m.best.?) {
            m.best = smooth;
            m.best_row = row;
        }
        const last: f64 = @floatFromInt(@as(i64, @intCast(text_len)) - 2);
        const elapsed: f64 = @floatFromInt(@as(i64, @intCast(row)) - @as(i64, @intCast(m.origin)));
        if (smooth >= last or text_done) {
            m.read_row = row;
            if (elapsed < m.target - m.margin) return .{ .miss = m.miss(.pace_fast, row, frontier, elapsed / m.target) };
            return null;
        }
        if (@as(f64, @floatFromInt(@as(i64, @intCast(row)) - @as(i64, @intCast(m.best_row.?)))) > m.stallFrames(text_len)) {
            const onset = m.best_row.?;
            return .{ .stall = .{ .kind = .hallucination, .detect_row = row, .onset_row = onset, .rollback_row = @max(onset -| m.c.rollback_margin_frames, m.min_rollback_row), .bin = frontier + 1, .frontier = frontier } };
        }
        if (elapsed > m.target + m.margin) {
            const first: f64 = @floatFromInt(m.pos.items[0]);
            const share = (smooth - first) / @max(1.0, last - first);
            const lower = elapsed / m.target;
            const ratio = if (share > 0) lower / share else ratio_range[1];
            return .{ .miss = m.miss(.pace_slow, row, frontier, @max(ratio, lower)) };
        }
        return null;
    }

    fn miss(m: *const Monitor, kind: detect.Kind, row: usize, frontier: usize, ratio: f64) Miss {
        const f: detect.Failure = .{ .kind = kind, .detect_row = row, .onset_row = m.min_rollback_row, .rollback_row = m.min_rollback_row, .bin = frontier, .frontier = frontier };
        return .{ .failure = f, .ratio = @min(ratio_range[1], @max(ratio_range[0], ratio)) };
    }

    /// The stream was rolled back to `row`: its observations from there on are dropped.
    pub fn truncate(m: *Monitor, row: usize) void {
        var keep: usize = 0;
        while (keep < m.rows.items.len and m.rows.items[keep] < row) keep += 1;
        m.rows.shrinkRetainingCapacity(keep);
        m.pos.shrinkRetainingCapacity(keep);
        if (m.read_row) |r| if (r >= row) {
            m.read_row = null;
        };
        m.best = null;
        m.best_row = null;
        for (m.rows.items, 0..) |r, k| {
            const smooth = m.median((k + 1) -| smooth_steps, k + 1);
            if (m.best == null or smooth > m.best.?) {
                m.best = smooth;
                m.best_row = r;
            }
        }
        // the regeneration gets a full stall budget
        if (m.best_row) |b| m.best_row = @max(b, row);
    }
};

test "rates: a bucket's rate, and the retry bucket a duration ratio asks for" {
    const t = std.testing;
    const r: Rates = .{ .index = 2, .edges = &.{ 10, 12, 14, 16 }, .n_buckets = 5 };
    try t.expectEqual(@as(f64, 9), r.rate(0));
    try t.expectEqual(@as(f64, 13), r.rate(2));
    try t.expectEqual(@as(f64, 17), r.rate(4));
    // twice as long as wanted: twice the rate, at least a bucket faster; half: slower
    try t.expectEqual(@as(?i64, 4), r.retryBucket(2, 2.0));
    try t.expectEqual(@as(?i64, 3), r.retryBucket(2, 1.05));
    try t.expectEqual(@as(?i64, 0), r.retryBucket(2, 0.5));
    try t.expectEqual(@as(?i64, null), r.retryBucket(0, 0.5));
    try t.expectEqual(@as(u32, 5), r.row(7));
}
