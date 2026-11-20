//! LACI's detector (zonos3.tts.laci.Detector): skips, stalls and truncations in a request's reading of its text.
const std = @import("std");
const Allocator = std.mem.Allocator;

/// LaciConfig's defaults, which the reference servers run.
pub const Config = struct {
    bin_tokens: usize = 8,
    cover_frac: f64 = 0.5,
    skip_frac: f64 = 0.15,
    stall_factor: f64 = 3.0,
    min_stall_frames: f64 = 45,
    tail_frames: usize = 15,
    rollback_margin_frames: usize = 8,
    max_retries: usize = 8,
    pace_window_bins: usize = 16,
    warmup_bins: usize = 2,
    /// Added to the stop id's logit in every codebook once the text is done and `stop_boost_delay` rows followed.
    stop_boost: f32 = 3.0,
    stop_boost_delay: usize = 6,
};

pub const Kind = enum { skip, hallucination, truncation, short_run_on };

pub const Failure = struct { kind: Kind, detect_row: usize, onset_row: usize, rollback_row: usize, bin: usize, frontier: usize };

/// The streaming skip, hallucination and truncation detector: one (row, pos) a decode step, rows rising.
pub const Detector = struct {
    a: Allocator,
    text_len: usize,
    prior: f64,
    c: Config,
    min_rollback_row: usize,
    nb: usize,
    obs_rows: std.ArrayList(usize) = .empty,
    obs_pos: std.ArrayList(usize) = .empty,
    /// Rows of the truncate() calls that still lie inside the kept observations.
    bumps: std.ArrayList(usize) = .empty,
    counts: []usize,
    first_row: []?usize,
    frontier: usize = 0,
    last_advance_row: ?usize = null,
    done_row: ?usize = null,
    /// The first row the reading was on the last text token.
    read_row: ?usize = null,

    pub fn init(a: Allocator, text_len: usize, frames_per_token: f64, c: Config, min_rollback_row: usize) !Detector {
        if (text_len < 1) return error.EmptyText;
        const nb = (text_len + c.bin_tokens - 1) / c.bin_tokens;
        const d: Detector = .{ .a = a, .text_len = text_len, .prior = frames_per_token, .c = c, .min_rollback_row = min_rollback_row, .nb = nb, .counts = try a.alloc(usize, nb), .first_row = try a.alloc(?usize, nb) };
        @memset(d.counts, 0);
        @memset(d.first_row, null);
        return d;
    }

    pub fn deinit(d: *Detector) void {
        d.obs_rows.deinit(d.a);
        d.obs_pos.deinit(d.a);
        d.bumps.deinit(d.a);
        d.a.free(d.counts);
        d.a.free(d.first_row);
    }

    fn reset(d: *Detector) void {
        @memset(d.counts, 0);
        @memset(d.first_row, null);
        d.frontier = 0;
        d.last_advance_row = null;
        d.done_row = null;
        d.read_row = null;
    }

    fn tokens(d: *const Detector, b: usize) usize {
        return @min(d.c.bin_tokens, d.text_len - b * d.c.bin_tokens);
    }

    /// The median frames a token over the last pace_window_bins bins below the frontier; the prior before warmup.
    pub fn paceTok(d: *const Detector) f64 {
        const lo = d.frontier -| d.c.pace_window_bins;
        const n = d.frontier - lo;
        if (n < d.c.warmup_bins) return d.prior;
        var rates: [64]f64 = undefined;
        std.debug.assert(n <= rates.len);
        for (rates[0..n], lo..) |*r, x| r.* = @as(f64, @floatFromInt(d.counts[x])) / @as(f64, @floatFromInt(d.tokens(x)));
        std.mem.sort(f64, rates[0..n], {}, std.sort.asc(f64));
        return if (n % 2 == 1) rates[n / 2] else (rates[n / 2 - 1] + rates[n / 2]) / 2.0;
    }

    fn share(d: *const Detector, b: usize) f64 {
        return d.paceTok() * @as(f64, @floatFromInt(d.tokens(b)));
    }

    /// The last observed pos at a row below `row`; 0 when there is none.
    pub fn lastPosBefore(d: *const Detector, row: usize) usize {
        var i: usize = 0;
        while (i < d.obs_rows.items.len and d.obs_rows.items[i] < row) i += 1;
        return if (i == 0) 0 else d.obs_pos.items[i - 1];
    }

    pub fn stopRow(d: *const Detector) ?usize {
        return if (d.done_row) |r| r + d.c.tail_frames else null;
    }

    fn rollback(d: *const Detector, onset: usize) usize {
        return @max(onset -| d.c.rollback_margin_frames, d.min_rollback_row);
    }

    /// Every skip or stall still to come (rows from `next_row`) rolls back to this row or later; null: none can come.
    pub fn commitRow(d: *const Detector, next_row: usize) ?usize {
        if (d.done_row != null) return null;
        var onset = @min(d.last_advance_row orelse next_row, next_row);
        if (d.frontier + 2 < d.nb) for (d.first_row[d.frontier + 2 ..]) |r| {
            if (r) |row| onset = @min(onset, row);
        };
        return d.rollback(onset);
    }

    pub fn observe(d: *Detector, row: usize, pos: usize) !?Failure {
        if (pos >= d.text_len) return error.PosOutsideText;
        if (d.obs_rows.items.len != 0 and row <= d.obs_rows.items[d.obs_rows.items.len - 1]) return error.RowsMustIncrease;
        try d.obs_rows.append(d.a, row);
        try d.obs_pos.append(d.a, pos);
        return d.step(row, pos);
    }

    fn step(d: *Detector, row: usize, pos: usize) ?Failure {
        const c = d.c;
        const b = pos / c.bin_tokens;
        d.counts[b] += 1;
        if (d.first_row[b] == null) d.first_row[b] = row;
        if (d.last_advance_row == null) d.last_advance_row = row;
        // the frontier, or a skip: a bin below the one now read that got far fewer steps than its share
        if (b > d.frontier and @as(f64, @floatFromInt(d.counts[b])) >= c.cover_frac * d.share(b)) {
            for (d.frontier + 1..b) |x| {
                if (@as(f64, @floatFromInt(d.counts[x])) < c.skip_frac * d.share(x)) {
                    var onset: usize = std.math.maxInt(usize);
                    for (d.first_row[x + 1 ..]) |r| {
                        if (r) |first| onset = @min(onset, first);
                    }
                    return .{ .kind = .skip, .detect_row = row, .onset_row = onset, .rollback_row = d.rollback(onset), .bin = x, .frontier = b };
                }
            }
            d.frontier = b;
            d.last_advance_row = row;
        }
        // a stall: the reading has not advanced for stall_factor bins at the pace so far
        if (d.done_row == null) {
            const limit = @max(c.stall_factor * d.paceTok() * @as(f64, @floatFromInt(c.bin_tokens)), c.min_stall_frames);
            if (@as(f64, @floatFromInt(row - d.last_advance_row.?)) > limit) {
                const onset = d.last_advance_row.?;
                return .{ .kind = .hallucination, .detect_row = row, .onset_row = onset, .rollback_row = d.rollback(onset), .bin = d.frontier + 1, .frontier = d.frontier };
            }
        }
        // the text is done: the last bin read for its share, and the reading has reached the last text token
        if (d.read_row == null and pos + 1 >= d.text_len) d.read_row = row;
        if (d.done_row == null and d.read_row != null and d.frontier + 2 >= d.nb and @as(f64, @floatFromInt(d.counts[d.nb - 1])) >= d.share(d.nb - 1)) d.done_row = row;
        return null;
    }

    /// The model emitted EOS at `row`: a truncation when two or more bins are unread.
    pub fn onEos(d: *const Detector, row: usize) ?Failure {
        if (d.frontier + 2 >= d.nb) return null;
        return .{ .kind = .truncation, .detect_row = row, .onset_row = row, .rollback_row = d.rollback(row), .bin = d.frontier + 1, .frontier = d.frontier };
    }

    /// Drops the observations at rows from `row` and replays the rest; what survives gets a full stall budget.
    pub fn truncate(d: *Detector, row: usize) !void {
        var keep: usize = 0;
        while (keep < d.obs_rows.items.len and d.obs_rows.items[keep] < row) keep += 1;
        var kept: usize = 0;
        for (d.bumps.items) |r| {
            if (r < row) {
                d.bumps.items[kept] = r;
                kept += 1;
            }
        }
        d.bumps.shrinkRetainingCapacity(kept);
        d.obs_rows.shrinkRetainingCapacity(keep);
        d.obs_pos.shrinkRetainingCapacity(keep);
        d.reset();
        var pending: usize = 0;
        for (d.obs_rows.items, d.obs_pos.items) |r, p| {
            // each earlier truncation still inside the kept rows moves the stall clock at the same point again
            while (pending < kept and r > d.bumps.items[pending]) : (pending += 1) {
                if (d.last_advance_row) |last| d.last_advance_row = @max(last, d.bumps.items[pending]);
            }
            if (d.step(r, p) != null) return error.ReplayRaisedAFailure;
        }
        if (d.last_advance_row) |last| d.last_advance_row = @max(last, row);
        try d.bumps.append(d.a, row);
    }
};

test "detector: a steady reading is done at the last bin's share, and an early EOS is a truncation" {
    const t = std.testing;
    var d = try Detector.init(t.allocator, 20, 2.0, .{}, 3);
    defer d.deinit();
    // three bins (8, 8 and 4 tokens) at two frames a token
    var row: usize = 3;
    for (0..20) |pos| for (0..2) |_| {
        try t.expectEqual(@as(?Failure, null), try d.observe(row, pos));
        if (pos < 4) try t.expect(d.onEos(row) != null);
        row += 1;
    };
    try t.expectEqual(@as(usize, 2), d.frontier);
    try t.expectEqual(@as(f64, 2.0), d.paceTok());
    // the last token is first read at row 41, where the last bin has seven of its eight frames; done a row later
    try t.expectEqual(@as(?usize, 42), d.done_row);
    try t.expectEqual(@as(?usize, 57), d.stopRow());
    try t.expectEqual(@as(?Failure, null), d.onEos(row));
    try t.expectEqual(@as(?usize, null), d.commitRow(row));
}

test "detector: a bin the reading jumps over is a skip, rolled back from the first row past it" {
    const t = std.testing;
    var d = try Detector.init(t.allocator, 40, 2.0, .{}, 0);
    defer d.deinit();
    var row: usize = 0;
    for (0..8) |pos| for (0..2) |_| {
        _ = try d.observe(row, pos);
        row += 1;
    };
    // bin 1 unread; bin 2 read from row 16 for half its share of 16 frames
    var failure: ?Failure = null;
    for (0..8) |i| {
        failure = try d.observe(row, 16 + i / 2);
        row += 1;
        if (failure != null) break;
    }
    try t.expectEqual(Failure{ .kind = .skip, .detect_row = 23, .onset_row = 16, .rollback_row = 8, .bin = 1, .frontier = 2 }, failure.?);
    try d.truncate(8);
    try t.expectEqual(@as(usize, 0), d.frontier);
    try t.expectEqual(@as(usize, 3), d.lastPosBefore(8));
    try t.expectEqual(@as(?usize, 8), d.last_advance_row);
}

test "detector: a reading that stops advancing is a hallucination after the stall limit" {
    const t = std.testing;
    var d = try Detector.init(t.allocator, 40, 1.0, .{}, 0);
    defer d.deinit();
    var failure: ?Failure = null;
    var row: usize = 0;
    while (failure == null) : (row += 1) failure = try d.observe(row, 2);
    // the limit is max(3 * 1 * 8, 45) frames past the last advance, row 0
    try t.expectEqual(Failure{ .kind = .hallucination, .detect_row = 46, .onset_row = 0, .rollback_row = 0, .bin = 1, .frontier = 0 }, failure.?);
}
