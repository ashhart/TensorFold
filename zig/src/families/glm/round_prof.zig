//! GLM Metal verify windows: GPU time, idle gaps and host phases by window shape, summarized when streams end.
const std = @import("std");
const platform = @import("sys");

const ticks_per_ms = 24e3;
const bin_ms = 0.05;
const n_bins = 400;

/// What a round measures: the window's own times, then the host phases between one window and the next.
pub const Metric = enum { gpu, gap, ahead, encode, wait, picks, lane, keeps, drafts };
const n_metrics = std.enums.values(Metric).len;
const row_buckets = [_][]const u8{ "1", "2", "3-4", "5-8", "9-16" };
const n_groups = row_buckets.len * 3 * 2;

/// A window's shape and the clock readings its times come from.
pub const Window = struct {
    rows: u32,
    streams: u32,
    drafted: u32,
    t_in: u64,
    committed: u64,
    ret: u64,
    gpu: [2]f64,
    ahead: [2]f64,
    last_end: f64,
};

const Stat = struct {
    sum: f64 = 0,
    max: f64 = 0,
    bins: [n_bins + 1]u32 = @splat(0),

    fn add(s: *Stat, ms: f64) void {
        s.sum += ms;
        s.max = @max(s.max, ms);
        const at: usize = @intFromFloat(@min(@max(ms, 0) / bin_ms, n_bins));
        s.bins[at] += 1;
    }

    /// The bin edge at or above fraction `q` of `n` values; the overflow bin answers with the largest value.
    fn quantile(s: *const Stat, n: u64, q: f64) f64 {
        const want: u64 = @max(1, @as(u64, @intFromFloat(@ceil(q * @as(f64, @floatFromInt(n))))));
        var seen: u64 = 0;
        for (s.bins, 0..) |c, i| {
            seen += c;
            if (seen >= want) return if (i == n_bins) s.max else @min(@as(f64, @floatFromInt(i + 1)) * bin_ms, s.max);
        }
        return s.max;
    }
};

const Group = struct { n: u64 = 0, stats: [n_metrics]Stat = @splat(.{}) };

/// Every field is valid all zero: `create` clears the whole record.
pub const Prof = struct {
    groups: [n_groups]Group,
    pend: [n_metrics]f64, // the round in progress: its window's times, then the host phases as they finish
    pend_group: usize,
    pend_live: bool,
    verify_end: u64,
    lane_open: bool,
    windows: u64,
    reported: u64,
    queued: u64, // one-row rounds queued ahead, which run no window
    keeps: u64,
    partial: u64, // keeps that rejected rows: each replays the KDA states
    kept_rows: u64,
    window_rows: u64,
    drafted_rows: u64,

    /// The round in progress joins its group's histograms.
    fn finish(p: *Prof) void {
        if (!p.pend_live) return;
        const g = &p.groups[p.pend_group];
        g.n += 1;
        for (&g.stats, p.pend) |*s, ms| s.add(ms);
        p.pend_live = false;
    }
};

fn rowBucket(rows: u32) usize {
    return if (rows <= 2) (if (rows == 0) 0 else rows - 1) else if (rows <= 4) 2 else if (rows <= 8) 3 else 4;
}

/// The group of a window's rows, streams and whether any draft rows rode in it.
pub fn group(rows: u32, streams: u32, drafted: bool) usize {
    return (rowBucket(rows) * 3 + @min(@max(streams, 1), 3) - 1) * 2 + @intFromBool(drafted);
}

fn alloc(gpa: std.mem.Allocator) !*Prof {
    const p = try gpa.create(Prof);
    @memset(std.mem.asBytes(p), 0);
    return p;
}

/// GLM_ROUND_PROF on: the record, else null (every hook then costs one branch and no clock read).
pub fn create(gpa: std.mem.Allocator, raw: ?[]const u8) !?*Prof {
    if (!try platform.environment.value(bool, "GLM_ROUND_PROF", raw, false)) return null;
    return try alloc(gpa);
}

pub fn fromEnvironment(gpa: std.mem.Allocator) !?*Prof {
    if (!try platform.environment.read(bool, "GLM_ROUND_PROF", false)) return null;
    return try alloc(gpa);
}

pub fn destroy(p: ?*Prof, gpa: std.mem.Allocator) void {
    if (p) |q| gpa.destroy(q);
}

/// A clock reading for `since`, or 0 with the profile off.
pub fn tick(p: ?*Prof) u64 {
    return if (p != null) platform.c.mach_absolute_time() else 0;
}

/// The host time from `t0` added to this round's `which` phase.
pub fn since(p: ?*Prof, which: Metric, t0: u64) void {
    const q = p orelse return;
    q.pend[@backingInt(which)] += millis(platform.c.mach_absolute_time() -| t0);
}

fn millis(ticks: u64) f64 {
    return @as(f64, @floatFromInt(ticks)) / ticks_per_ms;
}

/// A window ran: the previous round is complete, this one's times are taken.
pub fn window(p: *Prof, w: Window) void {
    p.finish();
    const first = if (w.ahead[1] > 0) w.ahead[0] else w.gpu[0];
    p.pend = @splat(0);
    p.pend[@backingInt(Metric.gpu)] = (w.gpu[1] - w.gpu[0]) * 1e3;
    p.pend[@backingInt(Metric.gap)] = if (w.last_end > 0) @max(first - w.last_end, 0) * 1e3 else 0;
    p.pend[@backingInt(Metric.ahead)] = (w.ahead[1] - w.ahead[0]) * 1e3;
    p.pend[@backingInt(Metric.encode)] = millis(w.committed -| w.t_in);
    p.pend[@backingInt(Metric.wait)] = millis(w.ret -| w.committed);
    p.pend_group = group(w.rows, w.streams, w.drafted > 0);
    p.pend_live = true;
    p.windows += 1;
    p.window_rows += w.rows;
    p.drafted_rows += w.drafted;
}

/// The window's picks are on the host: the lane core's accept time starts.
pub fn verified(p: ?*Prof) void {
    const q = p orelse return;
    q.verify_end = platform.c.mach_absolute_time();
    q.lane_open = true;
}

/// The lane core finished accepting: its time since `verified`, once a round.
pub fn laneEnd(p: ?*Prof) void {
    const q = p orelse return;
    if (!q.lane_open) return;
    q.lane_open = false;
    q.pend[@backingInt(Metric.lane)] += millis(platform.c.mach_absolute_time() -| q.verify_end);
}

/// One stream's keep: `kept` of the window's `rows` stay.
pub fn kept(p: *Prof, n: u32, rows: u32) void {
    p.keeps += 1;
    p.partial += @intFromBool(n < rows);
    p.kept_rows += n;
}

/// The summary, once there is a window since the last one.
pub fn report(p: ?*Prof) void {
    const q = p orelse return;
    q.finish();
    if (q.windows == q.reported) return;
    q.reported = q.windows;
    std.log.info("glm round prof: {d} windows, {d} queued one-row rounds, {d} keeps ({d} partial), {d} rows kept of {d} ({d} drafted)", .{ q.windows, q.queued, q.keeps, q.partial, q.kept_rows, q.window_rows, q.drafted_rows });
    var buf: [160]u8 = undefined;
    for (&q.groups, 0..) |*g, i| if (g.n > 0) {
        std.log.info("glm round prof: rows {s}, streams {d}{s}, {s}: {d} windows (ms: mean p50 p90 max)", .{ row_buckets[i / 6], (i / 2) % 3 + 1, if ((i / 2) % 3 == 2) "+" else "", if (i % 2 == 1) "drafted" else "plain", g.n });
        for (&g.stats, 0..) |*s, m| std.log.info("{s}", .{line(&buf, @tagName(@as(Metric, @fromBackingInt(@intCast(m)))), s, g.n)});
    };
}

fn line(buf: []u8, name: []const u8, s: *const Stat, n: u64) []const u8 {
    return std.fmt.bufPrint(buf, "  {s:<7} {d:8.2} {d:8.2} {d:8.2} {d:8.2}", .{ name, s.sum / @as(f64, @floatFromInt(n)), s.quantile(n, 0.5), s.quantile(n, 0.9), s.max }) catch "";
}

/// Draft rows (held and host drafts) in `wins`.
pub fn draftedRows(wins: anytype) u32 {
    var n: u32 = 0;
    for (wins) |w| n += w.held + @as(u32, @intCast(w.tokens.len));
    return n;
}

test "round profile: rows and streams bucket into thirty distinct groups" {
    var seen: [n_groups]bool = @splat(false);
    for ([_]u32{ 1, 2, 3, 4, 5, 8, 9, 16 }) |rows| for ([_]u32{ 1, 2, 3, 7 }) |streams| for ([_]bool{ false, true }) |d| {
        const g = group(rows, streams, d);
        try std.testing.expect(g < n_groups);
        seen[g] = true;
    };
    for (seen) |s| try std.testing.expect(s);
    try std.testing.expectEqual(group(3, 1, false), group(4, 1, false));
    try std.testing.expect(group(8, 2, true) != group(9, 2, true));
    try std.testing.expectEqual(group(1, 3, false), group(1, 9, false));
}

test "round profile: GLM_ROUND_PROF unset, blank or 0 is off and 1 on; text is refused by name" {
    const gpa = std.testing.allocator;
    for ([_]?[]const u8{ null, "", "  ", "0" }) |raw| try std.testing.expect((try create(gpa, raw)) == null);
    const p = (try create(gpa, "1")).?;
    defer destroy(p, gpa);
    try std.testing.expect(!p.pend_live and p.windows == 0 and p.groups[0].n == 0);
    try std.testing.expectError(error.InvalidEnvironment, create(gpa, "yes"));
    try std.testing.expectError(error.InvalidEnvironment, create(gpa, "2"));
    const why = try platform.environment.describe(gpa, "GLM_ROUND_PROF", "yes");
    defer gpa.free(why);
    try std.testing.expect(std.mem.startsWith(u8, why, "GLM_ROUND_PROF=\"yes\":"));
    defer _ = platform.env.unsetenv("GLM_ROUND_PROF");
    _ = platform.env.unsetenv("GLM_ROUND_PROF");
    try std.testing.expect((try fromEnvironment(gpa)) == null);
    _ = platform.env.setenv("GLM_ROUND_PROF", "   ", 1);
    try std.testing.expect((try fromEnvironment(gpa)) == null);
    _ = platform.env.setenv("GLM_ROUND_PROF", "1", 1);
    const env_p = (try fromEnvironment(gpa)).?;
    destroy(env_p, gpa);
    _ = platform.env.setenv("GLM_ROUND_PROF", "yes", 1);
    try std.testing.expectError(error.InvalidEnvironment, fromEnvironment(gpa));
}

test "round profile: off hooks read no clock and change nothing" {
    try std.testing.expectEqual(@as(u64, 0), tick(null));
    since(null, .picks, 5);
    verified(null);
    laneEnd(null);
    report(null);
}

test "round profile: histograms give mean, p50, p90 and max per group" {
    const gpa = std.testing.allocator;
    const p = (try create(gpa, "1")).?;
    defer destroy(p, gpa);
    for (1..11) |k| {
        const x: f64 = @floatFromInt(k);
        window(p, .{ .rows = 4, .streams = 2, .drafted = 2, .t_in = 0, .committed = 24000, .ret = 72000, .gpu = .{ 1.0, 1.0 + x * 1e-3 }, .ahead = .{ 0, 0 }, .last_end = 0 });
    }
    p.finish();
    const g = &p.groups[group(4, 2, true)];
    try std.testing.expectEqual(@as(u64, 10), g.n);
    const gpu = &g.stats[@backingInt(Metric.gpu)];
    try std.testing.expectApproxEqAbs(@as(f64, 5.5), gpu.sum / 10, 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 5), gpu.quantile(10, 0.5), bin_ms);
    try std.testing.expectApproxEqAbs(@as(f64, 9), gpu.quantile(10, 0.9), bin_ms);
    try std.testing.expectApproxEqAbs(@as(f64, 10), gpu.max, 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 1), g.stats[@backingInt(Metric.encode)].max, 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 2), g.stats[@backingInt(Metric.wait)].max, 1e-9);
    try std.testing.expectEqual(@as(u64, 10), p.windows);
    try std.testing.expectEqual(@as(u64, 20), p.drafted_rows);
    var buf: [160]u8 = undefined;
    try std.testing.expect(std.mem.startsWith(u8, line(&buf, "gpu", gpu, g.n), "  gpu         5.50 "));
}

test "round profile: keeps count partial accepts and the overflow bin reports the largest value" {
    const gpa = std.testing.allocator;
    const p = (try create(gpa, "1")).?;
    defer destroy(p, gpa);
    kept(p, 3, 5);
    kept(p, 5, 5);
    try std.testing.expectEqual(@as(u64, 2), p.keeps);
    try std.testing.expectEqual(@as(u64, 1), p.partial);
    try std.testing.expectEqual(@as(u64, 8), p.kept_rows);
    var s: Stat = .{};
    s.add(500);
    try std.testing.expectEqual(@as(f64, 500), s.quantile(1, 0.9));
}
