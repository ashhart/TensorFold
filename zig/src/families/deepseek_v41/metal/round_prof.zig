//! TF_DSV41_ROUND_PROF: a decode round's host phases, GPU time and the drafter's passes, summed by launch shape.
const std = @import("std");

/// The host phases of one round, in ns; `other` is what the named ones leave of the round's wall time.
pub const Phase = enum { prepare, encode, wait, choose, other, round, commit_done, gpu, gap, seam };
const nphase = std.enums.values(Phase).len;
/// The drafter's passes: host wall time of each command buffer, its GPU span, and a launch's `finish`.
pub const Event = enum { ingest_wall, ingest_gpu, propose_wall, propose_gpu, finish };
const nevent = std.enums.values(Event).len;

/// Command buffers a round can hold (forward.zig cuts a round into about five).
pub const max_cbs = 16;
/// Samples kept per series for the percentiles; later ones still count in the mean.
const max_samples = 1 << 15;
/// Rounds between summaries in the log.
pub const report_every = 4096;
const row_buckets = [_][]const u8{ "1", "2", "3-4", "5-8", "9-16" };

fn realNow() u64 {
    var t: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &t);
    return @as(u64, @intCast(t.sec)) * std.time.ns_per_s + @as(u64, @intCast(t.nsec));
}

/// A run of durations: count, sum, and the first `max_samples` for the percentiles.
const Series = struct {
    n: u64 = 0,
    sum: u64 = 0,
    at: std.ArrayList(u32) = .empty,

    fn add(s: *Series, a: std.mem.Allocator, ns: u64) void {
        s.n += 1;
        s.sum += ns;
        if (s.at.items.len < max_samples) s.at.append(a, @intCast(@min(ns, std.math.maxInt(u32)))) catch {};
    }

    fn deinit(s: *Series, a: std.mem.Allocator) void {
        s.at.deinit(a);
    }

    fn mean(s: *const Series) f64 {
        return if (s.n == 0) 0 else @as(f64, @floatFromInt(s.sum)) / @as(f64, @floatFromInt(s.n));
    }

    /// The nearest-rank percentile `q` (0 to 100) of the kept samples, in ns.
    fn pct(s: *const Series, a: std.mem.Allocator, q: usize) f64 {
        const n = s.at.items.len;
        if (n == 0) return 0;
        const tmp = a.dupe(u32, s.at.items) catch return 0;
        defer a.free(tmp);
        std.mem.sort(u32, tmp, {}, std.sort.asc(u32));
        return @floatFromInt(tmp[@min(n - 1, (q * n + 99) / 100 -| 1)]);
    }
};

/// Rounds of one launch shape.
const Group = struct { n: u64 = 0, phase: [nphase]Series = @splat(.{}) };

/// One round being timed: the phases book their time as the round passes through them.
pub const Round = struct {
    p: *RoundProf,
    slots: u32,
    rows: u32,
    drafted: bool,
    began: u64,
    last: u64,
    ns: [nphase]u64 = @splat(0),
    cbs: u32 = 0,
    cb_gpu: [max_cbs]u64 = @splat(0),
    cb_wall: [max_cbs]u64 = @splat(0),

    pub fn now(r: *const Round) u64 {
        return r.p.clock();
    }

    /// The time since the last lap (or the start) booked to `ph`.
    pub fn lap(r: *Round, ph: Phase) void {
        const t = r.p.clock();
        r.ns[@backingInt(ph)] += t -| r.last;
        r.last = t;
    }

    /// The time since the last lap is left to `other`.
    pub fn skip(r: *Round) void {
        r.last = r.p.clock();
    }

    /// Time that passed outside the host's laps (the GPU's own clock) booked to `ph`, without moving the lap.
    pub fn book(r: *Round, ph: Phase, ns: u64) void {
        r.ns[@backingInt(ph)] += ns;
    }

    /// One command buffer of the round done: its host wall time from commit and its GPU span, in ns.
    pub fn buffer(r: *Round, wall: u64, gpu: u64) void {
        if (r.cbs >= max_cbs) return;
        r.cb_wall[r.cbs] = wall;
        r.cb_gpu[r.cbs] = gpu;
        r.cbs += 1;
    }

    /// The round over: its wall time, the residue, and every phase into the shape's series.
    pub fn end(r: *Round) void {
        const t = r.p.clock();
        const named = r.ns[@backingInt(Phase.prepare)] + r.ns[@backingInt(Phase.encode)] + r.ns[@backingInt(Phase.wait)] + r.ns[@backingInt(Phase.choose)];
        r.ns[@backingInt(Phase.round)] = t -| r.began;
        r.ns[@backingInt(Phase.other)] = r.ns[@backingInt(Phase.round)] -| named;
        r.p.add(r);
        r.p.last_end = t;
    }
};

pub const RoundProf = struct {
    a: std.mem.Allocator,
    /// the clock in ns; a test replaces it
    clock: *const fn () u64 = realNow,
    last_end: u64 = 0,
    rounds: u64 = 0,
    groups: [17][row_buckets.len][2]?*Group = @splat(@splat(@splat(null))),
    cb_gpu: [max_cbs]Series = @splat(.{}),
    cb_wall: [max_cbs]Series = @splat(.{}),
    ev: [nevent]Series = @splat(.{}),

    pub fn init(a: std.mem.Allocator) !*RoundProf {
        const p = try a.create(RoundProf);
        p.* = .{ .a = a };
        return p;
    }

    pub fn deinit(p: *RoundProf) void {
        for (&p.groups) |*bs| for (bs) |*ds| for (ds) |g| if (g) |gg| {
            for (&gg.phase) |*s| s.deinit(p.a);
            p.a.destroy(gg);
        };
        for (&p.cb_gpu) |*s| s.deinit(p.a);
        for (&p.cb_wall) |*s| s.deinit(p.a);
        for (&p.ev) |*s| s.deinit(p.a);
        p.a.destroy(p);
    }

    /// A round of `slots` slots over `rows` rows starts; its `seam` is the time since the last one ended.
    pub fn begin(p: *RoundProf, slots: u32, rows: u32) Round {
        const t = p.clock();
        var r: Round = .{ .p = p, .slots = slots, .rows = rows, .drafted = rows > slots, .began = t, .last = t };
        if (p.last_end != 0) r.ns[@backingInt(Phase.seam)] = t -| p.last_end;
        return r;
    }

    /// A drafter event of `ns`.
    pub fn event(p: *RoundProf, e: Event, ns: u64) void {
        p.ev[@backingInt(e)].add(p.a, ns);
    }

    pub fn rowBucket(rows: u32) usize {
        return if (rows <= 2) rows -| 1 else if (rows <= 4) 2 else if (rows <= 8) 3 else 4;
    }

    fn add(p: *RoundProf, r: *const Round) void {
        const g = p.group(r);
        g.n += 1;
        p.rounds += 1;
        for (&g.phase, r.ns, 0..) |*s, ns, i| {
            const ph: Phase = @fromBackingInt(@intCast(i));
            if (ph == .seam and p.last_end == 0) continue;
            s.add(p.a, ns);
        }
        for (0..r.cbs) |i| {
            p.cb_gpu[i].add(p.a, r.cb_gpu[i]);
            p.cb_wall[i].add(p.a, r.cb_wall[i]);
        }
        if (p.rounds % report_every == 0) p.log();
    }

    fn group(p: *RoundProf, r: *const Round) *Group {
        const slot = &p.groups[@min(r.slots, 16)][rowBucket(r.rows)][@intFromBool(r.drafted)];
        if (slot.* == null) {
            slot.* = p.a.create(Group) catch @panic("round profile: out of memory");
            slot.*.?.* = .{};
        }
        return slot.*.?;
    }

    /// The summary: one line a launch shape (slots, rows, drafted or plain), the command buffers, the drafter's passes.
    pub fn write(p: *const RoundProf, w: *std.Io.Writer) !void {
        try w.print("dsv41 round profile: {d} rounds; ms mean/p50/p95 (gpu, gap, commit_done: the round's command buffers)\n", .{p.rounds});
        for (p.groups, 0..) |bs, slots| for (bs, 0..) |ds, b| for (ds, 0..) |g, drafted| if (g) |gg| {
            try w.print("slots {d} rows {s} {s} n {d}:", .{ slots, row_buckets[b], if (drafted == 1) "drafted" else "plain", gg.n });
            for (&gg.phase, 0..) |*s, i| if (s.n > 0) try p.cell(w, @tagName(@as(Phase, @fromBackingInt(@intCast(i)))), s);
            try w.writeByte('\n');
        };
        for (&p.cb_gpu, &p.cb_wall, 0..) |*g, *wl, i| if (g.n > 0) {
            try w.print("command buffer {d} n {d}:", .{ i, g.n });
            try p.cell(w, "gpu", g);
            try p.cell(w, "wall", wl);
            try w.writeByte('\n');
        };
        for (&p.ev, 0..) |*s, i| if (s.n > 0) {
            try w.print("drafter n {d}:", .{s.n});
            try p.cell(w, @tagName(@as(Event, @fromBackingInt(@intCast(i)))), s);
            try w.writeByte('\n');
        };
    }

    fn cell(p: *const RoundProf, w: *std.Io.Writer, name: []const u8, s: *const Series) !void {
        try w.print(" {s} {d:.3}/{d:.3}/{d:.3}", .{ name, s.mean() / 1e6, s.pct(p.a, 50) / 1e6, s.pct(p.a, 95) / 1e6 });
    }

    /// The summary to the log, a line each.
    pub fn log(p: *const RoundProf) void {
        var out: std.Io.Writer.Allocating = .init(p.a);
        defer out.deinit();
        p.write(&out.writer) catch return;
        var it = std.mem.splitScalar(u8, out.written(), '\n');
        while (it.next()) |line| if (line.len > 0) std.log.scoped(.dsv41).info("{s}", .{line});
    }
};

var fake_now: u64 = 0;

fn fakeClock() u64 {
    return fake_now;
}

test "phase sums are the round's wall time with a fake clock" {
    const a = std.testing.allocator;
    const p = try RoundProf.init(a);
    defer p.deinit();
    p.clock = fakeClock;
    fake_now = 1000;
    var r = p.begin(2, 5);
    fake_now += 300;
    r.lap(.prepare);
    fake_now += 500;
    r.lap(.encode);
    fake_now += 1200;
    r.lap(.wait);
    r.buffer(1700, 1100);
    fake_now += 40;
    r.lap(.choose);
    fake_now += 60;
    r.end();
    const g = p.groups[2][3][1].?;
    try std.testing.expectEqual(@as(u64, 1), g.n);
    try std.testing.expectEqual(@as(u64, 2100), g.phase[@backingInt(Phase.round)].sum);
    var sum: u64 = 0;
    for ([_]Phase{ .prepare, .encode, .wait, .choose, .other }) |ph| sum += g.phase[@backingInt(ph)].sum;
    try std.testing.expectEqual(@as(u64, 2100), sum);
    try std.testing.expectEqual(@as(u64, 60), g.phase[@backingInt(Phase.other)].sum);
    try std.testing.expectEqual(@as(u64, 0), g.phase[@backingInt(Phase.seam)].n);
    // the next round's seam is the time since this one ended
    fake_now += 250;
    var r2 = p.begin(2, 5);
    r2.end();
    try std.testing.expectEqual(@as(u64, 250), p.groups[2][3][1].?.phase[@backingInt(Phase.seam)].sum);
}

test "the summary names every launch shape and the drafter's passes, with percentiles" {
    const a = std.testing.allocator;
    const p = try RoundProf.init(a);
    defer p.deinit();
    p.clock = fakeClock;
    fake_now = 5;
    for ([_]u64{ 1_000_000, 2_000_000, 9_000_000 }) |ns| {
        var r = p.begin(1, 1);
        fake_now += ns;
        r.lap(.wait);
        r.end();
    }
    p.event(.ingest_wall, 4_000_000);
    var buf: [4096]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try p.write(&w);
    const text = w.buffered();
    try std.testing.expect(std.mem.indexOf(u8, text, "slots 1 rows 1 plain n 3:") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, " wait 4.000/2.000/9.000") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "drafter n 1: ingest_wall 4.000/4.000/4.000") != null);
}

test "row buckets" {
    try std.testing.expectEqual(@as(usize, 0), RoundProf.rowBucket(1));
    try std.testing.expectEqual(@as(usize, 1), RoundProf.rowBucket(2));
    try std.testing.expectEqual(@as(usize, 2), RoundProf.rowBucket(4));
    try std.testing.expectEqual(@as(usize, 3), RoundProf.rowBucket(8));
    try std.testing.expectEqual(@as(usize, 4), RoundProf.rowBucket(16));
}
