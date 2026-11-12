//! Priced plain rounds: each round's asks and drafts with the most expected tokens a ms, no pass at all among them.
const std = @import("std");
const Allocator = std.mem.Allocator;
const Costs = @import("costs.zig").Costs;
const depth = @import("depth.zig");

/// A probe's pass may cost at most this share of the modelled time of the rounds with a plain stream before it.
pub const probe_share = 0.01;
/// The newest pass's weight in a stream's running confidences (a pass after a plain run replaces them).
pub const pred_rate = 0.5;
const eps = 1e-12;

/// What a round was priced at: tokens a ms, the all-plain round's, its modelled ms, rows, asks and pass.
pub const Plan = struct { rate: f64, plain: f64, ms: f64, rows: u64, asks: usize = 0, pass: bool };

/// The planner's counts: rounds priced, with and without a pass, probes, and how far each sat above all-plain.
pub const Stats = struct {
    rounds: u64 = 0,
    passes: u64 = 0, // rounds whose plan runs a pass
    plain: u64 = 0, // rounds whose plan runs none
    probes: u64 = 0, // rounds a probe forced a pass
    fresh: u64 = 0, // asks forced by a stream without confidences yet
    asks: u64 = 0, // slots asked over all rounds
    below: u64 = 0, // unforced plans priced under the all-plain rate (never, by construction)
    plain_ms: f64 = 0.0, // modelled ms of rounds with a plain stream since the last probe
    plain_total_ms: f64 = 0.0, // modelled ms of every round with a plain stream
    probe_ms: f64 = 0.0, // the probes' passes
    low: f64 = std.math.inf(f64), // the lowest rate over the plain rate of an unforced plan that runs a pass
    last: ?Plan = null,
};

/// A round's modelled ms: its rows' window, each further window's overhead and the pass when one runs.
pub fn roundMs(c: Costs, windows: usize, rows: u64, pass: bool) f64 {
    return c.rowsMs(@intCast(rows)) + c.slot * @as(f64, @floatFromInt(windows -| 1)) + (if (pass) c.draft else 0.0);
}

/// The all-plain round's tokens a ms: one token a window, no drafts, no pass.
pub fn plainRate(c: Costs, windows: usize) f64 {
    const n = @max(windows, 1);
    return @as(f64, @floatFromInt(n)) / roundMs(c, n, n, false);
}

const Gain = struct { g: f64, stream: u32, pos: u32 };

fn larger(_: void, a: Gain, b: Gain) bool {
    if (a.g != b.g) return a.g > b.g;
    if (a.stream != b.stream) return a.stream < b.stream;
    return a.pos < b.pos;
}

/// One round's priced choice over `windows` windows of `base` rows committing `tokens` without drafts.
pub const Round = struct { windows: usize, tokens: f64, base: u64, max_rows: u64, paid: bool };

/// Drafts a stream (`ks`) with the most tokens a ms; `qss[i]` its chain's chances; no drafts and no pass unless paid.
pub fn scan(gpa: Allocator, c: Costs, r: Round, qss: []const []const f64, ks: []usize) !Plan {
    var gains: std.ArrayList(Gain) = .empty;
    defer gains.deinit(gpa);
    for (qss, 0..) |qs, i| {
        var run: f64 = 1.0;
        for (qs, 0..) |q, j| {
            run *= q;
            try gains.append(gpa, .{ .g = run, .stream = @intCast(i), .pos = @intCast(j) });
        }
    }
    // a chain's gains never grow along it, so the m largest are each chain's first drafts
    std.mem.sort(Gain, gains.items, {}, larger);
    const room = r.max_rows -| r.base;
    var best_m: usize = 0;
    var best = r.tokens / roundMs(c, r.windows, r.base, r.paid);
    var acc: f64 = 0.0;
    for (gains.items[0..@min(room, gains.items.len)], 1..) |x, m| {
        acc += x.g;
        const rate = (r.tokens + acc) / roundMs(c, r.windows, r.base + m, true);
        if (rate > best + eps) {
            best = rate;
            best_m = m;
        }
    }
    @memset(ks, 0);
    for (gains.items[0..best_m]) |x| ks[x.stream] += 1;
    const pass = r.paid or best_m > 0;
    return .{ .rate = best, .plain = plainRate(c, r.windows), .ms = roundMs(c, r.windows, r.base + best_m, pass), .rows = r.base + best_m, .pass = pass };
}

/// The next round's asks: forced ones, and every stream the priced round gives a draft; `out[i]`: stream i asks.
pub fn plan(gpa: Allocator, c: Costs, windows: usize, qss: []const []const f64, forced: []const bool, max_rows: u64, out: []bool) !Plan {
    const ks = try gpa.alloc(usize, qss.len);
    defer gpa.free(ks);
    var paid = false;
    for (forced) |f| paid = paid or f;
    const n = @max(windows, qss.len);
    var p = try scan(gpa, c, .{ .windows = n, .tokens = @floatFromInt(n), .base = n, .max_rows = max_rows, .paid = paid }, qss, ks);
    for (out, ks, forced) |*o, k, f| {
        o.* = f or k > 0;
        p.asks += @intFromBool(o.*);
    }
    return p;
}

/// A pass's raw confidences into the stream's running ones: replaced after a plain run, else blended.
pub fn observe(st: *depth.Slot, conf: []const f64) void {
    const n = @min(conf.len, depth.slot_rows);
    if (n == 0) return;
    if (st.pred_len == 0 or st.stale) {
        @memcpy(st.pred[0..n], conf[0..n]);
    } else for (st.pred[0..n], conf[0..n], 0..) |*p, x, j| {
        p.* = if (j < st.pred_len) p.* + pred_rate * (x - p.*) else x;
    }
    st.pred_len = n;
    st.stale = false;
}

/// Every DSpark slot's k of a priced round (its pass already paid), k = 0 included. Writes `ks`.
pub fn chooseRound(d: *depth.Depth, asks: []const depth.Depth.Ask, others: u64, other_windows: usize, ks: []usize) !Plan {
    d.close();
    const qbuf = try d.gpa.alloc([depth.slot_rows]f64, asks.len);
    defer d.gpa.free(qbuf);
    const qss = try d.gpa.alloc([]const f64, asks.len);
    defer d.gpa.free(qss);
    for (asks, qbuf, qss) |a, *qb, *qs| {
        const top = @min(@min(@min(a.most, a.conf.len), d.set.cap), depth.slot_rows - 1);
        for (a.conf[0..top], qb[0..top], 0..) |p, *q, j| q.* = d.cal.q(j, p);
        qs.* = qb[0..top];
        const st = try d.state(a.slot);
        st.first = null;
        st.q1 = if (top > 0) qb[0] else null;
        st.ms = null;
        observe(st, a.conf);
    }
    const windows = asks.len + other_windows;
    const r: Round = .{ .windows = windows, .tokens = @floatFromInt(windows), .base = asks.len + others, .max_rows = d.set.max_rows, .paid = asks.len > 0 };
    const p = try scan(d.gpa, d.costs, r, qss, ks);
    var rows: u64 = others;
    var drafts: u64 = 0;
    var zero: u64 = 0;
    for (asks, ks, qss) |a, k, qs| {
        const st = try d.state(a.slot);
        st.use(a.conf[0..k]);
        st.drafted = qs.len > 0;
        if (k == 0 and qs.len > 0) {
            if (a.first) |f| st.first = .{ .token = f, .p = a.conf[0] };
            zero += 1;
        }
        rows += 1 + k;
        drafts += k;
    }
    d.stats.joint[0] += 1;
    d.stats.joint[1] += rows;
    d.stats.joint[2] += drafts;
    d.stats.joint[3] += zero;
    return p;
}

test "a round priced at none, one or every stream: plain wins weak drafts, strong chains draft, the pass is shared" {
    const gpa = std.testing.allocator;
    var verify = m3Table();
    const c: Costs = .{ .verify = &verify, .draft = 12.0 };
    var out: [4]bool = undefined;
    const weak = [_]f64{ 0.5, 0.5, 0.5, 0.5, 0.5 };
    const strong = [_]f64{ 0.95, 0.95, 0.95, 0.95, 0.95 };
    const none = [_]bool{ false, false, false, false };
    // four weak streams: no ask, no pass, the plain rate exactly
    const p0 = try plan(gpa, c, 4, &.{ &weak, &weak, &weak, &weak }, &none, 64, &out);
    try std.testing.expectEqualSlices(bool, &none, &out);
    try std.testing.expect(!p0.pass and p0.rate == p0.plain);
    // one strong stream beside three weak ones: it alone asks, and the round beats plain
    const p1 = try plan(gpa, c, 4, &.{ &strong, &weak, &weak, &weak }, &none, 64, &out);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, false }, &out);
    try std.testing.expect(p1.pass and p1.asks == 1 and p1.rate > p1.plain);
    // a forced ask pays the pass: then a weak stream's draft is priced against the paid pass only
    const p2 = try plan(gpa, c, 1, &.{&weak}, &.{true}, 64, out[0..1]);
    try std.testing.expect(out[0] and p2.pass);
}

/// The M3 Ultra's measured verify windows (rows 1 .. 32), linear between the timed rows.
pub fn m3Table() [32]f64 {
    const at = [_]struct { r: u32, ms: f64 }{ .{ .r = 1, .ms = 30.6 }, .{ .r = 2, .ms = 41.1 }, .{ .r = 4, .ms = 56.6 }, .{ .r = 6, .ms = 74.9 }, .{ .r = 8, .ms = 90.7 }, .{ .r = 16, .ms = 169.5 }, .{ .r = 17, .ms = 190.4 }, .{ .r = 32, .ms = 323.2 } };
    var out: [32]f64 = undefined;
    for (&out, 1..) |*o, r| {
        var i: usize = 0;
        while (i + 2 < at.len and at[i + 1].r < r) i += 1;
        const a = at[i];
        const b = at[i + 1];
        o.* = a.ms + (b.ms - a.ms) * @as(f64, @floatFromInt(@as(i64, @intCast(r)) - a.r)) / @as(f64, @floatFromInt(b.r - a.r));
    }
    return out;
}
