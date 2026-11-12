//! The priced round's arithmetic: never under the all-plain rate unforced, equal to brute force, M3-shaped choices.
const std = @import("std");
const costs_mod = @import("costs.zig");
const depth = @import("depth.zig");
const priced = @import("priced.zig");

const gpa = std.testing.allocator;
const t_ = std.testing;

/// The chain's expected tokens past the pending row for its first k chances.
fn chain(qs: []const f64, k: usize) f64 {
    var run: f64 = 1.0;
    var e: f64 = 0.0;
    for (qs[0..k]) |q| {
        run *= q;
        e += run;
    }
    return e;
}

/// A random table: a launch's base, a price a row, a second launch's step from row 17 (sometimes none), a pass.
fn randomCosts(rnd: std.Random, buf: *[32]f64) costs_mod.Costs {
    const base = 10.0 + 30.0 * rnd.float(f64);
    const row = 2.0 + 12.0 * rnd.float(f64);
    const step = if (rnd.boolean()) 40.0 * rnd.float(f64) else 0.0;
    for (buf, 1..) |*v, r| v.* = base + row * @as(f64, @floatFromInt(r - 1)) + (if (r > 16) step else 0.0);
    const draft = 2.0 + 20.0 * rnd.float(f64);
    const ingest = if (rnd.boolean()) @min(draft, 3.0 * rnd.float(f64)) else 0.0;
    return .{ .verify = buf, .draft = draft, .slot = if (rnd.boolean()) 2.0 * rnd.float(f64) else 0.0, .ingest = ingest };
}

test "no unforced round is priced under the all-plain rate; with a flat pass the price is the best of every allocation" {
    var prng = std.Random.DefaultPrng.init(2984);
    const rnd = prng.random();
    for (0..3000) |case| {
        var buf: [32]f64 = undefined;
        const c = randomCosts(rnd, &buf);
        const n = 1 + rnd.uintLessThan(usize, if (case % 2 == 0) 3 else 8);
        var qbuf: [8][5]f64 = undefined;
        var qss: [8][]const f64 = undefined;
        for (qbuf[0..n], qss[0..n]) |*q, *qs| {
            const top = rnd.uintLessThan(usize, if (n <= 3) 4 else 6);
            for (q[0..top]) |*x| x.* = if (rnd.boolean()) rnd.float(f64) else 0.995 * (1.0 - rnd.float(f64) * rnd.float(f64));
            qs.* = q[0..top];
        }
        const forced_one = case % 5 == 4;
        var forced: [8]bool = @splat(false);
        if (forced_one) forced[0] = true;
        var catch_up: [8]bool = undefined;
        for (&catch_up) |*u| u.* = rnd.boolean();
        var joins: [8]f64 = undefined;
        for (&joins, catch_up) |*j, u| j.* = priced.joinMs(c, u);
        var out: [8]bool = undefined;
        const max_rows: u64 = if (case % 3 == 0) n + 3 else 64;
        const p = try priced.plan(gpa, c, n, qss[0..n], forced[0..n], catch_up[0..n], max_rows, out[0..n]);
        if (!forced_one) try t_.expect(p.rate >= p.plain);
        try t_.expectApproxEqRel(priced.plainRate(c, n), p.plain, 1e-15);
        var any_ask = false;
        for (out[0..n]) |o| any_ask = any_ask or o;
        try t_.expectEqual(any_ask, p.pass);
        if (n > 3) continue;
        // every allocation of up to three streams' chains (a pass whenever a draft is verified, or one is forced)
        var best = -std.math.inf(f64);
        var ks: [3]usize = .{ 0, 0, 0 };
        const tops = [3]usize{ qss[0].len, if (n > 1) qss[1].len else 0, if (n > 2) qss[2].len else 0 };
        while (true) {
            var rows: u64 = n;
            var tokens: f64 = @floatFromInt(n);
            for (ks[0..n], qss[0..n]) |k, qs| {
                rows += k;
                tokens += chain(qs, k);
            }
            var in: [3]bool = undefined;
            for (&in, ks, 0..) |*x, k, i| x.* = k > 0 or (i == 0 and forced_one);
            if (rows <= max_rows) best = @max(best, tokens / priced.roundMs(c, n, rows, priced.passMs(c, joins[0..n], in[0..n])));
            var i: usize = 0;
            while (i < n) : (i += 1) {
                if (ks[i] < tops[i]) {
                    ks[i] += 1;
                    break;
                }
                ks[i] = 0;
            }
            if (i == n) break;
        }
        // a pass priced by its asks: the largest gains first may miss a cheaper set of asks, never beat the best
        if (c.ingest == 0.0) try t_.expectApproxEqRel(best, p.rate, 1e-12) else try t_.expect(p.rate <= best * (1.0 + 1e-12));
    }
}

test "the M3 table with a 12 ms pass: weak streams decode plain, strong ones draft, a mixed round asks one slot" {
    var verify = priced.m3Table();
    const c: costs_mod.Costs = .{ .verify = &verify, .draft = 12.0 };
    var out: [4]bool = undefined;
    const none = [_]bool{ false, false, false, false };
    for (1..5) |n| {
        for ([_]f64{ 0.5, 0.7, 0.9 }, [_]bool{ false, n <= 2, true }) |q, drafts| {
            const qs = [5]f64{ q, q, q, q, q };
            const all = [_][]const f64{ &qs, &qs, &qs, &qs };
            const p = try priced.plan(gpa, c, n, all[0..n], none[0..n], none[0..n], 64, out[0..n]);
            try t_.expectEqual(drafts, p.pass);
            if (!drafts) try t_.expectEqual(p.plain, p.rate) else try t_.expect(p.rate > p.plain);
        }
    }
}

test "a round whose pass is paid prices k = 0 against the rows and never chooses under its no-draft option" {
    var prng = std.Random.DefaultPrng.init(17);
    const rnd = prng.random();
    for (0..500) |_| {
        var buf: [32]f64 = undefined;
        const c = randomCosts(rnd, &buf);
        var d = try depth.Depth.init(gpa, c, 5, .{ .priced = true });
        defer d.deinit();
        const n = 1 + rnd.uintLessThan(usize, 4);
        var conf: [4][5]f64 = undefined;
        var asks: [4]depth.Depth.Ask = undefined;
        for (asks[0..n], conf[0..n], 0..) |*a, *cf, i| {
            for (cf) |*x| x.* = rnd.float(f64);
            a.* = .{ .slot = @intCast(i), .conf = cf, .most = 5, .first = 7 };
        }
        var ks: [4]usize = undefined;
        try d.chooseJoint(asks[0..n], 0, 0, rnd.boolean(), ks[0..n]);
        const p = d.last.?;
        const asked = [_]bool{ true, true, true, true };
        const joins = [_]f64{ c.ingest, c.ingest, c.ingest, c.ingest };
        const zero = @as(f64, @floatFromInt(n)) / priced.roundMs(c, n, n, priced.passMs(c, joins[0..n], asked[0..n]));
        try t_.expect(p.rate >= zero);
        var rows: u64 = n;
        for (ks[0..n]) |k| rows += k;
        try t_.expectEqual(rows, p.rows);
        // a slot given none keeps its first draft for the bonus (its landing still teaches the calibration)
        for (ks[0..n], 0..) |k, i| if (k == 0) try t_.expect((try d.state(@intCast(i))).first != null);
    }
}

test "the pass priced by its asks: a further ask adds its ingest and a stream back from plain one more" {
    var verify = priced.m3Table();
    const c: costs_mod.Costs = .{ .verify = &verify, .draft = 12.0, .ingest = 1.5 };
    const joins = [_]f64{ priced.joinMs(c, false), priced.joinMs(c, true), priced.joinMs(c, false) };
    try t_.expectEqual(@as(f64, 0.0), priced.passMs(c, &joins, &.{ false, false, false }));
    try t_.expectEqual(@as(f64, 12.0), priced.passMs(c, &joins, &.{ true, false, false }));
    try t_.expectEqual(@as(f64, 13.5), priced.passMs(c, &joins, &.{ true, false, true }));
    try t_.expectEqual(@as(f64, 16.5), priced.passMs(c, &joins, &.{ true, true, true }));
    // a flat table (no ingest timed apart) prices any number of asks at the timed pass
    const flat: costs_mod.Costs = .{ .verify = &verify, .draft = 12.0 };
    try t_.expectEqual(@as(f64, 12.0), priced.passMs(flat, &.{ 0, 0, 0 }, &.{ true, true, true }));
    // a stream back from plain is not asked beside a strong one when its catch-up tips the price
    const strong = [_]f64{ 0.95, 0.95, 0.95, 0.95, 0.95 };
    const edge = [_]f64{ 0.8, 0.8, 0.8, 0.8, 0.8 };
    var out: [2]bool = undefined;
    const dear: costs_mod.Costs = .{ .verify = &verify, .draft = 12.0, .ingest = 6.0 };
    _ = try priced.plan(std.testing.allocator, dear, 2, &.{ &strong, &edge }, &.{ false, false }, &.{ false, false }, 64, &out);
    try t_.expectEqualSlices(bool, &.{ true, true }, &out);
    _ = try priced.plan(std.testing.allocator, dear, 2, &.{ &strong, &edge }, &.{ false, false }, &.{ false, true }, 64, &out);
    try t_.expectEqualSlices(bool, &.{ true, false }, &out);
}
