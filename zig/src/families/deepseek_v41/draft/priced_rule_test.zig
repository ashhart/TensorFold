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
    return .{ .verify = buf, .draft = 2.0 + 20.0 * rnd.float(f64), .slot = if (rnd.boolean()) 2.0 * rnd.float(f64) else 0.0 };
}

test "no unforced round is priced under the all-plain rate, and the price equals the best of every allocation" {
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
        var out: [8]bool = undefined;
        const max_rows: u64 = if (case % 3 == 0) n + 3 else 64;
        const p = try priced.plan(gpa, c, n, qss[0..n], forced[0..n], max_rows, out[0..n]);
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
            if (rows <= max_rows) best = @max(best, tokens / priced.roundMs(c, n, rows, forced_one or rows > n));
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
        try t_.expectApproxEqRel(best, p.rate, 1e-12);
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
            const p = try priced.plan(gpa, c, n, all[0..n], none[0..n], 64, out[0..n]);
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
        const zero = @as(f64, @floatFromInt(n)) / priced.roundMs(c, n, n, true);
        try t_.expect(p.rate >= zero);
        var rows: u64 = n;
        for (ks[0..n]) |k| rows += k;
        try t_.expectEqual(rows, p.rows);
        // a slot given none keeps its first draft for the bonus (its landing still teaches the calibration)
        for (ks[0..n], 0..) |k, i| if (k == 0) try t_.expect((try d.state(@intCast(i))).first != null);
    }
}
