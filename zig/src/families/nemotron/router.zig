//! A lesson's router: Fisher scores per layer, the best averaged; it acts past a bar read off left-out questions.
const std = @import("std");
const shape = @import("learn_shape.zig");
const subspace = @import("subspace.zig");

const n = shape.candidates;

const ridge = 0.1; // of the scatter's mean diagonal: questions are few against 64 directions
const most_reads = 12; // layers a router averages: past the best dozen, weaker layers blur the fact's questions back in
const odd_share = 4; // one fact question in this many may word itself apart from the rest and go untaken

pub const Fit = struct {
    dir: []f32, // [layers, n]: each layer's direction, unit, in its candidates' terms
    weight: []f32, // [layers]: each layer's share of the average over the other prompts' spread there (0: unread)
    mean: []f32, // [layers]: the other prompts' mean score there
    need: f32, // the average a reply's decision row must reach for the lesson to act on it
    reads: u32, // layers with a share
    fact_low: f32, // the average most fact questions left out of the fit reached (all but the oddest quarter)
    fact_least: f32, // the lowest any of them reached
    steady_high: f32, // the highest a steady prompt left out of the fit reached

    pub fn deinit(f: *Fit, gpa: std.mem.Allocator) void {
        gpa.free(f.dir);
        gpa.free(f.weight);
        gpa.free(f.mean);
    }

    /// Whether a fact question the fit never saw still scores above every steady prompt it never saw.
    pub fn apart(f: *const Fit) bool {
        return f.fact_low > f.steady_high;
    }

    /// Router row `i` of `rows` (rows[l]: layer l's decision rows along its candidates) as the lesson scores it.
    pub fn score(f: *const Fit, rows: []const []const f32, i: usize) f32 {
        var s: f32 = 0;
        for (rows, 0..) |r, l| {
            if (f.weight[l] == 0) continue;
            s += f.weight[l] * (subspace.dot(f.dir[l * n ..][0..n], r[i * n ..][0..n]) - f.mean[l]);
        }
        return s;
    }
};

/// Sums over a set of rows, from which a mean and scatter come with any one row left out.
const Stats = struct {
    count: f64 = 0,
    sum: [n]f64 = @splat(0),
    sq: [n * n]f64 = @splat(0),

    fn of(rows: []const f32) Stats {
        var s: Stats = .{};
        for (0..rows.len / n) |r| s.add(rows[r * n ..][0..n], 1);
        return s;
    }

    fn add(s: *Stats, p: []const f32, w: f64) void {
        s.count += w;
        for (0..n) |i| {
            s.sum[i] += w * p[i];
            for (0..n) |j| s.sq[i * n + j] += w * p[i] * p[j];
        }
    }
};

/// Fisher's direction (unit) between two sets: (scatter + ridge)^-1 (mean difference).
fn fisher(pos: *const Stats, neg: *const Stats, out: []f32) !void {
    var mean: [2][n]f64 = undefined;
    const sets = [2]*const Stats{ pos, neg };
    var a: [n * n]f64 = @splat(0);
    for (sets, 0..) |s, k| {
        for (0..n) |i| mean[k][i] = s.sum[i] / s.count;
        for (0..n) |i| for (0..n) |j| {
            a[i * n + j] += s.sq[i * n + j] / s.count - mean[k][i] * mean[k][j];
        };
    }
    var trace: f64 = 0;
    for (0..n) |i| trace += a[i * n + i];
    for (0..n) |i| a[i * n + i] += ridge * trace / n + 1e-12;
    var d: [n]f64 = undefined;
    for (&d, mean[0], mean[1]) |*x, p, q| x.* = p - q;
    try subspace.solve(&a, n, &d);
    var size: f64 = 0;
    for (d) |x| size += x * x;
    for (out, d) |*o, x| o.* = @floatCast(x / @sqrt(@max(size, 1e-30)));
}

/// One layer's direction from every row, and each row's score along the direction fitted without it.
fn layer(facts: []const f32, steady: []const f32, dir: []f32, fact_s: []f32, steady_s: []f32) !void {
    const pos = Stats.of(facts);
    const neg = Stats.of(steady);
    try fisher(&pos, &neg, dir);
    var d: [n]f32 = undefined;
    for (fact_s, 0..) |*s, i| {
        const p = facts[i * n ..][0..n];
        var less = pos;
        less.add(p, -1);
        try fisher(&less, &neg, &d);
        s.* = subspace.dot(&d, p);
    }
    for (steady_s, 0..) |*s, j| {
        const p = steady[j * n ..][0..n];
        var less = neg;
        less.add(p, -1);
        try fisher(&pos, &less, &d);
        s.* = subspace.dot(&d, p);
    }
}

/// The steady rows' mean and spread of `s`, with row `skip` left out.
fn spread(s: []const f32, skip: ?usize) [2]f32 {
    var sum: f64 = 0;
    var sq: f64 = 0;
    var count: f64 = 0;
    for (s, 0..) |x, j| if (j != skip) {
        sum += x;
        sq += x * x;
        count += 1;
    };
    const m = sum / count;
    return .{ @floatCast(m), @floatCast(@sqrt(@max(sq / count - m * m, 0)) + 1e-9) };
}

/// Each layer's share from fact scores `z` [layers, facts] less `skip`: top layers by median, above the steady mean.
fn shares(gpa: std.mem.Allocator, z: []const f32, layers: usize, nf: usize, skip: ?usize, out: []f32) !void {
    const col = try gpa.alloc(f32, nf);
    defer gpa.free(col);
    const sep = try gpa.alloc(f32, layers);
    defer gpa.free(sep);
    for (0..layers) |l| {
        var m: usize = 0;
        for (0..nf) |i| if (i != skip) {
            col[m] = z[l * nf + i];
            m += 1;
        };
        std.mem.sort(f32, col[0..m], {}, std.sort.asc(f32));
        sep[l] = col[m / 2];
    }
    const sorted = try gpa.dupe(f32, sep);
    defer gpa.free(sorted);
    std.mem.sort(f32, sorted, {}, std.sort.desc(f32));
    const bar = sorted[@min(most_reads, layers) - 1];
    var total: f32 = 0;
    for (out, sep) |*w, x| {
        w.* = if (x >= bar and x > 0) x else 0;
        total += w.*;
    }
    for (out) |*w| w.* = if (total > 0) w.* / total else 0;
}

/// The router from each layer's decision rows: facts[l] and steady[l] the same questions at every layer, [count, n].
pub fn fit(gpa: std.mem.Allocator, facts: []const []const f32, steady: []const []const f32) !Fit {
    const layers = facts.len;
    const nf = facts[0].len / n;
    const ns = steady[0].len / n;
    if (nf < 2 or ns < 3) return error.TooFewQuestions;
    const dir = try gpa.alloc(f32, layers * n);
    errdefer gpa.free(dir);
    const weight = try gpa.alloc(f32, layers);
    errdefer gpa.free(weight);
    const mean = try gpa.alloc(f32, layers);
    errdefer gpa.free(mean);
    const fact_s = try gpa.alloc(f32, layers * nf);
    defer gpa.free(fact_s);
    const steady_s = try gpa.alloc(f32, layers * ns);
    defer gpa.free(steady_s);
    const Share = struct {
        fn run(from: usize, step: usize, ctx: anytype, out: *?anyerror) void {
            var l = from;
            while (l < ctx.layers) : (l += step) layer(ctx.facts[l], ctx.steady[l], ctx.dir[l * n ..][0..n], ctx.fact_s[l * ctx.nf ..][0..ctx.nf], ctx.steady_s[l * ctx.ns ..][0..ctx.ns]) catch |e| {
                out.* = e;
            };
        }
    };
    const ctx = .{ .layers = layers, .facts = facts, .steady = steady, .dir = dir, .fact_s = fact_s, .steady_s = steady_s, .nf = nf, .ns = ns };
    var threads: [16]?std.Thread = @splat(null);
    var errs: [16]?anyerror = @splat(null);
    const k = @min(threads.len, layers);
    for (threads[0..k], 0..) |*t, i| t.* = std.Thread.spawn(.{}, Share.run, .{ i, k, ctx, &errs[i] }) catch null;
    for (threads[0..k], 0..) |t, i| if (t) |th| th.join() else Share.run(i, k, ctx, &errs[i]);
    for (errs[0..k]) |e| if (e) |x| return x;
    // every score standardised by the steady prompts' at its layer: a steady prompt's without its own
    const z_fact = try gpa.alloc(f32, layers * nf);
    defer gpa.free(z_fact);
    const z_steady = try gpa.alloc(f32, layers * ns);
    defer gpa.free(z_steady);
    const spread_all = try gpa.alloc([2]f32, layers);
    defer gpa.free(spread_all);
    for (0..layers) |l| {
        const s = steady_s[l * ns ..][0..ns];
        spread_all[l] = spread(s, null);
        for (0..nf) |i| z_fact[l * nf + i] = (fact_s[l * nf + i] - spread_all[l][0]) / spread_all[l][1];
        for (0..ns) |j| {
            const own = spread(s, j);
            z_steady[l * ns + j] = (s[j] - own[0]) / own[1];
        }
    }
    // each left-out question's average over the layers chosen without it
    const w = try gpa.alloc(f32, layers);
    defer gpa.free(w);
    const fact_avg = try gpa.alloc(f32, nf);
    defer gpa.free(fact_avg);
    for (fact_avg, 0..) |*avg, i| {
        try shares(gpa, z_fact, layers, nf, i, w);
        avg.* = 0;
        for (0..layers) |l| avg.* += w[l] * z_fact[l * nf + i];
    }
    try shares(gpa, z_fact, layers, nf, null, w);
    var high: f32 = -std.math.inf(f32);
    for (0..ns) |j| {
        var avg: f32 = 0;
        for (0..layers) |l| avg += w[l] * z_steady[l * ns + j];
        high = @max(high, avg);
    }
    std.mem.sort(f32, fact_avg, {}, std.sort.asc(f32));
    const low = fact_avg[nf / odd_share];
    var reads: u32 = 0;
    for (0..layers) |l| {
        weight[l] = w[l] / spread_all[l][1];
        mean[l] = spread_all[l][0];
        reads += @intFromBool(w[l] > 0);
    }
    // half way across the gap between left-out facts and steady prompts; on overlap, what most facts reach
    const need = if (low > high) (low + high) / 2 else low;
    return .{ .dir = dir, .weight = weight, .mean = mean, .need = need, .reads = reads, .fact_low = low, .fact_least = fact_avg[0], .steady_high = high };
}

test "a router clears questions it never saw about its fact, and steady prompts it never saw stay below it" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(11);
    const r = prng.random();
    const layers = 6;
    const nf = 8;
    const ns = 60;
    var facts: [layers][]f32 = undefined;
    var steady: [layers][]f32 = undefined;
    // facts share a direction the steady rows lack; every row then made unit
    for (0..layers) |l| {
        facts[l] = try gpa.alloc(f32, nf * n);
        steady[l] = try gpa.alloc(f32, ns * n);
        for (facts[l], 0..) |*x, i| x.* = r.floatNorm(f32) * 0.3 + @as(f32, if (i % n == l) 3 else 0);
        for (steady[l]) |*x| x.* = r.floatNorm(f32) * 0.3;
        for ([_][]f32{ facts[l], steady[l] }) |rows| for (0..rows.len / n) |q| {
            const row = rows[q * n ..][0..n];
            const size = @sqrt(subspace.dot(row, row));
            for (row) |*x| x.* /= size;
        };
    }
    defer for (0..layers) |l| {
        gpa.free(facts[l]);
        gpa.free(steady[l]);
    };
    var f = try fit(gpa, &facts, &steady);
    defer f.deinit(gpa);
    try std.testing.expect(f.apart());
    try std.testing.expect(f.need > f.steady_high and f.need < f.fact_low);
    try std.testing.expectEqual(@as(u32, layers), f.reads);
    // the fit's own rows: every fact above the bar, every steady prompt below it
    for (0..nf) |i| try std.testing.expect(f.score(&facts, i) >= f.need);
    for (0..ns) |j| try std.testing.expect(f.score(&steady, j) < f.need);
}
