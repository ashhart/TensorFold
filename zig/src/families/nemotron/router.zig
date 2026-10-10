//! A lesson's router: at each layer, Fisher's direction between its fact questions' decision rows and the steady
//! prompts', and how many layers must clear it before the lesson acts on a reply, read off questions left out of the fit.
const std = @import("std");
const shape = @import("learn_shape.zig");
const subspace = @import("subspace.zig");

const n = shape.candidates;

const ridge = 0.1; // of the scatter's mean diagonal: questions are few against 64 directions
const margins = [_]f32{ 0.005, 0.01, 0.015, 0.02 }; // how far above every steady row a layer's clearing must be
const min_need = 3; // a lesson never acts on fewer layers' say: one or two can clear by chance
const odd_share = 4; // one fact question in this many may word itself apart from the rest and go untaken

pub const Fit = struct {
    dir: []f32, // [layers, n]: each layer's direction, unit, in its candidates' terms
    tau: []f32, // [layers]: the cosine to clear there, or shut where most facts do not clear every steady row
    need: u32, // layers a reply's decision row must clear for the lesson to act on it
    margin: f32,
    fact_low: u32, // the layers most fact questions left out of the fit cleared (all but the oddest quarter)
    fact_least: u32, // the fewest any of them cleared
    steady_high: u32, // the most layers a steady prompt left out of the fit cleared

    pub fn deinit(f: *Fit, gpa: std.mem.Allocator) void {
        gpa.free(f.dir);
        gpa.free(f.tau);
    }

    /// Whether a fact question the fit never saw still clears more layers than any steady prompt it never saw.
    pub fn apart(f: *const Fit) bool {
        return f.fact_low > f.steady_high;
    }

    /// The layers whose router row `i` of `rows` clears (rows[l]: layer l's decision rows along its candidates).
    pub fn count(f: *const Fit, rows: []const []const f32, i: usize) u32 {
        var votes: u32 = 0;
        for (rows, 0..) |r, l| {
            if (f.tau[l] >= shape.shut) continue;
            votes += @intFromBool(subspace.dot(f.dir[l * n ..][0..n], r[i * n ..][0..n]) >= f.tau[l]);
        }
        return votes;
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

/// A fit's direction, the highest steady row's cosine, and the cosine most of its fact rows clear.
const Fold = struct { dir: [n]f32, top: f32, low: f32 };

fn fold(gpa: std.mem.Allocator, pos: *const Stats, neg: *const Stats, facts: []const f32, steady: []const f32, skip_fact: ?usize, skip_steady: ?usize) !Fold {
    var f: Fold = .{ .dir = undefined, .top = -2, .low = 2 };
    try fisher(pos, neg, &f.dir);
    for (0..steady.len / n) |r| if (r != skip_steady) {
        f.top = @max(f.top, subspace.dot(&f.dir, steady[r * n ..][0..n]));
    };
    const cos = try gpa.alloc(f32, facts.len / n);
    defer gpa.free(cos);
    var m: usize = 0;
    for (0..facts.len / n) |r| if (r != skip_fact) {
        cos[m] = subspace.dot(&f.dir, facts[r * n ..][0..n]);
        m += 1;
    };
    std.mem.sort(f32, cos[0..m], {}, std.sort.asc(f32));
    f.low = cos[m / odd_share];
    return f;
}

/// How far row `p` clears the fit's highest steady row (-inf where most of its facts do not clear it).
fn margin(f: Fold, p: []const f32) f32 {
    return if (f.low > f.top) subspace.dot(&f.dir, p) - f.top else -std.math.inf(f32);
}

/// One layer's fit and every left-out row's margin: facts [F, n] and steady [S, n] decision rows.
fn layer(gpa: std.mem.Allocator, facts: []const f32, steady: []const f32, dir: []f32, top: *f32, valid: *bool, fact_m: []f32, steady_m: []f32) !void {
    const pos = Stats.of(facts);
    const neg = Stats.of(steady);
    const all = try fold(gpa, &pos, &neg, facts, steady, null, null);
    @memcpy(dir, &all.dir);
    top.* = all.top;
    valid.* = all.low > all.top;
    for (fact_m, 0..) |*m, i| {
        const p = facts[i * n ..][0..n];
        var less = pos;
        less.add(p, -1);
        m.* = margin(try fold(gpa, &less, &neg, facts, steady, i, null), p);
    }
    for (steady_m, 0..) |*m, j| {
        const p = steady[j * n ..][0..n];
        var less = neg;
        less.add(p, -1);
        m.* = margin(try fold(gpa, &pos, &less, facts, steady, null, j), p);
    }
}

/// The router from each layer's decision rows: facts[l] and steady[l] the same questions at every layer, [count, n].
pub fn fit(gpa: std.mem.Allocator, facts: []const []const f32, steady: []const []const f32) !Fit {
    const layers = facts.len;
    const nf = facts[0].len / n;
    const ns = steady[0].len / n;
    if (nf < 2 or ns < 2) return error.TooFewQuestions;
    const dir = try gpa.alloc(f32, layers * n);
    errdefer gpa.free(dir);
    const tau = try gpa.alloc(f32, layers);
    errdefer gpa.free(tau);
    const top = try gpa.alloc(f32, layers);
    defer gpa.free(top);
    const valid = try gpa.alloc(bool, layers);
    defer gpa.free(valid);
    const fact_m = try gpa.alloc(f32, layers * nf);
    defer gpa.free(fact_m);
    const steady_m = try gpa.alloc(f32, layers * ns);
    defer gpa.free(steady_m);
    const Share = struct {
        fn run(from: usize, step: usize, ctx: anytype, out: *?anyerror) void {
            var l = from;
            while (l < ctx.layers) : (l += step) layer(std.heap.c_allocator, ctx.facts[l], ctx.steady[l], ctx.dir[l * n ..][0..n], &ctx.top[l], &ctx.valid[l], ctx.fact_m[l * ctx.nf ..][0..ctx.nf], ctx.steady_m[l * ctx.ns ..][0..ctx.ns]) catch |e| {
                out.* = e;
            };
        }
    };
    const ctx = .{ .layers = layers, .facts = facts, .steady = steady, .dir = dir, .top = top, .valid = valid, .fact_m = fact_m, .steady_m = steady_m, .nf = nf, .ns = ns };
    var threads: [16]?std.Thread = @splat(null);
    var errs: [16]?anyerror = @splat(null);
    const k = @min(threads.len, layers);
    for (threads[0..k], 0..) |*t, i| t.* = std.Thread.spawn(.{}, Share.run, .{ i, k, ctx, &errs[i] }) catch null;
    for (threads[0..k], 0..) |t, i| if (t) |th| th.join() else Share.run(i, k, ctx, &errs[i]);
    for (errs[0..k]) |e| if (e) |x| return x;
    var best: ?Fit = null;
    const fact_votes = try gpa.alloc(u32, nf);
    defer gpa.free(fact_votes);
    for (margins) |m| {
        for (fact_votes, 0..) |*votes, i| {
            votes.* = 0;
            for (0..layers) |l| votes.* += @intFromBool(fact_m[l * nf + i] > m);
        }
        std.mem.sort(u32, fact_votes, {}, std.sort.asc(u32));
        const low = fact_votes[nf / odd_share];
        var high: u32 = 0;
        for (0..ns) |j| {
            var votes: u32 = 0;
            for (0..layers) |l| votes += @intFromBool(steady_m[l * ns + j] > m);
            high = @max(high, votes);
        }
        const gap = @as(i64, low) - @as(i64, high);
        const old_gap = if (best) |b| @as(i64, b.fact_low) - @as(i64, b.steady_high) else std.math.minInt(i64);
        if (gap > old_gap or (gap == old_gap and gap > 0)) best = .{ .dir = dir, .tau = tau, .need = 0, .margin = m, .fact_low = low, .fact_least = fact_votes[0], .steady_high = high };
    }
    var f = best.?;
    // half way across the gap between the left-out facts and steady prompts; where they overlap, what most facts
    // clear, so the lesson still answers them (the steady prompts it then takes, it learns to leave as they were)
    const gap = @as(i64, f.fact_low) - @as(i64, f.steady_high);
    f.need = @max(min_need, if (gap > 0) f.steady_high + 1 + @as(u32, @intCast((gap - 1) >> 1)) else f.fact_low);
    for (0..layers) |l| tau[l] = if (valid[l]) top[l] + f.margin else shape.shut;
    return f;
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
    try std.testing.expect(f.need > f.steady_high and f.need <= f.fact_low);
    for (f.tau) |t| try std.testing.expect(t < shape.shut);
}
