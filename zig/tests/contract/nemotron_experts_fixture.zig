//! Nemotron routed-expert inputs: eight 4-bit experts, the engine's route tables and float64 at the bf16 x sums.
const std = @import("std");
const contract = @import("contract");
const cases = @import("nemotron_experts_cases.zig");
const c = contract.spec;
const g = contract.gates;
const Affine = contract.affine.Affine;

pub const experts = 8;
pub const top_k = 6;
pub const group = 64;
/// Threadgroups along z for a window, as layers.zig dispatches: min(pairs, the checkpoint's 128 experts).
const engine_experts = 128;

/// Rows each expert sits out of a window: two a row, weighted 0 to 7, so member counts differ and none passes rows.
pub fn exclusions(rows: usize) [experts]usize {
    var d: [experts]usize = undefined;
    var sum: usize = 0;
    for (&d, 0..) |*v, e| {
        v.* = 2 * rows * e / 28;
        sum += v.*;
    }
    var e: usize = experts - 1;
    while (sum < 2 * rows) : (e -= 1) {
        d[e] += 1;
        sum += 1;
    }
    return d;
}

/// A window's picks and the tables the engine binds: route's for one row (pair order), route_group's for more.
pub const Route = struct {
    ids: []u32,
    uids: []u32,
    start: []i32,
    count: []i32,
    members: []i32,
    ucount: i32,
    groups: usize,

    pub fn init(a: std.mem.Allocator, rows: usize, seed: u64) !Route {
        const pairs = rows * top_k;
        const ids = try a.alloc(u32, pairs);
        errdefer a.free(ids);
        var prng = std.Random.DefaultPrng.init(seed ^ 0x1494);
        var left = exclusions(rows);
        for (0..rows) |r| {
            var out = [2]usize{ experts, experts };
            for (&out, 0..) |*o, i| for (0..experts) |e| {
                if (i == 1 and e == out[0]) continue;
                if (o.* == experts or left[e] >= left[o.*]) o.* = e;
            };
            left[out[0]] -= 1;
            left[out[1]] -= 1;
            var picks: [top_k]u32 = undefined;
            var n: usize = 0;
            for (0..experts) |e| if (e != out[0] and e != out[1]) {
                picks[n] = @intCast(e);
                n += 1;
            };
            prng.random().shuffle(u32, &picks);
            @memcpy(ids[r * top_k ..][0..top_k], &picks);
        }
        const slots: usize = if (rows == 1) top_k else experts;
        var r: Route = .{ .ids = ids, .uids = try a.alloc(u32, slots), .start = try a.alloc(i32, slots), .count = try a.alloc(i32, slots), .members = try a.alloc(i32, pairs), .ucount = 0, .groups = if (rows == 1) pairs else @min(pairs, engine_experts) };
        if (rows == 1) {
            for (0..top_k) |k| {
                r.uids[k] = ids[k];
                r.start[k] = @intCast(k);
                r.count[k] = 1;
                r.members[k] = @intCast(k);
            }
            r.ucount = top_k;
            return r;
        }
        var u: usize = 0;
        var m: usize = 0;
        for (0..experts) |e| {
            const first = m;
            for (ids, 0..) |id, p| if (id == e) {
                r.members[m] = @intCast(p);
                m += 1;
            };
            if (m == first) continue;
            r.uids[u] = @intCast(e);
            r.start[u] = @intCast(first);
            r.count[u] = @intCast(m - first);
            u += 1;
        }
        r.ucount = @intCast(u);
        return r;
    }

    pub fn deinit(r: Route, a: std.mem.Allocator) void {
        for ([_][]u32{ r.ids, r.uids }) |s| a.free(s);
        for ([_][]i32{ r.start, r.count, r.members }) |s| a.free(s);
    }
};

/// Four bf16 inputs summed as tf_load16 does: fp32 adds, each rounded to bf16.
pub fn chain(x: []const u16) f64 {
    @setFloatMode(.strict);
    const s1: f32 = c.fromBf16(x[0]) + c.fromBf16(x[1]);
    const s2: f32 = c.fromBf16(c.toBf16(s1)) + c.fromBf16(x[2]);
    const s3: f32 = c.fromBf16(c.toBf16(s2)) + c.fromBf16(x[3]);
    return c.fromBf16(c.toBf16(s3));
}

/// relu(d)^2 with float64's NaN kept.
pub fn relu2(d: f64) f64 {
    if (std.math.isNan(d)) return d;
    const h = @max(d, 0);
    return h * h;
}

/// The bf16 value nearest `x` on its low (up false) or high side.
pub fn bfSide(x: f64, up: bool) f64 {
    var f: f32 = @floatCast(x);
    if (up and @as(f64, f) < x) f = std.math.nextAfter(f32, f, std.math.inf(f32));
    if (!up and @as(f64, f) > x) f = std.math.nextAfter(f32, f, -std.math.inf(f32));
    const t = @as(u32, @bitCast(f)) & 0xffff0000;
    if (@as(f32, @bitCast(t)) == f or (up == (f < 0))) return @as(f32, @bitCast(t));
    return @as(f32, @bitCast(t + 0x10000));
}

pub const Fixture = struct {
    a: std.mem.Allocator,
    proj: cases.Proj,
    shape: c.Shape,
    fill: g.Fill,
    q: Affine,
    route: Route,
    xf: []f64,
    chains: []f64,
    dots: []f64,
    mags: []f64,

    pub fn takes(proj: cases.Proj, s: c.Shape) bool {
        const nk = cases.dims(proj);
        return s.rows >= 1 and s.rows <= 128 and s.n == nk[0] and s.k == nk[1];
    }

    pub fn init(a: std.mem.Allocator, proj: cases.Proj, s: c.Shape, fill: g.Fill, seed: u64) !Fixture {
        const xrows = if (proj == .up) s.rows else s.rows * top_k;
        var q = try Affine.init(a, .{ .rows = xrows, .n = experts * s.n, .k = s.k }, 4, group);
        errdefer q.deinit();
        if (fill == .stress) stress(q, seed) else try q.fill(fill, seed);
        const route = try Route.init(a, s.rows, seed);
        errdefer route.deinit(a);
        const xf = try a.alloc(f64, q.x.len);
        errdefer a.free(xf);
        for (xf, q.x) |*v, b| v.* = c.fromBf16(b);
        const chains = try a.alloc(f64, q.x.len / 4);
        errdefer a.free(chains);
        for (chains, 0..) |*v, i| v.* = chain(q.x[4 * i ..][0..4]);
        const outs = s.rows * top_k * s.n;
        const dots = try a.alloc(f64, outs);
        errdefer a.free(dots);
        return .{ .a = a, .proj = proj, .shape = s, .fill = fill, .q = q, .route = route, .xf = xf, .chains = chains, .dots = dots, .mags = try a.alloc(f64, outs) };
    }

    pub fn deinit(f: Fixture) void {
        f.q.deinit();
        f.route.deinit(f.a);
        for ([_][]f64{ f.xf, f.chains, f.dots, f.mags }) |s| f.a.free(s);
    }

    /// Gate 4's inputs: scale 1, bias -8, codes q and 16 - q by halves, x in fours so bf16 x sums stay exact.
    fn stress(q: Affine, seed: u64) void {
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        @memset(q.scales, c.toBf16(1));
        @memset(q.biases, c.toBf16(-8));
        const k = q.shape.k;
        const h = k / 2;
        for (0..q.shape.n) |j| for (0..h) |i| {
            const v: u8 = @intCast(9 + rng.uintLessThan(u32, 7));
            q.codes[j * k + i] = v;
            q.codes[j * k + h + i] = 16 - v;
        };
        for (0..q.shape.rows) |r| for (0..h) |i| {
            const v: f32 = @floatFromInt(4 * (1 + rng.uintLessThan(u32, 62)));
            q.x[r * k + i] = c.toBf16(v + if (i < 8) @as(f32, 4) else 0);
            q.x[r * k + h + i] = c.toBf16(v);
        };
    }

    pub fn pairs(f: Fixture) usize {
        return f.shape.rows * top_k;
    }

    pub fn count(f: Fixture) usize {
        return f.pairs() * f.shape.n;
    }

    /// The x row pair p reads: its token's row (up) or its own activations (down).
    pub fn xRow(f: Fixture, p: usize) usize {
        return if (f.proj == .up) p / top_k else p;
    }

    /// The codes as the checkpoint packs them: eight 4-bit codes a word, the first in its low bits.
    pub fn words(f: Fixture, out: []u32) void {
        @memset(out, 0);
        for (f.q.codes, 0..) |q, i| out[i / 8] |= @as(u32, q) << @intCast(4 * (i % 8));
    }

    /// Pair p's output j before relu2: per group s * sum(q x) + b * (its bf16 x sums), and its terms' magnitude.
    fn dot(f: Fixture, p: usize, j: usize) [2]f64 {
        const k = f.shape.k;
        const col = f.route.ids[p] * f.shape.n + j;
        const xr = f.xRow(p);
        const x = f.xf[xr * k ..][0..k];
        const codes = f.q.codes[col * k ..][0..k];
        const sums = f.chains[xr * k / 4 ..][0 .. k / 4];
        var y: f64 = 0;
        var mag: f64 = 0;
        for (0..k / group) |gi| {
            const s: f64 = c.fromBf16(f.q.scales[col * (k / group) + gi]);
            const b: f64 = c.fromBf16(f.q.biases[col * (k / group) + gi]);
            var qx: f64 = 0;
            var qa: f64 = 0;
            for (gi * group..(gi + 1) * group) |i| {
                const t = @as(f64, @floatFromInt(codes[i])) * x[i];
                qx += t;
                qa += @abs(t);
            }
            var xs: f64 = 0;
            var xa: f64 = 0;
            for (sums[gi * group / 4 ..][0 .. group / 4]) |v| {
                xs += v;
                xa += @abs(v);
            }
            y += s * qx + b * xs;
            mag += @abs(s) * qa + @abs(b) * xa;
        }
        return .{ y, mag };
    }

    pub fn reference(f: Fixture, want: []f64, mag: []f64) void {
        for (0..f.pairs()) |p| for (0..f.shape.n) |j| {
            const d = f.dot(p, j);
            const i = p * f.shape.n + j;
            f.dots[i] = d[0];
            f.mags[i] = d[1];
            want[i] = if (f.proj == .up) relu2(d[0]) else d[0];
            mag[i] = d[1];
        };
    }

    /// Where relu2(bf16(d)) may land: d within its fp32 sums' error, then each rounding (the last reference's).
    pub fn intervals(f: Fixture, low: []f64, high: []f64) void {
        const e = if (f.fill == .stress) 0 else c.gamma(@floatFromInt(cases.depth(f.shape)), 0x1p-24);
        for (f.dots, f.mags, low, high) |d, m, *l, *h| {
            const lo = @max(bfSide(d - e * m, false), 0);
            const hi = @max(bfSide(d + e * m, true), 0);
            l.* = bfSide(lo * lo, false);
            h.* = bfSide(hi * hi, true);
        }
    }

    /// Each output's allowance: its interval's farther end from float64, at least the output's half ulp.
    pub fn bounds(f: Fixture, want: []const f64, allow: []f64) !void {
        const ends = try f.a.alloc(f64, 2 * want.len);
        defer f.a.free(ends);
        f.intervals(ends[0..want.len], ends[want.len..]);
        for (allow, want, ends[0..want.len], ends[want.len..]) |*l, w, lo, hi| l.* = @max(hi - w, w - lo, 0x1p-8 * @abs(w)) + 0x1p-133;
    }

    /// On gate 4's inputs: each dot as an accumulator leaves it, closest of three structures, then relu2 for up.
    pub fn simulate(f: Fixture, acc: c.Accum, order: g.Order, out: []f64) !void {
        const k = f.shape.k;
        const t = try f.a.alloc(f64, k + k / 4 + k / group);
        defer f.a.free(t);
        const qx = t[0..k];
        const xs = t[k..][0 .. k / 4];
        const gs = t[k + k / 4 ..];
        const sum = contract.rows.sumIn;
        for (0..f.pairs()) |p| for (0..f.shape.n) |j| {
            const col = f.route.ids[p] * f.shape.n + j;
            const xr = f.xRow(p);
            const s: f64 = c.fromBf16(f.q.scales[col * (k / group)]);
            const b: f64 = c.fromBf16(f.q.biases[col * (k / group)]);
            for (qx, 0..) |*v, i| v.* = @as(f64, @floatFromInt(f.q.codes[col * k + i])) * f.xf[xr * k + i];
            @memcpy(xs, f.chains[xr * k / 4 ..][0 .. k / 4]);
            const whole = sum(acc, order, qx) * s + sum(acc, order, xs) * b;
            for (gs, 0..) |*v, gi| v.* = sum(acc, .sequential, qx[gi * group ..][0..group]) * s + sum(acc, .sequential, xs[gi * group / 4 ..][0 .. group / 4]) * b;
            const grouped = sum(acc, order, gs);
            for (qx, 0..) |*v, i| v.* = (s * @as(f64, @floatFromInt(f.q.codes[col * k + i])) + b) * f.xf[xr * k + i];
            const direct = sum(acc, order, qx);
            const want = f.dots[p * f.shape.n + j];
            var best = whole;
            for ([_]f64{ grouped, direct }) |v| if (!std.math.isFinite(best) or @abs(v - want) < @abs(best - want)) {
                best = v;
            };
            out[p * f.shape.n + j] = if (f.proj == .up) relu2(best) else best;
        };
    }
};
