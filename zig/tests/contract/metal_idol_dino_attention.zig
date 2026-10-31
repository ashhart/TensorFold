//! dn_attention under the contract: softmax(q k^T / 8) v, bf16 heads of 64, against attention_ref's float64 formula.
const std = @import("std");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const dn = @import("metal_idol_dino_common.zig");

const c = dn.c;
const g = dn.g;
const D = dn.D;
const kref = dn.kref;

const heads = dn.cfg.heads;

/// Query rows repeat at this period (prime to the 64-row blocks): float64 costs 89 queries a head over every key.
const query_period = 89;

/// The scores' 64-deep sums, then the keys' sums.
fn attentionDepth(s: c.Shape) usize {
    return D + s.k;
}

pub const Attention = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    lib: *dn.Lib,
    name: []const u8,
    lq: usize = 0,
    lk: usize = 0,
    stress: bool = false,
    q: []u16 = &.{},
    k: []u16 = &.{},
    v: []u16 = &.{},
    ins: [3]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    /// Rows: the launch's 20 heads; n: query rows Lq; k: key rows Lk. The encoder runs 4,101 against 4,101.
    pub fn spec(name: []const u8) c.Spec {
        return .{ .name = name, .accum = .f32, .depth = attentionDepth, .ops = 8, .out = .bf16, .tiles = .{ .{}, .{ .size = 64 }, .{ .size = 32 } }, .served = &.{ .{ .rows = heads, .n = 4101, .k = 4101 }, .{ .rows = heads, .n = 100, .k = 64 } } };
    }

    pub fn deinit(t: *Attention) void {
        t.free();
        t.rig.deinit();
    }
    fn free(t: *Attention) void {
        t.a.free(t.q);
        t.a.free(t.k);
        t.a.free(t.v);
        t.q = &.{};
        t.k = &.{};
        t.v = &.{};
    }
    pub fn kernel(t: *Attention, control: ?g.Gate) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = control };
    }
    const vt = g.hooks(Attention);

    pub fn prepare(t: *Attention, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n == 0 or shape.k == 0) return false;
        t.rig.reset();
        t.free();
        t.lq = shape.n;
        t.lk = shape.k;
        t.stress = fill == .stress;
        const L = t.lk;
        t.q = try t.a.alloc(u16, heads * t.lq * D);
        t.k = try t.a.alloc(u16, heads * L * D);
        t.v = try t.a.alloc(u16, heads * L * D);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (t.stress) {
            // equal scores (q = k = 0); each column's values bf16 integers whose partial sums reach ~2^22, then cancel
            @memset(t.q, 0);
            @memset(t.k, 0);
            const half = L / 2;
            for (0..heads) |h| for (0..L) |j| {
                const at = if (j < half) j else j - half;
                const positive: f32 = if (at % 2 == 0) 127 else 4064;
                // the first four 4,064s come back as -4,032: a residual of 128 over the column
                const back: f32 = if (at < 8 and at % 2 == 1) 4032 else positive;
                const v: f32 = if (j < half) positive else if (j < 2 * half) -back else 0;
                @memset(t.v[(h * L + j) * D ..][0..D], dn.bits(v));
            };
        } else {
            dn.randomBf(rng, t.q, -2, 2);
            dn.randomBf(rng, t.k, -2, 2);
            dn.randomBf(rng, t.v, -1, 1);
        }
        if (fill == .nonfinite) {
            // head 0: values NaN, +Inf, -Inf, and both infinities in one column; head 1: a NaN query, an infinite key
            for ([_][2]usize{ .{ 0, 0 }, .{ 1, 1 }, .{ 2, 2 }, .{ 3, 3 }, .{ 4, 3 } }, [_]u16{ dn.nan, dn.inf, dn.ninf, dn.inf, dn.ninf }) |p, v| if (p[0] < L) {
                t.v[p[0] * D + p[1]] = v;
            };
            t.q[(t.lq + 0) * D + 5] = dn.nan;
            if (L > 7) t.k[(L + 7) * D + 3] = dn.inf;
        }
        // every head's queries repeat at the period, so float64 computes each distinct one once
        const p = @min(t.lq, query_period);
        for (0..heads) |h| for (p..t.lq) |i| {
            @memcpy(t.q[(h * t.lq + i) * D ..][0..D], t.q[(h * t.lq + i % p) * D ..][0..D]);
        };
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.q));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.k));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.v));
        t.out = try t.rig.output(heads * t.lq * D * 2);
        return true;
    }

    pub fn run(t: *Attention) !void {
        try t.lib.need("dn_attention");
        const p = t.rig.begin();
        t.lib.k.attention(p.e, kref(t.ins[0]), kref(t.ins[1]), kref(t.ins[2]), kref(t.out), heads, t.lq, t.lk);
        try t.rig.finish(p);
    }

    pub fn count(t: *Attention) usize {
        return heads * t.lq * D;
    }

    /// attention_ref.head's formula, one pass over the keys a distinct query; O rows [Lq, heads * 64].
    pub fn reference(t: *Attention, want: []f64, mag: []f64) !void {
        const L = t.lk;
        const scale: f64 = @as(f32, 1.0 / @sqrt(@as(f32, D)));
        const kf = try t.a.alloc(f64, L * D);
        defer t.a.free(kf);
        const vf = try t.a.alloc(f64, L * D);
        defer t.a.free(vf);
        const s = try t.a.alloc(f64, L);
        defer t.a.free(s);
        var qf: [D]f64 = undefined;
        var out: [D]f64 = undefined;
        var tol: [D]f64 = undefined;
        const p = @min(t.lq, query_period);
        for (0..heads) |h| {
            for (kf, vf, t.k[h * L * D ..][0 .. L * D], t.v[h * L * D ..][0 .. L * D]) |*kd, *vd, kb, vb| {
                kd.* = dn.value(kb);
                vd.* = dn.value(vb);
            }
            for (0..p) |i| {
                var zero = true;
                for (&qf, t.q[(h * t.lq + i) * D ..][0..D]) |*x, b| {
                    x.* = dn.value(b);
                    zero = zero and x.* == 0;
                }
                var m: f64 = -std.math.inf(f64);
                var nan = false;
                for (s, 0..) |*sj, j| {
                    var dot: f64 = 0;
                    if (!zero) {
                        const Vec = @Vector(8, f64);
                        var acc: Vec = @splat(0);
                        var e: usize = 0;
                        while (e < D) : (e += 8) acc += @as(Vec, qf[e..][0..8].*) * @as(Vec, kf[j * D + e ..][0..8].*);
                        dot = @reduce(.Add, acc);
                    }
                    sj.* = dot * scale;
                    nan = nan or std.math.isNan(sj.*);
                    m = @max(m, sj.*);
                }
                if (nan) m = std.math.nan(f64);
                var l: f64 = 0;
                @memset(&out, 0);
                @memset(&tol, 0);
                for (s, 0..) |sj, j| {
                    const pj = @exp(sj - m);
                    l += pj;
                    for (&out, &tol, vf[j * D ..][0..D]) |*o, *tl, vv| {
                        o.* += pj * vv;
                        tl.* += pj * @abs(vv);
                    }
                }
                var r = i;
                while (r < t.lq) : (r += p) {
                    const at = (r * heads + h) * D;
                    for (out, tol, 0..) |o, tl, e| {
                        want[at + e] = o / l;
                        // P enters P V as bf16 (2^-8 each), so the weights move by up to 2^-7; O rounds to bf16
                        mag[at + e] = tl / l * 0x1p-7 + @abs(o / l) * 0x1p-8 + 1e-6;
                    }
                }
            }
        }
    }

    pub fn bounds(t: *Attention, want: []const f64, tolerance: []const f64, allow: []f64) !void {
        for (want, tolerance, allow) |w, tl, *l| l.* = if (t.stress) c.boundExact(spec(t.name), w) else if (std.math.isFinite(w)) tl else 0;
    }

    pub fn simulate(t: *Attention, acc: c.Accum, order: g.Order, out: []f64) !void {
        if (!t.stress) return error.AttentionStressNeedsEqualScores;
        const L = t.lk;
        const terms = try t.a.alloc(f64, L);
        defer t.a.free(terms);
        const ones = try t.a.alloc(f64, L);
        defer t.a.free(ones);
        @memset(ones, 1);
        const l = contract.rows.sumIn(acc, order, ones);
        for (0..heads) |h| for (0..D) |e| {
            for (terms, 0..) |*v, j| v.* = dn.value(t.v[(h * L + j) * D + e]);
            const o = contract.rows.sumIn(acc, order, terms) / l;
            for (0..t.lq) |i| out[(i * heads + h) * D + e] = o;
        };
    }
};
