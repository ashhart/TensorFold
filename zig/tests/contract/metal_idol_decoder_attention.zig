//! The decoders' attention under the contract: softmax(q k^T / 8) v, P V as fp16 P hi + lo, or fp32 for 8 queries.
const std = @import("std");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const id = @import("metal_idol_common.zig");
const steps = @import("metal_idol_decoder_steps.zig");

const c = id.c;
const g = id.g;
const D = id.D;
const kref = id.kref;
const dk = id.tf.idol.decoder_kernels;
const dc = id.tf.idol.decoder_config;

const heads = dc.heads;
const C = dc.context_rows;

/// Query rows repeat at this period (prime to the 64-row blocks): float64 costs 89 queries a head over every key.
const query_period = 89;

/// Heads from here on make O one P's fp16 remainder alone (see `exact`): P without its lo half gives 0 there.
const exact_heads = 8;
const p_dot: f16 = -1;
const p_hi: f16 = @floatCast(@exp(-1.0 / 8.0));

/// Tiled forms by (Lq % 64 == 0)(Lk % 32 == 0), then the short form; rows: the 16 heads, n: Lq, k: Lk.
pub const forms = [5]struct { name: []const u8, served: []const c.Shape, tiles: [3]c.Tile }{
    .{
        .name = "dec_attention_00",
        // self attention over 1,025 and 8,193 anchors (32,800 and 262,176 Gaussians)
        .served = &.{ .{ .rows = heads, .n = 1025, .k = 1025 }, .{ .rows = heads, .n = 8193, .k = 8193 } },
        .tiles = .{ .{}, .{ .size = 64 }, .{ .size = 32 } },
    },
    .{
        .name = "dec_attention_01",
        // octree parents to the latent, the fewest past the short form, and 1,056 anchors' self attention
        .served = &.{ .{ .rows = heads, .n = 1000, .k = C }, .{ .rows = heads, .n = 9, .k = C }, .{ .rows = heads, .n = 1056, .k = 1056 } },
        .tiles = .{ .{}, .{ .size = 64 }, .{ .size = 64, .step = 32, .min = 32 } },
    },
    // Lq = Lk or Lk = 8,192 in every served call, so this form is never served: checked at its tails
    .{
        .name = "dec_attention_10",
        .served = &.{ .{ .rows = heads, .n = 4096, .k = 4101 }, .{ .rows = heads, .n = 128, .k = 70 } },
        .tiles = .{ .{}, .{ .size = 128, .step = 64, .min = 64 }, .{ .size = 32 } },
    },
    .{
        .name = "dec_attention_11",
        // 32k Gaussians' anchors, parents to the latent, then 262k's and 1M's anchors
        .served = &.{
            .{ .rows = heads, .n = 1024, .k = 1024 }, .{ .rows = heads, .n = 4096, .k = C },
            .{ .rows = heads, .n = 8192, .k = 8192 }, .{ .rows = heads, .n = 32768, .k = 32768 },
        },
        .tiles = .{ .{}, .{ .size = 128, .step = 64, .min = 64 }, .{ .size = 64, .step = 32, .min = 32 } },
    },
    .{
        .name = "dec_attention_few",
        // the first octree levels' 1 to 8 parents against the latent
        .served = &.{ .{ .rows = heads, .n = dk.few_rows, .k = C }, .{ .rows = heads, .n = 1, .k = C } },
        .tiles = .{ .{}, .{ .size = dk.few_rows }, .{ .size = 256 } },
    },
};
pub const few = 4;

pub fn form(lq: usize, lk: usize) usize {
    return @as(usize, @intFromBool(lq % 64 == 0)) * 2 + @intFromBool(lk % 32 == 0);
}

/// The scores' 64-deep sums, then the keys' sums.
fn attentionDepth(s: c.Shape) usize {
    return D + s.k;
}

pub const Attention = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    lib: *steps.Lib,
    form: usize,
    name: []const u8,
    lq: usize = 0,
    lk: usize = 0,
    stress: bool = false,
    q: []f16 = &.{},
    k: []f16 = &.{},
    v: []f16 = &.{},
    ins: [3]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub fn spec(name: []const u8, f: usize) c.Spec {
        return .{ .name = name, .accum = .f32, .depth = attentionDepth, .ops = 8, .out = .f16, .tiles = forms[f].tiles, .served = forms[f].served };
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
        return .{ .spec = spec(t.name, t.form), .ctx = t, .vt = &vt, .control = control };
    }
    const vt = g.hooks(Attention);

    fn takes(t: *const Attention, lq: usize, lk: usize) bool {
        if (lq == 0 or lk == 0) return false;
        if (t.form == few) return lq <= dk.few_rows and lk <= 32 * 256;
        return form(lq, lk) == t.form;
    }

    pub fn prepare(t: *Attention, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (!t.takes(shape.n, shape.k)) return false;
        t.rig.reset();
        t.free();
        t.lq = shape.n;
        t.lk = shape.k;
        t.stress = fill == .stress;
        const L = t.lk;
        t.q = try t.a.alloc(f16, heads * t.lq * D);
        t.k = try t.a.alloc(f16, heads * L * D);
        t.v = try t.a.alloc(f16, heads * L * D);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (t.stress) {
            // q = k = 0; odd fp16 integers, partial sums under 2^24 that cancel to 4; a thread's fp16 chain loses bits
            @memset(t.q, 0);
            @memset(t.k, 0);
            const half = L / 2;
            const big: f32 = if (half / 2 * (127 + 2047) < 1 << 24) 2047 else 1023;
            for (0..heads) |h| for (0..L) |j| {
                const at = if (j < half) j else j - half;
                const positive: f32 = if (at % 2 == 0) 127 else big;
                const residual: f32 = if (at < 8 and at % 2 == 0) 1 else 0;
                const v: f32 = if (j < half) positive else if (j < 2 * half) residual - positive else 0;
                @memset(t.v[(h * L + j) * D ..][0..D], @floatCast(v));
            };
        } else {
            id.randomF16(rng, t.q, -2, 2);
            id.randomF16(rng, t.k, -2, 2);
            id.randomF16(rng, t.v, -1, 1);
            if (L >= 2) for (exact_heads..heads) |h| t.exact(rng, h);
        }
        if (fill == .nonfinite) {
            // head 0: values NaN, +Inf, -Inf, and both infinities in one column; head 1: a NaN query, an infinite key
            for ([_][2]usize{ .{ 0, 0 }, .{ 1, 1 }, .{ 2, 2 }, .{ 3, 3 }, .{ 4, 3 } }, [_]f16{ id.nan16, id.inf16, -id.inf16, id.inf16, -id.inf16 }) |p, v| if (p[0] < L) {
                t.v[p[0] * D + p[1]] = v;
            };
            t.q[(t.lq + 0) * D + 5] = id.nan16;
            if (L > 7) t.k[(L + 7) * D + 3] = id.inf16;
        }
        // every head's queries repeat at the period, so float64 computes each distinct one once
        const p = @min(t.lq, query_period);
        for (0..heads) |h| for (p..t.lq) |i| {
            @memcpy(t.q[(h * t.lq + i) * D ..][0..D], t.q[(h * t.lq + i % p) * D ..][0..D]);
        };
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.q));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.k));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.v));
        t.out = try t.rig.output(t.lq * heads * D * 2);
        return true;
    }

    /// Head h: scores 0, -1/8 (last key) and -256 (P = 1, e^-1/8, 0), values -2^m fp16(P) and 2^m: O = 2^m lo / l.
    fn exact(t: *Attention, rng: std.Random, h: usize) void {
        const L = t.lk;
        for (0..t.lq) |i| {
            const q = t.q[(h * t.lq + i) * D ..][0..D];
            @memset(q, 0);
            q[0] = 1;
        }
        for (0..L) |j| {
            const k = t.k[(h * L + j) * D ..][0..D];
            @memset(k, 0);
            k[0] = if (j == 0) 0 else if (j == L - 1) p_dot else -2048;
        }
        for (0..D) |e| {
            const m: f32 = @floatFromInt(@as(u32, 1) << @intCast(rng.uintLessThan(u32, 11)));
            const s = id.sign(rng) * m;
            t.v[(h * L) * D + e] = @floatCast(-s * @as(f32, p_hi));
            t.v[(h * L + L - 1) * D + e] = @floatCast(s);
        }
    }

    pub fn run(t: *Attention) !void {
        try t.lib.need(forms[t.form].name);
        const p = t.rig.begin();
        const x = &t.lib.k;
        if (t.form == few) {
            x.attentionFew(p.e, kref(t.ins[0]), kref(t.ins[1]), kref(t.ins[2]), kref(t.out), t.lq, t.lk);
        } else x.attention(p.e, kref(t.ins[0]), kref(t.ins[1]), kref(t.ins[2]), kref(t.out), t.lq, t.lk);
        try t.rig.finish(p);
    }

    pub fn count(t: *Attention) usize {
        return heads * t.lq * D;
    }

    /// Float64 O [Lq, 16 * 64]; allowance 2^-17 sum P |v| / l + sum min(P, 2^-25) |v| / l + O's rounding.
    pub fn reference(t: *Attention, want: []f64, mag: []f64) !void {
        const L = t.lk;
        const scale: f64 = 0.125;
        const kf = try t.a.alloc(f64, L * D);
        defer t.a.free(kf);
        const vf = try t.a.alloc(f64, L * D);
        defer t.a.free(vf);
        const s = try t.a.alloc(f64, L);
        defer t.a.free(s);
        var qf: [D]f64 = undefined;
        var out: [D]f64 = undefined;
        var tol: [D]f64 = undefined;
        var floor: [D]f64 = undefined;
        const p = @min(t.lq, query_period);
        for (0..heads) |h| {
            for (kf, vf, t.k[h * L * D ..][0 .. L * D], t.v[h * L * D ..][0 .. L * D]) |*kd, *vd, kb, vb| {
                kd.* = kb;
                vd.* = vb;
            }
            for (0..p) |i| {
                var zero = true;
                for (&qf, t.q[(h * t.lq + i) * D ..][0..D]) |*x, b| {
                    x.* = b;
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
                @memset(&floor, 0);
                for (s, 0..) |sj, j| {
                    const pj = @exp(sj - m);
                    const least = @min(pj, 0x1p-25);
                    l += pj;
                    for (&out, &tol, &floor, vf[j * D ..][0..D]) |*o, *tl, *fl, vv| {
                        o.* += pj * vv;
                        tl.* += pj * @abs(vv);
                        fl.* += least * @abs(vv);
                    }
                }
                var r = i;
                while (r < t.lq) : (r += p) {
                    const at = (r * heads + h) * D;
                    for (out, tol, floor, 0..) |o, tl, fl, e| {
                        want[at + e] = o / l;
                        mag[at + e] = tl / l * 0x1p-17 + fl / l + @abs(o / l) * 0x1p-10 + 0x1p-24;
                    }
                }
            }
        }
    }

    pub fn bounds(t: *Attention, want: []const f64, tolerance: []const f64, allow: []f64) !void {
        for (want, tolerance, allow) |w, tl, *l| l.* = if (t.stress) c.boundExact(spec(t.name, t.form), w) else if (std.math.isFinite(w)) tl else 0;
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
            for (terms, 0..) |*v, j| v.* = t.v[(h * L + j) * D + e];
            const o = contract.rows.sumIn(acc, order, terms) / l;
            for (0..t.lq) |i| out[(i * heads + h) * D + e] = o;
        };
    }
};

test "the exact heads' P has a fp16 remainder far from both zero and a tie" {
    const p = @exp(@as(f64, p_dot) / 8);
    const ulp = 0x1p-11;
    const lo = (p - @as(f64, p_hi)) / ulp;
    try std.testing.expect(@abs(lo) > 0.25 and @abs(lo) < 0.4);
}
