//! QK norm and RoPE of one head in float64, the reference for every backend's attention inputs (cuda/qk_rows.zig).
const std = @import("std");
const bf = @import("bf16_round.zig");

/// One head: RMS-normed by w (none: as is), then rotate-half RoPE on its first `rot` dims (no cos/sin: none).
pub fn head(x: []const f64, w: ?[]const f64, eps: f64, cos: ?[]const f64, sin: ?[]const f64, rot: usize, y: []f64) void {
    const n = x.len;
    var ss: f64 = 0;
    for (x) |v| ss += v * v;
    const inv = if (w != null) 1 / @sqrt(ss / @as(f64, @floatFromInt(n)) + eps) else 1;
    var z: [512]f64 = undefined;
    for (x, 0..) |v, i| z[i] = v * inv * (if (w) |ww| ww[i] else 1);
    const half = rot / 2;
    for (y, 0..) |*o, i| {
        if (cos == null or i >= rot) {
            o.* = z[i];
            continue;
        }
        const p = if (i < half) i + half else i - half;
        const rotated = if (i < half) -z[p] else z[p];
        o.* = z[i] * cos.?[i] + rotated * sin.?[i];
    }
}

/// The options of a head beyond `head`'s: MLX's bf16 rows (bf16(x inv) before the weight, w bf16(x inv) before RoPE).
pub const Points = struct { row: bool = false, out: bool = false };

/// An fp32 run's stray: the inverse RMS by `spread`, each product or sum by `u`, cos and sin by `trig` (0: float64).
pub const Stray = struct { spread: f64 = 0, u: f64 = 0, trig: f64 = 0 };

/// a and b in order; NaN in either gives NaN for both.
fn order(a: f64, b: f64) [2]f64 {
    if (std.math.isNan(a) or std.math.isNan(b)) return @splat(std.math.nan(f64));
    return .{ @min(a, b), @max(a, b) };
}

/// The least and largest of the products of two intervals, widened by u of the largest magnitude.
fn mul(a: [2]f64, b: [2]f64, u: f64) [2]f64 {
    var r = [2]f64{ std.math.inf(f64), -std.math.inf(f64) };
    for (a) |x| for (b) |y| {
        const p = x * y;
        if (std.math.isNan(p)) return @splat(std.math.nan(f64));
        r = .{ @min(r[0], p), @max(r[1], p) };
    };
    return widen(r, u);
}

fn widen(r: [2]f64, u: f64) [2]f64 {
    const m = @max(@abs(r[0]), @abs(r[1])) * u;
    return .{ r[0] - m, r[1] + m };
}

/// A normed value's ends over the inverse's spread: (x inv) w in fp32, or w bf16(x inv) (`row`); bf16 with `out`.
fn normed(x: f64, inv: f64, w: f64, p: Points, st: Stray) [2]f64 {
    var r = [2]f64{ std.math.inf(f64), -std.math.inf(f64) };
    const a = st.spread + st.u + st.spread * st.u;
    for ([_]f64{ 1 - a, 1 + a }) |f| for ([_]f64{ 1 - st.u, 1 + st.u }) |f2| {
        const v = x * inv * f;
        var y = if (p.row) w * bf.bf16(v) else v * w * f2;
        if (p.out) y = bf.bf16(y);
        if (std.math.isNan(y)) return @splat(std.math.nan(f64));
        r = .{ @min(r[0], y), @max(r[1], y) };
    };
    return r;
}

/// One head's bf16 outputs as ranges: normed by w (none: as is) from `ss` at points `p`, then RoPE on `rot` dims.
pub fn headRange(x: []const f64, w: ?[]const f64, ss: f64, eps: f64, cos: ?[]const f64, sin: ?[]const f64, rot: usize, p: Points, st: Stray, lo: []f64, hi: []f64) void {
    const inv = 1 / @sqrt(ss / @as(f64, @floatFromInt(x.len)) + eps);
    var z: [512][2]f64 = undefined;
    for (x, 0..) |v, i| z[i] = if (w) |ww| normed(v, inv, ww[i], p, st) else .{ v, v };
    const half = rot / 2;
    for (lo, hi, 0..) |*l, *h, i| {
        var t = z[i];
        if (cos != null and i < rot) {
            const pi = if (i < half) i + half else i - half;
            const zp = if (i < half) [2]f64{ -z[pi][1], -z[pi][0] } else z[pi];
            const c = cos.?[i];
            const sn = sin.?[i];
            const a = mul(t, order(c - st.trig * @abs(c), c + st.trig * @abs(c)), st.u);
            const b = mul(zp, order(sn - st.trig * @abs(sn), sn + st.trig * @abs(sn)), st.u);
            t = widen(.{ a[0] + b[0], a[1] + b[1] }, st.u);
        }
        l.*, h.* = order(bf.bf16(t[0]), bf.bf16(t[1]));
    }
}

/// A v head by a unit RMS norm from its sum of squares: bf16(x inv), as ranges over the inverse's spread.
pub fn unitRange(x: []const f64, ss: f64, eps: f64, st: Stray, lo: []f64, hi: []f64) void {
    const inv = 1 / @sqrt(ss / @as(f64, @floatFromInt(x.len)) + eps);
    for (x, lo, hi) |v, *l, *h| l.*, h.* = normed(v, inv, 1, .{ .out = true }, .{ .spread = st.spread, .u = st.u });
}

/// A head's sum of squares in float64.
pub fn squares(x: []const f64) f64 {
    var s: f64 = 0;
    for (x) |v| s += v * v;
    return s;
}

test "rotate-half pairs (d, d + rot / 2) and leave the rest" {
    var y: [4]f64 = undefined;
    head(&.{ 1, 2, 3, 4 }, null, 0, &.{ 0, 0 }, &.{ 1, 1 }, 2, &y); // angle 90 degrees on the first pair
    try std.testing.expectEqualSlices(f64, &.{ -2, 1, 3, 4 }, &y);
    head(&.{ 2, 2, 2, 2 }, &.{ 1, 1, 1, 3 }, 0, null, null, 0, &y);
    try std.testing.expectEqualSlices(f64, &.{ 1, 1, 1, 3 }, &y);
}

test "rounding points: no options is head's value, MLX's rows round before the weight and before RoPE" {
    var prng = std.Random.DefaultPrng.init(7);
    var x: [64]f64 = undefined;
    var w: [64]f64 = undefined;
    var cs: [64]f64 = undefined;
    var sn: [64]f64 = undefined;
    for (&x, &w, &cs, &sn, 0..) |*a, *b, *c, *s, i| {
        a.* = bf.bf16(prng.random().floatNorm(f64));
        b.* = bf.bf16(0.9 + 0.2 * prng.random().float(f64));
        const ang = 0.37 * @as(f64, @floatFromInt(i % 16 + 1));
        c.* = @cos(ang);
        s.* = @sin(ang);
    }
    var want: [64]f64 = undefined;
    var lo: [64]f64 = undefined;
    var hi: [64]f64 = undefined;
    head(&x, &w, 1e-6, &cs, &sn, 32, &want);
    headRange(&x, &w, squares(&x), 1e-6, &cs, &sn, 32, .{}, .{}, &lo, &hi);
    for (want, lo, hi) |v, l, h| {
        try std.testing.expectEqual(l, h);
        try std.testing.expectEqual(bf.bf16(v), l);
    }
    headRange(&x, &w, squares(&x), 1e-6, null, null, 0, .{ .row = true, .out = true }, .{}, &lo, &hi);
    const inv = 1 / @sqrt(squares(&x) / 64 + 1e-6);
    for (x, w, lo) |v, ww, l| try std.testing.expectEqual(bf.bf16(ww * bf.bf16(v * inv)), l);
    headRange(&x, &w, squares(&x), 1e-6, &cs, &sn, 64, .{ .row = true, .out = true }, .{ .spread = 0x1p-20, .u = 0x1p-24, .trig = 0x1p-22 }, &lo, &hi);
    var wide: usize = 0;
    for (lo, hi) |l, h| {
        try std.testing.expect(l <= h);
        if (l != h) wide += 1;
    }
    try std.testing.expect(wide < 16); // most outputs stay decided
    unitRange(&x, squares(&x), 1e-6, .{}, &lo, &hi);
    for (x, lo) |v, l| try std.testing.expectEqual(bf.bf16(v * inv), l);
}
