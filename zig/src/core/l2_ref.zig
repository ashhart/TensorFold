//! nGPT rows in float64 (Dory's L2 norms and slerp residual), as ranges over an fp32 run at bf16 rounding points.
const std = @import("std");
const bf = @import("bf16_round.zig");

/// A value's least and largest ends; NaN ends for a NaN.
pub const Iv = [2]f64;

/// An fp32 run's stray: an inverse norm's relative spread, and each fp32 product or sum's (0, 0: float64's points).
pub const Stray = struct { spread: f64 = 0, u: f64 = 0 };

/// Sums of squares in float64; a test may pass its own `squares` (another accumulator, gate 4).
pub const Exact = struct {
    pub fn squares(_: Exact, v: []const f64) f64 {
        var s: f64 = 0;
        for (v) |x| s += x * x;
        return s;
    }
};

pub fn point(v: f64) Iv {
    return .{ v, v };
}

/// a and b in order; NaN in either gives NaN for both.
pub fn order(a: f64, b: f64) Iv {
    if (std.math.isNan(a) or std.math.isNan(b)) return @splat(std.math.nan(f64));
    return .{ @min(a, b), @max(a, b) };
}

/// The ends rounded to bf16 (rounding is monotone).
pub fn bf16(r: Iv) Iv {
    return order(bf.bf16(r[0]), bf.bf16(r[1]));
}

pub fn bf16If(round: bool, r: Iv) Iv {
    return if (round) bf16(r) else r;
}

/// The ends of x * y over both intervals, widened by u of the largest magnitude (one fp32 product).
pub fn mul(x: Iv, y: Iv, u: f64) Iv {
    var r = Iv{ std.math.inf(f64), -std.math.inf(f64) };
    for (x) |a| for (y) |b| {
        const p = a * b;
        if (std.math.isNan(p)) return @splat(std.math.nan(f64));
        r = .{ @min(r[0], p), @max(r[1], p) };
    };
    return widen(r, u);
}

/// r widened by u of its largest magnitude and, with u, fp32's least subnormal step; infinite ends stay as they are.
pub fn widen(r: Iv, u: f64) Iv {
    const m = @max(@abs(r[0]), @abs(r[1])) * u + @as(f64, if (u > 0) 0x1p-149 else 0);
    return .{ if (std.math.isFinite(r[0])) r[0] - m else r[0], if (std.math.isFinite(r[1])) r[1] + m else r[1] };
}


/// 1 / sqrt(sum(v^2) + eps) over every v in its interval, and the root's and quotient's spread `sp`.
pub fn invL2(v: []const Iv, eps: f64, sums: anytype, sp: f64, small: []f64, big: []f64) Iv {
    for (v, small, big) |x, *s, *b| {
        const lo, const hi = order(@abs(x[0]), @abs(x[1]));
        s.* = if (x[0] <= 0 and x[1] >= 0) 0 else lo;
        b.* = hi;
    }
    return .{ (1 - sp) / @sqrt(sums.squares(big) + eps), (1 + sp) / @sqrt(sums.squares(small) + eps) };
}

/// Every row value times the inverse norm over v, in place; bf16 with `round`.
pub fn normalize(v: []Iv, eps: f64, sums: anytype, st: Stray, round: bool, small: []f64, big: []f64) void {
    const inv = invL2(v, eps, sums, st.spread, small, big);
    for (v) |*x| x.* = bf16If(round, mul(x.*, inv, st.u));
}

/// tf_l2_rows: y = bf16(l2(s)), s = x, or x + r (fp32) rounded to bf16 with `sum`; y as ranges.
pub fn l2Rows(x: []const f64, r: ?[]const f64, sum: bool, eps: f64, sums: anytype, st: Stray, y: []Iv, small: []f64, big: []f64) void {
    for (y, x, 0..) |*o, v, i| {
        const s: f64 = if (r) |rr| @as(f32, @floatCast(v + rr[i])) else v; // the fp32 add rounds the exact sum once
        o.* = point(if (sum and r != null) bf.bf16(s) else s);
    }
    normalize(y, eps, sums, st, true, small, big);
}

/// The slerp's rounding points: the normalized residual a, the normalized branch b.
pub const Points = struct { a: bool = true, b: bool = true };

/// tf_slerp_rows: h = bf16(l2(a + lr (b - a))), a = l2(res), b = l2(out) (bf16 at points), lr = |fp32(alpha scale)|.
pub fn slerp(res: []const f64, out: []const f64, alpha: []const f64, scale: f32, p: Points, eps: f64, sums: anytype, st: Stray, h: []Iv, scratch: []Iv, small: []f64, big: []f64) void {
    const a = h;
    const b = scratch;
    for (a, b, res, out) |*x, *y, r, o| {
        x.* = point(r);
        y.* = point(o);
    }
    normalize(a, eps, sums, st, p.a, small, big);
    normalize(b, eps, sums, st, p.b, small, big);
    for (a, b, alpha) |*x, y, al| {
        const lr = @abs(@as(f64, @as(f32, @floatCast(al * scale))));
        var z = Iv{ std.math.inf(f64), -std.math.inf(f64) };
        var nan = false;
        for (x.*) |av| for (y) |bv| { // linear in a and b: the corners bound it
            const m = lr * (bv - av);
            const zc = av + m;
            nan = nan or std.math.isNan(zc);
            const e = if (std.math.isFinite(zc)) (@abs(zc) + 2 * @abs(m)) * st.u * 1.01 else 0; // b - a, lr d, a + m
            z = .{ @min(z[0], zc - e), @max(z[1], zc + e) };
        };
        x.* = if (nan) @splat(std.math.nan(f64)) else z;
    }
    normalize(a, eps, sums, st, true, small, big);
}

test "points: l2 rows are x / |x| in bf16, the sum rounds first, the slerp's ends meet at a and b" {
    var x = [_]f64{ 3, 4, 0, 0 };
    var y: [4]Iv = undefined;
    var s: [4]f64 = undefined;
    var g: [4]f64 = undefined;
    l2Rows(&x, null, true, 0, Exact{}, .{}, &y, &s, &g);
    try std.testing.expectEqual(bf.bf16(0.6), y[0][0]);
    try std.testing.expectEqual(bf.bf16(0.8), y[1][1]);
    const r = [_]f64{ 0x1p-9, 0, 0, 0 };
    var one = [_]f64{ 1, 0, 0, 0 };
    l2Rows(&one, &r, true, 0, Exact{}, .{}, &y, &s, &g);
    try std.testing.expectEqual(@as(f64, 1), y[0][0]); // bf16(1 + 2^-9) = 1 before the norm
    var h: [4]Iv = undefined;
    var t: [4]Iv = undefined;
    const alpha = [_]f64{ 0, 0, 0, 0 };
    slerp(&x, &[_]f64{ 0, 0, 1, 0 }, &alpha, 1, .{}, 0, Exact{}, .{}, &h, &t, &s, &g);
    try std.testing.expectEqual(bf.bf16(0.6), h[0][0]); // lr 0: the residual itself
    const ones = [_]f64{ -1, 1, 1, 1 }; // |alpha scale| = 1: the branch
    slerp(&x, &[_]f64{ 0, 0, 1, 0 }, &ones, 1, .{}, 0, Exact{}, .{}, &h, &t, &s, &g);
    try std.testing.expectEqual(@as(f64, 1), h[2][0]);
    try std.testing.expectEqual(@as(f64, 0), h[0][1]);
}

test "an fp32 run's ranges hold the points and stay far inside a bf16 step" {
    var prng = std.Random.DefaultPrng.init(11);
    var res: [256]f64 = undefined;
    var out: [256]f64 = undefined;
    var alpha: [256]f64 = undefined;
    for (&res, &out, &alpha) |*a, *b, *c| {
        a.* = bf.bf16(prng.random().floatNorm(f64));
        b.* = bf.bf16(4 * prng.random().floatNorm(f64));
        c.* = bf.bf16(0.3 * prng.random().floatNorm(f64));
    }
    var hp: [256]Iv = undefined;
    var hr: [256]Iv = undefined;
    var t: [256]Iv = undefined;
    var s: [256]f64 = undefined;
    var g: [256]f64 = undefined;
    slerp(&res, &out, &alpha, 2.8248587, .{}, 1e-6, Exact{}, .{}, &hp, &t, &s, &g);
    slerp(&res, &out, &alpha, 2.8248587, .{}, 1e-6, Exact{}, .{ .spread = 0x1p-20, .u = 0x1p-24 }, &hr, &t, &s, &g);
    var ranged: usize = 0;
    for (hp, hr) |p, r| {
        try std.testing.expect(r[0] <= p[0] and p[0] <= r[1]);
        if (r[0] != r[1]) ranged += 1;
    }
    try std.testing.expect(ranged < 32);
}

/// tf_qk_rows_l2's rounding points: the rotated head, q's normalized head before its scale.
pub const QkPoints = struct { rope: bool = true, norm: bool = true };

/// tf_qk_rows_l2's head as ranges: RoPE (cos, sin within `trig`), bf16; its L2 norm n; q bf16(qs n), k bf16(n).
pub fn qkHead(x: []const f64, cos: []const f64, sin: []const f64, rot: usize, qs: ?[]const f64, p: QkPoints, eps: f64, sums: anytype, st: Stray, trig: f64, y: []Iv, small: []f64, big: []f64) void {
    const half = rot / 2;
    for (y, x, 0..) |*o, v, i| {
        if (i >= rot) {
            o.* = point(v);
            continue;
        }
        const xp = if (i < half) -x[i + half] else x[i - half];
        const cs = order(cos[i] - trig * @abs(cos[i]), cos[i] + trig * @abs(cos[i]));
        const sn = order(sin[i] - trig * @abs(sin[i]), sin[i] + trig * @abs(sin[i]));
        const a = mul(point(v), cs, st.u);
        const b = mul(point(xp), sn, st.u);
        o.* = bf16If(p.rope, widen(order(a[0] + b[0], a[1] + b[1]), st.u));
    }
    const inv = invL2(y, eps, sums, st.spread, small, big);
    for (y, 0..) |*o, i| {
        const n = mul(o.*, inv, st.u);
        o.* = if (qs) |q| bf16(mul(bf16If(p.norm, n), point(q[i]), st.u)) else bf16(n);
    }
}

/// exp in fp32's range: past its largest finite value, infinity (expf overflows there).
fn exp32(x: f64) f64 {
    const e = @exp(x);
    return if (e > std.math.floatMax(f32)) std.math.inf(f64) else e;
}

/// exp's ends over x's interval, each within `e` relative (expf's ulps); monotone.
fn expIv(x: Iv, e: f64) Iv {
    return .{ exp32(x[0]) * (1 - e), exp32(x[1]) * (1 + e) };
}

/// 1 / (1 + exp(-x)) over x's interval in fp32 steps (expf within `e`, the sum and quotient by u): increasing in x.
pub fn sigmoid(x: Iv, e: f64, u: f64) Iv {
    const ex = expIv(.{ -x[1], -x[0] }, e);
    return .{ 1 / ((1 + ex[1]) * (1 + u)) * (1 - u), 1 / ((1 + ex[0]) * (1 - u)) * (1 + u) };
}

/// x / (1 + exp(-x)) at one value (a point x), its fp32 steps within e and u.
pub fn silu(x: f64, e: f64, u: f64) Iv {
    const ex = exp32(-x);
    const lo = x / ((1 + ex * (1 + e)) * (1 + u));
    const hi = x / ((1 + ex * (1 - e)) * (1 - u));
    return mul(order(lo, hi), point(1), u);
}

/// tf_gate_l2_rows' rounding point: the gated row before its norm.
pub const GatePoints = struct { gated: bool = true };

/// tf_gate_l2_rows: out = bf16(l2(t)), t = o sigmoid(g sg) (bf16 at `gated`), over the row's values; expf within e.
pub fn gateL2(o: []const f64, g: []const f64, sg: []const f64, p: GatePoints, eps: f64, sums: anytype, st: Stray, e: f64, y: []Iv, small: []f64, big: []f64) void {
    for (y, o, g, sg) |*out, ov, gv, sv| {
        const t = mul(point(gv), point(sv), st.u);
        if (std.math.isNan(t[0])) {
            out.* = @splat(std.math.nan(f64));
            continue;
        }
        out.* = bf16If(p.gated, mul(point(ov), sigmoid(t, e, st.u), st.u));
    }
    normalize(y, eps, sums, st, true, small, big);
}

/// tf_swiglu_l2_rows' rounding points: the scaled gate and up, their product.
pub const SwigluPoints = struct { gs: bool = true, us: bool = true, prod: bool = true };

/// tf_swiglu_l2_rows: out = bf16(l2(m)), m = silu(gs) us, gs = g sug, us = u suu (bf16 at the points); expf within e.
pub fn swigluL2(g: []const f64, u: []const f64, sug: []const f64, suu: []const f64, p: SwigluPoints, eps: f64, sums: anytype, st: Stray, e: f64, y: []Iv, small: []f64, big: []f64) void {
    for (y, g, u, sug, suu) |*out, gv, uv, sgv, suv| {
        const gs = bf16If(p.gs, mul(point(gv), point(sgv), st.u));
        const us = bf16If(p.us, mul(point(uv), point(suv), st.u));
        if (std.math.isNan(gs[0]) or std.math.isNan(us[0])) {
            out.* = @splat(std.math.nan(f64));
            continue;
        }
        // silu is not monotone below -1.28: its ends over gs's interval from both ends and the minimum between
        var s = silu(gs[0], e, st.u);
        const s2 = silu(gs[1], e, st.u);
        s = .{ @min(s[0], s2[0]), @max(s[1], s2[1]) };
        if (gs[0] < -1.2784645 and gs[1] > -1.2784645) s[0] = @min(s[0], -0.2784645 * (1 + 4 * st.u));
        out.* = bf16If(p.prod, mul(s, us, st.u));
    }
    normalize(y, eps, sums, st, true, small, big);
}
