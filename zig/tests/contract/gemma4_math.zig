//! Independent staged arithmetic and outward bounds for Gemma's fused prologues and GeGLU.
const std = @import("std");
const c = @import("contract").spec;
pub fn bf(x: f64) f64 {
    return c.fromBf16(c.toBf16(@floatCast(x)));
}
pub const Interval = struct {
    lo: f64,
    hi: f64,
    pub fn point(x: f64) Interval {
        return .{ .lo = x, .hi = x };
    }
    pub fn around(x: f64, e: f64) Interval {
        return .{ .lo = x - e, .hi = x + e };
    }
    pub fn add(x: Interval, y: Interval) Interval {
        return .{ .lo = x.lo + y.lo, .hi = x.hi + y.hi };
    }
    pub fn mul(x: Interval, y: Interval) Interval {
        const v = [4]f64{ x.lo * y.lo, x.lo * y.hi, x.hi * y.lo, x.hi * y.hi };
        return .{ .lo = @min(v[0], v[1], v[2], v[3]), .hi = @max(v[0], v[1], v[2], v[3]) };
    }
    pub fn rounded(x: Interval) Interval {
        const e = 0x1p-24 * @max(@abs(x.lo), @abs(x.hi)) + 0x1p-149;
        return .{ .lo = bf(x.lo - e), .hi = bf(x.hi + e) };
    }
    pub fn contains(x: Interval, y: f64) bool {
        return y >= x.lo and y <= x.hi;
    }
};
pub fn inverse(values: []const u16, eps: f64) struct { value: f64, bound: Interval } {
    var ss: f64 = 0;
    for (values) |h| {
        const v: @TypeOf(ss) = c.fromBf16(h);
        ss += v * v;
    }
    const n: f64 = @floatFromInt(values.len);
    const mean = ss / n + eps;
    const e = c.gamma(@floatFromInt(values.len + 16), 0x1p-24) * ss / n + 4 * 0x1p-24 * mean;
    const v = 1 / @sqrt(mean);
    return .{ .value = v, .bound = .{ .lo = (1 - 4 * 0x1p-24) / @sqrt(mean + e), .hi = (1 + 4 * 0x1p-24) / @sqrt(@max(mean - e, eps)) } };
}
pub fn prologue(kind: u8, aux: [5][]const u16, out: []u16, hidden: []u16, bounds: []Interval, hbounds: []Interval, eps: f64) void {
    const inv = inverse(aux[1], eps);
    for (hidden, 0..) |*h, i| {
        const y = c.fromBf16(aux[1][i]);
        const post = c.fromBf16(aux[2][i]);
        const residual = c.fromBf16(aux[0][i]);
        const b = bf(post * bf(y * inv.value));
        const scale: f64 = if (kind == 1) c.fromBf16(aux[3][0]) else 1;
        h.* = c.toBf16(@floatCast(bf(scale * bf(residual + b))));
        hbounds[i] = Interval.point(y).mul(inv.bound).rounded().mul(Interval.point(post)).rounded().add(Interval.point(residual)).rounded().mul(Interval.point(scale)).rounded();
    }
    const hinv = inverse(hidden, eps);
    var sq = Interval.point(0);
    for (hbounds) |v| {
        const lo = if (v.lo <= 0 and v.hi >= 0) 0 else @min(v.lo * v.lo, v.hi * v.hi);
        sq = sq.add(.{ .lo = lo, .hi = @max(v.lo * v.lo, v.hi * v.hi) });
    }
    const n: f64 = @floatFromInt(hidden.len);
    const gamma = c.gamma(@floatFromInt(hidden.len + 16), 0x1p-24);
    const inv2 = Interval{ .lo = (1 - 4 * 0x1p-24) / @sqrt(sq.hi / n * (1 + gamma) + eps), .hi = (1 + 4 * 0x1p-24) / @sqrt(sq.lo / n * (1 - gamma) + eps) };
    const norm = if (kind == 1) aux[4] else aux[3];
    for (out, 0..) |*x, i| {
        const w = c.fromBf16(norm[i]);
        x.* = c.toBf16(@floatCast(bf(w * bf(c.fromBf16(hidden[i]) * hinv.value))));
        bounds[i] = hbounds[i].mul(inv2).rounded().mul(Interval.point(w)).rounded();
    }
}
pub fn geglu(g: f64, u: f64) f64 {
    const x = bf(g);
    const up = bf(u);
    const cube = bf(x * x * x);
    const inner = bf(x + bf(bf(0.044715) * cube));
    const th = bf(std.math.tanh(bf(bf(0.7978845608028654) * inner)));
    return bf(bf(bf(0.5 * x) * bf(1 + th)) * up);
}
pub fn gegluBound(g: Interval, u: Interval) Interval {
    const x = g.rounded();
    const up = u.rounded();
    const cube = (Interval{ .lo = x.lo * x.lo * x.lo, .hi = x.hi * x.hi * x.hi }).rounded();
    const inner = x.add(cube.mul(Interval.point(bf(0.044715))).rounded()).rounded();
    const arg = inner.mul(Interval.point(bf(0.7978845608028654))).rounded();
    const th = (Interval{ .lo = std.math.tanh(arg.lo) - 4 * 0x1p-24, .hi = std.math.tanh(arg.hi) + 4 * 0x1p-24 }).rounded();
    return x.mul(Interval.point(0.5)).rounded().mul(th.add(Interval.point(1)).rounded()).rounded().mul(up).rounded();
}
pub fn roundedWeight(s: f64, q: u8, b: f64) f64 {
    const weight = @mulAdd(f32, @as(f32, @floatCast(s)), @as(f32, @floatFromInt(q)), @as(f32, @floatCast(b)));
    return c.fromBf16(c.toBf16(weight));
}
test "private dense rounds each weight, and GeGLU retains staged bf16 boundaries" {
    const s: f64 = c.fromBf16(c.toBf16(0.013));
    const b: f64 = c.fromBf16(c.toBf16(-0.03));
    try std.testing.expect(roundedWeight(s, 7, b) != s * 7 + b);
    try std.testing.expectEqual(@as(f64, 0), geglu(0, 9));
    try std.testing.expectEqual(@as(f64, 2), geglu(4, 0.5));
    try std.testing.expect(gegluBound(Interval.point(1), Interval.point(2)).contains(geglu(1, 2)));
}
