//! A head's attention in float64 at Metal's rounding points (the output's), and the range window_rows.cu's fp32 allows.
const std = @import("std");
const bfr = @import("bf16_round.zig");

const u: f64 = 0x1p-24;

fn gamma(n: f64) f64 {
    return n * u / (1 - n * u);
}

/// A form's fp32 arithmetic, as the range needs it; each count is in roundings of 2^-24.
pub const Form = struct {
    /// A score's spread over sum_e |q_e k_e|: lanes' fma chains and a xor tree, or mma steps and slice sums.
    score: f64,
    /// __expf (2 + 1.173 |x| ulps), else expf (2 ulps).
    fast_exp: bool,
    /// The merge's expf(m_c - M) of each chunk's weights (decode), on top of the weights' own.
    chunk_exp: bool,
    /// P's error into P x V, relative: bf16 2^-8, two bf16 halves 2^-16, fp32 0.
    p_rel: f64,
    /// The chains of o's terms p_j v_j and of l's terms p_j, as roundings of their terms' magnitude.
    num: f64,
    den: f64,
};

/// An mma k16 step's error in 2^-24 of |acc| + sum |a b|: 17 terms cut at the largest's ulp, then rounded down.
pub const mma_step = 36;

/// window_rows.cu's decode at head dim d over n keys: lanes' chains of d / 32 then a 5-level tree, 256-key chunks.
pub fn decode(d: usize, n: usize) Form {
    const chunks: f64 = @floatFromInt(n / 256 + 2);
    return .{ .score = @floatFromInt(d / 32 + 5), .fast_exp = false, .chunk_exp = true, .p_rel = 0, .num = 264 + chunks + 1, .den = 13 + chunks + 1 };
}

/// Prompt rows over n keys: 8 mma steps a 128-dim slice, slices in order, an fma a 32-key tile; P32: 4 P x V steps.
pub fn prompt(d: usize, n: usize, p32: bool) Form {
    const tiles: f64 = @floatFromInt(n / 32 + 2);
    const steps: f64 = if (p32) 4 else 2;
    return .{ .score = 8 * mma_step + @as(f64, @floatFromInt(d / 128)), .fast_exp = true, .chunk_exp = false, .p_rel = if (p32) 0x1p-16 else 0x1p-8, .num = mma_step * steps + tiles + 2, .den = tiles + 6 };
}

/// A head over n keys (k, v [n][d]): `point` float64's output, `half` how far fp32 may land (w, tau: n scratch).
pub fn head(q: []const f64, k: []const f64, v: []const f64, n: usize, d: usize, scale: f64, f: Form, w: []f64, tau: []f64, point: []f64, half: []f64) void {
    var m: f64 = -std.math.inf(f64);
    var dmax: f64 = 0;
    for (0..n) |j| {
        var s: f64 = 0;
        var a: f64 = 0;
        for (0..d) |e| {
            s += q[e] * k[j * d + e];
            a += @abs(q[e] * k[j * d + e]);
        }
        w[j] = s * scale;
        tau[j] = f.score * u * a * @abs(scale) + u * @abs(w[j]); // the score's spread; below, the weight's
        m = @max(m, w[j]);
        if (std.math.isFinite(tau[j])) dmax = @max(dmax, tau[j]);
    }
    var l: f64 = 0;
    var tmax: f64 = 0;
    var vmax: f64 = 0;
    for (w[0..n], tau[0..n]) |*p, *t| {
        const x = (m - p.*) + 2 * dmax; // |the computed s_j - max| at most
        var e = t.* + u * x + if (f.fast_exp) (3 + 1.2 * x) * 0x1p-23 else 0x1p-22;
        if (f.chunk_exp) e += 0x1p-22 + u * x;
        p.* = @exp(p.* - m);
        l += p.*;
        t.* = std.math.expm1(e);
        if (std.math.isFinite(t.*)) tmax = @max(tmax, t.*);
    }
    for (v[0 .. n * d]) |x| if (std.math.isFinite(x)) {
        vmax = @max(vmax, @abs(x));
    };
    const big = tmax >= 0.5; // scores too far apart for the spread's first order: any finite value
    const kappa = (1 + tmax) / (1 - tmax);
    const tiny = @as(f64, @floatFromInt(n + 8)) * 0x1p-120 * (1 + vmax);
    for (0..d) |e| {
        var o: f64 = 0;
        for (0..n) |j| o += w[j] * v[j * d + e];
        o /= l;
        point[e] = o;
        if (!std.math.isFinite(o) or big) {
            half[e] = if (big) std.math.inf(f64) else 0;
            continue;
        }
        var r: f64 = 0;
        var dv: f64 = 0;
        for (0..n) |j| {
            const x = v[j * d + e];
            r += w[j] * @abs(x);
            dv += w[j] * tau[j] * @abs(x - o);
        }
        r /= l;
        dv /= l;
        const e1 = dv / (1 - tmax);
        const en = (gamma(f.num) + f.p_rel) * r * kappa;
        const base = @abs(o) + e1 + en;
        half[e] = (e1 + en + (gamma(f.den) + u) * base) * (1 + 0x1p-20) + tiny;
    }
}

/// What a check saw: outputs whose range holds one bf16 value, several, and outputs non-finite in float64.
pub const Stats = struct { decided: usize = 0, ranged: usize = 0, nonfinite: usize = 0 };

/// bf16 bits in value order, -0 just below +0 (NaN bits fall outside every finite range).
fn key(b: u16) i32 {
    return if (b & 0x8000 != 0) -@as(i32, b & 0x7fff) - 1 else b;
}

fn bitsOf(v: f64) u16 {
    return bfr.bits(@floatCast(bfr.bf16(v)));
}

/// bf16 bits `got` in point +- half's ends rounded to bf16, in value order (-0 < +0); NaN for NaN; infinities by bits.
pub fn within(point: f64, half: f64, got: u16, stats: *Stats) bool {
    const nan = got & 0x7f80 == 0x7f80 and got & 0x7f != 0;
    if (std.math.isNan(point)) {
        stats.nonfinite += 1;
        return nan;
    }
    if (std.math.isInf(point)) {
        stats.nonfinite += 1;
        return got == bitsOf(point);
    }
    const lo = bitsOf(point - half);
    const hi = bitsOf(point + half);
    if (lo == hi) stats.decided += 1 else stats.ranged += 1;
    return !nan and key(lo) <= key(got) and key(got) <= key(hi);
}

test "one key gives its value; a range's ends, -0 below +0, NaN only for NaN, infinities by their bits" {
    var w: [2]f64 = undefined;
    var t: [2]f64 = undefined;
    var pt: [2]f64 = undefined;
    var hf: [2]f64 = undefined;
    head(&.{ 1, 0 }, &.{ 1, 0, 1, 0 }, &.{ 2, 4, 6, 8 }, 1, 2, 1, decode(256, 1), &w, &t, &pt, &hf);
    try std.testing.expectEqualSlices(f64, &.{ 2, 4 }, &pt);
    var st: Stats = .{};
    try std.testing.expect(within(2, hf[0], 0x4000, &st)); // 2.0
    try std.testing.expect(!within(2, hf[0], 0x4001, &st));
    try std.testing.expect(!within(2, hf[0], 0x0000, &st));
    try std.testing.expectEqual(@as(usize, 3), st.decided);
    // a range across zero holds both zeros; one below zero holds -0 and not +0
    try std.testing.expect(within(0, 0x1p-140, 0x8000, &st));
    try std.testing.expect(within(0, 0x1p-140, 0x0000, &st));
    try std.testing.expect(within(-0x1p-140, 0x1p-142, 0x8000, &st));
    try std.testing.expect(!within(-0x1p-140, 0x1p-142, 0x0000, &st));
    // NaN only for NaN (either sign), infinities by their bits
    try std.testing.expect(within(std.math.nan(f64), 0, 0xffc1, &st));
    try std.testing.expect(!within(std.math.nan(f64), 0, 0x7f80, &st));
    try std.testing.expect(!within(1e300, 1e300, 0x7fc0, &st));
    try std.testing.expect(within(-std.math.inf(f64), 0, 0xff80, &st));
    try std.testing.expect(!within(-std.math.inf(f64), 0, 0x7f80, &st));
    // a range whose upper end rounds to +Inf holds +Inf, and the largest finite, but not zero
    try std.testing.expect(within(3.3e38, 1e37, 0x7f80, &st));
    try std.testing.expect(within(3.3e38, 1e37, 0x7f7f, &st));
    try std.testing.expect(!within(3.3e38, 1e37, 0x0000, &st));
}

test "a weight's spread moves an output by about its share of |v - o|" {
    var w: [2]f64 = undefined;
    var t: [2]f64 = undefined;
    var pt: [1]f64 = undefined;
    var hf: [1]f64 = undefined;
    // two keys of equal score: o is the mean; the range grows with the scores' magnitude
    head(&.{1}, &.{ 3, 3 }, &.{ 0, 1 }, 2, 1, 1, prompt(256, 2, true), &w, &t, &pt, &hf);
    try std.testing.expectEqual(@as(f64, 0.5), pt[0]);
    const small = hf[0];
    head(&.{1}, &.{ 30, 30 }, &.{ 0, 1 }, 2, 1, 1, prompt(256, 2, true), &w, &t, &pt, &hf);
    try std.testing.expect(hf[0] > small and hf[0] < 0x1p-9);
}
