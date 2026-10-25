//! Cosmos3's Metal glue in float64 at its bf16 rounding points, and the range each output's fp32 steps allow.
const std = @import("std");
const contract = @import("contract");
const c = contract.spec;

const u: f64 = 0x1p-24;

pub fn gamma(n: f64) f64 {
    return n * u / (1 - n * u);
}

/// A value rounded to bf16 as the kernels round fp32 (to nearest even).
pub fn bf(v: f64) f64 {
    return c.fromBf16(c.toBf16(@floatCast(v)));
}

/// A literal as the kernels hold it: fp32.
fn lit(v: f64) f64 {
    return @as(f32, @floatCast(v));
}

/// exp and exp2 with fp32 functions fast (the engine's compile options): MSL's 3 + floor(2 |x|) ulp, relative.
pub fn fastExp(x: f64) f64 {
    return (3 + 2 * @abs(x)) * 0x1p-23;
}

/// metal::precise rsqrt, sin, cos and tanh: 5 ulp, above every limit the MSL table gives for them.
pub const precise: f64 = 5 * 0x1p-23;

/// An output's bf16 ends when its fp32 value lies within point +- half (fp32 rounding to bf16 is monotone).
pub fn ends(point: f64, half: f64, lo: *f64, hi: *f64) void {
    lo.* = bf(point - half);
    hi.* = bf(point + half);
}

/// cos3_norm on one row: v the row after its residual add (bf16), x = w * bf16(v * rsqrt(mean(v^2) + eps)).
pub fn rmsRow(v: []const f64, w: []const f64, eps: f32, point: []f64, lo: []f64, hi: []f64) void {
    var ss: f64 = 0;
    for (v) |x| ss += x * x;
    const inv = 1 / @sqrt(ss / @as(f64, @floatFromInt(v.len)) + eps);
    // 4 squares a thread, a simdgroup's tree, then 16 partials' tree: 14 roundings; the eps add, rsqrt, the product
    const rel = 0.5 * (gamma(14) + 2 * u) + precise + u;
    for (v, w, point, lo, hi) |x, wi, *p, *l, *h| {
        const y = x * inv;
        p.* = wi * y;
        const half = @abs(y) * rel * (1 + 0x1p-20) + 0x1p-140;
        const a = bf(wi * bf(y - half));
        const b = bf(wi * bf(y + half));
        l.* = @min(a, b);
        h.* = @max(a, b);
    }
}

/// cos3_layer_norm on one row of `v` (after its add): x = (v - mean) * rsqrt(var + eps) * w + b, fp32 statistics.
pub fn layerRow(v: []const f64, w: []const f64, b: []const f64, eps: f32, point: []f64, lo: []f64, hi: []f64) void {
    const n: f64 = @floatFromInt(v.len);
    var s: f64 = 0;
    var sa: f64 = 0;
    for (v) |x| {
        s += x;
        sa += @abs(x);
    }
    const mean = s / n;
    var q: f64 = 0;
    for (v) |x| q += (x - mean) * (x - mean);
    const inv = 1 / @sqrt(q / n + eps);
    // a thread's strided chain, a simdgroup's tree, the partials' tree; the mean's division
    const depth: f64 = @floatFromInt((v.len + 255) / 256 + 10);
    const em = gamma(depth) * sa / n + 2 * u * @abs(mean);
    // the variance about the computed mean: q + n em^2 at most, its squares and differences 3 roundings deeper
    const dv = (em * em + (gamma(depth + 3) + u) * (q / n + em * em) + u * eps) / (q / n + eps);
    const rel = 0.5 * dv / (1 - dv) + precise;
    for (v, w, b, point, lo, hi) |x, wi, bi, *p, *l, *h| {
        const d = x - mean;
        const y = d * inv * wi + bi;
        p.* = y;
        const half = (@abs(inv * wi) * (em * (1 + u) + u * @abs(d)) + @abs(d * inv * wi) * (rel + 2 * u)) * (1 + 0x1p-10) + 2 * u * @abs(y) + 0x1p-140;
        ends(y, half, l, h);
    }
}

/// cos3_bias's activations (0 none, 1 GELU tanh, 2 GELU erf) of v = bf16(y + bias): float64's value and fp32's spread.
pub fn act(kind: u32, v: f64, half: *f64) f64 {
    switch (kind) {
        0 => {
            half.* = 0;
            return v;
        },
        1 => {
            const inner = lit(0.7978845608028654) * (v + lit(0.044715) * v * v * v);
            const t = std.math.tanh(inner);
            const y = 0.5 * v * (1 + t);
            const dt = (1 - t * t) * @abs(inner) * 6 * u + precise * @abs(t) + 0x1p-140;
            half.* = (0.5 * @abs(v) * (dt + u * @abs(1 + t)) + 2 * u * @abs(y)) * (1 + 0x1p-10) + 0x1p-140;
            return y;
        },
        else => {
            const x = v * lit(0.7071067811865476);
            var e_abs: f64 = 0;
            const erf = erfAS(x, &e_abs);
            const y = 0.5 * v * (1 + erf);
            half.* = (0.5 * @abs(v) * (e_abs + u * @abs(1 + erf)) + 2 * u * @abs(y)) * (1 + 0x1p-10) + 0x1p-140;
            return y;
        },
    }
}

const as = [_]f64{ 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429 };

/// cos3_erf's Abramowitz and Stegun 7.1.26 in float64 with its fp32 constants, and fp32's absolute spread on it.
pub fn erfAS(x: f64, spread: *f64) f64 {
    const s: f64 = if (x < 0) -1 else 1;
    const a = @abs(x);
    const t = 1 / (1 + lit(0.3275911) * a);
    var poly: f64 = lit(as[4]);
    var mag: f64 = @abs(lit(as[4]));
    var slope: f64 = 4 * @abs(lit(as[4]));
    var i: usize = 4;
    while (i > 0) {
        i -= 1;
        poly = poly * t + lit(as[i]);
        mag = mag * t + @abs(lit(as[i]));
        slope = slope * t + @as(f64, @floatFromInt(i)) * @abs(lit(as[i]));
    }
    const e = @exp(-a * a);
    const q = poly * t * e;
    // Horner's 10 roundings on |terms|, t's three steps (5 roundings), exp's fast spread, the x product
    spread.* = t * e * (gamma(12) * mag + 5 * u * slope + @abs(poly) * gamma(3)) + @abs(q) * (fastExp(a * a) + gamma(3)) + 2 * u + 1.13 * u * a * 1.01;
    return s * (1 - q);
}

/// cos3_qk's rotation of pair (a, b) at rotary pair j and position pos: dst[j] and dst[j + 64] and each one's spread.
pub fn rotate(a: f64, b: f64, pos: i32, j: usize, log2_theta: f32, out: *[2]f64, half: *[2]f64) void {
    const x = -(@as(f64, @floatFromInt(2 * j)) / 128.0) * @as(f64, log2_theta);
    const inv = @exp2(x);
    const angle = @as(f64, @floatFromInt(pos)) * inv;
    // the exponent's product, exp2's fast spread, the angle's product; then precise cos and sin
    const dtheta = @abs(angle) * (@abs(x) * u * 0.7 + fastExp(x) + u) * (1 + 0x1p-10);
    const co = @cos(angle);
    const si = @sin(angle);
    out[0] = a * co - b * si;
    out[1] = b * co + a * si;
    const spread = (@abs(a) + @abs(b)) * (dtheta + precise);
    half[0] = (spread + 3 * u * (@abs(a * co) + @abs(b * si))) * (1 + 0x1p-10) + 0x1p-140;
    half[1] = (spread + 3 * u * (@abs(b * co) + @abs(a * si))) * (1 + 0x1p-10) + 0x1p-140;
}

/// A Metal attention form's fp32 arithmetic in roundings of 2^-24 (score, o's and l's chains), and P's own rounding.
pub const Form = struct { score: f64, blocks: f64, num: f64, den: f64, p_rel: f64 };

/// cos3_attn: a lane's dot chain, 32-key blocks of an online softmax, fp32 weights, then o / l.
pub fn decodeForm(d: usize, n: usize) Form {
    const blocks: f64 = @floatFromInt((n + 31) / 32);
    return .{ .score = @floatFromInt(d + 2), .blocks = blocks, .num = @as(f64, @floatFromInt(n)) + blocks + 1, .den = blocks + 9, .p_rel = 0 };
}

/// cos3_flash: 8 x 8 MMAs on bf16 tiles (any fp32 order, at most d roundings), P as bf16 into P x V, o * (1 / l).
pub fn flashForm(d: usize, n: usize) Form {
    const blocks: f64 = @floatFromInt((n + 31) / 32);
    return .{ .score = @floatFromInt(d + 2), .blocks = blocks, .num = @as(f64, @floatFromInt(n)) + blocks + 2, .den = blocks + 14, .p_rel = 0x1p-8 };
}

/// One head over n keys (k, v [n][d]) in float64: `point` its output, `half` how far fp32 lands (w, tau: n scratch).
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
        tau[j] = f.score * u * a * @abs(scale) + u * @abs(w[j]);
        if (std.math.isNan(w[j]) or w[j] > m) m = w[j];
        if (std.math.isFinite(tau[j])) dmax = @max(dmax, tau[j]);
    }
    var xmax: f64 = 0;
    for (w[0..n]) |p| if (std.math.isFinite(m - p)) {
        xmax = @max(xmax, m - p);
    };
    // every block's rescale exp(m_old - m_new): 3 ulp and a product each, and 2 ulp a unit of the drift, at most xmax
    const rescale = (4 * f.blocks + 2 * (xmax + 2 * dmax)) * 0x1p-23;
    var l: f64 = 0;
    var tmax: f64 = 0;
    var vmax: f64 = 0;
    for (w[0..n], tau[0..n]) |*p, *t| {
        const x = (m - p.*) + 2 * dmax;
        const e = t.* + u * x + fastExp(x) + rescale;
        p.* = @exp(p.* - m);
        l += p.*;
        t.* = std.math.expm1(e);
        if (std.math.isFinite(t.*)) tmax = @max(tmax, t.*);
    }
    for (v[0 .. n * d]) |x| if (std.math.isFinite(x)) {
        vmax = @max(vmax, @abs(x));
    };
    const big = tmax >= 0.5;
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
        half[e] = (e1 + en + (gamma(f.den) + 3 * u) * base) * (1 + 0x1p-20) + tiny;
    }
}

test "cosmos3 references: one key gives its value, erf's spread is small, rotation at position 0 is the identity" {
    var w: [1]f64 = undefined;
    var t: [1]f64 = undefined;
    var p: [2]f64 = undefined;
    var h: [2]f64 = undefined;
    head(&.{ 1, 0 }, &.{ 1, 0 }, &.{ 2, 4 }, 1, 2, 1, decodeForm(2, 1), &w, &t, &p, &h);
    try std.testing.expectEqualSlices(f64, &.{ 2, 4 }, &p);
    try std.testing.expect(h[0] < 0x1p-16 and h[1] < 0x1p-16); // a few fp32 roundings of |o|, far under a bf16 ulp
    var spread: f64 = 0;
    const e = erfAS(0.5, &spread);
    try std.testing.expect(@abs(e - 0.5204998778) < 2e-7 and spread < 4e-6);
    var o: [2]f64 = undefined;
    rotate(0.5, -0.25, 0, 7, 26.575, &o, &h);
    try std.testing.expectEqualSlices(f64, &.{ 0.5, -0.25 }, &o);
    try std.testing.expectEqual(@as(f64, 1.0), bf(1.00390625));
}
