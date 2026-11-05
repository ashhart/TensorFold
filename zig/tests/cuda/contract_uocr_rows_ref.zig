//! The rows group's float64 parts: values at the publisher's bf16 points and their spreads under nvcc's glue flags.
const std = @import("std");
const core = @import("core");

pub const c = core.contract.spec;
pub const g = core.contract.gates;
const rows = core.contract.rows;

/// fp32's unit roundoff: one correctly rounded fp32 step moves a value by at most this share of it.
pub const U: f64 = 0x1p-24;
/// x / y and sqrtf under --prec-div and --prec-sqrt: correctly rounded, one unit.
pub const div_u: f64 = 1;
/// 1.0f / sqrtf(x): the root's unit and the division's.
pub const rsqrt_u: f64 = 2;
/// sqrtf alone (LayerNorm2d's denominator): correctly rounded.
pub const sqrt_u: f64 = 1;
/// 1 / (1 + expf(-z)) and z / (1 + expf(-z)): expf's 2 ulps (4 units), the add's 1 and the division's 1, rounded up.
pub const sig_u: f64 = 8;
/// sinf and cosf within 2 ulps of values at most 1: an absolute spread, taken at twice that.
pub const trig_abs: f64 = 0x1p-21;
/// 1 + erf(x * f32(1 / sqrt 2)): the faithful erf, expf's 2 ulps inside it, the argument's rounding and the add.
pub const erf_abs: f64 = 0x1p-22;

/// The served widths (the release's config).
pub const hidden = 1280;
pub const router_dim = 128;
pub const dense_ffn = 6848;
pub const shared_ffn = 1792;
pub const expert_ffn = 896;
pub const sam_width = 768;
pub const sam_ffn = 3072;
pub const neck_width = 256;
pub const clip_width = 1024;
pub const clip_ffn = 4096;
pub const sam_eps: f32 = 1e-6;
pub const clip_eps: f32 = 1e-5;
pub const eps: f32 = 1e-6;
pub const max_context = 131072;

/// The page2d rotary's 63 token frequencies 1 / 10000^(i / 63), as fp32 (any fp32 values serve the contract).
pub fn invTok() [63]f32 {
    var out: [63]f32 = undefined;
    for (&out, 0..) |*o, i| o.* = @floatCast(1 / std.math.pow(f64, 10000, @as(f64, @floatFromInt(i)) / 63));
    return out;
}

pub const planted = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) };

pub fn noSums(_: c.Shape) usize {
    return 0;
}

/// A value as fp32 then bf16 rounds it (to nearest even, twice): an fp32 step on exact inputs, then the bf16 point.
pub fn bf(v: f64) f64 {
    return c.fromBf16(c.toBf16(@floatCast(v)));
}

/// A value's spread through fp32 steps and bf16 rounding points.
pub const R = struct {
    lo: f64,
    hi: f64,
    pub fn pt(v: f64) R {
        return .{ .lo = v, .hi = v };
    }
    pub fn wid(a: R, e: f64) R {
        return .{ .lo = a.lo - e, .hi = a.hi + e };
    }
    /// Widened by k fp32 units of its larger end (k = 1: one correctly rounded step).
    pub fn rel(a: R, k: f64) R {
        return a.wid(k * U * @max(@abs(a.lo), @abs(a.hi)));
    }
    pub fn bf16(a: R) R {
        return .{ .lo = bf(a.lo), .hi = bf(a.hi) };
    }
    pub fn add(a: R, b: R) R {
        return .{ .lo = a.lo + b.lo, .hi = a.hi + b.hi };
    }
    pub fn neg(a: R) R {
        return .{ .lo = -a.hi, .hi = -a.lo };
    }
    pub fn mul(a: R, b: R) R {
        const p = [4]f64{ a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi };
        return .{ .lo = @min(@min(p[0], p[1]), @min(p[2], p[3])), .hi = @max(@max(p[0], p[1]), @max(p[2], p[3])) };
    }
    pub fn times(a: R, v: f64) R {
        return a.mul(pt(v));
    }
    pub fn sq(a: R) R {
        if (a.lo >= 0) return .{ .lo = a.lo * a.lo, .hi = a.hi * a.hi };
        if (a.hi <= 0) return .{ .lo = a.hi * a.hi, .hi = a.lo * a.lo };
        return .{ .lo = 0, .hi = @max(a.lo * a.lo, a.hi * a.hi) };
    }
    pub fn rsqrt(a: R) R {
        return .{ .lo = 1 / @sqrt(a.hi), .hi = 1 / @sqrt(a.lo) };
    }
    pub fn clamp(a: R, m: f64) R {
        return .{ .lo = std.math.clamp(a.lo, -m, m), .hi = std.math.clamp(a.hi, -m, m) };
    }
};

pub fn sigmoid(x: f64) f64 {
    return 1 / (1 + @exp(-x));
}

/// torch.clamp: NaN stays NaN.
pub fn clampNan(v: f64, m: f64) f64 {
    return if (std.math.isNan(v)) v else std.math.clamp(v, -m, m);
}

/// erf in float64 to about 1e-15: its Maclaurin series below 3, erfc's continued fraction (60 terms) above, 1 past 6.
pub fn erf(z: f64) f64 {
    if (std.math.isNan(z)) return z;
    if (z < 0) return -erf(-z);
    if (z > 6) return 1;
    if (z < 3) {
        var term = z;
        var s: f64 = z;
        var n: f64 = 1;
        while (n < 200) : (n += 1) {
            term *= -z * z / n;
            const t = term / (2 * n + 1);
            s += t;
            if (@abs(t) < 1e-18 * @abs(s)) break;
        }
        return s * 2 / @sqrt(std.math.pi);
    }
    var f = z;
    var k: f64 = 60;
    while (k >= 1) : (k -= 1) f = z + (k / 2) / f;
    return 1 - @exp(-z * z) / (@sqrt(std.math.pi) * f);
}

/// A simulated accumulator for gate 4, or none: float64's own sums.
pub const Sim = ?struct { acc: c.Accum, order: g.Order };

pub fn sumOf(sim: Sim, terms: []const f64) f64 {
    if (sim) |s| return rows.sumIn(s.acc, s.order, terms);
    var s: f64 = 0;
    for (terms) |v| s += v;
    return s;
}

pub fn uni(r: std.Random, lo: f32, hi: f32) u16 {
    return c.toBf16(lo + r.float(f32) * (hi - lo));
}

pub fn val(h: u16) f64 {
    return c.fromBf16(h);
}
