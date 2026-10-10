//! The glue kernels' scalar math on the host, op for op as glue_math.cuh runs it: IEEE adds, muls, fmas, divs only.

const std = @import("std");

pub fn fma(a: f32, b: f32, c: f32) f32 {
    return @mulAdd(f32, a, b, c);
}

pub fn bf16ToF32(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

/// cvt.rn.bf16.f32: round to nearest even, any NaN to 0x7fff.
pub fn f32ToBf16(v: f32) u16 {
    if (std.math.isNan(v)) return 0x7fff;
    const u: u32 = @bitCast(v);
    return @intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

/// The value rounded through bf16.
pub fn rbf(v: f32) f32 {
    return bf16ToF32(f32ToBf16(v));
}

const l2e: f32 = 0x1.715476p+0; // log2(e)
const ln2_hi: f32 = 0x1.62e4p-1; // 15 significant bits: j * ln2_hi is exact for |j| <= 255
const ln2_lo: f32 = 0x1.7f7d1cp-20;
const shifter: f32 = 0x1.8p+23; // adding it rounds to an integer

/// e^x within one ulp: x = j ln2 + f, a degree-7 Taylor polynomial in f, then 2^j in two exact factors.
pub fn exp(x: f32) f32 {
    if (std.math.isNan(x)) return x;
    if (x > 104.0) return std.math.inf(f32);
    if (x < -104.0) return 0.0;
    const j = fma(x, l2e, shifter) - shifter;
    var f = fma(j, -ln2_hi, x);
    f = fma(j, -ln2_lo, f);
    var p: f32 = 0x1.a01a02p-13; // 1/7! .. 1/3!, each the nearest float
    p = fma(p, f, 0x1.6c16c2p-10);
    p = fma(p, f, 0x1.111112p-7);
    p = fma(p, f, 0x1.555556p-5);
    p = fma(p, f, 0x1.555556p-3);
    p = fma(p, f, 0.5);
    p = fma(p, f, 1.0);
    p = fma(p, f, 1.0);
    const i: i32 = @intFromFloat(j);
    const a: f32 = @bitCast(@as(u32, if (i > 0) 0x7f000000 else 0x02000000));
    const b: f32 = @bitCast(@as(u32, @intCast(if (i > 0) i << 23 else (i + 250) << 23)));
    return (p * a) * b;
}

/// ln(1 + u) for u in [0, 1], within one ulp: 2 atanh(t / (2 + t)) with the quotient carried in two floats.
pub fn log1pUnit(u: f32) f32 {
    const hi = u > 0.5;
    const t = if (hi) (u - 1.0) * 0.5 else u; // exact for u in (0.5, 1]: ln(1 + u) = ln2 + ln(1 + t)
    const s = 2.0 + t;
    const err = t - (s - 2.0); // 2 + t = s + err exactly
    const zh = t / s;
    const zl = (fma(-zh, s, t) - zh * err) / s; // t / (2 + t) = zh + zl to second order
    const z2 = zh * zh;
    var p: f32 = 0x1.745d18p-4; // 1/11 .. 1/3, each the nearest float
    p = fma(p, z2, 0x1.c71c72p-4);
    p = fma(p, z2, 0x1.24924ap-3);
    p = fma(p, z2, 0x1.99999ap-3);
    p = fma(p, z2, 0x1.555556p-2);
    const zz = zh + zh;
    const r = zz + fma(zz * z2, p, zl + zl);
    if (!hi) return r;
    return ln2_hi + (ln2_lo + r);
}

/// 1 / (1 + e^-x).
pub fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + exp(-x));
}

/// x / (1 + e^-x): SiLU with one rounding fewer than x * sigmoid(x).
pub fn silu(x: f32) f32 {
    return x / (1.0 + exp(-x));
}

/// max(v, 0) + ln(1 + e^-|v|), the dt softplus, without log(1 + e)'s loss for small e.
pub fn softplus(v: f32) f32 {
    return @max(v, 0.0) + log1pUnit(exp(-@abs(v)));
}

fn ulps(got: f32, want: f64) f64 {
    const w: f32 = @floatCast(want);
    if (w == 0 or !std.math.isFinite(w)) return if (got == w) 0 else std.math.inf(f64);
    const e = std.math.frexp(w).exponent;
    const ulp = std.math.ldexp(@as(f64, 1.0), @max(e - 24, -149));
    return @abs(@as(f64, got) - want) / ulp;
}

test "exp stays within 1 ulp of the f64 exp, including subnormal and overflowing results" {
    var worst: f64 = 0;
    var x: f32 = -103.9;
    while (x < 88.72) : (x += 0.000377) worst = @max(worst, ulps(exp(x), @exp(@as(f64, x))));
    try std.testing.expect(worst <= 1.0);
    try std.testing.expectEqual(std.math.inf(f32), exp(88.73));
    try std.testing.expectEqual(@as(f32, 1.0), exp(0.0));
    try std.testing.expectEqual(@as(f32, 1.0), exp(-0.0));
    try std.testing.expectEqual(@as(f32, 0.0), exp(-200.0));
    try std.testing.expect(std.math.isNan(exp(std.math.nan(f32))));
}

test "log1pUnit stays within 1 ulp of the f64 log1p over [0, 1]" {
    var worst: f64 = 0;
    var u: f32 = 0;
    while (u <= 1.0) : (u += 0.0000037) worst = @max(worst, ulps(log1pUnit(u), std.math.log1p(@as(f64, u))));
    for ([_]f32{ 1e-30, 1e-10, 5.9604645e-8, 3.861785e-3, 0.49958834, 0.5, 0.50000006, 1.0 }) |v| worst = @max(worst, ulps(log1pUnit(v), std.math.log1p(@as(f64, v))));
    try std.testing.expect(worst <= 1.0);
    try std.testing.expectEqual(@as(f32, 0.0), log1pUnit(0.0));
}

test "bf16 rounding is round to nearest even" {
    try std.testing.expectEqual(@as(u16, 0x3f80), f32ToBf16(1.0));
    try std.testing.expectEqual(@as(u16, 0x3f80), f32ToBf16(@bitCast(@as(u32, 0x3f808000)))); // a tie rounds to even
    try std.testing.expectEqual(@as(u16, 0x3f82), f32ToBf16(@bitCast(@as(u32, 0x3f818000))));
    try std.testing.expectEqual(@as(u16, 0x7fff), f32ToBf16(std.math.nan(f32)));
}
