//! Float64 pieces of the layer reference: the release's number formats, its quantisers, norms, RoPE, and stage bounds.
const std = @import("std");

/// fp32's unit roundoff: every fp32 operation's relative error is at most this.
pub const u: f64 = 0x1p-24;

pub fn bf(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

/// bf16 round to nearest even (NaN kept quiet).
pub fn toBf(v: f32) u16 {
    if (std.math.isNan(v)) return 0x7fc0;
    const b: u32 = @bitCast(v);
    return @truncate((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}

pub fn e4m3(code: u8) f32 {
    const e: i32 = (code >> 3) & 15;
    const m: f32 = @floatFromInt(code & 7);
    const v: f32 = if (e == 0) std.math.ldexp(m, -9) else std.math.ldexp(1 + m / 8, e - 7);
    return if (code & 0x80 != 0) -v else v;
}

pub fn e8m0(code: u8) f32 {
    return if (code == 0) std.math.ldexp(@as(f32, 1), -127) else std.math.ldexp(@as(f32, 1), @as(i32, code) - 127);
}

const e2m1_mag = [8]f32{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };

pub fn e2m1(code: u8) f32 {
    const v = e2m1_mag[code & 7];
    return if (code & 8 != 0) -v else v;
}

/// The nearest code to a magnitude over a sorted value table, ties to the even code: the definition, not a bit trick.
fn nearest(m: f32, values: []const f32) u8 {
    var best: usize = 0;
    for (values, 0..) |v, i| {
        const d = @abs(v - m);
        const bd = @abs(values[best] - m);
        if (d < bd or (d == bd and i % 2 == 0 and best % 2 == 1)) best = i;
    }
    return @intCast(best);
}

var e4m3_table: [127]f32 = undefined;
var e4m3_ready = false;

/// A magnitude of at most 448 to its nearest E4M3 code (0..126), ties to even.
pub fn e4m3Code(m: f32) u8 {
    if (!e4m3_ready) {
        for (&e4m3_table, 0..) |*t, i| t.* = e4m3(@intCast(i));
        e4m3_ready = true;
    }
    return nearest(m, &e4m3_table);
}

pub fn e2m1Code(m: f32) u8 {
    return nearest(m, &e2m1_mag);
}

/// ceil(log2(v)) of a positive float.
fn log2c(v: f32) i32 {
    const fr = std.math.frexp(v);
    return if (fr.significand == 0.5) fr.exponent - 1 else fr.exponent;
}

fn pow2(e: i32) f32 {
    return std.math.ldexp(@as(f32, 1), e);
}

/// act_quant's round trip of one 32-value bf16 block (E4M3 codes, power-of-two scale), back to bf16.
pub fn actq32(in: []const u16, out: []u16) void {
    var amax: f32 = 0;
    for (in) |h| amax = @max(amax, @abs(bf(h)));
    const e = log2c(@max(amax, 1e-4) * (1.0 / 448.0));
    for (in, out) |h, *o| {
        const v = bf(h);
        const q = e4m3(e4m3Code(@min(@abs(v) * pow2(-e), 448))) * pow2(e);
        o.* = toBf(if (std.math.signbit(v)) -q else q);
    }
}

/// fp4_act_quant of a 32-value block: E2M1 codes (low nibble first) and an E8M0 scale byte.
pub fn fp4pack32(in: []const u16, codes: []u8) u8 {
    var amax: f32 = 0;
    for (in) |h| amax = @max(amax, @abs(bf(h)));
    const e = log2c(@max(amax, 6 * pow2(-126)) * (1.0 / 6.0));
    for (0..16) |j| {
        var byte: u8 = 0;
        for (0..2) |k| {
            const v = bf(in[2 * j + k]);
            const code = e2m1Code(@min(@abs(v) * pow2(-e), 6)) | @as(u8, if (std.math.signbit(v)) 8 else 0);
            byte |= code << @intCast(4 * k);
        }
        codes[j] = byte;
    }
    return @intCast(e + 127);
}

/// The compressed KV's FP4 round trip of a 16-value block (E4M3 scale), back to bf16.
pub fn fp4rt16(in: []const u16, out: []u16) void {
    var amax: f32 = 0;
    for (in) |h| amax = @max(amax, @abs(bf(h)));
    const sv = @max(amax, 6 * pow2(-9)) / 6.0;
    const s = e4m3(e4m3Code(@min(sv, 448)));
    for (in, out) |h, *o| {
        const v = bf(h);
        const q = e2m1(e2m1Code(@min(@abs(v / s), 6))) * s;
        o.* = toBf(if (std.math.signbit(v)) -q else q);
    }
}

/// Half a bf16 unit in the last place at |v|: what rounding a nearby fp32 value to bf16 adds at most.
pub fn halfUlp(v: f64) f64 {
    const a = @abs(v);
    if (a < 0x1p-126) return 0x1p-134;
    return std.math.ldexp(@as(f64, 1), std.math.frexp(a).exponent - 1 - 8);
}

/// A bf16 result's bound: `depth` fp32 roundings on terms of magnitude sum `mag`, then the bf16 rounding.
pub fn bfBound(want: f64, mag: f64, depth: f64) f64 {
    return halfUlp(want + std.math.sign(want) * depth * u * mag) + depth * u * mag;
}

/// An fp32 result's bound.
pub fn fBound(want: f64, mag: f64, depth: f64) f64 {
    return depth * u * mag + u * @abs(want);
}

/// One E4M3 step at |v| (the act_quant grid's spacing), doubled for a scale that crosses a binade.
pub fn quantStep(v: f64) f64 {
    const a = @abs(v);
    if (a == 0) return 0x1p-20;
    return std.math.ldexp(@as(f64, 1), std.math.frexp(a).exponent - 1 - 2);
}

/// A stage's worst error relative to its bound, and where it was.
pub const Stage = struct {
    name: []const u8,
    n: usize = 0,
    bad: usize = 0,
    exact: usize = 0,
    worst_err: f64 = 0,
    worst_bound: f64 = 0,
    worst_ratio: f64 = 0,
    row: usize = 0,
    col: usize = 0,

    pub fn add(s: *Stage, got: f64, want: f64, bound: f64, row: usize, col: usize) void {
        s.n += 1;
        const err = @abs(got - want);
        if (got == want or (std.math.isNan(got) and std.math.isNan(want))) {
            s.exact += 1;
            return;
        }
        const ratio = if (bound > 0) err / bound else std.math.inf(f64);
        if (!(err <= bound)) s.bad += 1;
        if (!(ratio <= s.worst_ratio)) {
            s.worst_ratio = ratio;
            s.worst_err = err;
            s.worst_bound = bound;
            s.row = row;
            s.col = col;
        }
    }

    /// An exact stage: anything but equality is a failure.
    pub fn same(s: *Stage, eq: bool, row: usize, col: usize) void {
        s.n += 1;
        if (eq) {
            s.exact += 1;
        } else {
            if (s.bad == 0) {
                s.row = row;
                s.col = col;
            }
            s.bad += 1;
            s.worst_ratio = std.math.inf(f64);
        }
    }

    pub fn print(s: *const Stage, layer: usize) void {
        const word = if (s.bad == 0) "PASS" else "FAIL";
        std.debug.print("{s} dsv41 metal layer {d} {s}: {d} values, {d} bit-equal, worst |err| {e:.3} vs bound {e:.3} (x{d:.3}) at row {d} col {d}\n", .{ word, layer, s.name, s.n, s.exact, s.worst_err, s.worst_bound, s.worst_ratio, s.row, s.col });
    }
};

test "the quantisers' tables round to nearest even" {
    try std.testing.expectEqual(@as(u8, 0), e2m1Code(0.25));
    try std.testing.expectEqual(@as(u8, 2), e2m1Code(0.75));
    try std.testing.expectEqual(@as(u8, 2), e2m1Code(1.25));
    try std.testing.expectEqual(@as(u8, 6), e2m1Code(5));
    try std.testing.expectEqual(@as(u8, 0x7e), e4m3Code(448));
    try std.testing.expectEqual(@as(f32, 0.0625), e4m3(e4m3Code(0.0625)));
    try std.testing.expectEqual(@as(i32, 3), log2c(5));
    try std.testing.expectEqual(@as(i32, 2), log2c(4));
}
