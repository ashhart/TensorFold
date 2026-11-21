//! Independent float64 formulas for EXL3 epilogues, with explicit activation rounding and sum replays.
const std = @import("std");
const contract = @import("contract");
const c = contract.spec;
const rows = contract.rows;
const quant = @import("dsv41_glue_ref.zig");

pub const norm: f64 = 1 / @sqrt(@as(f64, 128));
pub const plant = [4]f64{ 8, 0x1p-9, -8, -0x1p-10 };

/// Round float64 directly to bf16, avoiding a float32 double rounding at a bf16 midpoint.
pub fn bf(v: f64) f64 {
    if (!std.math.isFinite(v) or v == 0) return v;
    const a = @abs(v);
    const exponent: i32 = @as(i32, @intCast((@as(u64, @bitCast(a)) >> 52) & 2047)) - 1023;
    if (exponent > 127) return std.math.copysign(std.math.inf(f64), v);
    const step = std.math.ldexp(@as(f64, 1), @max(-133, exponent - 7));
    const q = a / step;
    const low = @floor(q);
    const rounded = low + @as(f64, if (q - low > 0.5 or
        (q - low == 0.5 and @mod(low, 2) != 0)) 1 else 0);
    const out = rounded * step;
    return std.math.copysign(if (out >= 0x1p128) std.math.inf(f64) else out, v);
}

pub fn half(v: f64) f64 {
    return @as(f16, @floatCast(v));
}

/// A Walsh coefficient dot, not the device's shuffle butterfly.
pub fn hadamard(x: []const f64, column: usize, acc: c.Accum, order: rows.Order) f64 {
    std.debug.assert(x.len == 128 and column < 128);
    var terms: [128]f64 = undefined;
    for (&terms, x, 0..) |*t, v, i| t.* = if (@popCount(i & column) % 2 == 0) v else -v;
    return rows.sumIn(acc, order, &terms) * norm;
}

pub fn rotateIn(x: []const f64, su: []const f64, out: []f64, acc: c.Accum, order: rows.Order) void {
    std.debug.assert(x.len % 128 == 0 and su.len == x.len and out.len == x.len);
    var terms: [128]f64 = undefined;
    for (0..x.len / 128) |block| {
        const first = block * 128;
        for (&terms, 0..) |*v, i| v.* = x[first + i] * su[first + i];
        for (out[first..][0..128], 0..) |*v, i| v.* = half(hadamard(&terms, i, acc, order));
    }
}

pub fn rotateOut(sums: []const f64, sv: []const f64, bias: []const f64, out: []f64, acc: c.Accum, order: rows.Order) void {
    std.debug.assert(sums.len % 128 == 0 and sv.len == sums.len and out.len == sums.len);
    for (0..sums.len / 128) |block| {
        const first = block * 128;
        for (out[first..][0..128], 0..) |*v, i| {
            const col = first + i;
            v.* = bf(hadamard(sums[first..][0..128], i, acc, order) * sv[col] +
                (if (bias.len == 0) @as(f64, 0) else bias[col]));
        }
    }
}

pub fn rms(x: []const f64, w: []const f64, eps: f64, out: []f64, acc: c.Accum, order: rows.Order) void {
    std.debug.assert(x.len <= 16384 and x.len == w.len and x.len == out.len);
    var terms: [16384]f64 = undefined;
    for (x, terms[0..x.len]) |v, *t| t.* = v * v;
    const scale = 1 / @sqrt(rows.sumIn(acc, order, terms[0..x.len]) / @as(f64, @floatFromInt(x.len)) + eps);
    for (x, w, out) |v, weight, *o| o.* = bf((v * scale) * weight);
}

pub fn rope(x: []const f64, width: usize, rotary: usize, factors: []const f32, inverse: bool, out: []f64) void {
    std.debug.assert(x.len == out.len and x.len % width == 0 and rotary <= width and rotary % 2 == 0);
    @memcpy(out, x);
    for (0..x.len / width) |head| for (0..rotary / 2) |pair| {
        const at = head * width + width - rotary + pair * 2;
        const co: f64 = factors[2 * pair];
        const si: f64 = @as(f64, factors[2 * pair + 1]) * (if (inverse) @as(f64, -1) else 1);
        const a = x[at];
        const b = x[at + 1];
        out[at] = bf(a * co - b * si);
        out[at + 1] = bf(a * si + b * co);
    };
}

/// Metal min/clamp choose the finite operand when only one argument is NaN.
pub fn swiglu(gate: f64, up: f64, weight: f64) f64 {
    const g = if (std.math.isNan(gate)) 10 else @min(gate, 10);
    const u = if (std.math.isNan(up)) -10 else std.math.clamp(up, -10, 10);
    return bf(weight * ((g / (1 + @exp(-g))) * u));
}

pub fn e4code(v: f64) u8 {
    const a = @min(@abs(v), 448);
    var best: usize = 0;
    var distance = std.math.inf(f64);
    for (quant.e4m3, 0..) |value, i| {
        const d = @abs(a - value);
        if (d < distance or (d == distance and i % 2 == 0)) {
            best = i;
            distance = d;
        }
    }
    return @as(u8, @intCast(best)) | (if (std.math.signbit(v)) @as(u8, 128) else 0);
}

pub fn packKey(x: []const f64, out: []u8) void {
    std.debug.assert(x.len % 32 == 0 and out.len == x.len + x.len / 32);
    for (0..x.len / 32) |block| {
        const first = block * 32;
        var maximum: f64 = 1e-4;
        for (x[first..][0..32]) |v| maximum = @max(maximum, @abs(v));
        const exponent = quant.ceilLog2(maximum / 448);
        const scale = std.math.ldexp(@as(f64, 1), exponent);
        for (x[first..][0..32], 0..) |v, i| out[first + i] = e4code(v / scale);
        out[x.len + block] = @intCast(exponent + 127);
    }
}

test "bf16 oracle rounds float64 midpoint neighbours without double rounding" {
    try std.testing.expectEqual(@as(f64, 1), bf(1 + 0x1p-8));
    try std.testing.expectEqual(@as(f64, 1 + 0x1p-7), bf(1 + 0x1p-8 + 0x1p-30));
    try std.testing.expectEqual(@as(f64, -1 - 0x1p-7), bf(-1 - 0x1p-8 - 0x1p-30));
    try std.testing.expectEqual(@as(f64, 0), bf(0x1p-134));
    try std.testing.expectEqual(@as(f64, 0x1p-133), bf(0x1p-134 + 0x1p-145));
    try std.testing.expect(std.math.isInf(bf(0x1.ffp127)));
    try std.testing.expect(std.math.isNan(bf(std.math.nan(f64))));
}

test "Hadamard cancellation distinguishes FP32 from half and bf16 in every replay order" {
    var x: [128]f64 = @splat(0);
    @memcpy(x[0..4], &plant);
    const want = hadamard(&x, 0, .none, .sequential);
    for ([_]rows.Order{ .sequential, .pairwise, .strided }) |order| {
        try std.testing.expectEqual(want, hadamard(&x, 0, .f32, order));
        for ([_]c.Accum{ .f16, .bf16 }) |acc| {
            try std.testing.expect(@abs(hadamard(&x, 0, acc, order) - want) > 100 * 0x1p-11 * @abs(want));
        }
    }
}
