//! Scalar f64 values carry absolute F32 rounding envelopes through nonlinear nodes and exact BF16 storage cuts.
const std = @import("std");
const c = @import("core").contract.spec;
const u = 0x1p-24;
fn power(exponent: i32) f64 {
    return @bitCast(@as(u64, @intCast(exponent + 1023)) << 52);
}
pub fn bf64(value: f64) f64 {
    if (!std.math.isFinite(value) or value == 0) return value;
    const x = @abs(value);
    const bits: u64 = @bitCast(x);
    const exponent = @as(i32, @intCast((bits >> 52) & 2047)) - 1023;
    if (exponent > 127) return std.math.copysign(std.math.inf(f64), value);
    const step = @max(@as(i32, -133), exponent - 7);
    const scaled = x * power(-step);
    const low = @floor(scaled);
    const rest = scaled - low;
    const odd = @as(u64, @intFromFloat(low)) & 1 != 0;
    const rounded = (low + @as(f64, if (rest > 0.5 or (rest == 0.5 and odd)) 1 else 0)) * power(step);
    return std.math.copysign(if (rounded >= power(128)) std.math.inf(f64) else rounded, value);
}
fn reduced(acc: c.Accum, x: f64) f64 {
    return switch (acc) {
        .none => x,
        .f32 => @as(f32, @floatCast(x)),
        .f16 => @as(f16, @floatCast(x)),
        .bf16 => bf64(x),
    };
}
pub const Value = struct {
    v: f64,
    e: f64 = 0,
    pub fn exact(x: anytype) Value {
        return .{ .v = @floatCast(x) };
    }
    fn envelope(v: f64, error_before_round: f64, units: f64) Value {
        if (!std.math.isFinite(v)) return .{ .v = v };
        return .{ .v = v, .e = error_before_round + units * u * (@abs(v) + error_before_round) + 0x1p-149 };
    }
    pub fn add(a: Value, b: Value, acc: c.Accum) Value {
        const v = reduced(acc, reduced(acc, a.v) + reduced(acc, b.v));
        return envelope(v, a.e + b.e, 1);
    }
    pub fn neg(a: Value) Value {
        return .{ .v = -a.v, .e = a.e };
    }
    pub fn mul(a: Value, b: Value) Value {
        return envelope(a.v * b.v, @abs(a.v) * b.e + @abs(b.v) * a.e + a.e * b.e, 1);
    }
    pub fn div(a: Value, b: Value) Value {
        const v = a.v / b.v;
        const bottom = @abs(b.v) - b.e;
        return envelope(v, if (bottom > 0) (a.e + @abs(v) * b.e) / bottom else std.math.inf(f64), 1);
    }
    pub fn exp(a: Value) Value {
        const v = @exp(a.v);
        return envelope(v, @max(v - @exp(a.v - a.e), @exp(a.v + a.e) - v), 4);
    }
    pub fn log1p(a: Value) Value {
        const v = @log(1 + a.v);
        const bottom = 1 + a.v - a.e;
        return envelope(v, if (bottom > 0) a.e / bottom else std.math.inf(f64), 8);
    }
    pub fn rsqrt(a: Value) Value {
        const v = 1 / @sqrt(a.v);
        return envelope(v, @max(v - 1 / @sqrt(a.v + a.e), 1 / @sqrt(a.v - a.e) - v), 4);
    }
    pub fn bf(a: Value) Value {
        const v = bf64(a.v);
        if (!std.math.isFinite(v)) return .{ .v = v };
        return .{ .v = v, .e = @max(@abs(bf64(a.v - a.e) - v), @abs(bf64(a.v + a.e) - v)) };
    }
    pub fn sig(a: Value) Value {
        return exact(@as(f64, 1)).div(exact(@as(f64, 1)).add(a.neg().exp(), .none));
    }
    pub fn silu(a: Value) Value {
        return a.mul(a.sig());
    }
};
test "Qwen4 rounding bounds: direct binary64 BF16 ties and explicit nonlinear envelopes" {
    const t = std.testing;
    try t.expectEqual(@as(f64, 1), bf64(1 + 0x1p-8));
    try t.expectEqual(@as(f64, 1 + 0x1p-7), bf64(1 + 0x1p-8 + 0x1p-40));
    try t.expectEqual(@as(f64, 0), bf64(0x1p-134));
    try t.expectEqual(@as(f64, 0x1p-133), bf64(0x1p-134 + 0x1p-150));
    const v = Value.exact(@as(f64, 0.352294921875)).bf().silu().bf();
    try t.expectEqual(@as(f64, 0.2060546875), v.v);
    try t.expectEqual(@as(f64, 0), v.e);
}
