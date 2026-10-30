//! FP64 SiLU ranges carry the FP32 envelope through its BF16 store before the exact BF16 up product.
const std = @import("std");
const rounding = @import("rounding_bound.zig");
const V = rounding.Value;
pub const Range = struct { low: f64, high: f64 };

pub fn point(gate: f64, up: f64) f64 {
    const e = @exp(-@abs(gate));
    return rounding.bf64((if (gate < 0) gate * e else gate) / (1 + e)) * up;
}

pub fn interval(gate: f64, up: f64) Range {
    const e = V.exact(-@abs(gate)).exp();
    const numerator = if (gate < 0) V.exact(gate).mul(e) else V.exact(gate);
    const silu = numerator.div(V.exact(@as(f64, 1)).add(e, .none));
    const low = rounding.bf64(silu.v - silu.e) * up;
    const high = rounding.bf64(silu.v + silu.e) * up;
    // The finite fixtures' BF16 products have at most 16 significant bits and fit normal FP32 exactly.
    return .{ .low = rounding.bf64(@min(low, high)), .high = rounding.bf64(@max(low, high)) };
}

test "Qwen4 HC native contract: signed output intervals retain both BF16 storage cuts" {
    const t = std.testing;
    const ambiguous = interval(-0.00390625, 2.0625);
    try t.expectEqual(@as(f64, -0.0040283203125), point(-0.00390625, 2.0625));
    try t.expectEqual(@as(f64, -0.0040283203125), ambiguous.low);
    try t.expectEqual(@as(f64, -0.003997802734375), ambiguous.high);
    const negative_up = interval(-0.00390625, -2.0625);
    try t.expectEqual(-ambiguous.high, negative_up.low);
    try t.expectEqual(-ambiguous.low, negative_up.high);
    const unambiguous = interval(4, 1.5);
    try t.expectEqual(@as(f64, 5.875), unambiguous.low);
    try t.expectEqual(unambiguous.low, unambiguous.high);
    try t.expect(5.90625 > unambiguous.high);
    const zero = interval(4, 0);
    try t.expectEqual(@as(f64, 0), zero.low);
    try t.expectEqual(zero.low, zero.high);
}
