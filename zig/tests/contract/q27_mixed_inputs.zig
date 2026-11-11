//! Exact FP32 squares just below its exponent limit; FP16 and BF16 square sums overflow in all three orders.
const std = @import("std");
const ct = @import("contract");
pub fn stress(x: []u16, w: []u16) void {
    std.debug.assert(x.len % 256 == 0);
    @memset(x, 0x5d80); // 2^60, whose square is exactly 2^120
    for (0..x.len / 256) |row| x[row * 256 + 255] = 0x5d60; // (7/8) * 2^60
    @memset(w, 0x3f80);
}
test "qwen27 mixed: exact square sums distinguish both reduced accumulators in every order" {
    var x: [256]u16 = undefined;
    var w: [256]u16 = undefined;
    stress(&x, &w);
    var terms: [256]f64 = undefined;
    for (&terms, x) |*t, v| {
        const value: f64 = ct.spec.fromBf16(v);
        t.* = value * value;
    }
    const exact = @as(f64, 255.765625) * 0x1p120;
    for (std.enums.values(ct.gates.Order)) |order| {
        try std.testing.expectEqual(exact, ct.rows.sumIn(.f32, order, &terms));
        for ([_]ct.spec.Accum{ .f16, .bf16 }) |acc|
            try std.testing.expect(std.math.isInf(ct.rows.sumIn(acc, order, &terms)));
    }
}
