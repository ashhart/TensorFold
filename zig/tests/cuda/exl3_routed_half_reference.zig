//! Direct binary64 Walsh sums over independently rounded FP16 products, without the GPU's butterfly indexing.
const std = @import("std");
pub const scale: f64 = @as(f32, 0.08838834764831845);

pub fn half(x: f64) f64 {
    return @as(f16, @floatCast(x));
}

pub fn block(x: []const f16, suh: []const f16, want: []f64, mag: []f64) void {
    var p: [128]f64 = undefined;
    var magnitude: f64 = 0;
    for (&p, x, suh) |*v, a, b| {
        v.* = half(@as(f64, a) * @as(f64, b));
        magnitude += @abs(v.*);
    }
    for (want, mag, 0..) |*w, *m, row| {
        var total: f64 = 0;
        for (p, 0..) |v, col| total += if (@popCount(row & col) % 2 == 0) v else -v;
        w.* = total * scale;
        m.* = magnitude * scale;
    }
}

test "a half product cancellation witnesses the difference from widened multiplication" {
    var x: [128]f16 = @splat(0);
    var s: [128]f16 = @splat(1);
    x[0] = 1 + 0x1p-10;
    s[0] = 1 + 0x1p-10;
    x[1] = 1;
    s[1] = 1 + 0x1p-9;
    var want: [128]f64 = undefined;
    var mag: [128]f64 = undefined;
    block(&x, &s, &want, &mag);
    try std.testing.expectEqual(@as(f64, 0), want[1]);
    try std.testing.expect(half((@as(f64, x[0]) * s[0] - @as(f64, x[1]) * s[1]) * scale) != 0);
}

test "direct Walsh reference preserves an impulse and every exceptional output class" {
    var x: [128]f16 = @splat(0);
    const s: [128]f16 = @splat(1);
    var want: [128]f64 = undefined;
    var mag: [128]f64 = undefined;
    x[0] = 8;
    block(&x, &s, &want, &mag);
    for (want) |v| try std.testing.expectEqual(8 * scale, v);
    x[0] = std.math.inf(f16);
    block(&x, &s, &want, &mag);
    for (want) |v| try std.testing.expectEqual(std.math.inf(f64), v);
    x[0] = std.math.nan(f16);
    block(&x, &s, &want, &mag);
    for (want) |v| try std.testing.expect(std.math.isNan(v));
}
