//! bf16 rows times weight rows in float64, the reference for every backend's dense row kernels (cuda/dense_rows.zig).
const std = @import("std");

/// out[n] = x . w[n] for one row; tol: fp32 error of k / 32 sequential terms and a 5-step tree, plus the result's ulp.
pub fn row(x: []const f64, w: []const f64, k: usize, out: []f64, tol: []f64) void {
    for (out, tol, 0..) |*o, *t, j| {
        var s: f64 = 0;
        var mag: f64 = 0;
        for (x[0..k], w[j * k ..][0..k]) |a, b| {
            s += a * b;
            mag += @abs(a * b);
        }
        o.* = s;
        t.* = (@as(f64, @floatFromInt(k / 32 + 6)) * mag + @abs(s)) * std.math.pow(f64, 2, -24) + 1e-30;
    }
}

test "a row against hand values" {
    var out: [2]f64 = undefined;
    var tol: [2]f64 = undefined;
    row(&.{ 1, 2 }, &.{ 3, 4, -1, 0.5 }, 2, &out, &tol);
    try std.testing.expectEqualSlices(f64, &.{ 11, 0 }, &out);
    try std.testing.expect(tol[0] > 0 and tol[1] > 0);
}
