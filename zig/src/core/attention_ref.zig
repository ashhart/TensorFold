//! Attention of one query head in float64, the reference for every backend's attention kernels.
const std = @import("std");

/// softmax(scale q . k_j) v over `keys`; tol 2^-7 of sum p_j |v_j| / sum p_j (bf16 weights) + a bf16 half-ulp.
pub fn head(q: []const f64, k: []const f64, v: []const f64, keys: []const usize, d: usize, scale: f64, out: []f64, tol: []f64) void {
    var m: f64 = -std.math.inf(f64);
    for (keys) |j| {
        var s: f64 = 0;
        for (0..d) |e| s += q[e] * k[j * d + e];
        m = @max(m, s * scale);
    }
    var l: f64 = 0;
    @memset(out, 0);
    @memset(tol, 0);
    for (keys) |j| {
        var s: f64 = 0;
        for (0..d) |e| s += q[e] * k[j * d + e];
        const p = @exp(s * scale - m);
        l += p;
        for (0..d) |e| {
            out[e] += p * v[j * d + e];
            tol[e] += p * @abs(v[j * d + e]);
        }
    }
    for (out, tol) |*o, *t| {
        o.* /= l;
        t.* = t.* / l * std.math.pow(f64, 2, -7) + @abs(o.*) * std.math.pow(f64, 2, -8) + 1e-6;
    }
}

test "one key returns its value; equal scores average" {
    var out: [2]f64 = undefined;
    var tol: [2]f64 = undefined;
    head(&.{ 1, 0 }, &.{ 1, 0, 1, 0 }, &.{ 2, 4, 6, 8 }, &.{1}, 2, 1, &out, &tol);
    try std.testing.expectEqualSlices(f64, &.{ 6, 8 }, &out);
    head(&.{ 0, 0 }, &.{ 1, 0, 1, 0 }, &.{ 2, 4, 6, 8 }, &.{ 0, 1 }, 2, 1, &out, &tol);
    try std.testing.expectEqualSlices(f64, &.{ 4, 6 }, &out);
}
