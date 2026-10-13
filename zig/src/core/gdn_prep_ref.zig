//! Gated DeltaNet inputs in float64, the reference every backend's GDN prep kernels are checked against.
const std = @import("std");

/// A head's conv over `taps` rows (x [taps][dk], w [dk][taps]), then SiLU -> c; err: a channel's fp32 bound.
pub fn conv(x: []const []const f64, w: []const f64, taps: usize, c: []f64, err: []f64) void {
    for (c, err, 0..) |*o, *e, i| {
        var acc: f64 = 0;
        var mag: f64 = 0;
        for (0..taps) |t| {
            acc += x[t][i] * w[i * taps + t];
            mag += @abs(x[t][i] * w[i * taps + t]);
        }
        o.* = acc / (1 + @exp(-acc));
        e.* = 1.1 * @as(f64, @floatFromInt(taps)) * std.math.pow(f64, 2, -24) * mag; // silu' is at most 1.1
    }
}

/// q (is_q) or k from a head's c: c / sqrt(mean(c^2) + eps) * (1 / dk for q, 1 / sqrt(dk) for k).
pub fn norm(c: []const f64, is_q: bool, eps: f64, out: []f64) f64 {
    var ss: f64 = 0;
    for (c) |v| ss += v * v;
    const dk: f64 = @floatFromInt(c.len);
    const f = 1 / @sqrt(ss / dk + eps) * (if (is_q) 1 / dk else 1 / @sqrt(dk));
    for (out, c) |*o, v| o.* = v * f;
    return f;
}

/// g = exp(-exp(a_log) * softplus(a + dt_bias)).
pub fn gate(a: f64, dt_bias: f64, a_log: f64) f64 {
    const x = a + dt_bias;
    const sp = if (x > 20) x else std.math.log1p(@exp(x));
    return @exp(-@exp(a_log) * sp);
}

test "conv, norms and the gate against hand values" {
    var c: [2]f64 = undefined;
    var e: [2]f64 = undefined;
    conv(&.{ &.{ 1, 2 }, &.{ 3, 4 } }, &.{ 1, 0, 0, 1 }, 2, &c, &e); // channel 0 tap 0, channel 1 tap 1
    try std.testing.expectApproxEqRel(@as(f64, 1.0 / (1.0 + @exp(-1.0))), c[0], 1e-15);
    try std.testing.expectApproxEqRel(@as(f64, 4.0 / (1.0 + @exp(-4.0))), c[1], 1e-15);
    var o: [2]f64 = undefined;
    _ = norm(&.{ 3, 4 }, false, 0, &o);
    try std.testing.expectApproxEqRel(@as(f64, 0.6), o[0], 1e-15);
    try std.testing.expectApproxEqRel(@as(f64, @exp(-@exp(0.5) * 25.0)), gate(25, 0, 0.5), 1e-15);
}
