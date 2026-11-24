//! idol_birefnet.metal's br_erf on the host, the same fp32 steps: its error against float64's erf.
const std = @import("std");

/// The Maclaurin series' terms past x for |x| < 1: 2 / sqrt(pi) - 1, then 2 / sqrt(pi) (-1)^n / (n! (2n + 1)).
pub const series = [11]f32{
    1.283791661e-01,
    -3.761263788e-01,
    1.128379181e-01,
    -2.686617151e-02,
    5.223977845e-03,
    -8.548327023e-04,
    1.205533263e-04,
    -1.492565025e-05,
    1.646211444e-06,
    -1.636584415e-07,
    1.480719281e-08,
};
/// R(1 / x) ~ erfc(x) exp(x^2) on 1 <= x < 4: our degree-10 least-squares fit (relative error 2.3e-9 in float64).
pub const tail = [11]f32{
    3.088846279e-05,
    5.636186004e-01,
    3.914452624e-03,
    -2.898161113e-01,
    -5.404456705e-02,
    8.660649061e-01,
    -1.474402905e+00,
    1.374662042e+00,
    -7.791740298e-01,
    2.530610263e-01,
    -3.633073717e-02,
};

fn horner(c: []const f32, z: f32) f32 {
    var p = c[c.len - 1];
    var i = c.len - 1;
    while (i > 0) {
        i -= 1;
        p = @mulAdd(f32, p, z, c[i]);
    }
    return p;
}

pub fn erf(x: f32) f32 {
    if (std.math.isNan(x)) return x;
    const ax = @abs(x);
    if (ax < 1) return @mulAdd(f32, x, horner(&series, x * x), x);
    if (ax >= 4) return std.math.copysign(@as(f32, 1), x);
    const hi = ax * ax;
    const lo = @mulAdd(f32, ax, ax, -hi);
    const e = @exp(-hi) * (1 - lo);
    return std.math.copysign(1 - e * horner(&tail, 1 / ax), x);
}

const c_erf = @extern(*const fn (f64) callconv(.c) f64, .{ .name = "erf" });

test "br_erf's steps stay within 1.6 ulp of float64's erf, odd, and saturate" {
    // every 97th positive fp32 from 1e-30 to 6 (the whole range gives 1.516 ulp at 1.0072)
    var worst: f64 = 0;
    var b: u32 = @bitCast(@as(f32, 1e-30));
    while (b <= @as(u32, @bitCast(@as(f32, 6)))) : (b += 97) {
        const x: f32 = @bitCast(b);
        const w = c_erf(x);
        const wf: f32 = @floatCast(w);
        const ulp = @as(f64, std.math.nextAfter(f32, wf, 2)) - wf;
        worst = @max(worst, @abs(@as(f64, erf(x)) - w) / ulp);
        try std.testing.expectEqual(-erf(x), erf(-x));
    }
    try std.testing.expect(worst < 1.6);
    try std.testing.expectEqual(@as(f32, 1), erf(std.math.inf(f32)));
    try std.testing.expectEqual(@as(f32, -1), erf(-std.math.inf(f32)));
    try std.testing.expect(std.math.isNan(erf(std.math.nan(f32))));
}
