//! RMS norm rows in float64, the reference every backend's norm kernels are checked against (CUDA: cuda/norms.zig).
const std = @import("std");

/// y = x / sqrt(mean(x^2) + eps) * w for one row (w null: no weight).
pub fn rms(x: []const f64, w: ?[]const f64, eps: f64, y: []f64) void {
    var ss: f64 = 0;
    for (x) |v| ss += v * v;
    const inv = 1 / @sqrt(ss / @as(f64, @floatFromInt(x.len)) + eps);
    for (x, y, 0..) |v, *o, i| o.* = v * inv * (if (w) |ww| ww[i] else 1);
}

/// The residual step for one row: res += rms(x) * w1 (w1 null: res += x), then y = rms(res) * w2.
pub fn addRms(x: []const f64, res: []f64, w1: ?[]const f64, w2: []const f64, eps: f64, y: []f64) void {
    if (w1) |w| {
        rms(x, w, eps, y); // y as scratch: rms(x) * w1
        for (res, y) |*r, v| r.* += v;
    } else for (res, x) |*r, v| {
        r.* += v;
    }
    rms(res, w2, eps, y);
}

/// One once-rounded output's tolerance: half an ulp of its kind at |want| plus 2^-20 of |want| for fp32 arithmetic.
pub fn tolerance(want: f64, kind: Kind) f64 {
    const half_ulp: f64 = switch (kind) {
        .f32 => std.math.pow(f64, 2, -24),
        .f16 => std.math.pow(f64, 2, -11),
        .bf16 => std.math.pow(f64, 2, -8),
    };
    return @abs(want) * (half_ulp + std.math.pow(f64, 2, -20)) + 1e-30;
}

/// The residual step's bounds: res_i 2^-20 of |res0_i| + |add_i| (add: what it adds), y_i that error carried on.
pub fn addRmsBounds(res0: []const f64, add: []const f64, w2: []const f64, y: []const f64, res: []const f64, kind: Kind, res_tol: []f64, y_tol: []f64) void {
    var ss: f64 = 0;
    for (res) |v| ss += v * v;
    const inv = 1 / @sqrt(ss / @as(f64, @floatFromInt(res.len)));
    for (res0, add, w2, y, res_tol, y_tol) |r0, a, w, yy, *rt, *yt| {
        rt.* = (@abs(r0) + @abs(a)) * std.math.pow(f64, 2, -20) + 1e-30;
        yt.* = tolerance(yy, kind) + rt.* * inv * @abs(w);
    }
}

/// y's sums over each 64 values.
pub fn groupSums(y: []const f64, xs: []f64) void {
    for (xs, 0..) |*o, j| {
        var t: f64 = 0;
        for (y[64 * j ..][0..64]) |v| t += v;
        o.* = t;
    }
}

/// A sum of 64 fp32 values' tolerance: 64 roundings of at most 2^-24 of the running |sum| <= sum |y|.
pub fn groupSumTolerance(y: []const f64) f64 {
    var t: f64 = 0;
    for (y) |v| t += @abs(v);
    return t * 64 * std.math.pow(f64, 2, -24) + 1e-30;
}

/// silu(z) * rms(y) * w for one head.
pub fn gatedRms(y: []const f64, z: []const f64, w: []const f64, eps: f64, out: []f64) void {
    rms(y, w, eps, out);
    for (out, z) |*o, zz| o.* *= zz / (1 + @exp(-zz));
}

/// The element kinds the kernels read and write, numbered as they are (hip/ops/common.hpp: 0 fp32, 1 fp16, 2 bf16).
pub const Kind = enum(u8) { f32 = 0, f16 = 1, bf16 = 2 };

test "a row's norm and the residual step against hand values" {
    var y: [4]f64 = undefined;
    rms(&.{ 1, -1, 1, -1 }, &.{ 2, 2, 2, 2 }, 0, &y);
    try std.testing.expectEqualSlices(f64, &.{ 2, -2, 2, -2 }, &y);
    rms(&.{ 3, 4 }, null, 0, y[0..2]);
    try std.testing.expectApproxEqRel(@as(f64, 3.0 / @sqrt(12.5)), y[0], 1e-15);
    var res = [_]f64{ 1, 1, 1, 1 };
    addRms(&.{ 1, -1, 1, -1 }, &res, null, &.{ 1, 1, 1, 1 }, 0, &y);
    try std.testing.expectEqualSlices(f64, &.{ 2, 0, 2, 0 }, &res);
    try std.testing.expectApproxEqRel(@as(f64, @sqrt(2.0)), y[0], 1e-15);
    res = .{ 0, 0, 0, 0 };
    addRms(&.{ 2, -2, 2, -2 }, &res, &.{ 3, 3, 3, 3 }, &.{ 1, 1, 1, 1 }, 0, &y); // res = rms(x) * 3 = +-3
    try std.testing.expectEqualSlices(f64, &.{ 3, -3, 3, -3 }, &res);
    try std.testing.expect(tolerance(1, .bf16) > tolerance(1, .f32));
}
