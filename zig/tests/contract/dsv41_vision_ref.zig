//! Float64 oracles of DeepSeek V4.1's vision kernels from the math alone: epilogues, attention, SwiGLU.
const std = @import("std");

/// erf by its series below |x| = 4 (about 1e-9 absolute) and the complementary asymptotic series past it; erf(inf) = 1.
pub fn erf(x: f64) f64 {
    if (std.math.isNan(x)) return x;
    const ax = @abs(x);
    if (ax >= 6) return std.math.sign(x);
    if (ax < 4) {
        var term = ax;
        var sum = ax;
        var n: f64 = 0;
        while (@abs(term) > 1e-18 * @abs(sum) and ax > 0) {
            n += 1;
            term *= -ax * ax / n;
            sum += term / (2 * n + 1);
        }
        return std.math.sign(x) * sum * 2 / @sqrt(std.math.pi);
    }
    var t: f64 = 1;
    var s: f64 = 1;
    for (1..14) |i| {
        t *= -@as(f64, @floatFromInt(2 * i - 1)) / (2 * ax * ax);
        s += t;
    }
    return std.math.sign(x) * (1 - @exp(-ax * ax) / (ax * @sqrt(std.math.pi)) * s);
}

pub fn gelu(x: f64) f64 {
    return 0.5 * x * (1 + erf(x * 0.7071067811865476));
}

pub fn silu(x: f64) f64 {
    return x / (1 + @exp(-x));
}

/// One query row of one head against `n` keys: the softmax-weighted values, and the weights' magnitude of the values.
pub fn attend(out: []f64, mag: []f64, q: []const f64, keys: []const f64, vals: []const f64, n: usize, hd: usize, stride: usize) void {
    var scores: [512]f64 = undefined;
    var mx: f64 = -std.math.inf(f64);
    for (0..n) |j| {
        var s: f64 = 0;
        for (0..hd) |d| s += q[d] * keys[j * stride + d];
        scores[j] = s / 8.0;
        if (std.math.isNan(scores[j])) mx = std.math.nan(f64) else if (!std.math.isNan(mx)) mx = @max(mx, scores[j]);
    }
    var sum: f64 = 0;
    for (0..n) |j| {
        scores[j] = @exp(scores[j] - mx);
        sum += scores[j];
    }
    for (0..hd) |d| {
        var acc: f64 = 0;
        var m: f64 = 0;
        for (0..n) |j| {
            acc += scores[j] * vals[j * stride + d];
            m += @abs(scores[j] * vals[j * stride + d]);
        }
        out[d] = acc / sum;
        mag[d] = m / sum;
    }
}

test "the oracle's erf and attention" {
    try std.testing.expectApproxEqAbs(@as(f64, 0.8427007929497149), erf(1), 1e-12);
    try std.testing.expectEqual(@as(f64, 1), erf(std.math.inf(f64)));
    var out: [2]f64 = undefined;
    var mag: [2]f64 = undefined;
    // two equal keys: the mean of the values
    attend(&out, &mag, &.{ 1, 0 }, &.{ 1, 0, 1, 0 }, &.{ 2, 4, 6, 8 }, 2, 2, 2);
    try std.testing.expectApproxEqAbs(@as(f64, 4), out[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 6), out[1], 1e-12);
}
