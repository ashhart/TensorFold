//! Softmax top-k routing in float64, the reference every backend's routing kernels are checked against.
const std = @import("std");

/// x . w for one router row.
pub fn dot(x: []const f64, w: []const f64) f64 {
    var t: f64 = 0;
    for (x, w) |a, b| t += a * b;
    return t;
}

/// An fp32 dot's tolerance: every product exact, at most one rounding a term of 2^-24 of the running |sum|.
pub fn dotTolerance(x: []const f64, w: []const f64) f64 {
    var t: f64 = 0;
    for (x, w) |a, b| t += @abs(a * b);
    return t * @as(f64, @floatFromInt(x.len)) * std.math.pow(f64, 2, -24) + 1e-30;
}

/// The `ids.len` largest logits (the lower id among equal ones), largest first, and weights e^(l_k - l_0) / their sum.
pub fn topk(logits: []const f64, ids: []usize, weights: []f64) void {
    for (ids, 0..) |*id, k| {
        var best: ?usize = null;
        for (logits, 0..) |v, i| {
            if (std.mem.indexOfScalar(usize, ids[0..k], i) != null) continue;
            if (best == null or v > logits[best.?]) best = i;
        }
        id.* = best.?;
    }
    var total: f64 = 0;
    for (ids) |id| total += @exp(logits[id] - logits[ids[0]]);
    for (ids, weights) |id, *w| w.* = @exp(logits[id] - logits[ids[0]]) / total;
}

test "the top picks are the largest, ties to the lower id, weights summing to one" {
    const l = [_]f64{ 1, 3, 3, -1, 2 };
    var ids: [3]usize = undefined;
    var w: [3]f64 = undefined;
    topk(&l, &ids, &w);
    try std.testing.expectEqualSlices(usize, &.{ 1, 2, 4 }, &ids);
    try std.testing.expectApproxEqAbs(@as(f64, 1), w[0] + w[1] + w[2], 1e-12);
    try std.testing.expectEqual(w[0], w[1]);
    try std.testing.expect(dotTolerance(&.{ 1, 2 }, &.{ 3, -4 }) > 0);
}
