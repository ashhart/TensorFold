//! Greedy's host step: a row's largest logit by `x > best`, the lowest column on ties (vectorized, same pick).

const std = @import("std");

const lanes = 16;
const V = @Vector(lanes, f32);

/// The column the scalar loop `best = 0; for i: if (x[i] > x[best]) best = i` returns.
pub fn argmax(vals: []const f32) usize {
    if (vals.len == 0 or std.math.isNan(vals[0])) return 0;
    var acc: V = @splat(-std.math.inf(f32));
    var i: usize = 0;
    while (i + lanes <= vals.len) : (i += lanes) {
        const x: V = vals[i..][0..lanes].*;
        acc = @select(f32, x > acc, x, acc);
    }
    var m: f32 = -std.math.inf(f32);
    const lane: [lanes]f32 = acc;
    for (lane) |x| m = if (x > m) x else m;
    for (vals[i..]) |x| m = if (x > m) x else m;
    // vals[0] is not NaN, so the maximum is a value of the row (-inf at least): its first column
    const want: V = @splat(m);
    i = 0;
    while (i + lanes <= vals.len) : (i += lanes) {
        const x: V = vals[i..][0..lanes].*;
        if (@reduce(.Or, x == want)) for (vals[i..][0..lanes], i..) |y, k| if (y == m) return k;
    }
    for (vals[i..], i..) |x, k| if (x == m) return k;
    unreachable;
}

fn scalar(vals: []const f32) usize {
    var best: usize = 0;
    for (vals, 0..) |x, i| if (x > vals[best]) {
        best = i;
    };
    return best;
}

test "greedy argmax: the scalar loop's column on ties, signed zeros, infinities and NaN" {
    const nan = std.math.nan(f32);
    const inf = std.math.inf(f32);
    const cases = [_][]const f32{
        &.{ 1, 3, 3, 2 },
        &.{ -0.0, 0.0, -0.0 },
        &.{ 0.0, -0.0 },
        &.{ -inf, -inf, -inf },
        &.{ nan, 5, 7 },
        &.{ 2, nan, 7, nan },
        &.{ -inf, nan, nan },
        &.{ inf, inf },
    };
    for (cases) |c| try std.testing.expectEqual(scalar(c), argmax(c));
    // random rows of a rank's vocabulary width with planted ties, zeros and NaNs, past the vector width's tail
    var prng = std.Random.DefaultPrng.init(4101);
    const r = prng.random();
    const row = try std.testing.allocator.alloc(f32, 64_647);
    defer std.testing.allocator.free(row);
    for (0..200) |t| {
        for (row) |*x| x.* = @floatFromInt(r.intRangeAtMost(i32, -400, 400));
        for (0..(t % 7)) |_| row[r.uintLessThan(usize, row.len)] = nan;
        if (t % 5 == 0) row[r.uintLessThan(usize, row.len)] = 401;
        if (t % 5 == 0) row[r.uintLessThan(usize, row.len)] = 401;
        if (t % 11 == 0) @memset(row, if (t % 2 == 0) 0.0 else -0.0);
        if (t % 13 == 0) row[0] = nan;
        try std.testing.expectEqual(scalar(row), argmax(row));
        try std.testing.expectEqual(scalar(row[0 .. 1 + t]), argmax(row[0 .. 1 + t]));
    }
}
